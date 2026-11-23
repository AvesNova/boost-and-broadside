# Training recipe: BC, weight averaging, KL stop, advantage scale, loss balance, entropy

Literature and code review, 2026-11-23, for the learner in `lean/learn` (branch core). It
answers docs/rewrite/author-review.md items 7.1 to 7.5 and the entropy question in the
author's answer to 7.5 (docs/rewrite/core-decisions.md, "Answers from the author review,
follow-ups and sections 7 to 8"). Minibatch count, advantage timing and per-update against
per-minibatch normalisation are in docs/rewrite/literature-ppo-targets-minibatches.md and are
not repeated here.

Each claim is labelled **measured** (a source ran an experiment), **default** (a library or
system uses it, without a published ablation), or **inference** (mine). For each source I
note its year and how close it is to this project: a 4v4 multi-agent game, a recurrent
policy of about 2M parameters, PPO with about 1M environment ticks per update, 32
minibatches, 4 epochs (128 Adam steps per update), five reward levels each with a 64-bin
HL-Gauss critic, a next-state head and a hit head, a scripted teacher, half self-play and
half a PFSP league.

The project's current numbers (`lean/learn/config.py`): entropy 0.003 on the joint entropy
of one 42-way action (`:58`, `update.py:92`); value 0.05 on the mean HL-Gauss
cross-entropy over levels (`:59`, `update.py:95-98`); next-state 0.1 on a squared error
(`:60`, `update.py:104-105`; decided to become HL-Gauss cross-entropy); hit 0.05 (`:63`); BC
2.0 x max(0, 1 - w / 0.45) on the teacher's sampled action (`:61-62`, `:118-120`,
`update.py:93`); advantages summed over levels, divided by their standard deviation over the
update, mean not subtracted (`returns.py:46-66`); no KL stop, approximate KL and clip
fraction logged (`update.py:121-122`); Adam epsilon 1e-5, warm-up 10 updates, then constant
3e-4 (`:54-56`, `:123-125`); no weight averaging.

Library code was read on 2026-11-23 at these commits: Stable-Baselines3 `0cdcd200`, CleanRL
`fe8d8a03`, Spinning Up `038665d6`, rsl_rl `4c9900fe`, Sample Factory `7a837983`, rl_games
`81924de5`, RLlib at branch `releases/2.40.0`, PufferLib at branch `3.0`.

## Summary

| Question | Recommendation | Evidence |
|---|---|---|
| 7.1 BC | A BC stage (full teacher distribution, on the student's own states, critics trained too), then RL from its averaged weights with a cross-entropy-to-teacher term that decays geometrically and is latched off once the measured win rate against the scripted agent passes the gate | Moderate. Several large systems keep a decaying or permanent KL to the cloned or teacher policy (AlphaStar, VPT, kickstarting, DAPG); kickstarting measured that a constant weight plateaus at the teacher. No source compares a stage with a gated in-PPO term head to head |
| 7.2 Weight averaging | Keep an EMA and a uniform average as side copies (never fed back into training); rate them against the live weights; hand BC's average to RL | Strong for BC (supervised evidence). Moderate for RL as a deployed policy (Nikishin 2018, Generals.io 2026, v1's own measurement). "A uniform average from the end of warm-up instead of a decay" is untested in RL; in LLM pre-training averaging narrows but does not close the gap to a cooldown |
| 7.3 KL early stop | Not a default in the most used libraries: monitor and warn (epoch-mean approximate KL above 0.02, clip fraction, advantage std). A guard-rail stop at 0.02 is cheap if wanted | Mixed. Measured to help PPO on MuJoCo with many epochs; default off in SB3, CleanRL, Sample Factory; on in Spinning Up; rsl_rl adapts the learning rate by KL; RLlib uses an adaptive penalty |
| 7.4 Advantage scale | Keep the standard deviation over the update (scale only); log each level's share of the summed advantage's variance; per-level normalisation as an experiment arm | Weak to moderate. Sullivan et al. 2023 measured DreamerV3's percentile scaling in PPO: no general gain, best combined with standard normalisation. Per-reward normalisation before summing is measured only in LLM GRPO (GDPO 2026) |
| 7.5 Loss balance | Fixed coefficients on scale-free terms (normalised advantages; every other output a cross-entropy in nats); stop the gradient-share loop; gradient shares and cosines as diagnostics only; a doubtful term gets an on/off arm, not a retune | Moderate. Adaptive balancers (GradNorm, uncertainty weighting, PCGrad) were not better than tuned fixed weights in the two large comparisons (Kurin et al. 2022, Xin et al. 2022); DreamerV3 runs fixed coefficients across 150+ tasks because each term is normalised |
| Entropy | Keep a constant bonus on the joint 42-way entropy, in nats; keep 0.003 (inside the common 0.001 to 0.01); monitor H / ln 42; an adaptive (target-entropy) coefficient only if entropy collapses | Weak. Measured as of little benefit on MuJoCo (Andrychowicz et al. 2021); defaults range 0 to 0.01; adaptive coefficients are measured in SAC and LLM RL, not in PPO games |

## 1. Behaviour cloning then RL (item 7.1)

### 1.1 What the sources did

| Source (year) | What | How the teacher signal ends | Label |
|---|---|---|---|
| AlphaStar, Vinyals et al. 2019, Nature | Supervised stage on human replays; RL "combined with a KL loss towards the supervised agent", quoted as there "to aid in exploration and to preserve strategic diversity"; also pseudo-rewards for following a sampled human build order (statistic z) | Kept through RL (no end) | default for the KL; the paper's Fig. 3 ablates human-data use, numbers not re-checked here |
| OpenAI Five, Berner et al. 2019 | No BC; self-play from scratch | n/a | default |
| Kickstarting, Schmitt et al. 2018 | RL loss plus lambda_k x cross-entropy H(pi_teacher, pi_student) on the student's own trajectories (A3C/IMPALA, DMLab) | lambda_k to 0. Measured: "constant schedules can perform well early on, but ultimately cause the agent to plateau near the teacher's final performance"; linear decays to 0 work better, "it is important to reduce this weight quickly"; PBT-controlled lambda nearly as good as the best hand schedule. About 10x fewer steps than from scratch, 42% higher final score (multi-teacher) | measured |
| DAPG, Rajeswaran et al. 2018 | Policy gradient augmented with a demonstration term weighted lambda0 x lambda1^k x max advantage | Geometric decay (0.1, 0.95 in all experiments; "not very sensitive") | measured (robot hands) |
| DQfD, Hester et al. 2018 | Pre-training on demonstrations, then a large-margin supervised loss kept during Q-learning | Kept (no end) | measured (Atari) |
| VPT, Baker et al. 2022 | BC on video, then PPG fine-tuning with rho x KL(pi_pretrained, pi_theta) replacing the entropy bonus | rho 0.2, decayed by 0.9995 per iteration (Table 6). Measured: without the KL term the agent only obtained items early in the sequence and needed a 7x lower learning rate to learn at all. Quote: a high rho "would prevent the agent from properly optimizing the reward function while a low coefficient rho was ineffective" | measured |
| Jump-Start RL, Uchendu et al. 2023 (ICML) | A fixed guide policy (scripted, BC or RL) plays the first h steps, the learner the rest; h shrinks | Gated by measured performance: advance "if the current moving average is close enough to the best previous moving average"; too loose a tolerance hurt (Table 11). Also measured: initialising the actor from a prior policy with a fresh critic lost the prior's performance (its Figs. 2, 7, 8) | measured |
| Wolczyk et al. 2024 (ICML), fine-tuning is forgetting mitigation | BC-pretrained NetHack policy fine-tuned with BC-on-pretraining-data, kickstarting (KL to the pretrained policy), EWC, episodic memory | KS 0.5 decayed by 0.99998 per step; BC 2.0 not decayed. Measured: KS best on NetHack (score from 5K to over 10K), BC best on Montezuma and RoboticSequence, KS "fails completely" on RoboticSequence; EWC poor on NetHack | measured |
| Policy distillation, Rusu et al. 2016; Czarnecki et al. 2019 | Supervised matching of the teacher's distribution; Czarnecki compares distillation variants and finds the best depends on the task | n/a (pure distillation) | measured |
| RLHF: Ziegler et al. 2019; InstructGPT, Ouyang et al. 2022 | PPO with a per-token KL penalty to the supervised (SFT) reference; Ziegler adapts beta to a KL target; InstructGPT uses a fixed beta | Kept | default |
| Recent reasoning RL: DAPO, Yu et al. 2025 | Drops the KL-to-reference term for long reasoning RL | n/a | default (argued, not isolated) |
| v2 release run (this project) | 100M steps of BC to Imugi's full distribution, then RL from the clone's EMA with no BC term and no entropy | Stage only | the project's own |

### 1.2 What this says

- Every large system that used demonstrations did a supervised stage first (AlphaStar, VPT,
  DQfD, NetHack BC, RLHF) (default). The only head-to-head result on the in-RL term is
  about its schedule, not about stage against term: a teacher term that never ends caps the
  student near the teacher (kickstarting, measured); one that ends too slowly costs reward,
  one that ends too fast or is absent forgets the clone's skills (VPT, Wolczyk, measured).
- Three ways sources end the teacher signal: a fixed decay per iteration (VPT, DAPG,
  Wolczyk's KS), a hyperparameter search (kickstarting with PBT), or a measured-performance
  gate (JSRL's curriculum). I found no source that gates a BC/KL term by the win rate
  against the teacher, which is lean's rule (inference: the closest precedent is JSRL's
  moving-average gate). AlphaStar and RLHF never end it.
- The form of the term: kickstarting, VPT and Wolczyk's KS all use the cross-entropy (or KL)
  to the teacher's **full distribution** on the **student's** states. Lean uses the teacher's
  **sampled** action. The sampled action is an unbiased but noisier one-sample estimate of
  the same cross-entropy gradient (inference); `scripted.marginals` already provides the
  distribution (author-review 7.1), so the distribution costs nothing.
- The critic: a BC stage that trains only the actor hands RL a fresh critic, and JSRL
  measured the cloned policy's performance collapsing in that case. v2's stage trained its
  critics (BC's own discounts, recipe-v2-vs-lean "BC / distillation"), which avoids it
  (inference). Lean's BC stage, if built, should train the critics on the same rollouts.

### 1.3 Risks

| Option | Risks |
|---|---|
| Stage only, then pure RL (v2) | Forgetting of cloned skills in states RL does not revisit early (Wolczyk, VPT, measured); a fresh optimiser and warm-up at the hand-over (v2 did this); no way back if RL drifts |
| Stage, then decaying KL/cross-entropy to the teacher (kickstarting, VPT) | Two schedules to set (decay rate, cut-off); a too-slow decay caps the student at the teacher (kickstarting, measured) |
| In-PPO term only, from random weights (lean now) | Early updates mix a large supervised gradient with a noisy policy gradient through a fresh critic; the gate reads a noisy win rate (lean's window is about 100 games) and can switch on and off without a latch (v1 latched after 3 updates) |
| KL kept forever (AlphaStar, RLHF) | Caps play near the teacher's style; acceptable when the teacher is a human population, not when it is a weaker scripted agent (inference) |

### 1.4 Recommendation

The author's lean (a stage, then distillation) is consistent with the literature. Concretely:
a BC stage on the full teacher distribution, on the student's own rollouts (half self-play,
half against the scripted agent, as v2), with the critics trained; RL from the stage's
averaged weights; a cross-entropy-to-teacher term in RL that decays per update (VPT-style
geometric) and is latched to 0 once the measured win rate against the scripted agent passes
the gate. The decay alone ends it even if the gate is never reached; the gate ends it early
if the student is already better. Evidence strength: moderate for "stage first" and for "a
decaying teacher term after it"; none for the exact gate. The gated in-PPO-only variant
(lean now) is the alternative with the least support; a head-to-head of stage-plus-decay
against in-PPO-gated is a candidate experiment (untested in any domain I found).

## 2. Weight averaging (item 7.2)

### 2.1 Who averages weights in RL

| Source (year) | Use | Label, relevance |
|---|---|---|
| Polyak target networks (DDPG 2016, TD3 2018, SAC 2018; tau about 0.005) | EMA of the critic (and actor) as the bootstrap target, fed back into training | default; off-policy value learning, not a deployed-policy average |
| Averaged-DQN, Anschel et al. 2017 | Average of the last K Q-estimates as the target | measured (Atari); value-based |
| Nikishin et al. 2018 (UAI workshop), SWA in RL | Uniform SWA of actor and critic after a pre-training phase, every 100 updates (A2C, Atari) or every 2,000 steps (DDPG, MuJoCo); the average is not fed back | measured: A2C+SWA better on 5 of 6 Atari games (Breakout 522 to 703, Qbert 18,777 to 21,272), worse on CrazyClimber; DDPG+SWA better on 4 of 4 MuJoCo tasks; 3 seeds, wide intervals. Old and small, but the nearest test of a uniform average of a policy |
| PPO-EWMA, Hilton et al. 2022 (NeurIPS) | EMA of the policy weights as the proximal (trust-region anchor) policy, decoupled from the behaviour policy | measured (Procgen): batch-size invariance over a 256-fold range; the EMA is used inside the loss, not deployed |
| WARP, Rame et al. 2024 | EMA of the policy as a moving KL anchor in RLHF, plus merging of fine-tuned policies | measured (Gemma, reward vs KL front); LLM |
| EMAgnet, Maidment et al. 2026 (preprint) | PPO self-play regularised by KL towards an EMA of its own weights | measured (exact exploitability, small games, 24 seeds): lower exploitability in most variants; the EMA itself sometimes, not always, less exploitable than the last iterate |
| Generals.io, Straka et al. 2026 (preprint, revised Oct 2026) | Policy-gradient self-play; EMA of parameters, decay 0.999 per iteration, used for deployment | measured: "From 4,000 iterations on, the EMA is ahead at every checkpoint of every run, by 65 Elo on average" (500 games per checkpoint, 3 runs); notes that StarCraft II, Dota 2 and Stratego work "does not discuss parameter EMA as a way to improve the deployed policy". The closest domain match found (two-player self-play game, sparse win/loss) |
| PolicyIK 2026 (robot arm IK, PPO) | Exponentially weighted average of the last 20 checkpoints for deployment | measured: 92.30 to 93.49 strict success; single application |
| v1 (this project) | Uniform running mean after BC | measured here: averaged weights held 0.08 to 0.11 more break-even ratio at every size (author-review 7.2) |
| v2 (this project) | EMA 0.95 per update and uniform from 10M steps, both league members and rated; BC hands its EMA to RL | the project's own; which file shipped was not recorded |

### 2.2 Averaging against learning-rate decay (supervised and LLM)

- Kaddour 2022: averaging the latest k checkpoints (LAWA) saves "~68 and ~30 GPU hours"
  on ImageNet ResNet-50 and RoBERTa-Base (measured).
- Sanyal et al. 2023 (LAWA for LLM pre-training): "models trained with high learning rates
  observe higher gains due to checkpoint averaging"; outperforms EMA and SWA (measured).
- Hagele et al. 2024 (NeurIPS), constant learning rate with a cooldown: SWA on a constant
  rate gives "a significant performance boost", "Yet, it does not reach the loss values of
  explicit cooldowns" (measured, LLM pre-training, up to about 1B parameters).
- Ajroldi, Orvieto, Geiping 2025 (AlgoPerf, 7 workloads): averaging "cannot fully replace a
  learning rate annealing"; it speeds training and combining both is best; "The benefits of
  WA diminish when annealing the learning rate to zero" (measured).
- Morales-Brotons, Vogels, Hendrikx 2024 (TMLR): EMA "requires less learning rate decay"
  (measured, vision).
- Defazio et al. 2024, schedule-free AdamW: the averaging is part of the optimiser (the
  gradient is taken at an interpolation of the iterate and its average), with no schedule;
  "state-of-the-art performance compared to schedules", winner of AlgoPerf 2024 self-tuning
  (measured, supervised). I found no published PPO or self-play test of schedule-free
  optimisers.

### 2.3 Has "a uniform average from the end of warm-up instead of a decay" been tested in RL?

Not that I found. Nikishin 2018 used a uniform average after pre-training at a constant
learning rate but compared it with the live weights only, not with a decayed run. Generals.io
2026 compares an EMA with the last iterate, not with a decay. In LLM pre-training the answer
is "it narrows the gap, it does not close it". This looks untested in RL, and it is a
reasonable experiment arm under the author's rule (theory plus a supervised result, no test
in this domain).

Two inferences that bear on the design of that arm:
- Self-play is non-stationary: a uniform average from 10M steps weights the weak early
  policies as much as the latest ones. In supervised training the loss landscape stays put;
  in self-play the opponent distribution moves. An EMA or a window of recent checkpoints
  (LAWA) tracks this; a uniform average does not. Nikishin's uniform SWA worked on
  single-agent tasks with fixed reward.
- Averaging assumes the checkpoints lie in one basin (they are linearly connected). Along
  one continuous run at a moderate rate this usually holds (inference from the supervised
  results above); RL has no measurement of it I could find.

### 2.4 Recommendation

- BC stage: average (EMA, or uniform over the stage's second half) and hand the average to
  RL. Strong evidence from supervised learning; BC is supervised.
- RL: keep an EMA (and, if wanted, a uniform average) as side copies only, never fed back
  into the loss; add them to the league and rate them against the live weights. Moderate
  evidence (Generals.io +65 Elo, Nikishin, v1's own measurement).
- The decay question as an experiment: four arms from one run's checkpoints where possible
  — live, EMA, uniform from warm-up, and a short learning-rate cooldown at the end (as
  Hagele: about 10 to 20% of the steps, decayed to 0). Without the cooldown arm the author's
  question ("does the average give the benefit of a decay?") is not answered.
- v2's EMA decay 0.95 per update is a half-life of about 13.5 updates (about 14M steps);
  Generals.io used 0.999 per iteration. Neither is tuned against the other (default).

## 3. KL early stopping (item 7.3)

### 3.1 Defaults

| Library | Rule | Default | Code |
|---|---|---|---|
| Spinning Up | Stop the policy's gradient steps when the mean KL exceeds 1.5 x target_kl | 0.01 (on) | `spinup/algos/pytorch/ppo/ppo.py:91,269` |
| Stable-Baselines3 | Stop the epoch loop at a minibatch whose approximate KL exceeds 1.5 x target_kl | None (off) | `stable_baselines3/ppo/ppo.py:100,267` |
| CleanRL | After each epoch, stop if the last minibatch's approximate KL exceeds target_kl | None (off) | `cleanrl/ppo_atari.py:77,309` |
| RLlib 2.40 | Adaptive KL penalty, not a stop: kl_coeff 0.2, kl_target 0.01 | on (penalty) | `rllib/algorithms/ppo/ppo.py:133-135` |
| rsl_rl (legged robots, Isaac Lab) | Adaptive learning rate: divide by 1.5 when KL > 2 x 0.01, multiply by 1.5 when KL < 0.005 | on (schedule "adaptive") | `rsl_rl/algorithms/ppo.py:56-57,249-266` |
| rl_games / Sample Factory | KL-adaptive learning-rate schedules, threshold 0.008 | rl_games per config; Sample Factory off (`lr_schedule` "constant") | `sample_factory/cfg/cfg.py:299-310` |
| v2 | Stop after an epoch whose mean approximate KL exceeds 0.02 | 0.02 (on) | recipe-v2-vs-lean, row "KL early stop" |

### 3.2 Measured

- Andrychowicz et al. 2021 (250k runs, MuJoCo) did not study early stopping. They found KL
  penalties and constraints redundant with the clipped loss: "the PPO policy loss already
  enforces the trust region which makes KL penalties or constraints redundant" (measured).
- Dossa, Huang, Ontanon, Matsubara 2021 (IEEE Access): PPO is sensitive to the number of
  updates per iteration; KL early stopping reduces that sensitivity and could replace tuning
  it (measured, MuJoCo and others; full text not read, abstract only).
- Sun et al. 2022 (preprint): KL early stopping at 0.01 "significantly boosts the
  performance of the vanila ratio clipping PPO on Ant, HalfCheetah, Humanoid and
  HumanoidStandup"; "as the number of optimization epochs increases, the ratios grow
  without bound" (measured).
- VPT 2022: their KL-to-pretrained term let them use a 7x higher learning rate; a related
  but different mechanism (measured).
- Generals.io 2026 lists target KL 0.02 but with one epoch "never triggers" (default).

All the measurements are small-batch continuous control. None is at 128 Adam steps on
minibatches of about 150k ship-steps, where each step's gradient is far less noisy (inference:
less noise means each step moves the policy more consistently in one direction, so the drift
over 128 steps can be larger than on MuJoCo's few small steps).

### 3.3 Recommendation

By the author's rule (early stop if standard, else monitor and warn): it is not a default in
the most used libraries (SB3, CleanRL, Sample Factory off; RLlib uses a penalty; Spinning Up
is on, and robotics libraries adapt the learning rate instead), so monitor and warn. Warn
when an epoch's mean approximate KL exceeds 0.02 (v2's stop value; Spinning Up's effective
threshold is 0.015), and log the clip fraction and the advantage standard deviation (Hilton
et al. 2022). If the warning fires routinely in the first run, the measured evidence (Dossa,
Sun) supports turning the same 0.02 into a stop checked after each epoch, as v2. Evidence:
moderate that a stop helps when many epochs drift; weak for this batch size.

## 4. Advantage scaling (item 7.4)

### 4.1 The options and what measured them

| Option | Where | Evidence |
|---|---|---|
| Mean and std per minibatch | CleanRL `norm_adv` (`ppo_atari.py:65`), SB3 (`ppo.py:218`) | measured as unimportant (Andrychowicz 2021, "seems not to affect the performance too much") |
| Std over the update, scale only | lean; Acme uses scale-only EMA | default (see literature-ppo-targets-minibatches.md 1.4) |
| Percentile range EMA, divide by max(1, S) | DreamerV3 (Hafner et al. 2023/2025): 5th to 95th percentile of lambda-returns, EMA 0.99. Reason given: under sparse rewards advantage normalisation amplifies noise until it outweighs the entropy term | measured in DreamerV3 only as part of a robustness ablation ranked second after the KL objective; no direct comparison with advantage normalisation |
| The same in PPO | Sullivan, Kumar, Huang, Dickerson, Suarez 2023 (NeurIPS), CleanRL base, over 10,000 A100 hours on Atari and DMC | measured: comparable to the baseline with reward clipping, slightly better without; "works best in combination with the standard advantage normalization used in PPO"; the DreamerV3 tricks together "do not generally improve" PPO; two-hot critics consistently hurt in their PPO (fixed bin range suspected) |
| PopArt | van Hasselt et al. 2016; Hessel et al. 2019 (multi-task IMPALA) | measured: makes one agent work across Atari games with very different reward scales. It renormalises a regression critic's targets; with fixed categorical supports it does not apply directly (inference) |
| Symlog targets | DreamerV3 | measured in DreamerV3; Sullivan found symlog the most impactful trick for PPO without reward clipping, harmful with clipping |
| Normalise each reward before summing | GDPO, Liu et al. 2026 (ICML), LLM GRPO with several rewards: normalise each reward in its group, then sum, then a batch-wise normalisation | measured: better than GRPO on the sum for tool calling, maths and coding. LLM, outcome-level rewards, no critic |
| Monitoring | Hilton et al. 2022 | advises EWMA statistics at small batch sizes and watching the advantage std |

### 4.2 For this project

- The update holds about 4.7M ship-steps, so per-update statistics are stable (inference,
  as in the earlier review). DreamerV3's concern (sparse reward, noise amplified) applies
  weakly: the damage and kill levels pay often (inference).
- The summed advantage's std is dominated by the level with the largest advantage variance.
  Which level that is (damage with weight 2.74 per 100 points, or outcome with weight 10 and
  discount 0.9997) is not measured. If one level dominates, the others' signals are
  effectively down-weighted (inference). This is the GDPO argument, in a different setting.

### 4.3 Recommendation

Keep the std over the update, scale only, and delete the half-ported "percentile" option
(no source prefers it over standard normalisation; Sullivan measured it as best used on top
of it). Log each level's share of the summed advantage's variance per update. Per-level
normalisation (each level's advantage divided by its own std before the weighted sum) is an
experiment arm; it is untested in PPO with multi-horizon critics. Evidence: weak to
moderate.

## 5. Balancing the loss terms (item 7.5)

### 5.1 What others do

| Approach | Who | Evidence |
|---|---|---|
| Fixed coefficients from library defaults | PPO paper (value 1 or 0.5, entropy 0.01), CleanRL (`vf_coef` 0.5, `ppo_atari.py:73`), SB3, rsl_rl (`value_loss_coef` 1.0, `ppo.py:50`), PufferLib 3.0 (`vf_coef` 2.0, `ent_coef` 0.001), MAPPO | default |
| Fixed coefficients on normalised terms, unchanged across domains | DreamerV3: categorical critics trained by cross-entropy so "gradient size doesn't depend on target magnitude", return normalisation, symlog, free bits; beta_pred 1, beta_dyn 1, beta_rep 0.1; one set for 150+ tasks | measured (robustness ablation; fixed hyperparameters across domains) |
| Categorical value loss | HL-Gauss, Farebrother et al. 2024 (ICML): "significantly improves performance and scalability" over regression, attributed to noisy targets and non-stationarity | measured, mostly value-based RL (Atari, chess, robotics, Wordle); not PPO |
| Separate policy and value training | PPG, Cobbe et al. 2021: separate phases to avoid interference | measured (Procgen); Andrychowicz: separate networks better on 4 of 5 tasks (measured) |
| GradNorm (Chen et al. 2018), uncertainty weighting (Kendall et al. 2018), PCGrad (Yu et al. 2020) | Supervised multi-task; PCGrad also Meta-World | measured in their papers; but Kurin et al. 2022 (NeurIPS, includes Meta-World RL) and Xin et al. 2022 (NeurIPS) found none reliably better than the plain weighted sum with tuning and standard regularisation (measured) |
| Auxiliary weight by gradient cosine | Du et al. 2018 (keep the auxiliary gradient only when its cosine with the main gradient is non-negative); Lin et al. 2019 (NeurIPS, adaptive auxiliary weighting for RL) | measured (gridworld, Atari, small scale) |
| Gradient-share tuning | No paper found | — |

### 5.2 Why tuning by gradient share chases its own tail (inference)

- The share is measured on the gradient, but Adam's step divides each parameter's gradient
  by its own running second moment. A term's share of the gradient norm is not its share of
  the parameter change, and heads' parameters (touched by one term) are rescaled separately
  from the trunk.
- The shares are not stationary. A cross-entropy's gradient shrinks as it fits (the HL-Gauss
  loss approaches the target's entropy, its gradient approaches 0); the policy gradient
  depends on the clip fraction and the advantage scale. Changing a coefficient changes the
  trajectory, which changes the shares that motivated the change.
- There is no principled target share. GradNorm's target (equal training rates) is also a
  choice, and the comparisons above did not find it better than fixed weights.
- The global gradient clip (1.0) couples the terms: when it binds, every term is scaled
  together, and the shares say nothing about the step size.

### 5.3 What makes terms scale-free

- Policy: normalised advantages give an O(1) surrogate gradient per token regardless of the
  reward weights (measured property of normalisation; DreamerV3's argument).
- Values and auxiliaries: a cross-entropy in nats has logit gradient (p - q), bounded and
  independent of the return's units (property of the loss; DreamerV3, Farebrother). The
  author's decision that every output trains by cross-entropy in nats (core-decisions,
  item 6.3) puts all non-policy terms on one scale. The next-state squared error
  (`update.py:104-105`) is the one term still in target units.
- Entropy: in nats, bounded by ln 42 = 3.74.
- A loss value of an HL-Gauss cross-entropy includes the target's own entropy (a floor
  above 0). For monitoring, cross-entropy minus target entropy (the KL) compares across
  levels; the gradient is the same (inference).

### 5.4 Recommendation

Fixed coefficients, set once and not retuned: policy 1 on normalised advantages; value,
next-state and hit as cross-entropies in nats with coefficients from the current values or
v1/v2 defaults; entropy as in section 6. Drop the gradient-share loop. Keep gradient shares
and the cosine between each auxiliary gradient and the policy gradient on the trunk as
diagnostics. A term suspected of harm gets an on/off ablation arm (Du et al.'s cosine test is
the cheap check), not a retune. Separate value training (PPG) is the measured alternative if
policy-value interference shows; it costs extra passes. Evidence: moderate (fixed weights
match adaptive balancers in the two large comparisons; scale-free losses carry one
coefficient set across domains in DreamerV3). Untested here: whether nats-valued terms let
coefficients transfer across this project's game changes.

## 6. Entropy

### 6.1 Types

| Type | Who | Evidence |
|---|---|---|
| Constant bonus on the policy's entropy | PPO paper (Atari 0.01), CleanRL Atari 0.01 (`ppo_atari.py:71`), SB3 0.0 (`ppo.py:93`), RLlib 0.0 (`ppo.py:137`), rsl_rl 0.01 (`ppo.py:51`), Sample Factory 0.003 (`cfg.py:206-207`), PufferLib 3.0 0.001, DreamerV3 3e-4 on return-normalised advantages | default; Andrychowicz 2021 measured little benefit except HalfCheetah; Ahmed et al. 2019 (ICML) measured that entropy smooths the optimisation landscape and allows higher learning rates |
| Scheduled bonus | Generals.io 2026: max(0.05 (t+1)^-0.2, 1e-3); RLlib `entropy_coeff_schedule` | measured weakly: one run per schedule, "the shape of a schedule matters less than its range" |
| Target entropy, adaptive coefficient | SAC (Haarnoja et al. 2018b); discrete SAC (Christodoulou 2019, target 0.98 ln A; Xu et al. 2021 found that target problematic and annealed it); Skywork-OR1 2025 (LLM RL: coefficient raised by a step when entropy falls below a target) | measured in SAC and LLM RL; not in PPO games |
| KL to a prior instead of entropy | VPT (KL to the pretrained policy replaces entropy); EMAgnet (KL to an EMA magnet); Generals.io tried regularising towards heuristic policies "in place of the entropy bonus, but observed no improvement" | measured in those settings |
| Entropy of a factored action | Per-factor entropies summed (v2: three factors, power, stick given power, fire given both) | default |
| Covariance-based clipping | Cui et al. 2025 (Clip-Cov, KL-Cov) against entropy collapse in LLM RL | measured, LLM only |

### 6.2 The factored action

Lean's policy is one softmax over 42 joint actions (7 stick x 3 power x 2 fire). By the chain
rule, H(joint) = H(power) + H(stick | power) + H(fire | power, stick); v2's three per-factor
bonuses with one shared coefficient equal the joint bonus, and unequal coefficients only
reweight the factors (inference, exact identity). The uniform policy over 42 is uniform on
each factor, so a KL-to-uniform term equals ln 42 minus the joint entropy and is the same
term (exact). A joint softmax needs no per-head entropy; per-head coefficients would only
matter to push one factor (say fire) harder than the others, and I found no measurement of
that.

### 6.3 Recommendation

Keep a constant bonus on the joint entropy at 0.003 (inside the 0.001 to 0.01 range of the
defaults, with normalised advantages); log the normalised entropy H / ln 42 per update and
warn on a fast fall. Do not add a schedule or an adaptive coefficient for the first run; if
entropy collapses, a target-entropy controller (SAC or Skywork style) is the documented fix,
untested in PPO games. During the KL-to-teacher phase (section 1), VPT's practice was to drop
the entropy bonus because the teacher term regularises; v2's RL stage ran with no entropy
bonus at all. Evidence: weak (the effect of the coefficient is small and task-dependent in the
measured work).

## Untested in this domain (candidate experiments)

1. A uniform weight average from the end of warm-up at a constant learning rate, against a
   short learning-rate cooldown, in RL (or self-play). Not found in any RL paper; in LLM
   pre-training averaging narrows the gap to a cooldown without closing it.
2. EMA against uniform average as the deployed policy in self-play. Generals.io (2026)
   measured EMA against the last iterate only; Nikishin (2018) uniform against the last
   iterate only.
3. A BC stage plus decaying distillation against a win-rate-gated BC term inside PPO from
   random weights, with a scripted teacher. No head-to-head found.
4. Gating a teacher term by the measured win rate against the teacher. Nearest: JSRL's
   moving-average curriculum gate.
5. Per-level advantage normalisation before summing, in PPO with several critics at
   different horizons. Measured only for LLM GRPO (GDPO).
6. KL early stopping at very large minibatches (about 150k ship-steps) and 128 Adam steps per
   update. Measurements are small-batch MuJoCo.
7. Whether nats-valued auxiliary terms let one coefficient set survive game and model
   changes in PPO (DreamerV3's claim is for its own world-model losses).
8. Adaptive target-entropy coefficients in PPO for games.

## What I could not verify

- AlphaStar's Fig. 3 ablation numbers for the KL-to-supervised term and the KL coefficient;
  the Nature text was not accessible. The KL's existence and stated purpose come from
  quoted excerpts in secondary sources.
- Dossa et al. 2021: abstract only; thresholds and environments not read.
- Lin et al. 2019 adaptive auxiliary weighting: known from NeurIPS reviews, paper not read.
- Kurin et al. 2022 and Xin et al. 2022 findings, PPG's Procgen results, DQfD, OpenAI Five,
  InstructGPT's fixed beta, Ziegler's adaptive controller and DAPO's removal of the KL term
  are from my knowledge of those papers, not re-read today.
- Hagele et al.'s size of the remaining gap between SWA and a cooldown (only shown in a
  figure).
- DreamerV3's loss weights differ between paper versions (beta_dyn 0.5 in the first arXiv
  version, 1 in the current one as read today).
- The Generals.io and EMAgnet papers are 2026 preprints, not peer reviewed.
- rl_games' default `kl_threshold` depends on each config file; not checked.

## Sources

- Vinyals et al. 2019, Grandmaster level in StarCraft II using multi-agent RL, Nature 575.
  https://www.nature.com/articles/s41586-019-1724-z
- Berner et al. 2019, Dota 2 with Large Scale Deep RL. https://arxiv.org/abs/1912.06680
- Schmitt et al. 2018, Kickstarting Deep Reinforcement Learning. https://arxiv.org/abs/1803.03835
- Rajeswaran et al. 2018, Learning Complex Dexterous Manipulation with Deep RL and
  Demonstrations (DAPG). https://arxiv.org/abs/1709.10087
- Hester et al. 2018, Deep Q-learning from Demonstrations. https://arxiv.org/abs/1704.03732
- Baker et al. 2022, Video PreTraining (VPT), Appendix G and Table 6.
  https://arxiv.org/abs/2206.11795
- Uchendu et al. 2023, Jump-Start Reinforcement Learning (ICML). https://arxiv.org/abs/2204.02372
- Wolczyk et al. 2024, Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
  Mitigation Problem (ICML). https://arxiv.org/abs/2402.02868
- Rusu et al. 2016, Policy Distillation. https://arxiv.org/abs/1511.06295
- Czarnecki et al. 2019, Distilling Policy Distillation (AISTATS). https://arxiv.org/abs/1902.02186
- Ziegler et al. 2019, Fine-Tuning Language Models from Human Preferences.
  https://arxiv.org/abs/1909.08593
- Ouyang et al. 2022, Training language models to follow instructions with human feedback.
  https://arxiv.org/abs/2203.02155
- Yu et al. 2025, DAPO. https://arxiv.org/abs/2503.14476
- Nikishin et al. 2018, Improving Stability in Deep RL with Weight Averaging.
  https://izmailovpavel.github.io/files/swa_rl/paper.pdf
- Anschel et al. 2017, Averaged-DQN. https://arxiv.org/abs/1611.01929
- Hilton, Cobbe, Schulman 2022, Batch size-invariance for policy optimization (NeurIPS).
  https://arxiv.org/abs/2110.00641
- Rame et al. 2024, WARP. https://arxiv.org/abs/2406.16768
- Maidment et al. 2026, EMAgnet (preprint). https://arxiv.org/abs/2606.23995
- Straka, Lisy, Schmid 2026, Superhuman AI for Generals.io Using Self-Play RL (preprint).
  https://arxiv.org/abs/2606.23348
- PolicyIK 2026 (preprint). https://arxiv.org/abs/2611.05372
- Kaddour 2022, Stop Wasting My Time! (LAWA). https://arxiv.org/abs/2209.14981
- Sanyal et al. 2023, Early Weight Averaging meets High Learning Rates for LLM Pre-training.
  https://arxiv.org/abs/2306.03241
- Hagele et al. 2024, Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations (NeurIPS). https://arxiv.org/abs/2405.18392
- Ajroldi, Orvieto, Geiping 2025, When, Where and Why to Average Weights?
  https://arxiv.org/abs/2502.06761
- Morales-Brotons, Vogels, Hendrikx 2024, Exponential Moving Average of Weights in Deep
  Learning: Dynamics and Benefits (TMLR). https://arxiv.org/abs/2411.18704
- Defazio et al. 2024, The Road Less Scheduled (schedule-free). https://arxiv.org/abs/2405.15682
- Andrychowicz et al. 2021, What Matters in On-Policy RL? (ICLR). https://arxiv.org/abs/2006.05990
- Dossa, Huang, Ontanon, Matsubara 2021, An Empirical Investigation of Early Stopping
  Optimizations in PPO, IEEE Access 9. https://doi.org/10.1109/ACCESS.2021.3106662
- Sun et al. 2022, You May Not Need Ratio Clipping in PPO. https://arxiv.org/abs/2202.00079
- Spinning Up PPO. https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Hafner et al., Mastering Diverse Domains through World Models (DreamerV3).
  https://arxiv.org/abs/2301.04104
- Sullivan et al. 2023, Reward Scale Robustness for PPO via DreamerV3 Tricks (NeurIPS).
  https://arxiv.org/abs/2310.17805
- van Hasselt et al. 2016, Learning values across many orders of magnitude (PopArt).
  https://arxiv.org/abs/1602.07714
- Hessel et al. 2019, Multi-task Deep RL with PopArt (AAAI). https://arxiv.org/abs/1809.04474
- Liu et al. 2026, GDPO (ICML). https://arxiv.org/abs/2601.05242
- Farebrother et al. 2024, Stop Regressing (ICML). https://arxiv.org/abs/2403.03950
- Cobbe et al. 2021, Phasic Policy Gradient (ICML). https://arxiv.org/abs/2009.04416
- Chen et al. 2018, GradNorm (ICML). https://arxiv.org/abs/1711.02257
- Kendall, Gal, Cipolla 2018, Multi-task learning using uncertainty to weigh losses (CVPR).
  https://arxiv.org/abs/1705.07115
- Yu et al. 2020, Gradient Surgery for Multi-Task Learning (PCGrad). https://arxiv.org/abs/2001.06782
- Kurin et al. 2022, In Defense of the Unitary Scalarization for Deep Multi-Task Learning
  (NeurIPS). https://arxiv.org/abs/2201.04122
- Xin et al. 2022, Do Current Multi-Task Optimization Methods in Deep Learning Even Help?
  (NeurIPS). https://arxiv.org/abs/2209.11379
- Du et al. 2018, Adapting Auxiliary Losses Using Gradient Similarity.
  https://arxiv.org/abs/1812.02224
- Lin et al. 2019, Adaptive Auxiliary Task Weighting for RL (NeurIPS).
  https://papers.nips.cc/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html
- Haarnoja et al. 2018, Soft Actor-Critic Algorithms and Applications.
  https://arxiv.org/abs/1812.05905
- Christodoulou 2019, Soft Actor-Critic for Discrete Action Settings.
  https://arxiv.org/abs/1910.07207
- Xu et al. 2021, Target Entropy Annealing for Discrete Soft Actor-Critic.
  https://arxiv.org/abs/2112.02852
- Ahmed et al. 2019, Understanding the Impact of Entropy on Policy Optimization (ICML).
  https://arxiv.org/abs/1811.11214
- He et al. 2025, Skywork Open Reasoner 1 Technical Report. https://arxiv.org/abs/2505.22312
- Cui et al. 2025, The Entropy Mechanism of RL for Reasoning Language Models.
  https://arxiv.org/abs/2505.22617
- Code: Stable-Baselines3 https://github.com/DLR-RM/stable-baselines3, CleanRL
  https://github.com/vwxyzjn/cleanrl, Spinning Up https://github.com/openai/spinningup,
  rsl_rl https://github.com/leggedrobotics/rsl_rl, Sample Factory
  https://github.com/alex-petrenko/sample-factory, rl_games https://github.com/Denys88/rl_games,
  RLlib https://github.com/ray-project/ray (branch releases/2.40.0), PufferLib
  https://github.com/PufferAI/PufferLib (branch 3.0).
