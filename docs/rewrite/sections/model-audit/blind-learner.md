# Blind review of the learner (rewards, returns, value, losses, PPO, BC, league)

Reviewer worked from the inventory and from source under `src/bnb/` only; nothing under `docs/` was read.

**Contamination disclosure.** While confirming defects in `learn/ppo.py`, `learn/bc.py` and `learn/rewards.py` I unavoidably saw four designer comments: (1) per-level discounts were "squared for the x2 game" and one shared value "let a run learn to stop fighting"; (2) the warm-start entropy 0 was set after the -0.0081 bonus "drove the shoot share from 1.7% to 6.8% and the score against Imugi from 0.23 to 0.03 in 24 updates"; (3) BC uses outcome gamma 1 because "the outcome head's Brier skill was negative in the first 10M steps" with PPO's values; (4) rewards are "v1's values times 10" so returns fill the critic bins, and the next-state weight was chosen "for the same trunk share". I treat these as unverified claims of project measurement (category c, evidence not seen by me) and argue each point independently below.

**Citation tags.** [FETCHED-T] = I fetched the arXiv record this session and confirmed the id and title only; what I say the paper shows is from memory. [MEMORY] = not checked at all.

**Arithmetic used throughout** (derived from the inventory, not measured):
- One update = 2048 x 128 x 4 = 1,048,576 environment steps. BC 100M steps is about 95 updates (about 97,000 optimiser steps at 4 x 256). RL 50M steps is about 48 updates (about 6,100 optimiser steps at 4 x 32).
- Horizons 1/(1-gamma) at 30 ticks/s: damage 56 ticks (1.9 s), kill_death 100 (3.3 s), objective_progress 200 (6.7 s), objective 1000 (33 s), outcome 3333 (111 s). gamma x lambda for damage = 0.795, an effective trace of about 5 ticks (0.16 s).
- Adam is invariant to a global rescale of the loss (apart from eps), so only ratios of the coefficients matter. Dividing by the policy's 27: entropy default 3.0e-4, value 2.6e-4, next-state and correction 1.2e-4, enemy-action 4.1e-3, density 7.8e-4, shot-trajectory 2.6e-4, shot-outcome 7.0e-4, geometry 2.6e-4.

---

## Row-by-row verdicts

### Section 5: heads as training targets

**5.1 Policy head, 42-way joint categorical per ship**
- Purpose: one distribution over power x stick x shoot, so correlations between factors are representable.
- Standard?: factored independent categoricals are more common (OpenAI Five, Dota 2 with Large Scale Deep RL [FETCHED-T]); a joint head over 42 actions is small and strictly more expressive. Fine.
- Size generalisation: neutral (per ship, no count dependence).
- Verdict: KEEP. Provenance: a. Confidence: high.

**5.2 Value head, 4 per-ship levels x 77 symexp bins, two-hot, expectation decode**
- Purpose: scale-robust critic; one value per reward stream so each can have its own discount.
- Standard?: categorical critics trained by cross-entropy are published and beat regression (Stop Regressing [FETCHED-T]; DreamerV3, Mastering Diverse Domains through World Models [FETCHED-T], which uses symexp two-hot bins). Stop Regressing reports HL-Gauss targets beating two-hot [MEMORY]; the project already has HL-Gauss code for its auxiliary heads.
- Size generalisation: neutral for zero-shot play, because the critic is never used at play. It becomes relevant the moment training samples sizes (fixed support +-19.09 against returns whose rate grows with enemies in range).
- Verdict: KEEP the form; SIMPLIFY the target to HL-Gauss (one code path for value and auxiliaries); log the fraction of targets clamped at the support edge per level. Provenance: b. Confidence: high on the form, medium on HL-Gauss mattering.

**5.3 Outcome head (4-way on the global token) + 7.6 outcome targets + 8.4 outcome loss**
- Purpose: a win-probability critic for the terminal reward.
- Standard?: no. A terminal +-10 reward with the same per-level value form does the same job. The custom backward recursion over 4-way distributions with an "unresolved" class is a bespoke distributional TD(lambda).
- Size generalisation: neutral.
- Verdict: DROP as a separate mechanism (owner already decided). Becomes unnecessary with it: the 7.6 recursion, the reward-unit conversion (`_in_reward_units`), the per-view-step denominator for value, the BC-only gamma/lambda override for the outcome level, the outcome reliability figures. Provenance: none. Confidence: high.

**5.4 + 8.5 Next-state head and loss (12 HL-Gauss groups, delta from an extrapolation baseline)**
- Purpose: representation learning; in the old design also the engine of the belief.
- Standard?: auxiliary prediction is published for model-free agents (UNREAL, RL with Unsupervised Auxiliary Tasks [FETCHED-T]) but evidence that one-tick observation prediction helps PPO is mixed; no source known that it is reliably positive. At 30 Hz a one-tick change is almost fully explained by extrapolation, so the residual carries little signal about tactics.
- Size generalisation: neutral (per row). Mildly helps if hidden-enemy rows are predicted from privileged truth, because that is the only pressure that makes the recurrent state track unseen ships once the hand-built belief is gone.
- Verdict: KEEP (owner decision) but SIMPLIFY: drop `physics_hidden_decay` weighting, the hidden-row special cases and the decode path (all belief machinery); report it as mean nats per group; TEST one variant that predicts the privileged true state of currently hidden enemies (and/or a 0.5-1 s horizon) against the one-tick delta. Provenance: c-needed. Confidence: medium.

**5.5 + 8.6 Error head and correction loss**: DROP (owner decision). Nothing else depends on it in the learner. Provenance: none. Confidence: high.

**5.6 + 8.7 Enemy-action head and loss (weight 0.11, the largest auxiliary)**
- Purpose: opponent modelling; in the old design its softmax fed back as an input.
- Standard?: opponent-action prediction as an auxiliary is published (Agent Modeling as Auxiliary Task for Deep RL [FETCHED-T]). With the belief removed its feedback role disappears and it is a pure auxiliary. Its targets mix a scripted agent, cheese agents and network snapshots, so the irreducible entropy is large and its gradient is mostly noise at the floor.
- Size generalisation: neutral.
- Verdict: DROP from the baseline; first auxiliary to re-add as an experiment. Its relative weight (35x the next-state weight) is the most suspicious number in the table. Provenance: b for the idea, none for the weight. Confidence: medium.

**5.7 + 8.8 Density head and Poisson loss**: DROP (owner leaning). It is the only non-cross-entropy auxiliary, it has a wrong normaliser (see defects), and its targets scale with fleet size (12.10). Provenance: none. Confidence: high.

**5.8 + 8.10 Shot-outcome head and loss**
- Purpose: teach "would a shot fired now hit", a counterfactual label the policy gradient delivers very slowly (shoot credit is delayed by bullet flight and bullets are not observed).
- Standard?: game-feature prediction auxiliaries are published (Playing FPS Games with Deep RL [FETCHED-T]). Privileged counterfactual labels are a reasoned choice.
- Size generalisation: neutral to helps (purely local target).
- Verdict: KEEP. SIMPLIFY the label generation (8.11): label a random quarter of (tick, row) pairs rather than all. Provenance: d / c-needed. Confidence: medium.

**5.9 + 8.9 Shot-trajectory head and loss**: DROP (owner decision). The 4-age sampling and the age-feature query path go with it. Confidence: high.

**5.10 + 8.12 Geometry head and loss**
- Purpose: forces the trunk to represent pairwise range/angle/closure, i.e. checks that relative-position attention carries usable geometry.
- Standard?: no source known for this specific auxiliary. Reasoned.
- Size generalisation: neutral in training; coverage per tick falls from 8/9 to 8/99 partners, which only matters if it is trained at large sizes.
- Verdict: KEEP (owner decision), tagged as an untested reasoned choice; TEST by ablation once (cheap: one arm without it). Provenance: d. Confidence: low.

**5.11** (which heads run at play): KEEP; after the removals `step` computes policy and value only.

### Section 7: rewards and returns

**7.1 Five reward levels with weights 2.74 / 2.83 / 20 / 20 / 10**
- Purpose: separate streams so each can have its own discount; weights set relative importance; absolute size chosen to fill the critic bins.
- Standard?: several value heads with different discounts whose advantages are summed is published for two streams (RND, Exploration by Random Network Distillation [FETCHED-T]: separate extrinsic and intrinsic heads with different gammas). Hybrid Reward Architecture [FETCHED-T] decomposes value by reward component. Five streams is more than any precedent I know, and a sum of differently-discounted advantages is not the gradient of any single objective; it is a heuristic that works.
- Size generalisation: neutral (all per ship).
- Verdict: KEEP the multi-level structure (the reported single-discount collapse is the strongest evidence in the whole learner). SIMPLIFY to three levels as the first experiment, not in the baseline: combat (damage + kill_death), objective (progress + objective), outcome. The odd weights 2.74 and 2.83 need a recorded measurement or should be rounded. Provenance: b for the structure, c-needed for five levels and the weights. Confidence: medium.

**7.2 Damage events with a (B, N, N) ledger: taken, dealt, friendly fire, recharge refund, charge-back, unforced**
- Purpose: dense combat shaping that is zero-sum between teams and does not pay for damage that later heals.
- Standard?: the principled standard is potential-based shaping (Ng, Harada, Russell 1999, Policy invariance under reward transformations [MEMORY]) with potential = own shield minus enemy shield; the ledger is an attribution refinement of that. No source known for charge-back or for paying opponents for unforced errors.
- Analysis: recharge refund and charge-back must stand or fall together. Keeping the refund without the charge-back makes hit-and-heal positive-sum for the pair, a collusion exploit in self-play. Dropping both leaves dealt/taken, which is zero-sum and needs no ledger. "Unforced" pays ships for events they did not cause, which is pure variance in their advantage; the erring ship is already penalised by damage taken or friendly fire.
- Size generalisation: per-ship events are neutral; unforced shares shrink as 1/team size (harmless).
- Verdict: SIMPLIFY to taken, dealt, friendly fire. DROP unforced. TEST the ledger (refund + charge-back) as one arm, since it may matter with a 1.9 s damage horizon. Provenance: d for the simplified form, none for charge-back/unforced. Confidence: medium.

**7.3 Kill/death events (death -1, kill split half by final-tick hits and half by ledger, ally kill, unforced)**
- Verdict: SIMPLIFY. Death -1; kill credit split among enemies that hit the ship in a recent window (or by final-tick hits only); ally kill negative; DROP unforced. Provenance: d. Confidence: medium.

**7.4 Objective and outcome events**
- Purpose: the real task signal. Outcome +-result to every ship, dead included, not divided by team size.
- Standard?: yes (team terminal reward shared by all agents is the cooperative MARL norm, e.g. MAPPO, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games [FETCHED-T]).
- Size generalisation: neutral per ship. How objective amounts divide among ships is not traced (inventory "Uncertain"); if a capture is split over the team, per-ship objective reward shrinks with team size while per-ship combat reward does not, which shifts the balance toward combat at larger training sizes.
- Verdict: KEEP. Confirm the objective split is per-ship-constant. Provenance: a. Confidence: medium.

**7.5 Per-level GAE with level-specific gamma and lambda; separate BC values**
- Purpose: short horizons for dense combat signal (low variance), long for objectives and outcome.
- Standard?: GAE is standard (High-Dimensional Continuous Control Using GAE [FETCHED-T]). Per-stream gamma has the RND precedent. Per-stream lambda has no source known; standard is one lambda, 0.95.
- Concern: damage lambda 0.81 gives gamma x lambda = 0.795. A bullet lives up to 11 moves and bullets are not observed, so the critic cannot see a bullet in flight; credit for a shot that lands 8-11 ticks later is attenuated to 8-16% unless the recurrent critic memorises the shot. This couples the lowest lambda to exactly the action whose reward is most delayed.
- The different BC and RL discounts for the outcome level (1.0 vs 0.9997) change what the outcome value means at the hand-over (see defects).
- Size generalisation: neutral.
- Verdict: KEEP per-level gamma. SIMPLIFY lambda to 0.95 for every level and use the same gamma/lambda in BC and RL. TEST damage lambda 0.81 vs 0.95 (watch shoot share and hit rate). Provenance: b for gamma, a for lambda 0.95. Confidence: medium.

**7.6 Outcome targets**: DROP with 5.3.

**7.7 128-tick lookahead never trained on**
- Purpose: makes the lambda-return at the end of the 512 trained ticks nearly exact instead of bootstrapped.
- Standard?: no; standard PPO bootstraps from the value at the rollout end (PPO, Proximal Policy Optimization Algorithms [FETCHED-T]). The lookahead costs 25% extra rollout ticks to improve the targets of roughly the last 30-100 of 512 ticks.
- Verdict: DROP. TEST once if rollout is cheap relative to learning: compare explained variance on the last 64 ticks with and without. Provenance: none. Confidence: medium.

**7.8 Level sum; team mixing coefficient 0**
- Sum of level advantages: KEEP (RND precedent). Team mixing at 0 is dead code that still builds an (S, S) matrix each update: DROP the code. Team spirit is published (OpenAI Five [FETCHED-T]) and is a legitimate later experiment. Size: per-ship credit with zero mixing is the size-neutral choice. Provenance: a/b. Confidence: high.

**7.9 + 7.10 Return scale (EMA of the 5th-95th percentile gap of summed returns, floor 1) dividing the advantage; no mean subtraction**
- Purpose: make the policy gradient invariant to reward scale. This is DreamerV3's return normalisation [FETCHED-T], including decay 0.99 and the floor of 1. Note 0.0081 / 27 = 3.0e-4, DreamerV3's entropy scale [MEMORY], so the trio (range normaliser, entropy 3e-4, multiplier 27) looks like DreamerV3's actor loss multiplied by a constant.
- Standard?: not for PPO. DreamerV3 normalises by the return range because its advantages come from imagined lambda-returns of comparable size to that range. Here advantages are per-tick GAE quantities at 30 Hz with short traces; they are probably two to three orders of magnitude smaller than the return range, which is what the multiplier 27 is compensating. The standard PPO choice is to divide by the advantage standard deviation of the batch (PPO [FETCHED-T]; examined in What Matters In On-Policy RL [FETCHED-T] and Implementation Matters in Deep Policy Gradients [FETCHED-T]).
- Size generalisation: neutral for play; the scale is a batch statistic over the mode mix (12.45), so changing the mix or the sizes changes every mode's step size. Standard-deviation normalisation has the same property but tracks the quantity that actually multiplies the gradient.
- Verdict: REPLACE with division by the advantage standard deviation computed over the whole update (no EMA, no carried state, no mean subtraction needed at 5M samples). Removes 7.9, the checkpoint field, the BC-to-RL carry and the reason for 27. Provenance of the replacement: a. Confidence: medium-high.

**7.11 Critic targets in reward units, clamped, two-hot, no normalisation**: KEEP. This is exactly what a categorical critic buys. Provenance: b. Confidence: high.

**7.12 Valued mask (critic on alive or waiting to respawn; policy on alive)**: KEEP the idea, but see defects: GAE reads the values of ships outside the mask, and benched ships would count as valued. Provenance: d. Confidence: medium.

### Section 8: losses and balancing

**8.1 Clipped surrogate, coefficient 27**: KEEP the surrogate (a). DROP the 27 together with 7.9 (coefficient 1 on standardised advantages). Confidence: medium-high.

**8.2 Entropy terms, coefficient 0 in both stages**
- Purpose: keep exploration alive.
- Standard?: PPO uses a small bonus (0.01 in the Atari settings of PPO [FETCHED-T], 0 for continuous control [MEMORY]). Zero is defensible straight after cloning a stochastic teacher, and the project claims a measurement that the bonus damaged the clone.
- My reading of that measurement: at relative weight 3e-4 against advantages of order 1e-3 after range normalisation, the entropy gradient is of the same order as the policy gradient and is systematic where the policy gradient is noisy. So the result shows the bonus was mis-scaled, and does not show that an entropy bonus is harmful. With standardised advantages the same relative 3e-4 would be about a thousand times weaker.
- Also: the three chain-rule coefficients are always equal, so three knobs do the job of one; and the config default (-0.0081) is silently replaced by 0 on warm start, so the printed default is not what runs.
- Verdict: SIMPLIFY to one coefficient on the joint entropy, stated explicitly in the config. Baseline value 0 (the only measured value), then TEST 0.001 and 0.003 after the advantage normalisation changes; watch joint entropy and shoot share over the RL stage for collapse. Provenance: c for 0 under the old scaling; c-needed after the change. Confidence: medium.

**8.3 Value loss, coefficient 0.007 on the sum of levels**: KEEP the loss; REPLACE the coefficient with a rule (section 1 below). Provenance: b for the loss, none for the number. Confidence: medium.

**8.5-8.12 auxiliary weights**: see section 2. All become "mean nats per softmax group" with one shared auxiliary coefficient.

**8.11 Shot re-simulation labels**: KEEP for shot-outcome, subsampled. It is 88% of the prepare phase but, by my arithmetic from the L record, prepare is only about 5% of an update's ops, so this is not a cost priority. Confidence: medium.

**8.13 No label smoothing**: KEEP. a. High.

**8.14 Custom grouped cross-entropy autograd function with a bf16 gradient factor**
- Purpose: speed/memory.
- Verdict: SIMPLIFY to plain `log_softmax` per group unless a recorded timing shows it matters; with the 500-, 342- and 154-logit heads gone the largest coded head is geometry (241). Provenance: c-needed. Confidence: medium.

**8.15 Non-finite guard**: KEEP, but treat any nonzero count as an alarm, not a statistic. a. High.

### Section 9: PPO and optimisation

**9.1 Clip 0.2**: KEEP. a. High.

**9.2 Rollout log-probabilities**: KEEP. a. High. The drift diagnostic stays useful because rollout and learner run different compiled paths in bf16.

**9.3 Batch of 1.05M environment steps per update**
- Purpose: low-variance gradients, throughput.
- Problem: at the launch budget the RL stage gets about 48 policy-improvement iterations, each bounded by clip 0.2 and KL 0.02. That is very few; whatever RL adds over the clone has to happen in 48 trust-region steps, 10 of which are learning-rate warm-up. BC gets 16x more optimiser steps than RL.
- Standard?: large batches are normal at scale (OpenAI Five [FETCHED-T]) but with run lengths of tens of thousands of updates.
- Verdict: TEST 1 shard per update (262k steps, about 190 updates at the same budget, same minibatch size, 8 minibatches). If the snapshot schedule (to 160M and beyond) reflects real runs, the concern weakens proportionally. Provenance: c-needed. Confidence: medium.

**9.4 4 epochs x 32 minibatches; whole 128-tick segments per view**: KEEP. a. High.

**9.5 Gradient clip at global norm 1.0**: KEEP (a; PPO commonly 0.5). Check the fraction of minibatches clipped: if it is near 100% the clip is acting as a gradient normaliser and, under Adam, is nearly a no-op that only removes outlier minibatches. Confidence: high.

**9.6 Micro-batches with exact minibatch-total denominators**: KEEP micro-batching (memory). SIMPLIFY the denominators to per-micro-batch means if the count pre-pass is only there for bit-exactness. d. Medium.

**9.7 KL early stop at 0.02**: KEEP (a, common optional detail). Log how often it fires; if never, delete it. Confidence: medium.

**9.8 Adam, torch defaults**: KEEP. Adam eps 1e-5 is the usual PPO detail [MEMORY: Huang et al. 2022, The 37 Implementation Details of PPO]; with clipped gradients over 2.85M parameters the per-parameter gradient is about 6e-4, so eps is immaterial either way. a. High.

**9.9 Linear warm-up over 10M steps, then constant; fresh warm-up per stage**
- Warm-up with a fresh Adam is reasonable. In RL it consumes 20% of the 50M stage. No decay: standard PPO anneals linearly to zero [MEMORY, same source]; constant is defensible in non-stationary self-play.
- Verdict: KEEP constant; SIMPLIFY the RL warm-up to 2M steps (about 250 optimiser steps is enough for Adam's second moment). d. Medium.

**9.10 No weight decay**: KEEP. a. High.

**9.11 bf16 autocast, float32 losses**: KEEP. a (for cost). High.

**9.12 Whole-loss compilation**: KEEP; with bit-exactness dropped, allow `dynamic` shapes if that removes per-shape compiles when sizes are sampled. d. Medium.

**9.13 Two weight averages (EMA 0.95 per update; uniform after warm-up), both league entries and both rated**
- Purpose: smoother policies for evaluation, hand-over and opponents.
- Standard?: weight averaging in RL has some support (Nikishin et al. 2018, Improving Stability in Deep RL with Weight Averaging [MEMORY]); two averages has none. EMA 0.95 per update is a 20-update window, 40% of the RL stage. A uniform average over a whole RL run averages policies from different league eras.
- Verdict: SIMPLIFY to one EMA; DROP the uniform average. The lanes already rate live vs EMA vs uniform, so the project can decide from existing logs which (if any) beats live. Provenance: c-needed. Confidence: medium.

**9.14 Diagnostics**: KEEP on cadence; the per-term gradient decomposition is exactly the instrument section 2 needs.

### Section 10: BC stage

**10.1 + 10.2 Teacher distribution labels on the student's own states, cross-entropy**
- This is DAgger with the student always in control (A Reduction of Imitation Learning ... to No-Regret Online Learning [FETCHED-T]) with soft labels. It is the right way to clone a queryable teacher.
- Verdict: KEEP. Coefficient becomes 1 (section 1). a. High.

**10.3 Critic and auxiliaries train during BC**
- Pre-training the critic on the student's own returns before RL is good practice: a random critic at hand-over is the usual cause of post-cloning collapse. KEEP, with the same gamma/lambda as RL. d. High.

**10.4 Half self-play, half versus the teacher**: KEEP. d. Medium.

**10.5 Budget-only stop; plateau detector computed but disabled**
- 100M of 150M steps go to cloning a scripted agent. The project already logs the BC KL plateau; if it flattens at 30-40M, the remainder is better spent on RL (which is starved, 9.3).
- Verdict: TEST a 40M/110M split, reading the plateau from existing logs first. DROP the disabled stop rules from the baseline code. c-needed. Medium.

**10.6 Hand-over: EMA weights, carried return scale, fresh Adam, new warm-up, entropy 0, all slots = clone**
- Standard alternative: kickstarting, an annealed distillation term toward the teacher during RL (Kickstarting Deep RL [FETCHED-T]). It needs the teacher in the RL loop, which costs here (Imugi is 4,290 ops per tick as an opponent), and the teacher is weak, so a hard switch with a pre-trained critic is acceptable.
- Verdict: KEEP the hard switch; the carried return scale disappears with 7.9. TEST kickstarting only if the first RL updates show a score dip. d. Medium.

### Section 11: league

**11.1 50% self-play, 50% league, three modes in one batch, seat redrawn per reset**: KEEP. Mixtures of self-play and past opponents are standard (OpenAI Five used 80/20 [MEMORY]). b. High.

**11.2 Four slots, vmapped, 1 redrawn every 2 updates**: KEEP. With 48 RL updates each slot changes about 6 times in the stage. d. Medium.

**11.3 Hard prioritised sampling (1 - p)^2 over Imugi, averages, snapshots, cheese agents**: KEEP. This is prioritised fictitious self-play from AlphaStar (Vinyals et al. 2019, Grandmaster level in StarCraft II, Nature [MEMORY]). Add a small floor (e.g. 0.05) so that beaten opponents are not forgotten; floor 0 lets cycles reopen. b. Medium.

**11.4 Keep the opponent's carry when a slot changes mid-game**: SIMPLIFY: change a slot's entry only at an environment reset for those rows; removes the cut handling and the untallied games. d. Medium.

**11.5 1,024 rating lanes (28% of rollout ops), all versus Imugi at training size**
- Verdict: SIMPLIFY to 256 lanes or a periodic evaluation, and REPLACE part of it with a small large-size lane (see section 3, risk 1). A rating at 5v5 against one scripted agent does not measure the headline goal. c-needed. High on the second point.

**11.6 Opponent identity not an input**: KEEP. a. High.

**11.7 Fixed lineup, equal full teams**: for the headline goal this hurts. REPLACE with sampled lineups and sampled team sizes (section 3). d. High.

---

## 1. The baseline learner I would build

Rewards
- Per-ship streams in three or five levels with per-level discount; start from the existing five to keep the one measured safeguard against "stops fighting" [c, claimed], with the three-level merge as the first experiment [d].
- Combat events: damage taken, damage dealt, friendly fire; death, kill split over recent hitters, ally kill. No ledger, no charge-back, no unforced shares [d]. Ledger variant returns as an experiment.
- Objective events from the mode rules, per ship [a]. Outcome: terminal +-w to every ship of the team, dead included, as an ordinary reward level [a].
- Level weights: keep current values until re-measured, but record them as tuned numbers [c-needed].

Returns and advantages
- GAE per level, gamma per level as now, lambda 0.95 for all levels [a for GAE and lambda; b for per-level gamma]. Same values in BC and RL [d].
- Time-limit ends are true terminations (the game clock is observed) [a: Time Limits in RL, FETCHED-T].
- Bootstrap from the value at the rollout end; no lookahead [a].
- Values of ships outside the valued mask are set to zero before GAE, or dead-without-respawn ships are included in the mask [d, fixes a defect].
- Policy advantage = sum of level advantages for the ship [b: RND]; no team mixing [a].
- Advantage divided by its standard deviation over the whole update [a].

Value
- One head form for every level including outcome: per own ship, categorical over symexp bins, HL-Gauss targets, expectation decode [b: Stop Regressing; DreamerV3]. 77 bins over +-19 is adequate; log the clamp fraction per level [d].

Losses (every term a mean over its tokens, in nats per softmax group)
- Policy surrogate, coefficient 1 [a].
- Value: mean over levels, coefficient 0.5 [d; the PPO default number transferred to a cross-entropy critic, so it must be confirmed by one gradient-share measurement].
- Entropy: one coefficient on the joint entropy, 0 at first [c, claimed], then tested at 0.001-0.003 [c-needed].
- Auxiliaries kept: next-state, shot-outcome, geometry; each as mean nats per group, one shared coefficient 0.1 [d]. Rule: measure the trunk-gradient norm of each term once at the start of RL with the existing decomposition; if the auxiliaries together exceed about a third of the policy's, lower the single shared coefficient. No per-head tuning in the baseline.
- No density, shot-trajectory, correction, enemy-action terms.

PPO and optimisation
- Clip 0.2, 4 epochs, rollout log-probabilities, KL stop 0.02 [a]. Adam 3e-4, short warm-up, constant, no weight decay, global-norm clip 1.0 [a]. bf16 autocast [a].
- Update size: 1 shard (262k steps) unless the 4-shard batch is shown better at equal budget [c-needed].

BC and hand-over
- DAgger-style soft-label cloning, coefficient 1, critic and auxiliaries on, same returns settings as RL [a/d].
- Stop at the KL plateau or a fixed 40M, whichever the logs support [c-needed].
- Hand over the BC weight average and a fresh optimiser; no carried normaliser state [d].

Sizes in training
- Allocate a maximum of 8 slots per team and bench a random subset per episode per team (existing `benched` mechanism, `mask_benched` on), so team sizes 2-8 and unequal teams appear inside one tensor shape; sample lineups, field counts and arena scale with the options that already exist [b: size randomisation with entity-attention policies in Emergent Tool Use From Multi-Agent Autocurricula, FETCHED-T; REFIL, FETCHED-T; UPDeT, FETCHED-T; curriculum over agent count in Evolutionary Population Curriculum, FETCHED-T].

What the league needs from the learner
- Policy snapshots and at most one weight average; win/loss tallies per opponent entry for the sampling probability; opponent swaps at resets only [b/d].
- One periodic evaluation at a size never trained on (for example 20v20 against the scripted agent, a few hundred games every N updates) reported next to the training-size rating [d]. Without it no experiment above can be judged against the headline goal.

---

## 2. Balancing machinery: what the move to cross-entropy does and does not make redundant

What cross-entropy gives: the gradient of each softmax group with respect to its logits is (prediction - target), bounded by sqrt 2 in norm whatever the size of the quantity being predicted. That removes outliers and removes any dependence of gradient size on target units. What it does not give: equal gradients between terms. Three things still differ: (i) how many groups are summed per token, (ii) how much of each loss is irreducible noise (a two-hot code of a noisy return can never be matched, so its gradient stays of order 1 per sample for ever, while a well-predicted physics delta's gradient shrinks), (iii) the surrogate is not a likelihood at all; its gradient is proportional to the advantage.

Rows that exist to balance scales:

| Row | What it balances | Made redundant by cross-entropy everywhere? |
|---|---|---|
| v1 per-component return scaler, Huber, normalised value space | regression target scales | Yes, and already gone (7.11 uses raw reward units). This is the real dividend. |
| 7.1 the x10 on every reward weight | returns versus the fixed bin grid | No. It no longer affects loss scale, but resolution near zero still depends on reward units. It is a support-fitting choice now, and should be recorded as one. |
| 7.9 / 7.10 return scale | surrogate versus reward scale | No. The surrogate is not a cross-entropy. Some normaliser is still required; the range-of-returns one is the wrong one for 30 Hz GAE advantages. Replace with advantage standard deviation. |
| 8.1 policy 27 | offsets 7.9 making advantages tiny; places policy against everything else | Redundant only if 7.9 is replaced. Under Adam the 27 has no meaning by itself; with the clip at norm 1.0 it probably means the clip is always active. |
| 8.2 entropy -0.0081 (off) | entropy versus policy at 3e-4 relative | Not addressed by cross-entropy. Its failure is a consequence of 7.9 (bonus comparable to the advantage scale). |
| 8.3 value 0.007 | critic versus policy in the shared trunk | No. Five levels are summed, each with a large irreducible floor (a two-hot of a Monte-Carlo-like return over 77 bins sits at perhaps 2-3 nats), so the value gradient is persistently of order 1 per level while the policy's is of order of the scaled advantage. A weight is still needed; what is no longer needed is per-level weights, schedules or target normalisation. |
| 8.5 next-state 0.0033, 8.6 correction 0.0033 | 12 and 14 summed groups per row, over all N rows | Partly. The tiny number is mostly the count of summed groups times twice as many rows as own ships. Reported as mean nats per group it becomes an ordinary auxiliary weight. |
| 8.7 enemy-action 0.11 | one 42-way group, high irreducible entropy | No, and its size relative to the others is not explained by units: per group it is weighted about 400x the next-state head. |
| 8.8 density 0.021 | Poisson deviance summed over 662 cells divided by own-ship count | Not a cross-entropy; the owner's premise does not hold for this row. Going anyway. |
| 8.9 shot-trajectory 0.007 | 15 summed groups | Partly, same as 8.5. Going anyway. |
| 8.10 shot-outcome 0.019, 8.12 geometry 0.007 | single group; mean of 5 groups | Yes in the sense that these are already per-group nats; they can share one coefficient. |
| 10.2 BC policy 0.025 versus RL 27 | keeps the policy's share of trunk gradient continuous across the hand-over | Redundant once advantages are standardised: a cross-entropy at coefficient 1 and a surrogate on unit-scale advantages at coefficient 1 are already of the same order. |
| 9.5 clip at 1.0 | total gradient size | Independent of the loss form; keep. |
| 12.19 / 12.47 per-token denominators | per-view-step terms versus per-ship terms | Not affected by the loss form. Disappears when outcome becomes a per-ship level and density goes: every remaining term is per ship or per pair. |
| 8.5 `physics_hidden_decay` weights | down-weights unpredictable hidden rows | Belief machinery; goes with it. |

Conclusion. The move to cross-entropy removed the need for target normalisation and per-level scaling, and that has already been cashed in. It did not remove the need for (a) an advantage normaliser, (b) one policy-to-value weight, (c) one auxiliary weight. The nine hand-set coefficients spanning 0.0033 to 27 can collapse to three numbers of order 1 if terms are expressed per group and advantages are standardised; the existing per-term gradient decomposition is the check.

---

## 3. Size-related risks in the learner, ranked

1. **Training shows exactly one size, one lineup, full teams, 10 fields, two arena radii.** Nothing in the learner prepares the policy for other counts, and the switches meant for other sizes (`mask_benched`, `mask_absent_fields`) are never trained on. Cheapest test: evaluation only. Take the current champion and play the scripted agent at 5v5, 10v10, 20v20 and 5v8 (a few hundred games each); the score curve against size is the baseline number for the whole project. Then one training arm with random benching (2-8 per team).
2. **No measurement of the headline goal inside training.** Ratings, opponent sampling and checkpoint choice all use 5v5 results. Test: add the periodic large-size evaluation and check whether the 5v5 rating and the 20v20 score even rank checkpoints the same way.
3. **Benched and dead ships under the valued mask** (defects 4 and 5 below) become live as soon as sizes are sampled by benching: benched ships count as valued, and GAE reads untrained values. Test: assert on a benched batch that no benched ship enters any loss denominator.
4. **Per-ship combat reward rate grows with enemies in range; critic support is fixed.** Only matters when training at larger sizes. Test: log per-level clamp fraction and the 99th percentile of |return| per level by team size.
5. **Terminal reward credit dilutes with team size.** Each ship gets the full +-10 but its influence on the result falls roughly as 1/S, so the outcome advantage becomes mostly noise at large training sizes while combat rewards stay sharp; behaviour shifts toward individual fighting. Test: outcome-level explained variance and the outcome share of advantage variance by team size.
6. **Objective reward split with team size is unknown.** Test: read `game/modes/`; one assertion on the per-ship sum at two sizes.
7. **Batch-level normaliser over a mixed batch.** Any single normaliser (return scale or standard deviation) lets the mode or size with the largest advantages set the step for the rest. Published remedy if it bites: per-task normalisation (Multi-task Deep RL with PopArt [FETCHED-T]). Test: log advantage standard deviation per mode and per size; act only if they differ by more than about 3x.
8. **One size per batch (12.46).** Benching avoids it; otherwise sizes must alternate per update, which makes each update's normaliser size-specific. No test needed; choose benching.
9. **Geometry coverage and team-deathmatch score target** scale with N. Low priority; training-only or input-side.

---

## 4. Things that look like defects

1. **Return-scale EMA has no bias correction and never converges within the launch budget. Confirmed in code** (`models/yemong/critic.py`, `ReturnScale`: `low` and `high` start at zero, decay 0.99, updated once per update). After 95 BC updates the EMAs hold 1 - 0.99^95 = 61% of the stationary gap; after the 48 RL updates, 76%. So the advantage divisor is 39% to 24% too small through the whole RL stage and the effective policy weight drifts down by about 20% during it. Assumes one BC update is also about 1.05M steps (the BC docstring's "8,192-view batch" supports this).
2. **Density normaliser divides by the wrong count. Confirmed in code** (`learn/ppo.py:727`: deviance over both sides' cells divided by valued own ships, clamp 1). Moot if the head goes.
3. **RL entropy coefficient is 0 through a silent override. Confirmed in code** (`cli.warm_start_entropy` replaces any factor not named on the command line by `WARM_START_ENTROPY = 0.0`; the dataclass default still reads -0.0081). This is deliberate, so not a bug, but the printed default misstates the champion and the justification rests on the scale mismatch described under 8.2.
4. **GAE bootstraps through values the critic is never trained on. Confirmed in code that there is no masking** (`Learner.chained_returns` passes `chain["values"]` straight to `per_level_gae`; `valued` is used only in the loss). A ship that is dead with `respawn_wait == 0` (waiting for a closed wave, or dead for good) is outside the valued mask, yet its value output is the bootstrap for the tick on which it died and for the trace before it. **Not confirmed**: how often that state occurs per mode (`game/physics/step.py:335-342` shows the wait counting down to 0 and respawn only when the wave is open; elimination's respawn length was not traced).
5. **Benched ships are "valued". Confirmed in code** (`game/physics/step.py:149-156` holds a benched ship's `respawn_wait` at the full respawn length; `runtime/rollout.py:459-460` sets valued = alive or wait > 0). Harmless today because training never benches; wrong the moment benching is used for size sampling.
6. **Outcome value changes meaning at the BC-to-RL hand-over. Confirmed from the two configs** (`learn/bc.py` DISCOUNT[-1] = 1.0, TRACE_DECAY[-1] = 1.0; RL 0.9997, 0.97). In BC the head learns the undiscounted result; in RL the target puts 1 - gamma per tick on "unresolved", which early in a 600 s game is nearly all the mass (0.9997^18000 = 0.005). The first RL updates therefore see a systematically wrong outcome value, hence biased outcome advantages, exactly when the policy is leaving the clone. Disappears when BC and RL share settings.
7. **Damage trace shorter than bullet flight** (gamma x lambda = 0.795 per tick against up to 11 ticks of flight, bullets unobserved). From the inventory only; not checked against bullet life in ticks. Confirm by regressing the damage-level advantage of the shoot action on eventual hit.
8. **Gradient clip probably always active.** Inferred from coefficient 27 and norm 1.0; not checked. Confirm from the logged pre-clip norm.
9. **RL stage is about 48 updates.** Arithmetic from launch.toml as reported by the inventory; not a code defect, a budget one.
10. **Outcome discount half-life is 77 s in a 600 s match** (0.9997 at 30 Hz). OpenAI Five's 0.9997 was at a much lower decision rate [MEMORY], giving minutes. If the number was carried over without converting for tick rate, the outcome barely reaches the first two thirds of a game. Inventory only; test 0.9999.

---

## 5. What the inventory does not tell me

- Typical magnitude of the advantage relative to the return scale, per level; clamp fractions; explained variance per level. These decide the whole of section 2 quantitatively.
- How often the KL stop fires, the pre-clip gradient norm, the measured per-term trunk-gradient shares.
- What initial recurrent state the learner's `unroll` uses for a 128-tick segment (stored rollout state, zeros, or burn-in), and therefore how stale it is by epoch 4.
- Whether the lookahead ticks count against the step budget and whether the same ticks are replayed as the next update's first shard.
- How objective rewards divide among ships; elimination's respawn rule.
- Whether BC updates have the same size as RL updates; where the BC KL plateaus.
- Actual run lengths used beyond launch.toml's 150M (the snapshot schedule suggests longer).
- Wall-clock shares (ops are not seconds): shot re-simulation, lanes, lookahead.
- Whether lanes' results show the weight averages beating the live policy.
- Evaluation-time switches at 20v20/50v50, and whether any such evaluation has ever been run.
- Whether Adam state, the uniform average or the return scale are meant to survive a resume inside a stage.

---

## 6. Papers cited

- Schulman et al., Proximal Policy Optimization Algorithms, arXiv 1707.06347 [FETCHED-T]
- Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation, arXiv 1506.02438 [FETCHED-T]
- Andrychowicz et al., What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study, arXiv 2006.05990 [FETCHED-T]
- Engstrom et al., Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO, arXiv 2005.12729 [FETCHED-T]
- Huang et al. 2022, The 37 Implementation Details of Proximal Policy Optimization (ICLR blog track) [MEMORY]
- Burda et al., Exploration by Random Network Distillation, arXiv 1810.12894 [FETCHED-T]
- van Seijen et al., Hybrid Reward Architecture for Reinforcement Learning, arXiv 1706.04208 [FETCHED-T]
- Hafner et al., Mastering Diverse Domains through World Models (DreamerV3), arXiv 2301.04104 [FETCHED-T]
- Farebrother et al., Stop Regressing: Training Value Functions via Classification for Scalable Deep RL, arXiv 2403.03950 [FETCHED-T]
- Hessel et al., Multi-task Deep Reinforcement Learning with PopArt, arXiv 1809.04474 [FETCHED-T]
- Pardo et al., Time Limits in Reinforcement Learning, arXiv 1712.00378 [FETCHED-T]
- Berner et al. (OpenAI), Dota 2 with Large Scale Deep Reinforcement Learning, arXiv 1912.06680 [FETCHED-T]
- Vinyals et al. 2019, Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature [MEMORY]
- Yu et al., The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games, arXiv 2103.01955 [FETCHED-T]
- Baker et al., Emergent Tool Use From Multi-Agent Autocurricula, arXiv 1909.07528 [FETCHED-T]
- Iqbal et al., Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning, arXiv 2006.04222 [FETCHED-T]
- Hu et al., UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers, arXiv 2101.08001 [FETCHED-T]
- Long et al., Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning, arXiv 2003.10423 [FETCHED-T]
- Jaderberg et al., Reinforcement Learning with Unsupervised Auxiliary Tasks, arXiv 1611.05397 [FETCHED-T]
- Hernandez-Leal et al., Agent Modeling as Auxiliary Task for Deep Reinforcement Learning, arXiv 1907.09597 [FETCHED-T]
- Lample and Chaplot, Playing FPS Games with Deep Reinforcement Learning, arXiv 1609.05521 [FETCHED-T]
- Ross et al., A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, arXiv 1011.0686 [FETCHED-T]
- Schmitt et al., Kickstarting Deep Reinforcement Learning, arXiv 1803.03835 [FETCHED-T]
- Ng, Harada, Russell 1999, Policy invariance under reward transformations, ICML [MEMORY]
- Nikishin et al. 2018, Improving Stability in Deep Reinforcement Learning with Weight Averaging [MEMORY]

Fetched but not relied on: arXiv 1902.06865 (Hyperbolic Discounting and Learning over Multiple Horizons), 1802.01561 (IMPALA), 1707.01495 (Hindsight Experience Replay).
