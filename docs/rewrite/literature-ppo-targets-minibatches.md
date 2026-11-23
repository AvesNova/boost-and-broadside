# PPO: when to compute advantages and value targets, and how many minibatches

Literature and code review, 2026-11-23, for the learner in `lean/learn` (branch core). It
answers two questions from docs/rewrite/core-decisions.md ("Learner"):

1. Should advantages and value targets come once per update from the values stored at
   collection, or be recomputed from the current critic (every epoch, or more often)?
2. How many minibatches per epoch for an update of about 1M environment ticks?

Each claim is marked as one of: **measured** (a source ran an experiment), **default** (a
library or system uses it, without a published ablation), or **inference** (mine). Library
code was read on 2026-11-23; where a repository was cloned, the commit is given in the
source list.

The project's numbers used below: an update is 2,048 environments x 512 ticks = 1,048,576
environment ticks, about 4.7M ship-steps (4 to 5 learner ships per environment); 4 epochs;
32 minibatches of whole environments (64 environments, 32,768 environment ticks, about 150k
ship-steps each), so 128 Adam steps per update; learning rate 3e-4; clip 0.2; gradient
clip 1.0; advantages summed over the 5 levels and divided by their standard deviation over
the whole update (no mean subtraction) (`lean/learn/config.py`, `lean/learn/returns.py`,
`lean/learn/update.py`).

## Summary

| Question | Recommendation | Evidence | Speed |
|---|---|---|---|
| Advantages and value targets | Once per update, from the values stored at collection (`targets_every_epoch=False`) | Weak either way by direct measurement; most implementations compute once per update (CleanRL, SB3, PureJaxRL, Sample Factory, RLlib, Acme, TRL, verl); Brax and PufferLib recompute inside each minibatch at no extra forward cost | Measured here: loop 5,160 to 6,457 steps/s (+25%) at 256 environments on the laptop (docs/rewrite/scoreboard.md) |
| If per-epoch is kept | Skip the recompute in epoch 1: the weights have not changed since collection | Follows from the definition (inference) | Saves about a quarter of the target-pass cost |
| Advantage normalisation | Keep it over the whole update | Measured as unimportant at small scale; at 150k samples per minibatch per-minibatch and per-update statistics are practically the same (inference) | None |
| Minibatches per epoch | Keep 32 (about 150k ship-steps each, 128 Adam steps per update) | No source supports a different count at this update size; the best-supported rule is about changes: keep minibatch size and learning rate, change the count | On the 8 GB laptop nearly neutral (micro-batches); on a large GPU fewer, larger minibatches may be faster |

## 1. Advantages and value targets

### 1.1 The options

| Option | What it costs | Who does it |
|---|---|---|
| A. Once per update from the values stored at collection | Nothing extra | PPO paper's Algorithm 1, OpenAI Baselines, CleanRL, Stable-Baselines3, PureJaxRL, Sample Factory (GAE path), TRL, verl |
| B. Once per update, values recomputed by the learner at the start of the update | One no-grad forward over the rollout per update | RLlib (new API stack), Acme PPO. In a synchronous learner the weights equal the collection weights, so A and B coincide up to rounding |
| C. Recomputed at the start of every epoch | One no-grad forward over the rollout per epoch | Proposed by Andrychowicz et al. 2021; TorchRL PPO tutorial; this project today |
| D. Recomputed inside every minibatch from that minibatch's own value forward | Nothing extra if minibatches are whole trajectory fragments (the loss already computes the values) | Brax PPO; PufferLib 5.0 (optionally with V-trace clipping) |
| E. Recomputed before every minibatch over the whole batch, values updated in place, V-trace-style clipping by the current ratio | GAE over the whole batch per minibatch (no extra forward) | PufferLib 3.0 |
| F. Off-policy targets (V-trace, UPGO) from the current critic per minibatch | Usually none beyond the loss forward | IMPALA, AlphaStar, Sample Factory with `--with_vtrace`; designed for asynchronous actors whose data lags the learner |

### 1.2 What has been measured

**Andrychowicz et al. 2021, "What Matters in On-Policy Reinforcement Learning?"** (ICLR
2021, [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)), Section 3.5 and Appendices
B.1 and H.

- Setup (measured): MuJoCo (Ant, HalfCheetah, Hopper, Humanoid, Walker2d), 1M or 2M
  environment steps, small MLPs, defaults of 2,048 transitions per iteration, 10 epochs,
  minibatch 64. Metric: 95th percentile of performance over randomly sampled configurations
  of the choices in one group, 3 seeds each.
- Choice C5 ("batch_mode") had four variants: *Fixed trajectories*, *Shuffle trajectories*,
  *Shuffle transitions* (advantages precomputed once per iteration, which the paper says is
  how OpenAI Baselines' ppo2 works) and *Shuffle transitions (recompute advantages)*
  (recomputed at the start of each epoch). The paper says shuffling transitions "makes the
  estimation of advantages impossible in each minibatch ... so we precompute all advantages
  at the beginning of each iteration"; from that I infer (not stated outright) that the
  two trajectory variants estimate advantages inside each minibatch, i.e. already fresh.
- Result, in the paper's words: "The plots suggest that stale advantages can in fact hurt
  performance and that recomputing them at the beginning of each pass at least partially
  mitigates the problem and performs best among all variants." Recommendation: "Shuffle
  individual transitions before assigning them to minibatches and recompute advantages once
  per data pass."
- Size of the effect (read off Figures 53 and 58 by me; approximate): the per-epoch
  recompute variant has the highest or near-highest 95th-percentile score on four of five
  tasks and is slightly below the trajectory variants on Humanoid. The clearest gap is
  against the stale *Shuffle transitions* variant on HalfCheetah (very roughly 1,400 against
  1,800); on the other tasks the variants are within roughly 10-20% of each other, with
  overlapping error bars on several. In the "top 5% of configurations" panel the recompute
  variant is the most frequent overall. No effect size is given in the text.
- What it does not test: recurrent policies, multi-agent or self-play settings, updates
  larger than 4,096 transitions, and option A against option C with whole-trajectory
  minibatches (the configuration this project would use).

**Pleines et al. 2022, "Generalization, Mayhems and Limits in Recurrent PPO"**
([arXiv 2205.11104](https://arxiv.org/abs/2205.11104), Section 5.3 and Appendix E).
Measured: recurrent PPO (GRU/LSTM) on two memory tasks (Mortar Mayhem, Dancing the Ballet),
refreshing hidden states and advantages after every epoch or every other epoch. "Refreshing
stale hidden states and advantages does not yield promising results. Applying the refresh
every other epoch slightly improves the performance in Mortar Mayhem on novel seeds ... This
minor improvement is likely not worth the cost of another forward pass on the entire data."
Gradient norms and KL showed no substantial difference. Small study (few seeds, two
environments), but it is the only measurement I found with a recurrent policy, which is the
case here.

**Indirect evidence.**
- Phasic Policy Gradient (Cobbe et al. 2020, [arXiv 2009.04416](https://arxiv.org/abs/2009.04416),
  Section 2): the value targets of the auxiliary phase are "the same targets computed during
  the policy phase. They remain fixed throughout the auxiliary phase", which runs 6 epochs.
  So the authors were content with fixed targets over many value epochs (default, not an
  ablation of freshness).
- OpenAI Five (Berner et al. 2019, [arXiv 1912.06680](https://arxiv.org/abs/1912.06680),
  Section 4.4, Figure 5b) measured that *data* staleness of about 8 parameter versions caused
  significant slowdowns; it targets staleness 0-1. That concerns the behaviour policy, not
  advantage freshness within one update. Where Five computed GAE is not stated in the paper;
  I could not verify it.
- Hilton, Cobbe and Schulman 2022 ([arXiv 2110.00641](https://arxiv.org/abs/2110.00641),
  NeurIPS 2022) show PPO tolerates data from several iterations ago if the proximal policy
  is decoupled from the behaviour policy. Not about advantage staleness.

**2022-2026.** A search for later ablations of per-epoch or per-minibatch advantage
recomputation in on-policy RL found nothing beyond the two studies above. The LLM-RLHF PPO
stacks (TRL, verl) compute once per update. I found no study that measures option C against
option A at large scale or with self-play.

### 1.3 What the main implementations do

| Implementation | When advantages and targets are computed | Values from | Advantage normalisation | Source |
|---|---|---|---|---|
| PPO paper (Schulman et al. 2017) | Once per iteration, then K epochs | Collection | Not described | [arXiv 1707.06347](https://arxiv.org/abs/1707.06347), Algorithm 1 |
| CleanRL `ppo.py` | Once, before the epoch loop | Collection | Per minibatch, mean and std (`norm_adv=True`) | `cleanrl/ppo.py` lines 230 (GAE), 244 (epochs), 262 (normalisation) |
| CleanRL `ppo_atari_lstm.py` | Once (recurrent) | Collection | Per minibatch | `cleanrl/ppo_atari_lstm.py` lines 284, 301 |
| Stable-Baselines3 PPO | Once, at the end of `collect_rollouts` | Collection | Per minibatch (`normalize_advantage=True`) | `common/on_policy_algorithm.py` line 262; `ppo/ppo.py` line 219 |
| PureJaxRL | Once per update | Collection | Per minibatch | `purejaxrl/ppo.py` lines 166, 190, 248 |
| Sample Factory (default GAE) | Once per batch; "in case of V-trace it is done separately for each minibatch" | Collection (bootstrap value recomputed) | Per minibatch | `algo/learning/learner.py` lines 993-994, 647, 707 |
| RLlib (new API stack) | Once per train batch, in the learner connector `GeneralAdvantageEstimation` | Recomputed by the learner (`module.compute_values`) | Whole train batch, per module | `rllib/connectors/learner/general_advantage_estimation.py` lines 104, 156-157 |
| Acme PPO (DeepMind) | Once per update | Recomputed by the learner at the start of the update | Divided by an EMA of mean absolute advantage, no mean subtraction | `acme/agents/jax/ppo/learning.py` lines 294, 331, 356-372 |
| TorchRL PPO tutorial | Every epoch: "We re-compute it at each epoch as its value depends on the value network which is updated in the inner loop." | Current critic | (per tutorial) | [TorchRL PPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html) |
| Brax PPO | Inside the loss, per minibatch of whole unrolls | Current critic (`value_apply(params.value, ...)`, stop-gradient) | Per minibatch | `brax/training/agents/ppo/losses.py` lines 205, 221, 237; `train.py` lines 544, 553 |
| PufferLib 3.0 | Before every minibatch, over the whole batch, with V-trace clipping by the latest ratios; values overwritten as samples are trained | Mix of collection and current | Per minibatch, with priority weights | `pufferlib/pufferl.py` (branch 3.0) lines 348-375, 394, 409 |
| PufferLib 5.0 (current) | Inside every minibatch (`puff_advantage` kernel) | Current critic ("live V in"); V-trace optional (`vtrace=0` default) | (in the CUDA loss) | `src/pufferl.cu` lines 1511-1539; `src/algo.cu` line 1244 |
| TRL `PPOTrainer` (v0.24.0; no longer in TRL main) | Once, then `num_ppo_epochs=4` | No-grad forward after generation | Whole batch (`masked_whiten`) | `trl/trainer/ppo_trainer.py` lines 519, 574-585, 589 |
| verl | Once per step: compute values, then advantages, then update critic and actor | Critic forward before the update | Whole batch (`masked_whiten`) | `verl/trainer/ppo/v1/trainer_base.py` lines 656-664; `trainer/ppo/core_algos.py` line 262 |
| OpenAI Five | Not stated | Not stated | Not stated | arXiv 1912.06680 |
| IMPALA, AlphaStar | V-trace (AlphaStar also UPGO) targets with the current critic on each learner batch | Current critic | n/a | [arXiv 1802.01561](https://arxiv.org/abs/1802.01561); [Nature 2019](https://www.nature.com/articles/s41586-019-1724-z) |

Reading of the table (inference): option A/B (once per update) is the classic and still the
majority default, including the RLHF stacks. The libraries that recompute are either cheap
to do so (Brax, PufferLib: in-minibatch, no extra forward) or are a tutorial (TorchRL). No
library I checked pays a separate forward over the whole rollout per epoch by default except
the TorchRL tutorial.

### 1.4 Advantage normalisation

- Measured: per-minibatch normalisation against none "seems not to affect the performance
  too much" (Andrychowicz et al., Section 3.3, choice C67, Figure 35).
- Hilton et al. 2022 (Section 6): at small batch sizes the normalisation statistics can be
  noisy; they estimate them with EWMAs over iterations and advise monitoring the advantage
  standard deviation ("If estimates oscillate by a factor of 10 or more ...").
- Defaults: per minibatch with mean subtraction in CleanRL, SB3, PureJaxRL, Brax, Sample
  Factory; whole batch in RLlib, TRL, verl; scale-only (no mean subtraction) EMA in Acme.
- Inference for this project: with about 150k ship-steps per minibatch, per-minibatch and
  per-update statistics are practically equal, so the choice does not matter for learning.
  Per-update is one statistic per update, which is simpler with option A. The project's
  scale-only form (no mean subtraction) differs from most defaults but has a precedent in
  Acme; I found no measurement of mean subtraction in either direction.

### 1.5 For this project

- Cost, measured here (docs/rewrite/scoreboard.md, 256 environments, laptop, compiled):
  the per-epoch target pass is about 1.3 s of a 5.1 s update; loop throughput 5,160 steps/s
  with it, 6,457 without (+25%).
- Epoch 1 recompute is redundant (inference): the update starts from the weights that
  collected the rollout, so the epoch-1 values equal the stored ones up to rounding (bf16,
  sequence pass against step-by-step pass). If per-epoch targets are kept, start them at
  epoch 2; that removes a quarter of the pass.
- The policy is recurrent and minibatches are whole environments; the closest measurement
  (Pleines et al.) found no meaningful gain from refreshing. The Andrychowicz gain is
  modest, on small MuJoCo tasks, and mostly against a stale *transition*-shuffled variant.
- A zero-extra-forward alternative exists (option D, as Brax and PufferLib 5.0): compute GAE
  inside each minibatch from the detached values of the minibatch's own forward. It is
  exact only if each minibatch holds an environment's whole rollout (all 512 ticks, or the
  bootstrap value at each fragment end); with 128-tick fragments the long-horizon levels
  (discount 0.999 and 0.9997) would be cut at the fragment boundary. It also moves
  normalisation and range monitoring into the minibatch. Not recommended now; worth knowing
  if fresher targets become wanted.

**Recommendation: option A (`targets_every_epoch=False`).** Strength: moderate by
consensus of implementations, weak by direct measurement in either direction. Gains 25%
loop throughput at the measured setting. If the author prefers to keep per-epoch targets,
skip epoch 1.

## 2. Number of minibatches

### 2.1 What has been measured

**Andrychowicz et al. 2021** (Section 3.5, Appendix H; MuJoCo, iterations of 512-4,096
transitions). Minibatch size {64, 128, 256}: "Increasing the batch size (C4) does not appear
to hurt the sample complexity in the range we tested (Fig. 57) which suggests that it should
be increased for faster iteration speed." Transitions per iteration (C2) "influences the
performance quite significantly". Epochs {1, 3, 10}: "going over the experience multiple
times appears to be crucial". The paper also notes that learning rate and minibatch size
interact and must be tuned together (Section 2).

**MAPPO** (Yu et al. 2022, NeurIPS Datasets and Benchmarks,
[arXiv 2103.01955](https://arxiv.org/abs/2103.01955), Section 5.3). Measured on SMAC, MPE,
GRF, Hanabi with 1, 2 or 4 minibatches: "when using 4 mini-batches, MAPPO fails to solve any
of the selected maps while using 1 mini-batch produces the best performance on 22/23 maps."
Suggestion 3: "Use at most 10 training epochs on difficult environments and 15 training
epochs on easy environments. Additionally, avoid splitting data into mini-batches."
Context that limits transfer (inference): their batches are small (SMAC: 8 environments x
400 steps x agents; GRF: 128 x 25), so one minibatch there is thousands to tens of
thousands of samples, and fewer minibatches also means fewer optimiser steps per update;
the paper attributes the effect to non-stationarity and to "using more data to estimate
gradients". Section 5.5: there is a critical batch size below which MAPPO is poor, and
beyond it larger batches can hurt sample efficiency.

**OpenAI Five** (arXiv 1912.06680, Section 4.3, Appendix M). One optimiser step per batch
of about 1M to 3M timesteps (Rerun: 983,040), sample reuse targeted at about 1 (reuse of
2-3 caused about a 2x slowdown, 8 could prevent learning; Section 4.4). Batch 983k against
123k: speed-up about 2.5x early in training, less than linear, learning rate not retuned
per batch size.

**McCandlish et al. 2018, "An Empirical Model of Large-Batch Training"**
([arXiv 1812.06162](https://arxiv.org/abs/1812.06162), Table 1). Measured critical batch
sizes (where compute efficiency falls to half): Atari (A2C) 400-8,000 frames averaged over
training; Dota 1v1 (PPO) 3,000,000; Dota 5v5 estimated above 8,000,000. The critical batch
grows during training. This is about samples per *gradient step*, i.e. the minibatch.

**Hilton, Cobbe, Schulman 2022** (arXiv 2110.00641, Sections 3 and 6). Measured: with Adam,
dividing the batch by c is compensated by dividing the step size by sqrt(c) (Appendix C
verifies the square-root rule against the linear one), plus keeping the proximal policy's
age constant (PPO-EWMA). Advice (labelled by the authors as largely folklore): the learning
rate is "by far the most important hyperparameter"; monitor the clipped fraction: much less
than 1% suggests a larger iteration batch, "much more than 10% with 1 policy epoch or 20%
with multiple policy epochs" suggests the learning rate is too high.

**Beukman et al. 2026, "Preventing Learning Stagnation in PPO by Scaling to 1 Million
Parallel Environments"** ([RLJ 2026](https://rlj.cs.umass.edu/2026/papers/Paper19.pdf),
Sections 4-5). Measured (Kinetix, Isaac Gym robotics): when the number of environments
grows, three options were compared: more minibatches of the same size; larger minibatches
with the same learning rate; larger minibatches with the square-root learning-rate rule.
"Keeping the inner optimisation process unchanged performs best"; "we recommend a
stability-first procedure: increase the number of minibatches while keeping the learning
rate and minibatch size fixed. Only increase minibatch size (adjusting the learning rate
appropriately) if hardware utilization is a bottleneck." Reverting Isaac Gym's minibatch
from 98,304 to the default 16,384 at 24,576 environments "significantly outperforms" for
PPO and SAPG. More epochs plateau earlier unless regularisation (clip or EWMA proximal
policy) is stronger.

**PPG** (Cobbe et al. 2020, Section 3.2, Appendix D). In PPO on Procgen, 3 epochs per
rollout was best; once value training is decoupled, one policy epoch is near-optimal: "the
PPO baseline benefits from greater sample reuse only because the extra epochs offer
additional value function training."

**Li et al. 2026, "When Do Larger Batches Help Scale LLM Reinforcement Learning?"**
([arXiv 2608.29296](https://arxiv.org/abs/2608.29296)). From the abstract only (I did not
read the full paper): square-root learning-rate scaling with Adam keeps learning curves
roughly consistent over a bounded range of batch sizes, for GRPO and PPO; a larger batch
helps only when its throughput gain outweighs its samples-to-target penalty.

### 2.2 Defaults

| Source | Samples per update | Minibatches | Minibatch size | Epochs | Adam steps per update |
|---|---|---|---|---|---|
| PPO paper, MuJoCo | 2,048 | 32 | 64 | 10 | 320 |
| PPO paper, Atari | 1,024 (8 x 128) | 4 | 256 | 3 | 12 |
| PPO paper, Roboschool | 16,384 or 65,536 | 4 or 16 | 4,096 | 15 | 60 or 240 |
| CleanRL `ppo.py` (Atari-style) | 512 (4 x 128) | 4 | 128 | 4 | 16 |
| CleanRL `ppo_continuous_action.py` | 2,048 | 32 | 64 | 10 | 320 |
| SB3 | 2,048 per env | 32 (one env) | 64 | 10 | 320 |
| RLlib | 4,000 | about 31 | 128 | 30 | about 940 |
| Brax | 16 x 32 x 10 = 5,120 | 16 | 320 (32 unrolls of 10) | 2 | 32 |
| PufferLib 3.0 / 5.0 | all agents x 64 | batch / 8,192 | 8,192 (fixed) | 1 (`update_epochs=1`, `replay_ratio=1.0`) | batch / 8,192 |
| Sample Factory | 1,024 | 1 | 1,024 | 1 | 1 |
| MAPPO (SMAC) | 3,200 x agents | 1 | whole batch | 5-15 | 5-15 |
| OpenAI Five | 1M-3M timesteps | 1 step per batch | whole batch | reuse about 1 | about 1 |
| TRL | per step | 1 | whole batch | 4 | 4 |
| verl | 1,024 prompts | 4 | 256 prompts | 1 | 4 |
| This project | about 4.7M ship-steps | 32 | about 150k ship-steps | 4 | 128 |

Defaults are not measurements; most were tuned for much smaller updates.

### 2.3 How the knobs interact

- Epochs x minibatches is the number of Adam steps per update; with the learning rate it sets
  how far the policy can move before the clip (or KL) stops it. Changing the minibatch count
  at a fixed update changes both the step count and the gradient noise per step (Beukman et
  al.; Hilton et al.).
- Minibatch size is about gradient noise. In this project a minibatch is 64 whole
  environments, so its samples are strongly correlated within an environment; the
  effective number of independent samples is much smaller than 150k (inference). The Dota
  measurements put the critical batch for a team game in the millions of frames (McCandlish
  et al.), which suggests (inference) that 150k ship-steps per step is not wastefully large.
- Changing the minibatch size calls for a square-root change of the Adam learning rate
  (Hilton et al., measured; Li et al. 2026, LLMs).
- The clipped fraction and approximate KL are the agreed monitors; Hilton et al.'s bands
  (above about 1%, below about 20% with several epochs) are advice, not measurement.

### 2.4 For this project

- No source measured minibatch counts at an update of about 1M environment ticks with a
  recurrent policy, so no number is "best practice" in the sense of evidence. MAPPO's "one
  minibatch" comes from updates hundreds to thousands of times smaller, where one minibatch
  is the only way to have a large gradient batch.
- 32 minibatches of about 150k ship-steps, 4 epochs, 128 Adam steps per update, sits inside
  the range of common practice (12 to 940 steps per update) with minibatches larger than any
  library default except OpenAI Five's.
- The best-supported rule (Beukman et al. 2026, measured) governs changes, not the starting
  value: when the update size changes, keep the minibatch size and learning rate and change
  the count. Move to fewer, larger minibatches only for hardware speed, then scale the
  learning rate by the square root of the size ratio (Hilton et al.).
- Speed (inference from scoreboard.md): on the 8 GB laptop a minibatch is built from
  micro-batches with gradient accumulation, so the forward and backward cost per tick does
  not depend on the count; Adam and clipping together are under 0.1 s per update. The count
  is nearly speed-neutral there. On a large rented GPU, fewer and larger minibatches can be
  faster until the GPU saturates.

**Recommendation: keep 32 minibatches** (64 environments per minibatch at 2,048
environments), 4 epochs, learning rate 3e-4. Watch the clipped fraction and approximate KL
per update. Strength: weak (no direct evidence for any count at this scale); it is a
defensible default, not an optimum. If the environment count changes, keep 64
environments per minibatch and change the count.

## What I could not verify

- Where OpenAI Five computed GAE (rollout workers or optimisers) and from which values.
- The exact effect sizes in Andrychowicz et al. Figures 53 and 58; the text gives none, and
  my figures are read off the bar charts.
- That the two trajectory variants in Andrychowicz et al. compute advantages inside each
  minibatch (inferred from the text quoted above).
- Li et al. 2026 beyond its abstract.
- RLlib's learner connector is called once per train batch (`Learner._make_batch_if_necessary`,
  `core/learner/learner.py` around line 1677); I did not trace every update path.

## Sources

Papers:
- Schulman et al. 2017, Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- Andrychowicz et al. 2021, What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study. https://arxiv.org/abs/2006.05990
- Engstrom et al. 2020, Implementation Matters in Deep Policy Gradients. https://arxiv.org/abs/2005.12729 (studies code-level optimisations; it does not ablate the minibatch count or advantage recomputation, so it is not cited above)
- Huang et al. 2022, The 37 Implementation Details of Proximal Policy Optimization (ICLR blog track). https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (details 6 and 7: minibatch updates, `nminibatches` 4 Atari, 32 MuJoCo, 8 Procgen; advantage normalisation "at the minibatch level")
- Yu et al. 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. https://arxiv.org/abs/2103.01955
- Pleines et al. 2022, Generalization, Mayhems and Limits in Recurrent Proximal Policy Optimization. https://arxiv.org/abs/2205.11104
- Cobbe et al. 2020, Phasic Policy Gradient. https://arxiv.org/abs/2009.04416
- Hilton, Cobbe, Schulman 2022, Batch size-invariance for policy optimization. https://arxiv.org/abs/2110.00641
- McCandlish et al. 2018, An Empirical Model of Large-Batch Training. https://arxiv.org/abs/1812.06162
- Berner et al. 2019, Dota 2 with Large Scale Deep Reinforcement Learning. https://arxiv.org/abs/1912.06680
- Espeholt et al. 2018, IMPALA. https://arxiv.org/abs/1802.01561
- Vinyals et al. 2019, Grandmaster level in StarCraft II using multi-agent reinforcement learning. https://www.nature.com/articles/s41586-019-1724-z
- Beukman et al. 2026, Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments (RLJ 2026). https://rlj.cs.umass.edu/2026/papers/Paper19.pdf
- Li et al. 2026, When Do Larger Batches Help Scale LLM Reinforcement Learning? https://arxiv.org/abs/2608.29296

Code (read 2026-11-23):
- CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py, `ppo_continuous_action.py`, `ppo_atari_lstm.py`
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py, `common/on_policy_algorithm.py`
- PureJaxRL: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
- Brax: https://github.com/google/brax/blob/main/brax/training/agents/ppo/losses.py, `train.py`
- Sample Factory: https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/learning/learner.py, `cfg/cfg.py` (defaults `num_epochs=1`, `num_batches_per_epoch=1`, `batch_size=1024`)
- RLlib: https://github.com/ray-project/ray, commit 13b350de (2026-11-21), `python/ray/rllib/algorithms/ppo/ppo.py` lines 118-135 (lr 5e-5, `num_epochs=30`, `minibatch_size=128`, `train_batch_size=4000`, KL loss on, `clip_param=0.3`), `python/ray/rllib/connectors/learner/general_advantage_estimation.py`
- Acme: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py (constructor defaults `num_epochs=4`, `num_minibatches=1`)
- TorchRL PPO tutorial: https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html
- PufferLib: https://github.com/PufferAI/PufferLib, branch 5.0 commit ea7f9893 (2026-10-14), `src/pufferl.cu`, `src/algo.cu`, `config/default.ini`; branch 3.0 `pufferlib/pufferl.py`, `pufferlib/config/default.ini`
- TRL: https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/ppo_trainer.py and `ppo_config.py`, `utils.py` (`num_mini_batches=1`); the PPO trainer is absent from TRL main at ba6fb013 (2026-11-21)
- verl: https://github.com/volcengine/verl, commit 453954b3 (2026-11-20), `verl/trainer/ppo/v1/trainer_base.py`, `verl/trainer/ppo/core_algos.py`, `verl/trainer/config/actor/actor.yaml` (`ppo_epochs: 1`, `ppo_mini_batch_size: 256`)
