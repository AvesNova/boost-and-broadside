# Literature check: rollout shape, lookahead, optimiser, precision, next-state targets, rotary bands (2026-11-23)

Scope: author-review.md 6.1 (rotary, the open parts), 6.3 and the follow-up table (next-state
target form), 7.6 (lookahead), 7.7 (optimiser), 7.8 (precision), 7.10 and "Rollout shape"
(how the million ticks are split). It builds on literature-model-choices.md (HL-Gauss for
next-state, the target form (a)/(b)/(c) with Nagabandi and GNS, the band ratio and the
coarsest/finest limits) and literature-ppo-targets-minibatches.md (targets once per update,
32 minibatches, advantage normalisation), and on provenance-check.md section 7 (the partial
rotary table and Barbero et al.'s Table 2). What those settled is referred to, not repeated.

Labels on each claim:
- **measured**: a primary source (or this tree, where said) reports an experiment.
- **default**: what released code or a paper does, without an ablation.
- **inference**: my reasoning or arithmetic; nothing measured it.

Papers were downloaded from arXiv on 2026-11-23 and the named passages read in the text
extracted from the PDF, unless marked otherwise. Lean line numbers are from the tree at
6785907d; v2 line numbers from `src/` in the same checkout.

## Project numbers used below

- 30 ticks per second, one decision per tick. Discounts (0.982, 0.990, 0.995, 0.999, 0.9997)
  and GAE lambdas (0.81, 0.90, 0.95, 0.97, 0.97) (`lean/learn/config.py:25-26`).
- Arithmetic (inference): the GAE horizon 1/(1 − γλ) is 4.9, 9.2, 18.3, 32.3 and 33.0 ticks
  (0.16 to 1.10 s). The discount horizon 1/(1 − γ) is 56, 100, 200, 1,000 and 3,333 ticks
  (1.9 s to 111 s). With λ ≤ 0.97 no level's advantage reaches past about 100 ticks with
  more than 5% weight. Beyond that, credit flows only through the critic.
- An update: 2,048 environments × 4 segments × 128 ticks = 1,048,576 ticks
  (`config.py:43-44`); 12,288 views; 32 minibatches × 4 epochs = 128 Adam steps.
- Laptop measurement (docs/rewrite/scoreboard.md, v2 game row): collect 22,032 steps/s,
  update 8,482, loop 6,124. Collection is about 28% of the loop, and the policy dominates a
  collection tick (scoreboard line 82).

---

## 1. Lookahead before the bootstrap

### What lean does, and what v2 did

- Lean computes GAE **per 128-tick segment**. Each view's segment is bootstrapped from the
  value stored at its own end (`lean/learn/store.py:211-217` `end_segment` stores a
  bootstrap per view; `lean/learn/update.py:148-149` runs `generalised_advantages` per view
  with `store.bootstrap`). The four segments of one environment are consecutive in time, but
  their returns are still cut at ticks 128, 256 and 384, although the real rewards after
  each cut are already in the store.
- v2 chained the returns over the whole update: "the returns over the whole update's ticks,
  the shards in order and then the lookahead ... bootstrapped only at the end"
  (`src/bnb/learn/ppo.py:6-8`; `prepare`, `ppo.py:1015-1029`, concatenates the shards and
  the lookahead before `chained_returns`). The lookahead was 128 ticks, played after the
  fourth shard and never trained on, and the next update continued from its end
  (`src/bnb/runtime/train.py:158-161, 210, 577-590`). So v2 collected 640 ticks per
  environment per update and trained on 512 (20% of collection never trained). It also used
  the lookahead for shot labels.
- This is a regression from v2 that author-review 7.6 does not mention. Lean dropped the
  lookahead, which is what 7.6 asks about, and it also dropped the chaining, which 7.6 does
  not mention.

### Sources

- **Pardo, Tavakoli, Levdik, Kormushev, "Time Limits in Reinforcement Learning", ICML 2018**,
  https://arxiv.org/abs/1712.00378 (PDF read, §3 and §3.2). Partial-episode bootstrapping
  (PEB): bootstrap from the value of the last state when an episode is cut for a reason other
  than the environment. PPO with PEB on Hopper and Walker, trained on 300-step partial
  episodes, "significantly outperforms the standard PPO" (Fig. 5). For tasks that are
  time-limited by nature, the paper's other proposal is to put the remaining time in the input
  (time-awareness). (measured) The paper is about treating a cut as a cut. It does not study
  the bias of bootstrapping from an imperfect value, and it does not study playing extra ticks
  past the cut. (finding)
- **Andrychowicz et al. 2021, "What Matters in On-Policy RL"**, https://arxiv.org/abs/2006.05990
  (§3.5). At a fixed number of transitions per iteration, more parallel environments make
  performance "decrease sharply on some of the environments (Fig. 55). This is likely caused
  by shortened experience chunks ... and earlier value bootstrapping." (measured, MuJoCo,
  MLP policies; the cause is the authors' guess)
- **Rudin, Hoeller, Reist, Hutter 2021, "Learning to Walk in Minutes Using Massively Parallel
  Deep RL"**, https://arxiv.org/abs/2109.11978 (§2.2, §4). "The algorithm struggles when we
  provide fewer than 25 consecutive steps, corresponding to 0.5 s of simulated time", which
  they attribute to GAE needing "rewards from multiple time steps". They bootstrap at
  time-outs as Pardo. (measured, legged locomotion, 50 Hz)
- **Berner et al. 2019, OpenAI Five**, https://arxiv.org/abs/1912.06680 (§3, Fig. 8). Rollout
  workers send 256-step segments (34 s at 7.5 decisions/s), not whole games: "Because we use
  GAE with λ = 0.95, the GAE rewards need to be smoothed over a number of timesteps ≫ 1/λ = 20;
  using 256 timesteps causes relatively little loss." (an assertion; no experiment is
  reported for it)
- **The 37 implementation details of PPO**,
  https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (detail 5). GAE
  bootstraps from the value of `next_obs` at the rollout's end. Truncation is "not
  implemented correctly" in the reference code. (default)
- R2D2's overlapping replay sequences and burn-in (Kapturowski et al. 2019) concern the
  recurrent **state** at a sequence's start, not the return at its end (see 2). IMPALA's
  V-trace bootstraps from the value at the end of each 20- or 100-step unroll
  (https://arxiv.org/abs/1802.01561, Table D.1: unroll length 100 on DMLab). (default)
- **No source plays ticks past the training segment only to complete the returns, and none
  measures what that buys.** The usual practice is to bootstrap at the cut (PPO, IMPALA,
  OpenAI Five, Brax, Rudin) and make the cut rare enough. (finding)

### What it means here

- The weight a segment cut puts on the bootstrap, averaged over a segment of L ticks, is
  (1/L) Σₖ (γλ)ᵏ (inference, arithmetic):

  | level | L = 128 (lean now) | L = 512 (chained, as v2) | L = 512 + 128 lookahead |
  |---|---|---|---|
  | damage | 3.0% | 0.8% | ≈ 0 |
  | kill_death | 6.4% | 1.6% | ≈ 0 |
  | objective_progress | 13.5% | 3.4% | ≈ 0.02% |
  | objective | 24.0% | 6.1% | ≈ 0.5% |
  | outcome | 24.5% | 6.3% | ≈ 0.5% |

  Lean now puts a quarter of the long levels' advantage weight on the segment-end
  bootstraps. Chaining the four segments removes three quarters of that at no collection
  cost. A lookahead after the chain removes most of what is left.
- With λ = 0.97 the λ-return already leans on the critic at every tick (weight 1 − λ per
  step). A cut does not add a new kind of bias; it puts the remaining tail weight (γλ)ᵏ on one
  value. The bias matters most while the critic is poor, which is early training, and for the
  outcome level. (inference)
- Cost of a 128-tick lookahead after a 512-tick chain: 25% more collection, which is about 7%
  of loop time at the laptop numbers (0.25 × (1/22,032) ÷ (1/6,124)). The cost is the
  policy's forward passes, not the environment: "the environment is cheap" does not make the
  lookahead cheap, because the model acts on every lookahead tick. (inference, from
  scoreboard numbers)
- A lookahead that wastes nothing: play the lookahead ticks as the next update's first
  segment and finish this update's tail returns from them. The cost is that the tail of
  segment 4 waits for the next update, or that segment 1 of the next update is trained one
  policy version late. OpenAI Five measured that staleness of about 8 versions "causes
  significant slowdowns" (Fig. 5b) and targeted 0 to 1. One version for a quarter of the data
  is inside what they ran. (inference on top of a measurement; nobody measured this scheme)
- Time limits: lean treats the clock's end as a true termination (`returns.py:24-25`) and
  puts the elapsed fraction tick/clock_ticks in the game token's inputs
  (`lean/game/core.py:1391`). That is Pardo's time-awareness option for a time-limited task.
  Segment ends are bootstrapped, which is Pardo's PEB. Both follow the paper. (default)
- Shot labels have the same cut. A bullet still flying at a segment's end leaves its row
  unlabelled, and a bullet fired in the previous segment is dropped (`lean/learn/NOTES.md`,
  hit labels, lines 311-316). With 11-tick bullets, about 11 of every 128 ticks lose their
  labels at each of the three interior cuts, although the next segment holds the outcome.
  (inference from the notes; not measured)

### Recommendation

1. **Chain GAE across the update's four segments**, as v2 did. For each seat, run the reverse
   scan over 512 ticks with the stored values and bootstrap only after segment 4. Do the same
   for the hit labels across segment boundaries. There is no collection cost, and the scan is
   the same doubling scan over 512 ticks. Strength: **strong** as a correction. Every
   implementation bootstraps only where its data ends, and lean's data does not end at tick
   128. It is also v2's behaviour.
2. **No lookahead by default after chaining.** It would cost about 7% of loop time to move
   6% of the long levels' weight off the bootstrap. Strength: **weak**. No source measures a
   lookahead.
3. **Cheap check before deciding** (inference): from one collected update plus 128 extra
   ticks, compute the long levels' targets for ticks 0 to 511 with the bootstrap at 512 and
   with the real tail. Report the mean |Δ target| over the target's standard deviation, at
   initialisation and after some training. If the change is small next to the targets'
   spread, the lookahead is not worth 7%.

---

## 2. Rollout (truncation) length

Two lengths are separate in lean, though the literature often merges them. (a) The **return
horizon**: where GAE is cut and bootstrapped. Section 1 fixes this by chaining, whatever the
training sequence length. (b) The **BPTT length**: how many ticks a training sequence runs
from a stored carry. Gradients through the recurrence stop there; the forward state does not.

### Sources

| Source | Rate | BPTT / sequence | Rollout per env per update | Kind |
|---|---|---|---|---|
| OpenAI Five, https://arxiv.org/abs/1912.06680 §3, Fig. 8 | 7.5 Hz | 16 steps (2.1 s), LSTM | 256 steps (34 s) | default |
| Sample Factory, https://arxiv.org/abs/2006.11751 §3.4, Table A.5 | 35/4 or 35/2 Hz (frameskip 4/2) | 32 | 32 | default; "we generally want larger T, in the 2⁵–2⁷ range for backpropagation through time with recurrent policies" |
| Pleines et al. 2022, https://arxiv.org/abs/2205.11104 App. D, Table 1 | — | grid 8 to "Max" (whole episode); chosen Max or 128 | 512 worker steps | measured: "longer sequences and greater hidden state sizes are favorable" (GRU/LSTM, memory tasks) |
| POPGym, Morad et al. 2023, https://arxiv.org/abs/2303.01859 Table 2 | — | ∞ (whole episodes, max 1,024) | batch 65,536 | default |
| Lu et al. 2023 (S5 in RL), https://arxiv.org/abs/2303.03982 Table 3 | — | 1,024 (unroll, parallel scan with resets) | 1,024 × 64 envs | default; S5 "running over six times faster" than a GRU (measured, runtime) |
| AGaLiTe (Pramanik et al., TMLR 2024), https://arxiv.org/abs/2310.15719 Tables 3, 6, 7 | — | 128 (Mystery Path), 100 (Memory Maze) | 128 to 256 rollout | default |
| PufferLib 3.0, `pufferlib/config/default.ini` (branch 3.0) | — | `bptt_horizon = 64` | auto | default |
| CleanRL `ppo_atari_lstm.py` | 15 Hz | 128 | 128 | default |
| Ni et al. 2022, https://arxiv.org/abs/2110.05038 §5 | — | 1 to 2,048 in prior work; theirs 64 or task-specific | — | measured: "the choice of context length seems to be problem-specific"; "a medium length (64) dominates" in some groups |
| Ni et al. 2023, "When Do Transformers Shine in RL?", https://arxiv.org/abs/2307.03864 | — | — | — | measured: Transformers improve memory up to 1,500 steps but "do not improve long-term credit assignment" |

- **R2D2** (Kapturowski et al., ICLR 2019; read only through search summaries, the OpenReview
  PDF was blocked): replay sequences of 80 steps, overlapping by 40, with a burn-in prefix of
  40 or 20 steps. Burn-in and stored state reduce the stale-state discrepancy in replay.
  (measured there, off-policy replay)
- **Pleines et al. 2022, §3.2**: refreshing stale hidden states and advantages before each
  epoch "does not provide a gain in performance". (measured, recurrent PPO)
- **Griffin (De et al. 2024)**, https://arxiv.org/abs/2402.19427 §4.2 and App. D.2: the RG-LRU
  is memory-bound. A linear-scan kernel was about 3× faster than JAX's native scan on TPU-v3,
  and the associative scan was "significantly slower ... up to 50% slower" in overall step
  time. (measured, TPU, language models)

### Cost of a longer sequence in lean (measured here)

Lean's `Temporal` block alone, forward and backward, bf16 autocast, eager, RTX 4070 laptop,
platform profile "performance", at a fixed 8,192 view-ticks (views × T), M = 9 tokens, width
128 (script: scratchpad `bench/scan_cost.py`, not committed):

| T | views | block time | block peak | scan alone time | scan alone peak |
|---|---|---|---|---|---|
| 32 | 256 | 94 ms | 1,528 MiB | 40 ms | 424 MiB |
| 128 | 64 | 111 ms | 1,662 MiB | 58 ms | 567 MiB |
| 512 | 16 | 129 ms | 1,803 MiB | 76 ms | 711 MiB |
| 1,024 | 8 | 138 ms | 1,875 MiB | 85 ms | 783 MiB |

- Each doubling of T at fixed ticks adds about 4% of the block's memory and 8% of its time.
  That is the log₂ T rounds of the doubling scan: O(T log T) work and saved tensors, against
  O(T) for a sequential scan. From 128 to 512: +8.5% memory and +16% time in the temporal
  block. Attention, MLPs and heads cost the same per tick at any T, so the whole update
  changes by less. (measured here for the block; the whole-model share is inference; compiled
  numbers not measured)
- At a fixed micro-batch size in ticks, a 512-tick view means 16 views per micro-batch, not
  64. (arithmetic)

### What it means here

- **Credit assignment.** At these lambdas the advantage reaches at most about 1.1 s. A BPTT
  length above about 100 ticks adds nothing to how far rewards reach once the returns are
  chained; long horizons go through the critic. Ni et al. 2023 found that longer memory
  models do not fix long-term credit assignment. (inference, with Ni et al. measured)
- **Memory.** BPTT length bounds how far back the gradient can teach the recurrence to keep
  something. The RG-LRU's retention is initialised a^c ∈ [0.9, 0.999] per tick
  (`policy.py:63-64`), with timescales up to about 1,000 ticks (33 s). With 128-tick BPTT the
  slow channels are carried forward but trained only on 4.3 s windows. Pleines measured longer
  better on memory tasks; Ni et al. 2022 measured it task-dependent. Whether air combat needs
  more than 4 s of trained memory (for example a hidden enemy seen 10 s ago) is a game
  question. (inference)
- In seconds, lean's 128 ticks (4.3 s) is already longer than OpenAI Five's 16-step BPTT
  (2.1 s), which reached top human level, and it lies in Sample Factory's 32 to 128 range.
  (default comparison)

### Recommendation

Keep 128-tick training sequences, with the returns chained over the update (section 1).
Strength: **weak to moderate**. 128 sits in the middle of practice: OpenAI Five 16, Sample
Factory 32 to 128, PufferLib 64, AGaLiTe 100 to 128, with whole episodes for memory
benchmarks. The only ablations (Pleines; Ni et al. 2022) are on memory tasks and disagree.
If a memory-heavy behaviour fails to appear (tracking hidden enemies), a 512-tick arm costs
about +16% temporal-block time and +8% of its memory (measured here). It is cheap to try.

---

## 3. Shape of a fixed ~1M-tick update

The three shapes: (A) 2,048 × 4 × 128 trained as 128-tick views from stored carries (lean
now); (B) 8,192 × 128; (C) 2,048 × 512 as one sequence.

### Sources

- **Rudin et al. 2021** (§4, Fig. 4). At a fixed batch, raising the number of robots
  shortens each robot's steps. "When the number of robots is too high, the performance drops
  sharply, which can be explained by the time horizon of each robot becoming too small." With
  too few robots there is "a slow decrease in performance", explained by samples that are "very
  similar with many steps per robot ... less diversity". Best trade-off there: 2,048 to 4,096
  robots at a batch of about 100k to 200k (24 to 49 steps of 20 ms). (measured, both effects)
- **Andrychowicz et al. 2021** (§3.5): more environments at a fixed iteration size hurt
  through shorter chunks and earlier bootstrapping. "For faster wall-clock time training use
  many parallel environments and increase the batch size (both might hurt the sample
  complexity)." (measured)
- **Policy lag.** In synchronous PPO every tick of an update is collected by the same weights
  θ_old, so all three shapes have the same lag. Lag here is the epochs' drift, which is the
  same in every shape. Sample Factory §3.4 and OpenAI Five's staleness (Fig. 5b) concern
  asynchronous systems. (inference; the measurements are for asynchronous lag)
- **Stale carries.** In (A), epochs 2 to 4 start segments 2 to 4 from carries computed by
  θ_old. In (C) the current weights recompute them. Pleines measured no gain from refreshing
  hidden states in recurrent PPO. (measured there)
- **Large-batch practice.** Brax PPO's default `unroll_length = 10`
  (`brax/training/agents/ppo/train.py`, read); IsaacGym practice (Rudin) 24 to 50 steps; OpenAI
  Five 256-step segments. Many short rollouts is the GPU-simulator norm for MLP policies with
  short credit horizons. (default)

### What it means here

| | (A) 2,048 × 4 × 128 | (B) 8,192 × 128 | (C) 2,048 × 512 |
|---|---|---|---|
| Return cut (outcome level, mean weight) | 6.3% if chained (24.5% now) | 24.5%, cannot chain | 6.3% |
| BPTT length | 128 | 128 | 512 |
| Carry at ticks 128/256/384 | stored (θ_old) | — | recomputed |
| Distinct environments in a minibatch (of 32) | ≈ 370 of 384 views (four segments of one environment can share a minibatch) | 384 | 96 |
| Temporal-block cost (measured, §2) | base | base | +16% time, +8.5% memory |
| Collection | 512 sequential policy steps of 2,048 | 128 steps of 8,192 (better GPU use per step; 4× environment state, carries and league slots) | as (A) |
| Episode progress per update | 512 ticks | 128 ticks | 512 ticks |

(inference and arithmetic, except the measured cost row)

- (B) is the worst for returns. A quarter of the long levels' advantage weight would rest on
  bootstraps, with no way to chain. Andrychowicz and Rudin both measured the "too short per
  environment" failure, though at much shorter chunks (0.5 s in Rudin, where lean's 128 ticks
  is 4.3 s). (B) gives the most diverse minibatches and the fastest collection on a large GPU.
  Its memory cost is four times the environments' state on the device. (inference)
- (A) chained and (C) have the same returns. They differ only in BPTT length (128 against
  512), in stale against recomputed carries, and in minibatch diversity (≈ 370 against 96
  distinct environments). At 2,048 environments, (A) sits at the upper end of Rudin's measured
  sweet spot for diversity. (inference)

### Recommendation

Keep (A), 2,048 × 4 × 128, **with GAE and hit labels chained across the four segments**.
Strength: **moderate** for preferring (A) or (C) over (B), since two measured studies show the
short-chunk loss; **weak** between (A) and (C), since no source compares BPTT 128 against 512
at a fixed update for a linear recurrence. (C) is the arm to run if memory turns out to matter
(section 2).

---

## 4. Optimiser

### Sources

- **Adam against AdamW.** PPO's reference code and CleanRL use Adam without weight decay
  (37 details 3 and 4; CleanRL `ppo_atari_lstm.py:212`:
  `optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)`). Decision Transformer
  uses AdamW with weight decay 0.1 (Atari) and 1e-4 (Gym) (https://arxiv.org/abs/2106.01345,
  App. A; `gym/experiment.py:237-244, 297-299` in kzl/decision-transformer). Pleines et al. use
  AdamW (Table 1), weight decay not stated. Andrychowicz et al. found Adam against RMSprop
  "rather small" (§3.7). (default; measured only Adam against RMSprop)
- **Epsilon.** 1e-5 in PPO's reference code, kept for fidelity: "neither mentioned in the
  paper nor a configurable parameter" (37 details 3). Sample Factory uses 1e-6 (Table A.5),
  PufferLib 3.0 1e-12 (with Muon as its default optimiser), and PyTorch's default is 1e-8.
  **Andrychowicz et al. swept 1e-9 to 1e-4 and "did not find any difference between different
  values of the ε coefficients" (§3.7, Fig. 68).** (measured, MuJoCo, MLPs)
- **Fused and foreach** (PyTorch 2.15 docs, `torch.optim.Adam`,
  https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). foreach is the default
  on CUDA. Fused supports float64, float32, float16 and bfloat16. "The foreach and fused
  implementations are typically faster than the for-loop"; fused is "theoretically fastest".
  The only caveat given is that fused "is relatively new", so it is not the default, to allow
  "bake-in time". (default)
- **Gradient clipping.** Global norm 0.5 (37 details 11; CleanRL), 1.0 (Decision Transformer
  Atari), 0.25 (Decision Transformer Gym), 1.5 (PufferLib), 4.0 (Sample Factory). OpenAI Five
  clipped per parameter at ±5√v (§3). Andrychowicz: "gradient clipping (C68) provides a small
  performance boost with the exact clipping threshold making little difference (Fig. 34)",
  swept 0.5 to 5.0 and none. (measured)
- **Warm-up.**
  - Ma and Yarats, "On the Adequacy of Untuned Warmup for Adaptive Optimization", AAAI 2021,
    https://arxiv.org/abs/1910.04209 (abstract and recommendation through the search engine's
    summary): linear warm-up over 2/(1 − β₂) iterations as an untuned default, which performs
    "more-or-less identically to RAdam". (measured, supervised)
  - Decision Transformer: Gym, linear warm-up over 10,000 steps, then constant
    (`warmup_steps=10000`, `LambdaLR(min((steps+1)/warmup_steps, 1))`); Atari, warm-up over
    512 × 20 tokens, then cosine decay. (default)
  - GTrXL (https://arxiv.org/abs/1910.06764, §1): standard transformers in RL are "significantly
    difficult to optimize", and the supervised remedies of warm-up and initialisation "do not
    seem to be sufficient for RL". Its fix is architectural (gating, identity-map reordering).
    No warm-up length found in the text. (measured: architecture; finding: no warm-up number)
  - OpenAI Five ran learning rate 0 "for the first several hours of training after surgery"
    so that Adam's moments could adjust (§3.3, "Smooth Training Restart"). (default)
  - PPO libraries (CleanRL, SB3, Sample Factory) have no warm-up. (default)
- **Constant against annealed.** PPO's reference decays linearly to 0 (37 details 4).
  Andrychowicz: decaying to 0 "increases the performance on 4 out of 5 tasks but the gains are
  very small apart from Ant, where it leads to 15% higher scores" (§3.7, Fig. 65). OpenAI Five
  lowered the learning rate by hand (5e-5 to 5e-6 and lower). Pleines decayed 3e-4 to 3e-5.
  (measured, small effect; default)
- **Skipping non-finite steps.** v2 skipped a step whose gradient norm was non-finite. Under
  float16, `GradScaler` skips steps with inf or NaN gradients (from general knowledge of the
  PyTorch API; the 2.15 AMP page fetched today does not state it). Under bf16 there is no
  scaler, so nothing skips unless the code does. No RL source measures a skip guard.

### What it means here

- Lean: `torch.optim.Adam(..., eps=1e-5)`, not fused (`lean/learn/update.py:37`); clip 1.0 via
  `clip_grad_norm_` (`update.py:251`); warm-up 10 updates, then constant 3e-4
  (`config.py:54-57`, `:125`).
- Warm-up in Adam steps: 10 updates × 128 = 1,280 steps. Ma and Yarats' rule at β₂ = 0.999 is
  2,000 steps, which is 15.6 updates, about 16.4M environment ticks. v2's 10M ticks is
  1,220 steps. (arithmetic)
- Fused against foreach: literature-ppo-targets-minibatches.md 2.4 found Adam and clipping
  together under 0.1 s per update on the laptop, so fused saves little there. It has no
  numerical downside beyond float rounding and accepts the float32 master weights that
  autocast keeps. (inference)
- Annealing to 0 conflicts with open-ended self-play and with a league that keeps adding the
  live policy. Weight averaging at a constant learning rate (author-review 7.2) is the
  intended substitute; that it substitutes in RL is untested (literature-model-choices.md
  points this out too). (inference)

### Recommendation

| Item | Recommendation | Strength |
|---|---|---|
| Optimiser | Adam, no weight decay (AdamW only as an arm) | moderate (standard PPO practice; no RL ablation of decay found) |
| Epsilon | Keep 1e-5 | moderate that it does not matter (Andrychowicz measured no effect over 1e-9 to 1e-4); 1e-5 is PPO's convention |
| Implementation | `fused=True` on CUDA | weak (doc-level; the gain on the laptop is under 0.1 s per update) |
| Clip | Keep global norm 1.0; keep logging the pre-clip norm | moderate (threshold measured as unimportant) |
| Warm-up | Linear, 2,000 Adam steps (16 updates at 128 steps each), then constant | weak (supervised rule of thumb; no RL measurement of warm-up length) |
| Schedule | Constant, with weight averaging; a final linear decay only for a fixed-budget release run | weak (measured gains from decay are small and from single-agent MuJoCo) |
| Non-finite | Skip the step when the clipped norm is non-finite, and count skips (v2's rule) | weak (cheap guard; no measurement) |

---

## 5. Mixed precision: what stays in float32

### Sources

- **Micikevicius et al. 2018, "Mixed Precision Training"**, https://arxiv.org/abs/1710.03740
  (§3). An FP32 master copy of the weights; dot products accumulate in FP32; "large reductions
  (sums across elements of a vector) should be carried out in FP32. Such reductions mostly
  come up in batch-normalization layers when accumulating statistics and softmax layers";
  point-wise operations in either precision. Loss scaling is for float16's range. (measured
  there, float16)
- **PyTorch 2.15 autocast op lists** (https://docs.pytorch.org/docs/2.15/amp.html, CUDA,
  listed for float16). Lower precision: linear, matmul, bmm, conv, the RNN cells, prelu.
  Float32: exp, expm1, log, log1p, log_softmax, softmax, layer_norm, group_norm, norm, rsqrt,
  sum, cumsum, cumprod, pow, softplus, cross_entropy, nll_loss, mse_loss, kl_div and others.
  Widest input type: addcmul, addcdiv, index_put, scatter_add and others. The page gives the
  CUDA lists for float16 and does not say that bf16 uses the same ones. "Backward passes under
  autocast are not recommended." (default)
- **Measured here** (torch 2.15.0, CUDA, bf16 autocast): `nn.RMSNorm` on a bf16 input with
  its float32 weight returns **float32**. It warns "Mismatch dtype between input and weight
  ... Cannot dispatch to fused implementation" and gives the same result as float32 on the
  same input, so norm statistics are float32. softmax, log_softmax, exp and sum return
  float32; sigmoid, gelu and **mean** return bf16. (measured here)
- **recurrentgemma** (official PyTorch code, google-deepmind/recurrentgemma, `torch/layers.py`
  and `torch/modules.py`, as downloaded for provenance-check.md). `rnn_scan(...,
  acc_dtype=torch.float32)` accumulates h_t = a_t h_{t−1} + x_t in float32 and casts outputs
  back. "RG-LRU cache always in float32." The gates and √(1 − a²) are computed in the
  activation dtype, with a bounded-derivative sqrt "to prevent NaNs during training in
  bfloat16". Attention logits are cast to float32 before softmax. RMSNorm computes in the
  input's dtype. The Griffin paper trained with "bfloat16 representation for model parameters
  and activations" (§4.1). (default)
- **Mamba README** (https://github.com/state-spaces/mamba, "Troubleshooting / Precision"):
  "AMP keeps model parameters in float32 and casts to half precision when necessary";
  "higher precision for the main model parameters may be necessary, because SSMs are sensitive
  to their recurrent dynamics". (default, stated from experience)
- **Hugging Face Llama** (transformers v4.46.0 `modeling_llama.py:71, 154-157, 373`):
  RMSNorm upcasts to float32; rotary cos/sin "Force float32" under
  `torch.autocast(enabled=False)`; eager attention softmax in float32. (default)
- Not verified: the internal softmax precision of PyTorch's flash and memory-efficient SDPA
  kernels (believed float32 accumulation; not read).

### What lean does, against the sources

| Quantity | Standard | Lean (file:line) | Match |
|---|---|---|---|
| Master weights, optimiser state | float32 (Micikevicius; AMP; Mamba) | autocast keeps parameters and Adam state float32 | yes |
| Matmuls | bf16 inputs, float32 accumulation (hardware) | autocast | yes |
| Norm statistics | float32 (Micikevicius; HF; autocast list for layer_norm) | `nn.RMSNorm` returns float32 under autocast (measured here) | yes, but the norm misses the fused kernel (warning); compiled code may fuse anyway (not checked) |
| Attention softmax | float32 (recurrentgemma; HF) | `F.scaled_dot_product_attention` on bf16 (`policy.py:225`) | kernel-internal (not verified) |
| Rotary angles, position phases | float32 (HF Llama) | float32, autocast off (`position.py:7, 38-39`) | yes |
| Recurrence state and scan | float32 accumulation (recurrentgemma; Mamba's caution) | convolution, gates, log-retention, √(1 − a²) and scan all float32 (`temporal.py:95-121`); carry float32 | yes, stricter than recurrentgemma (gates in float32 too) |
| Losses, log-softmax, HL-Gauss cross-entropy | float32 (autocast list; Micikevicius) | heads cast `.float()` before every loss (`update.py:52, 64, 85-97`); value softmax float32 (`value.py:53`) | yes |
| Reductions (masked sums, means over tokens) | float32 (Micikevicius) | on float32 outputs | yes; note `mean` stays bf16 under autocast if applied to a bf16 tensor (measured here) |
| Storage of the rollout | (no standard) | float16 for most stored floats, float32 for positions and PPO numbers (`store.py:21-25`) | own choice |

### Recommendation

Keep the present split. It matches every float32 requirement in the sources: master weights,
optimiser state, accumulations, norm statistics, softmax and log-softmax, losses, the scan's
state, and the position phases. Strength: **moderate** (consistent defaults across
Micikevicius, PyTorch, recurrentgemma, Mamba and HF; no ablation for this model). Two small
items: (1) the RMSNorm dtype mismatch forces the non-fused path in eager code; measure whether
it matters compiled before acting; (2) never take a `mean` of a bf16 tensor for a loss or a
statistic, because autocast leaves it in bf16.

---

## 6. Next-state targets per feature

The target form (absolute, change, change against dead reckoning) and HL-Gauss were covered
in literature-model-choices.md section 1, with Nagabandi et al. and GNS. New here: PETS,
MBPO, residual physics and one measured comparison.

### Sources

- **PETS**, Chua et al. 2018, https://arxiv.org/abs/1805.12114 (App. A.6): "instead of
  learning a forward dynamics in the form s_{t+1} = f(s_t, a_t), we learn a model that predicts
  the difference to the current state", following Deisenroth et al. 2014. Angles are fed as
  [sin, cos] inputs. (default)
- **MBPO**, Janner et al. 2019, code jannerm/mbpo `mbpo/models/fake_env.py`, `step`:
  `ensemble_model_means[:,:,1:] += obs`, so the model predicts the change. (default)
- **Residual physics, TossingBot**, Zeng et al. 2020, https://arxiv.org/abs/1903.11239v3
  (§V, Table I): a learned residual on top of a ballistic controller against pure regression
  and physics alone. Throwing success on unseen objects (simulation): Regression 28.4%,
  Regression pre-trained on physics 35.0%, Physics-only 50.0%, Residual-physics 66.5%.
  Residual-physics was best on every object set. (measured; a control output, not a
  prediction target, but the same structure of known model plus learned residual)
- **Neural-Lander**, Shi et al. 2019, https://arxiv.org/abs/1811.08027: a nominal quadrotor
  model plus a DNN for the unmodelled ground-effect force; it outperforms the baseline
  controller in landing (§V). (measured, residual dynamics in control)
- **Delta against absolute, measured.** Wang et al. 2020, multi-agent rendezvous,
  https://arxiv.org/abs/2003.06906 (App. D): "delta-pose-lidar performs the best while the
  other two prediction models perform equally well" (absolute pose, and delta in a different
  frame). (measured, a small ablation)
- DreamerV3, TD-MPC2, MuZero, EfficientZero: their dynamics targets are reconstructions or
  latents, not state changes (literature-model-choices.md §1). (default)
- **No source compares change against dead-reckoning residual for individual physical
  quantities**, and none does it for an RL auxiliary head. (finding)

### Lean's state and physics (read in code)

- Position: kick-drift-kick per tick; the drift uses the velocity after half a kick, bent by
  the arc nose_rate × h (`lean/game/core.py:604-644`). The state carries speed, angle of attack
  and nose_rate (`lean/contract.py:89-100`).
- Shield (`core.py:879-886`): damage from hits (about 21 per hit at full closeness and aspect,
  `card.py:31`) and from the boundary (up to 40/s, 1.33 per tick); otherwise, on a quiet tick
  (no power, small stick, not shooting) and after the 6 s wait, recharge at 12/s (0.4 per
  tick) to the maximum. The wait is a ship input (`core.py:1366`). Whether a tick is quiet
  depends on the action of that tick, which the next-state head does not see until actions
  apply one tick late (decision 3.3, not built).
- Energy, core flight model: the store changes only by the speed-store exchange; firing costs
  no store (`core.py:70-76, 166`). In the v2 flight model firing costs `bullet_energy_cost`
  (`core.py:154, 754`), which is what made v2's energy column heavy-tailed.

### Recommended per-column table (lean `next_state_fields`, `policy.py:78-86`)

| Column | Target form | Extrapolation subtracted | Bins | Notes |
|---|---|---|---|---|
| x, y → **along, across** | change against dead reckoning, **in the ship's frame** | velocity × h (author's decision); optionally the constant-turn arc (nose_rate) as well | linear, range from measured residuals | A world-frame residual depends on the heading. Along-track and cross-track residuals are rotation-invariant and map onto thrust/drag and turn (inference). GNS's acceleration target (default). |
| heading_cos, heading_sin → **one heading change** | wrapped heading change (radians) against constant turn | nose_rate × h | linear | Two columns for one degree of freedom, with Δcos ≈ −sin θ·Δθ: the two columns' scales depend on the heading itself. A single angle change is the standard delta (PETS predicts the raw angle's change). (inference) |
| speed | plain change | none in the state (no stored acceleration) | linear | Bounded by thrust, drag and dive per tick. |
| energy (store) | plain change | none (the exchange is action-dependent) | **linear** in the core game; symlog only under the v2 flight model | Firing costs no store in the core model, so the change is bounded and continuous. Check with a histogram. This differs from literature-model-choices.md, which recommended symlog for energy on v2's behaviour. |
| shield | change against the recharge schedule, i.e. minus the damage taken this tick | the scheduled recharge (0.4 per tick when quiet and the wait has run out, to the maximum) | symlog (mostly 0, rare hits of about 21) | Exact only once the pending action is an input (3.3); until then subtract nothing (plain change), because quiet depends on the unseen action. Residual-physics structure (TossingBot, Neural-Lander, measured). |

Ranges for every column from measured changes (as decided); the check is literature-model-
choices.md's histogram of each target over one collection.

### Recommendation

The table above. Strength: **moderate** for change over absolute (PETS, MBPO, Nagabandi as
defaults; one small measured ablation). **Weak** for the dead-reckoning residuals (GNS default;
residual physics measured in control, not as targets), for the ship frame, and for the single
heading column (both reasoned only). Shield against the schedule should wait for actions to
apply one tick late.

---

## 7. Rotary bands: irrational ratio, and the rotated share of a 64-wide head

### Sources

- **One band contributes one sinusoid.** With RoPE the score's contribution from a pair at
  wave vector k is Re[q k* e^{i k·Δ}], a single cosine of k·Δ (Heo et al. 2024, RoPE for ViT,
  https://arxiv.org/abs/2403.13298, Eqs. 3-4). No band creates harmonics in the score; the
  softmax's exponential and later layers can. (default, from the definition)
- **LLM frequency ladders.** θ_t = base^(−2t/d) (RoPE; Heo et al. Eq. 5). The ratio between
  successive frequencies is base^(2/d), irrational in general (10000^(1/32) ≈ 1.33 for a
  64-wide head), by construction and not by design. Vision axial RoPE uses 100^(−t/(d/4))
  (Heo et al. Eq. 10). NeRF-style positional encoding uses exact octaves. (default)
- **Tancik et al. 2020, "Fourier Features Let Networks Learn High Frequency Functions"**,
  https://arxiv.org/abs/2006.10739 (§5, §6, Table 1, App. E). Random Gaussian frequencies
  (incommensurate and off-axis) beat log-linear axis-aligned "positional encoding" on every
  low-dimensional regression task in Table 1. "The standard deviation of the sampled
  frequencies alone is enough to predict test set performance, regardless of the underlying
  distribution's shape." The positional encoding "performs worse for frequencies that are not
  axis-aligned". (measured; MLP inputs, not rotary)
- **Spiral RoPE**, https://arxiv.org/abs/2602.03227 (§3, Tables 5-6). Spreads 2D rotary
  frequencies over K uniformly spaced directions, interleaving one geometric frequency ladder
  across directions so that each direction gets different frequencies. On ImageNet with
  DeiT-B: axial (K = 2) 83.31%, K = 4: 83.31, 8: 83.34, 16: 83.39, 32: 83.32. Frequency scale
  1.5 was best, with 2.0 worse "because excessively high frequencies cause the rotations to
  vary too rapidly". (measured; small differences)
- **RoPE-Mixed** (Heo et al. 2024, §3.2): learnable frequencies mixing both axes, to handle
  diagonal directions. (measured there against axial; numbers not re-read)
- **Rotated share.** provenance-check.md §7 holds the table (GPT-J 25%, NeoX 25%, Phi-1.5 50%,
  Phi-2 40%, StableLM-2 25%, Qwen3-Next 25%, MLA 33% of query-key dims, Llama/Mistral/gpt-oss
  100%) and Barbero et al.'s ablation (Gemma 2B: 75% rotated as good as or slightly better than
  full, 25% worse; dropping the lowest frequencies better than dropping the highest). New here,
  **heads of width 64** (config.json files, read): StableLM-2-1.6B rotates 25% (16 dims, 8
  pairs); Phi-1.5 50% (32 dims, 16 pairs); gpt-oss-20b 100% (32 pairs). Small heads do not
  push practice in one direction. (default)
- Barbero et al. (https://arxiv.org/abs/2410.06205, as summarised in provenance-check.md §7):
  Gemma uses the highest frequencies for positional patterns and the lowest as near
  position-free "semantic" channels. p-RoPE removes the lowest frequencies. (measured, 1D,
  language)
- **No source studies whether band ratios should be irrational** for multi-scale rotary or
  Fourier position codes. The nearest results are Tancik's (random frequencies best, with scale
  mattering more than shape) and Spiral RoPE's (spreading directions helps a little). (finding)

### What it means here

- **Irrational ratio.** Three reasons it buys little in lean's layout (inference):
  1. In the score each pair is a pure first harmonic, so harmonics can only line up after
     nonlinearities.
  2. Successive bands are turned by an irrational fraction of 90° (decided). In 2D a harmonic
     coincides with another band only if the wave vectors match in length *and* direction.
     2·k_j = k_{j+1} cannot hold when k_{j+1} points elsewhere, whatever the length ratio. The
     turn already does what an irrational ratio would do for collinear axes.
  3. With the coarsest band above 16.8 km (decided), one band alone is unambiguous over the
     map. A rational ratio makes the joint code periodic at the coarsest period; an irrational
     ratio makes it quasi-periodic beyond that, which helps only a residue-style decoder
     reading past the coarsest period (literature-model-choices.md §5 judged that unlikely).
  An irrational ratio costs nothing either. If the author wants the extra margin against
  near-coincidences of low-order combinations n·k_i + m·k_j, a ratio such as 1.9 or 2^(15/16)
  serves as well as any; no source supports one over another.
- **Rotated share.** literature-model-choices.md §5 sized the layout at ratio 2: 10 bands from
  64 m to 32.8 km plus 2 heading harmonics = 22 pairs, 69% of a 64-wide head; at ratio 1.65, 28
  pairs = 88%. Barbero's one controlled ablation found 75% best, 100% about equal and 25%
  worse, in 1D language over thousands of tokens. In lean the unrotated dimensions are the only
  position-free channels for content matching (identity, team, shield). With 2 heads of 64,
  10 unrotated pairs at ratio 2 is 20 dimensions per head; 4 pairs at ratio 1.65 is 8.
  (inference, arithmetic)

### Recommendation

- Ratio: choose it by the nested condition (literature-model-choices.md §5: 1.65 to 2), keep
  the irrational turn, and treat an irrational ratio as optional. Strength: **weak**. No
  source tests irrational ratios. The reasoning above shows the turn already prevents exact
  coincidence in 2D.
- Rotated share: about 70 to 75% of each head (22 to 24 of 32 pairs), leaving at least 8
  unrotated pairs. This argues for a ratio near 2 rather than 1.65 at width 64, unless the
  width per head grows. Strength: **weak** (one LLM ablation in 1D; practice at width 64 spans
  25% to 100%).

---

## Summary

| # | Question | Recommendation | Strength |
|---|---|---|---|
| 1 | Lookahead | Chain GAE (and hit labels) across the update's 4 segments, as v2 did (lean now cuts at every 128 ticks: 24.5% of the long levels' advantage weight on bootstraps, 6.3% when chained). No lookahead by default; check its effect on one rollout first (about 7% loop cost) | strong for chaining; weak for no lookahead |
| 2 | Rollout length | Keep 128-tick training sequences; returns chained. A 512-tick arm costs +16% temporal-block time, +8.5% memory (measured here) | weak to moderate |
| 3 | Update shape | 2,048 × 4 × 128 chained; not 8,192 × 128 (cannot chain; short-chunk loss measured by Andrychowicz and Rudin); 2,048 × 512 as the memory arm | moderate against (B); weak between (A) and (C) |
| 4 | Optimiser | Adam, eps 1e-5 (measured irrelevant), fused, clip 1.0, warm-up 2,000 Adam steps (16 updates), constant learning rate with averaging, skip non-finite steps | moderate (eps, clip); weak (warm-up, schedule, fused) |
| 5 | Precision | Keep the present split; avoid bf16 means; check the RMSNorm non-fused path when compiled | moderate |
| 6 | Next-state targets | Ship-frame along/across residual against velocity × h; one heading-change column against nose_rate × h; speed and energy plain change, linear bins (core game); shield against the recharge schedule, symlog, once actions are delayed | moderate (delta); weak (residual forms) |
| 7 | Rotary | Ratio by the nested condition, irrational optional (the turn already prevents 2D coincidences); rotate about 70 to 75% of each 64-wide head | weak |

## Untested in this domain

- Chaining against per-segment GAE was not measured anywhere as an ablation (it is the
  definition of GAE over contiguous data). The lookahead before the bootstrap has no paper.
- BPTT 128 against 512 at a fixed update for a linear recurrence in PPO. Recurrent-PPO
  sequence-length studies use GRU and LSTM on memory benchmarks.
- A warm-up length for PPO with attention models. The 2/(1 − β₂) rule is from supervised
  learning.
- Constant learning rate plus weight averaging as a substitute for decay, in RL.
- Ship-frame and dead-reckoning residual targets; a single heading-change column; the shield
  residual against the recharge schedule as an auxiliary target.
- Irrational band ratios, and the rotated share for 2D rotary in RL.

## Could not verify

- R2D2 (Kapturowski et al. 2019): the OpenReview PDF was blocked. The sequence, overlap and
  burn-in figures come from search summaries quoting the paper.
- Ma and Yarats 2021: read only through search summaries (abstract and recommendation).
- Memory Gym (Pleines et al., the journal follow-up): not read.
- GTrXL's optimiser settings: not found in the extracted text.
- The internal softmax precision of PyTorch's SDPA kernels, and the dtype of Mamba's
  selective-scan state (README only).
- GradScaler's skip of non-finite steps: from general knowledge; not on the AMP page fetched.
- RoPE-Mixed's numbers: not re-read. Barbero et al.: through provenance-check.md §7, not
  re-read today.
- Whether the RMSNorm non-fused fallback persists under torch.compile.
