# Simplification audit

A running record of the audit of v2's complexity, started 2026-11-21. The aim is to find
problems, not to fix them yet. Each finding names what exists, what it costs and the
assumption behind it. Decisions are recorded with their date. Update this file whenever
something is decided or measured, and commit it.

Measurement scripts and raw records: `docs/rewrite/sections/simplification/`.

## Why this audit

- **Compile time.** Cold compiles take 5–10 minutes per stage on the laptop and about an
  hour per stage on rented boxes, where Dynamo's tracing is single-threaded
  (`sections/launch/progress.md`). Every new shape is another cold compile. Evaluation
  spends longer compiling than playing.
- **Eager speed.** Uncompiled, v2 is far slower than v1. Uncompiled, every operation is at
  least one kernel launch, so the cost follows the operation count.
- **Steady speed.** Not clearly better than v1 on the laptop (see "Steady speed" below).
- **Size.** `src/` is about 67k lines against v1's 33k, plus 45k lines of tests and 18k of
  benchmarks.

## Principles (the author, 2026-11-21)

- Rounding differences do not matter. Bit-exact equality with earlier runs is not a goal.
- Insignificant model changes (bin counts, code layouts) are acceptable.
- Remove all complexity that is not needed now. Add it back when it is needed.
- Challenge the assumptions that led here.
- Prefer ideas from the literature and industry over bespoke pipelines.
- The code must be simple, easy to maintain, and fast both in steady state and in compile
  time.
- Expensive components earn their place with an A/B.

## Decisions

| Date | Decision |
|---|---|
| 2026-11-21 | **Fog: no belief system.** Hidden enemies leave the observation; the Griffin layers carry memory. Belief pipelines, Mamba, a causal transformer and autoregressive models are later alternatives to test. |
| 2026-11-21 | **No correction (error) head.** Follows from no belief. |
| 2026-11-21 | **Drop the shot trajectory head.** This also removes shot capture in the rollout, bullet re-simulation in the learner and the shot labels' share of the lookahead. |
| 2026-11-21 | **Drop the density head** (tentative: "maybe"). |
| 2026-11-21 | **Outcome merges into value.** Value itself is to be simplified. |
| 2026-11-21 | **Keep:** a next-state head (simplest form), the shot outcome head, the geometry head. |
| 2026-11-21 | **Positions:** the 236 km range and the random offset served varied map sizes, which are not needed now. The map is not toroidal (v1's was). Start with a much simpler encoding, preferably grid-cell-like (see "Position encoding"). |
| 2026-11-21 | **League matchmaking keeps a live rating**, or failing that a precomputed schedule. |

## Open questions

- **Enemy action head:** keep or drop. Cheap (one head, one cross-entropy); its only rollout
  use was the belief.
- **Value:** what the simplified form is. Today it is per level, 77 symlog bins each.
- **Next-state head:** what it predicts and in what form. Without a belief it is a
  learner-only auxiliary loss, so a plain regression on next-tick truth may do.
- **Rating lanes:** drop them and rate from training games? Matchmaking already reads the
  league's own games (`bnb.league.league`); the lanes add 1,024 environments (half again
  the 2,048 training ones), a stacked forward and Imugi every tick.
- **Imugi:** how much of it is needed (it is bloated; see the measurements).
- **Mixed batches:** do the three modes, two pools per mode and drawn lineups need to share
  one batch? This generality makes every rule a per-environment tensor.
- **Dynamic shapes:** which dimensions can be dynamic after compilation (see below).
- **Constants as Python values** instead of tensors (see below).
- **MoE tooling** (grouped GEMM) for several weight sets per tick, if any remain after the
  lanes question.
- **Bit-level reproducibility machinery:** how much to remove (finding 9).
- **Shot outcome labels from the real game** (the author's idea, 2026-11-21): record, for
  each bullet, whether it hit an enemy, an ally or nothing, and the damage, when
  `bnb.game.physics.combat.hits` resolves it; written back to the shooter's record at the
  launch tick (the label is known within a bullet's life, at most 11 ticks). Real bullets
  only (labels only where the policy fires), or real and ghost bullets (ghosts flown in the
  rollout's tick, dense labels). Either removes the learner's re-simulation (finding L2).
  Leaning: real bullets first, ghosts as an A/B.
- **The Triton kernels** (finding 14): drop all five and use standard tools, re-adding a
  kernel only where a profile shows it pays?

## Measurements

### Method

`regions.py` counts every operation PyTorch dispatches in eager mode (a
`TorchDispatchMode`), attributed to named regions: the stages, the model's sections and
the game's functions. Operations that only make a view (`OpOverload.is_view`) launch no
kernel and are counted apart; the counts below are the others, the ones that launch at
least one kernel on a GPU. v2's Triton kernels (attention, gated MLP, heads) count as one
operation each, as on the GPU, with their plain twin computing the values.

Run on the CPU (no GPU in the session); the counts do not depend on the device. They do not
depend on the batch size either: identical at small and full launch sizes (v2 27,019 at
256 + 128 and at 2,048 + 1,024 environments; v1 7,574 at 288 and at 2,592).

Configurations: v1 is the `rl` profile `wild-universe-731` trained with (4v4, 4 fields,
action repeat 2, ego pass, 4 league slots, Elo evaluation in the rollout). v2 is
`launch.toml`: `real_scale_all_modes`, lineup Yongma x2 + Jujak x3, the champion's model
(the delta next state on the extrapolation baseline, split, hidden physics decay), PPO
with league slots and 1,024 lane environments.

These are counts, not timings. GPU timings are still to be measured.

### One rollout step (one decision)

| | v1 | v2 |
|---|---|---|
| RL rollout step | 7,574 | 27,019 |
| BC rollout step | | 20,787 |
| Policy forwards per step | 3 | 3 (learner, league slots, lanes) |

### One policy forward

| Section | v1 | v2 |
|---|---|---|
| Inputs (features, encoders, belief update, carry reset) | 148 | 1,180 |
| Trunk (4 spatial + 2 temporal sublayers) | 340 | 295 |
| Output head MLPs | 108 | 12 |
| Forecast decoding into the belief | | 3,588 |
| (of which the correction decode) | | (3,023) |
| (of which the delta decode) | | (506) |
| Action sampling | 48 | 77 |
| **Total** | **649** | **5,172** |

The trunk is the one part that shrank. About 92% of v2's forward is input encoding and
forecast decoding.

### The correction decode, per call

| Step | Operations |
|---|---|
| Softmax over each code | 32 |
| Median, per component (14 calls) | 518 |
| Quantile spread, per component (14 calls) | 2,114 |
| Combining axes, confidence, step caps | 164 |
| Applying the step, covariance, clamps, cooldown | 195 |
| **Total** | **3,023** |

87% is a Python loop over 14 components with different bin counts. Each runs three
quantiles (cumulative sum, search, gathers, two `erfinv`, interpolation). Moot after the
decisions above, but the pattern recurs: the delta decode and the input codes have the same
shape.

### Environment, per decision

v1 runs two physics ticks per decision (action repeat 2); v2 runs one.

| Section | v1 | v2 |
|---|---|---|
| Flight physics and bullets | 1,482 | 1,861 |
| Hits, damage, deaths, mode rules | 242 | 1,866 |
| Observation (fog in v2) | 26 | 828 |
| Rewards and statistics | 598 | 445 |
| Resets, records, other | about 200 | about 667 |
| **Total** | **2,549** | **5,667** |

### The game by function (v2, per step, exclusive)

| Source | Operations | Note |
|---|---|---|
| Flight model (thrust and lift budgets via `softmin`, 2x2 exponentials, substeps) | about 1,300 | real game content |
| `bnb.game.layout.environment` (2 calls) | 844 | per-environment rules rebuilt every tick; folded when compiled, paid every tick eager |
| `bnb.game.ship_types.per_ship` (121 calls) | 484 | ship-card lookups as `where` chains, because types can differ per environment and per episode |
| Three modes' machinery in every batch (capture, zones, phases, spawns, mode globals) | about 600 | |
| Fog line of sight and view ordering | about 350 | |
| Refractive medium | about 250 | |

About a third is physics. Most of the rest is the generality of a mixed batch.

### Imugi by function (v2, per step, two calls)

| Function | Operations | What it does |
|---|---|---|
| `logistic` (86 calls) | 774 | logistic switches parameterising the doctrines |
| `choice_uniforms` (14 calls) | 616 | random draws hashed by purpose so they match earlier runs |
| Frontline layer | 302 | |
| Combat, manoeuvres, doctrine | about 700 | |
| Memory | about 150 | |
| **Total** | **4,256** | 12.6x v1's scripted agent per call |

It runs twice per tick because each controller calls it on its own rows (once for the
league slots, once for the lanes). Each player should run once per tick over all the rows
it plays.

### Steady speed (from earlier records)

| | Steps per second | Source |
|---|---|---|
| v1, `wild-universe-731` (RTX 4070 Laptop) | 2,670 median | `train/sps` in the W&B export; whole-run wall time, compile and in-training Elo included |
| v1, `good-leaf-719` (same laptop) | 2,707 median | same |
| v2, BC, about 2026-11-10 (laptop) | about 4,200 | `sections/gpu-box-log.md`; steady updates only |
| v2, real-scale RL after the learner work (laptop) | about 5,050 | `sections/compile-overhaul-notes.md`; steady updates only |

Not like for like (whole run against steady updates; different games and models). The
laptop limits v2's learner to 160-row micro-batches, its slowest regime. A direct A/B on
the laptop is still to do.

## Findings

Each finding is an assumption that brought in complexity, with its status.

1. **A hand-built belief for fogged enemies.** About 3,600 operations per forward, 70% of
   the model. The evidence for the error head was within seed noise (deadreckon-split over
   absolute: +27 Elo mean, under the +40/+50 bar; seed noise about 110 Elo). Arm A14 never
   ran. **Decided: removed.**
2. **Every output is a distribution over bins** (symlog HL-Gauss, ranges up to ±26 km),
   decoded with medians and quantiles, encoded again as learner labels, trained with a
   custom grouped cross-entropy. Assumption: distributional outputs everywhere. **Open:**
   largely resolved by the head decisions; value and next state remain.
3. **Positions for a world 30 times the arena.** A random offset over 236 km per match
   needs an 8-level colour code down to 4 m and 8 rotary bands up to 340 km, for a 7.8 km
   arena. Assumption: very varied map sizes, and absolute positions must not be
   memorisable. **Decided: simplify.**
4. **Input encoding generally.** 1,037 operations per forward against 148 for v1's whole
   encoder. Assumption: the model owns all feature encoding, as two-hot codes of everything.
   **Open.**
5. **Eight auxiliary heads** and the rollout machinery they need (shot capture, bullet
   re-simulation, a 128-tick lookahead after every segment that is never trained on, about
   25% extra rollout). Assumption: auxiliary heads help; arm A13 never ran. **Decided in
   part** (see Decisions). The lookahead's remaining users (the returns' tail, shot outcome
   labels) are to check.
6. **Rating lanes inside training.** Half again the training environments, for a live
   rating that the lanes' own analysis says cannot reach its ±25 Elo target. **Open.**
7. **Imugi.** 2,145 operations per call, run twice per tick; about 2,500 lines plus doctrine
   and population files. **Open.**
8. **Mixed batches.** Three modes, two pools each and drawn lineups in one batch make every
   rule a per-environment tensor and every ship constant a lookup. **Open.**
9. **Bit-level reproducibility machinery:** 173 call sites of keyed draws, a Dynamo
   workaround for the crc32 stream keys, about 64 bitwise or ulp assertions in tests,
   autotune caches kept to repeat the bits, game-version fingerprints, Imugi's purpose-keyed
   draws (616 operations per tick). Assumption: bit-exact reproducibility. **Contradicted by
   the principles; open how far to cut.**
10. **Fixed shapes everywhere** (`dynamic=False`). Set in `decisions.md` on 2026-10-16
    ("counts are fixed for a given run... not from dynamic shapes") when the framework was
    JAX, where static shapes are mandatory. PyTorch 2.15 defaults to automatic dynamic
    shapes. No v2 measurement of the cost of dynamic shapes was found. Each shape is a cold
    compile: the multiplier behind long compiles. **Open.**
11. **Constants as tensors.** Partly forced by eager CUDA-graph capture, which refuses host
    copies; plain arithmetic with Python numbers makes none. **Open.**
12. **Diagnostics in the hot path** (analytics 286 operations per tick, head statistics 960
    lines, triggers 647 lines). Cheap at run time, a maintenance load. **Open.**
13. **N-team generality** ("free-for-all must not require a rewrite", `decisions.md`).
    Not yet measured. **Open.**
14. **Custom Triton kernels** (attention, gated MLP, heads, scan, temporal mix):
    1,287 lines plus 695 lines of tests; each has a plain twin (two implementations to keep
    equal), custom autograd and autotuning (about 60 s of a cold stage), handling so that
    autotuning repeats the bits, and no `vmap` rule (the stacked lanes call each once per
    lane). Measured gain (`sections/fast-yemong-notes.md`, laptop, 160-row micro-batches):
    training bf16 51.5 -> 37.7 ms (1.23x of training throughput; fp32 1.10x), of which
    attention 9.0 ms, gated MLP 2.0, scan 1.7, temporal mix 0.9, heads 0.2. The staged
    shot-head kernel goes with the shot trajectory head. Standard alternatives: SDPA or
    FlexAttention with rotary applied before (the compiler fuses it), plain compiled code
    for the rest. **Open.**

## Notes on the open questions

### Dynamic shapes

Reasoning, not measured:

- **Batch-like dimensions** (environments, micro-batch rows, lane counts) should be cheap to
  make dynamic. Matrix multiplies go to cuBLAS, which picks kernels at run time anyway.
  Inductor's elementwise and reduction kernels take the size as an argument. The expected
  loss is a few per cent at most, and it removes most recompiles.
- **Small inner dimensions** (ships, tokens, bins, heads) gain more from specialisation:
  keep them static, or round them to two or three fixed sizes.
- CUDA graphs still need one capture per shape, but a capture takes seconds.

### Constants as Python values

- Compiled steady state: small gain, probably low single-digit per cent (constant folding).
- Eager: a real gain. Python numbers add no operations; tensor constants built per call do
  (450 `scalar_tensor` in the correction decode alone).
- Recompiles: a Python value is baked in, so changing it recompiles. Values fixed for a run
  should be Python numbers. Values that vary per environment or during a run stay tensors.

### Position encoding

Grid cells are multi-scale periodic codes on three axes 60 degrees apart. Projecting the
position (relative to the arena centre) onto three hexagonal wave vectors at a few scales,
then taking cos and sin, is one small matrix multiply plus two elementwise operations.
That is both the brain-like option and close to the cheapest.

### MoE tooling

League slots, rating lanes and averaged policies apply several weight sets to different,
variable numbers of rows. That is the MoE setting. A grouped matrix multiply with per-group
offsets (`torch._grouped_mm`, or Triton's grouped GEMM) does it in one launch, without
padding or `RUN_SLACK` grouping, at a fixed total row count. To check: support on the
laptop's GPU and the 5090 in this build. First settle how many weight sets a tick needs.

## The learner (counted 2026-11-21)

`v2_learn.py` and `v1_learn.py` count one update's learner phases: the preparation done once
per update, and one micro-batch's label codes, forward and loss, and backward. Every
function of the model and the learner is a region. v2's Triton kernels count as one
operation forward and one backward (their plain twin computes the values and gradients
uncounted). Each was run at segment lengths 16 and 32 to see what grows with the length.
Records: `records/v2_learn_ppo_L16_L32.txt`, `records/v1_learn_L16_L32.txt`.

### Per micro-batch

| Phase | v1 (L 16 / 32) | v2 L 16 | v2 L 32 | v2 L 128 (extrapolated) |
|---|---|---|---|---|
| Label codes | (once per update) | 1,538 | 2,338 | about 7,100 |
| Forward and loss | 1,043 / 1,053 | 3,996 | 4,796 | about 9,600 |
| Backward | 1,153 / 1,175 | 782 | 782 | 782 |
| **Total** | **about 2,200** | **6,316** | **7,916** | **about 17,500** |

v1 barely grows with the length (its log-depth scan adds about 10 operations per doubling).
v2 grows by about 50 operations per tick in both the label codes and the forward. The
extrapolation: v2's growth is linear in the two measured points.

v2's backward is smaller than v1's because its kernels count as one operation each. Most of
v2's forward is not differentiable: label encodings, decodes, the per-tick loop below.

The clip and optimizer phase is not comparable here: on the CPU both versions run an
unfused Adam (v2 also saves and restores every tensor); on a GPU v2's Adam is fused.

### Once per update

| | v1 | v2 L 16 | v2 L 32 |
|---|---|---|---|
| Prepare (returns, advantages, labels written to the store) | 1,761 | 236,562 | 353,330 |
| Heads' sums (every 10th update, a sample of micro-batches) | | 8,516 | 10,916 |

### Learner findings

L1. **A Python loop over every tick in the extrapolation baseline.** `Yemong.rates`, via
    `Yemong.segment_rates`, steps tick by tick: about 50 operations per tick, in the label
    codes and again in the forward. At a segment length of 128 that is about 5,600
    operations in each graph, more than half of it, so the compiled graphs and their
    compile time grow with the segment length. Part of the belief: removed by the
    2026-11-21 decision.

L2. **Shot re-simulation for the shot labels** (`bnb.learn.shots.resimulate`): about 300
    operations per simulated tick, in a Python loop of L + 11 ticks per chunk of views,
    chunks the size of a loss micro-batch. At the launch's RL settings (12,288 views,
    64-view micro-batches, L 128) that is 192 x 139 = about 26,700 calls of the shot tick
    per update: about 8 million operations eager, and 26,700 calls of a compiled function
    compiled. It flies a hypothetical shot for every ship at every tick, so it serves the
    shot trajectory head and the shot outcome head alike. **Keeping the shot outcome head
    keeps this machinery** unless its labels change (for example, real shots only, labelled
    by the game's own hit events).

L3. **Labels recomputed every epoch.** The label codes (about 7,100 operations per
    micro-batch at L 128) are rebuilt for every micro-batch of every epoch, though they do
    not change within an update. The BC teacher's labels were already moved to `prepare`
    for this reason.

L4. **The heads' sums diagnostic** (8,500 to 10,900 operations per sampled micro-batch) uses
    the same per-tick loop and the same decodes.

### Per update, uncompiled (estimate)

- v2 RL at the launch's settings: 4 epochs x 32 minibatches x 6 micro-batches = 768
  micro-batch steps x about 17,500, plus about 8 million for the shot labels: **about 21
  million operations**.
- v1 (`wild-universe-731`): 4 x 32 x 15 = 1,920 micro-batch steps x about 2,200, plus
  preparation: **about 4.3 million operations**.

Both updates are about a million environment steps. These are estimates from the counts
above, not measurements of an update.

## Not yet examined

1. The learner on the GPU: time and memory traffic per section (the laptop). The learner's
   steady time is GPU-bound (220 of 227 ms of kernels per step), so the counts above
   explain compile time and eager speed, not its steady speed.
2. Resolve and perceive split by mode, fog and bookkeeping (in more detail than above).
3. A laptop GPU benchmark of compile time and step time for: fully static; dynamic batch
   dimension; dynamic ship dimension; Python constants instead of tensors.
4. A like-for-like steady-speed A/B of v1 and v2 on the laptop.
