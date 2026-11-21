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
| 2026-11-21 | **Drop the custom Triton kernels.** They treated the symptom (too many small operations) rather than its cause. Use standard tools (SDPA or FlexAttention, plain compiled code); a kernel comes back only if a profile of the simplified model shows it pays. |

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
  Ghost bullets respect the gun's cooldown (the author): a ghost fills a slot a real shot
  could have used, so real and ghost bullets together never exceed a ship firing at its
  full rate. The game already sizes its slots for that rate (`most_bullet_slots`) and tests
  every slot each tick, so ghosts cost about the existing worst case, with perhaps one or
  two extra slots. To settle: ghosts spend no energy; one cooldown timer for real and ghost
  shots, so every tick the gun is ready carries a real or a ghost label.
- **Other treatments of the same symptom**, to re-examine once the simplified model exists:
  CUDA-graph capture of the rollout, `vmap` stacking with `RUN_SLACK` grouping, the raised
  recompile limit.

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
    for the rest. **Decided 2026-11-21: dropped.**

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

## Band-aids, by root cause (2026-11-21)

A band-aid is a mechanism added to mitigate a symptom instead of removing its cause. Four
read-only reviews (runtime and compilation; model; learner and league; game and Imugi)
listed them with file and note evidence; the surprising claims were checked against the
code (marked "checked"). Paths are under `src/bnb/` unless given in full. "Goes" means it
should not be needed once its root cause is fixed or the decisions above are carried out.

### R1. Far too many small operations per tick

The tick and the policy step are launch-bound, so the fixes made launches cheaper instead
of making fewer of them.

| Band-aid | Where | Goes? |
|---|---|---|
| CUDA-graph capture of the rollout, and the rules it imposes: static buffers, every output copied back into the carry, every change between segments copied in place, a device tick counter | `runtime/rollout.py` (capture at 681-695), `runtime/opponents.py`, `runtime/lanes.py`, `runtime/cloning.py`, `runtime/analytics.py` | Measure. The rollout is 16-18 s of a 90-250 s update, and v1's CUDA graphs were measured as a wash (`decisions.md`, checked). Capture's warm cost is 33-38 s (laptop) to 87-91 s (3090) |
| Constants filled on the device, index tables as buffers, "no host copy" rules (about 22 sites in the game and Imugi, about 15 in the model) | e.g. `game/modes/phase.py`, `controllers/imugi_frontline.py`, `models/yemong/state_code.py`, `models/yemong/delta.py` | Goes with eager capture |
| Five custom Triton kernels | `models/yemong/kernels/` | Decided: dropped |
| AOTInductor builds of the play tick and Imugi's step, in a child process, with an eager fallback until built | `runtime/built.py`, `runtime/realtime.py`, `runtime/imugi_step.py`, `play/sides.py` | Probably, if the tick and Imugi shrink about 10x |
| The play agent's B=1 Yemong step in a CUDA graph (eager GPU 15.5 ms, CPU 7.2 ms) | `play/sides.py:302-381` | Probably |
| A compiled flight tick for live tuning; ship characteristics flown at run time, cached on disk, in a child process | `runtime/flights.py`, `game/handling/characteristics.py` | Store characteristics with the card |
| Hand op-batching ("families", `*_at` twins of every code function) | `models/yemong/delta.py:840-881`, `models/yemong/codes.py` | Goes with the codes |
| Four separately compiled rollout stages; the bootstrap left eager | `runtime/rollout.py:195-203` | Measure once graphs are small |
| Per-tick analytics accumulated inside the graph (286 operations per tick) | `runtime/analytics.py` | Compute from the stored segment |

### R2. Every shape compiled separately, and graphs too large to recompile cheaply

| Band-aid | Where | Goes? |
|---|---|---|
| Training choices bent so the learner compiles once: 2,048 environments, mode shares in sixteenths, minibatch counts that divide the views, micro-batch size families, equal-row lane weights | `learn/ppo.py:301-316`, `runtime/train.py:67-72`, `runtime/lanes.py:86-111`, `runs/session.py:217-235` | Goes with a dynamic batch dimension |
| Dynamo's recompile limit raised to 64; every `Rollout` recompiles its stages (guards read the instance) | `cli.py:789-797` | Goes: compile pure stage functions |
| vmap stacking of weight sets; players grouped by row count (`RUN_SLACK`, which pads up to about 2x, not 10% as the notes say: checked) | `runtime/stacked.py`, `runtime/lanes.py:239-326` | Grouped matmul, or fewer weight sets per tick |
| Compile-cache logistics: queue warm keys, cache shipping, pruning (which once corrupted the cache) | `runs/queue.py`, `docs/rewrite/sections/launch/cache.sh` | Goes with short compiles |
| `compile_split`, measured as not paying | `runtime/train.py`, `learn/ppo.py:399-405` | Goes |
| The heads' sums run eagerly on a sample ("compiled it was the largest single compile of a cold start, 554 s") | `learn/ppo.py:285-291, 409-411` | Goes with the heads |
| Live-tunable rules as 0-d tensors with every branch computed (play shell only) | `game/tuning.py:435-541`, about 16 dual tensor/number paths | Goes if the play tick is cheap to rebuild or eager |
| Bullet slots sized for the longest life any tuning allows; the hit test over all slots (about 5% active) | `game/rules.py:36-38, 285-293`, `game/physics/combat.py:58-85` | Fixed slots for training are fine |
| Recompile and memory telemetry, recompile alert | `runs/recording.py`, `runs/triggers.py:448-456` | Recompile alert goes; the memory probe stays |
| Inductor's comprehensive padding off for the learner | `runtime/train.py:360-371` | Goes with the 154-wide shot code |

### R3. Mixed batches

Three modes, two pools per mode and drawn lineups share one batch, so every rule is a
per-environment tensor and every player kind runs on every row.

| Band-aid | Where | Goes? |
|---|---|---|
| Every opponent kind runs on every opponent row, then `where` picks one: the stacked network forward, a full Imugi pass, all six cheese agents (checked) | `runtime/opponents.py:327-347` | Goes: group rows by player |
| Per-environment rules rebuilt every tick (844 operations) | `game/layout.py:149-200` | Goes: one game per batch |
| `per_ship` and `lookup` as chains of `where` (484 operations) | `game/ship_types.py:216-228`, `game/physics/numerics.py:61-69` | Goes |
| Imugi's per-view parameter tables, and mode and ship set inferred from the observation (map size in gun ranges, gravity, type count) | `controllers/imugi_parameters.py:311-337`, `controllers/imugi.py:33-77`, `controllers/imugi_doctrine.py:337-348` | Goes: pass the game at construction |
| Imugi runs its combat and frontline layers on every row | `controllers/imugi.py:125-137` | Goes |
| Episode-end work (new map, mode state, spawn placement) built for every environment every tick, then selected by `where(done, ...)` (checked) | `game/physics/step.py:347-371`, `game/physics/spawning.py:208` | Mostly: reset only finished environments |
| Padding to one shape across modes: 5 zone slots everywhere (+3.4% rollout), absent fields as fake fields, fog sight computed in unfogged environments | `contracts/obs.py:123-126`, `game/physics/medium.py:19-22`, `game/perception/sight.py:106-108` | Mostly |

### R4. Bit-exact reproducibility

| Band-aid | Where | Goes? |
|---|---|---|
| Keyed counter-based random draws (173 call sites), crc32 keys made Dynamo-constant, salts kept so draws match earlier runs (Imugi: 616 operations per tick) | `streams.py`, `controllers/imugi_combat.py:98-105` | Goes: ordinary generators |
| Exact resume of the rollout carry and generators, about 630 lines of resume tests | `runs/checkpoint.py:360-430`, `tests/runs/test_resume.py` | Partly: weights, optimizer and league state stay |
| The step/unroll bit-agreement machinery: a replay record per tick, int16 actions, float32 spreads, the rule "never store a rounded copy a compiled stage also reads unrounded" | `models/yemong/inputs.py:42-53, 155-171` | Goes with the belief |
| Kernel autotune caches kept to repeat the bits; fixed 32-row groups | `models/yemong/kernels/` | Goes with the kernels |
| The step/unroll ratio "drift" check, with its own eager unroll | `learn/ppo.py:293-298, 1682-1706` | Goes |
| Game fingerprint hash (the version number stays) | `game/__init__.py`, `tests/sections/test_game_version.py` | Goes |
| Small patches: a -0.0 turned into +0.0 for tie order, a gun range "to the last bit", float32 tempering, truncated series matched to a float64 reference | `game/physics/combat.py:69-77`, `controllers/scripted.py:17-25`, `models/yemong/policy.py:36-42`, `game/physics/numerics.py` | Goes |

### R5. The hand-built belief feeding the model's own decoded outputs back in

| Band-aid | Where | Goes? |
|---|---|---|
| The error head's confidence-shrunk, rate-limited, capped step toward a median (it "ran away" without them) | `models/yemong/correction.py` (842 lines) | Decided: removed |
| Held, decaying extrapolation rates (a velocity error of 1,040 m/s ran away without the decay) and the per-tick loop | `models/yemong/model.py:904-995`, `models/yemong/delta.py` | Removed |
| Covariance caps, the decoupled Jacobian, the fade, the non-finite scrub and its counter | `models/yemong/delta.py:723-752, 1304-1321`, `models/yemong/belief.py:65-73` | Removed |
| The decode-correction stack: padding bins, the mean-inverse table, the median window, probit quantiles | `models/yemong/codes.py:550-849` | Mostly removed |
| The `sigma_miscalibrated` alert, hidden-row weighting of the next-state loss, belief sums in `prepare` | `runs/triggers.py:592`, `learn/ppo.py:244-260, 1100-1112` | Removed |

### R6. Categorical outputs over fixed, very wide ranges

| Band-aid | Where | Goes? |
|---|---|---|
| Return units: every reward weight x10, value support widened to 77 bins over ±3, the return-scale floor of 1, the support-clipping alert, zero entropy at warm start (the EMA of the scale also starts at 0 without bias correction) | `learn/rewards.py:45-63`, `models/yemong/model.py:250-251`, `models/yemong/critic.py:87-110`, `learn/ppo.py:149, 207-212` | Goes: normalise advantages per batch and predict normalised returns |
| The outcome as its own 4-class value level with its own γ and λ (BC needed different ones) | `learn/ppo.py:374-482`, `models/yemong/critic.py`, `learn/bc.py:43-50` | Decided: merged into value |
| Hard-coded marginal bias tables for the next-state head (about 510 lines of quantiles) and the baseline floor | `models/yemong/delta.py:134-648`, `models/yemong/codes.py:529-545` | Probably |
| Shot-code memory tricks: the bf16 kept factor, encoding inside the loss, the custom grouped cross-entropy | `models/yemong/codes.py:482-526`, `models/yemong/shots.py` | Goes with the shot trajectory head |

### R7. Positions in a 236 km frame, and map-size sampling

| Band-aid | Where | Goes? |
|---|---|---|
| Float32 precision fixes for the offset frame: a remainder per level, exact periods, unwrapping, absent rows placed at the centre "which would reveal the offset" | `models/yemong/codes.py:38-50`, `models/yemong/layers.py:31-46`, `models/yemong/decode.py:152-157`, `models/yemong/inputs.py:405-408` | Decided: simpler positions |
| A cascade of floors for sampled small maps: zone radius and spacing floors, then an arena floor so the floored zones fit | `game/modes/rules.py:88-109, 266-272`, `game/layout.py:105-113` | Goes if fixed maps stay |

### R8. Labels and targets rebuilt after the fact

| Band-aid | Where | Goes? |
|---|---|---|
| Shot re-simulation (about 26,700 compiled calls per update) | `learn/shots.py:163-200` | Goes with game-recorded labels |
| A 128-tick lookahead after every segment, played and never trained on (about 25% extra rollout), for the returns' tail and the shot labels | `runtime/train.py:159-161, 210`, `learn/ppo.py:1024-1029` | Goes: bootstrap the tail from the value, as standard PPO does; shot labels need at most 11 ticks |
| Label codes rebuilt every epoch | `learn/ppo.py:567-597` | Goes: compute once per update |

### R9. Memory pressure from bloated per-tick records

| Band-aid | Where | Goes? |
|---|---|---|
| The pinned-host segment store: a copy stream, events, double-buffered staging on a worker thread. One race fixed (1f5558ac); pinning about 11 GB of host RAM made two rented boxes stop answering | `runtime/store.py`, `runtime/rollout.py:697-712`, `runtime/train.py:244` | Probably: without the shot capture, the belief replay and the long lookahead, the segment may fit on the device |
| View indices kept on the host, pinned copies per micro-batch | `learn/ppo.py:1008-1015, 1157-1170` | With the store |

### R10. Live rating through extra machinery

| Band-aid | Where | Goes? |
|---|---|---|
| Rating lanes | `runtime/lanes.py`, `league/lanes.py` | Leaning: rate from training games |
| Bookkeeping for weights swapped mid-game and staggered starts: a row's first game not counted, games in progress at a cut not credited, lane generations | `runtime/opponents.py:214-219, 389-406` | Mostly, if an opponent changes only at its row's reset |
| Corrections inside the rating fit: prior games, a moving-player window, a snapshot credited with the live policy's games, the gauge reset, `GAUGE_DISTANCE` and the `gauge_runaway` alert (the runaway was a fit bug, since fixed) | `league/ratings.py:76-94`, `league/gauge.py:76-84`, `league/league.py:477-491` | The alert goes; some regularisation stays |

### R11. Monitoring a symptom, or keeping an option nobody uses

| Item | Where | Goes? |
|---|---|---|
| The trigger framework (647 lines), with placeholder thresholds | `runs/triggers.py` | Mostly |
| Non-finite steps skipped and counted, via fused Adam's private `found_inf`. The one real NaN came from the game (a 0/0); the learner never produced one | `learn/ppo.py:1289-1327` | Goes: check once per update and stop the run |
| KL early stop ("never fires"); BC's plateau rule and score window (only the budget fires; about 240 lines) | `learn/ppo.py:1412-1419`, `learn/bc.py:382-663` | Goes |
| The combat-collapse alert. Its cause is the reward and discount design: one discount for every level let a run stop fighting | `runs/triggers.py:361-414` | Fix the cause; the monitor can stay |
| Four trust regions, and the rollout recording the full 42-way distribution for one of them | `learn/ppo.py:84, 204-206` | Keep one |
| Team mixing: `team_spirit` is 0 on every level, so it computes an identity (checked) | `learn/returns.py:55-93`, `learn/ppo.py:147` | Goes |
| Loss-weight balancing machinery (gradient decomposition on a cadence), mostly for heads being dropped | `learn/ppo.py:1532-1681`, `learn/gradients.py` | Mostly |
| Legacy weight: 20+ record-default switches in `YemongConfig`, ONNX-shaped logic for a contract test only, legacy observation columns, `imugi_populations.json` (5,170 lines read only by one test: checked), three standard Imugi parameter sets if training is real-scale only | `models/yemong/model.py:104-342`, `contracts/obs.py`, `controllers/imugi_defaults.py` | Goes |
| An `Adam` subclass that skips the capture health check on the CPU | `learn/ppo.py:94-108` | Unclear |

### Suspected defects found on the way (unconfirmed)

- **The next-state code's bounds may predate the game the model trains on.** The constants
  in `models/yemong/delta.py` were last changed at game version 3 and were measured on the
  standard ship set at x1 speed. Training runs the real-scale set at x2 native speed
  (game v13), where a tick's move may pass the position bound of 8.7 m. To measure before
  relying on any next-state result; moot if the head is rebuilt simply.
- **The return scale's EMA starts at 0 without bias correction**, so early updates sit on
  the floor (`scale_floored 1` in the 3090 logs).
- **Stale notes:** `RUN_SLACK` described as 10%; a gradient-clip rationale from v1 ("fires
  every step") that does not hold in v2 (clip fire rate 0).

## Root causes, ranked

Each band-aid above traces to one of these. They are what to remove.

1. **Far too many small operations** (R1): the encodings, the decodes, the belief, Imugi,
   and the game's generality. Cause of the eager slowness and of most compile time; the
   kernels, CUDA graphs and AOT builds treat it.
2. **The hand-built belief** (R5): a closed loop through decoded outputs. Each fix for a
   runaway added more. Decided: removed.
3. **Mixed batches** (R3): every rule per environment, every player kind on every row.
4. **Every shape compiled separately** (R2): `dynamic=False`, inherited from the JAX plan;
   the configuration bent to fit it.
5. **Bit-exact reproducibility** (R4): keyed RNG, exact resume, bit agreement, fingerprints.
   Contradicted by the principles.
6. **Categorical outputs over fixed wide ranges** (R6): reward scaling, floors, bias tables.
7. **Labels built after the fact** (R8) and **bloated per-tick records** (R9): the
   re-simulation, the long lookahead, the pinned-host store.
8. **Absolute positions in a huge frame** (R7) and **live rating via extra environments**
   (R10).
9. **Monitoring and options instead of decisions** (R11).

## Not yet examined

1. The learner on the GPU: time and memory traffic per section (the laptop). The learner's
   steady time is GPU-bound (220 of 227 ms of kernels per step), so the counts above
   explain compile time and eager speed, not its steady speed.
2. Resolve and perceive split by mode, fog and bookkeeping (in more detail than above).
3. A laptop GPU benchmark of compile time and step time for: fully static; dynamic batch
   dimension; dynamic ship dimension; Python constants instead of tensors.
4. A like-for-like steady-speed A/B of v1 and v2 on the laptop.
