# Compile overhaul: notes

Branch `perf/compile-overhaul` (from `feat/real-scale` at 15fca8d1), started 2026-11-10
21:20 KST; soft deadline 2026-11-11 10:00 KST.

The author's complaint: everything compiles slowly (a 40v40 ratings match about 1000 s,
evaluation spending five times longer compiling than playing, play mode freezing up to a
minute), and end-to-end training is only about 25% faster than v1. The aim is to find and
fix the structural causes, not micro-optimise.

## The diagnosis (before any change)

Traced op counts (Dynamo with a recording backend, no Inductor), identical at 5v5 and
40v40, so the graphs do not grow with the fleet:

| Graph | Ops |
|---|---|
| Game tick (move + resolve + perceive, B = 1) | 2,867 |
| Scripted agent step | 1,352 |
| Yemong controller step | 3,663 |

Compile time follows the op count, and the stages stack these units with every Python
loop unrolled: one network forward per lane or weight set, one per Yemong block, one per
code level, one per field. `dynamic=False` makes every new shape (size, preset, lane
count, micro-batch) a full new compile.

## The items

From my own reading and two read-only review passes (one over the section notes, one over
the hot code paths), checked in the code before listing. Status is kept current below.

| # | Item | Status |
|---|---|---|
| 1 | `bnb.game.VERSION` never raised (36 game changes under version 1) | done (v2 + fingerprint test) |
| 2 | Ratings lanes: 16 serial Yemong forwards per tick in one graph (`MatchSeat._network`) | done: one vmapped forward (`runtime/stacked.py`) |
| 3 | `default_lanes` gives one network group all 16 slots; at most 8 can play | done (capped at `blocks x min(players, 2)`) |
| 4 | `bnb ratings` builds a new Arena and compiles everything per preset (3x) | |
| 5 | Training act stage: 11 forwards, about 5 duplicates (LIVE = learner, league_k = slot k) | the slots and the lanes vmapped; the duplicates remain (below) |
| 6 | Learner loss compiled in several variants (heads flag, micro-batch remainders, cadence, a second learner step for bootstrap) | bootstrap uncompiled; heads' sums their own small stage (one loss graph) |
| 7 | BC teacher labels computed inside the loss, every epoch | done (made once in `prepare`, in micro-batch-sized chunks) |
| 8 | Several scripted passes per act stage; the frontline layer on every row | |
| 9 | Triton autotune keyed on exact shapes | |
| 10 | Arena counts only first games but plays until the slowest ends | kept: deliberate (see below) |
| 11 | Arena runs the training bookkeeping (records, store, analytics) and syncs 3x per segment | |
| 12 | Non-finite step guard clones every parameter and Adam state per step | done (fused Adam's `found_inf`; GPU test exact) |
| 13 | Diagnostics over their 2% cap; cadence OOM at 4096 environments on 8 GB | |
| 14 | Yemong code helpers: ~2,100 of 3,663 ops from Python loops | |
| 15 | `sight.sighted` unrolls one copy per field | |
| 16 | Dynamo's recompile limit (8 per code object): a process building more than 8 Rollouts may fall back to eager | done: it fails hard, not eager; the CLI raises the limit to 64 |

Not mine to change: the real-scale damage per second (360) rests on one seed-1 episode
(`game/real_scale.py:112-115`); the author's balance work builds on it. Yemong.VERSION is
also left at 1: raising it refuses every existing policy, and the two model fixes since it
was introduced (a2619189 the error head's spread, 59ea32b2 the delta covariance) change
auxiliary heads, not the policy's decisions, as far as I can tell. Both are for the author.

## Baseline (cold caches)

Method: `bench/compile_baseline.py` wraps every `torch.compile` site and times each call
that compiles (device synchronised before and after, labelled by function), times the
CUDA-graph capture separately, then the case's phases. `bench/compile_baseline.sh` runs
each case in its own process with empty `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` and
`BNB_CACHE`, as a systemd user unit (MemoryMax 20G). Power profile: performance. The
author's `bnb play` (27% of a core) was killed before the timed cases; nothing else ran.

Cases: realtime (the play tick, CPU); ratings (two policies of one build, arm2d's
exponential and live averages, plus the scripted agent, frontline 5v5, default
ArenaConfig: 8 blocks x 128 environments); bc (arm2d's TrainConfig); rl (rl-aux-x0's
TrainConfig). The runs' stored ship cards are dropped (`drop=ship_card`): the card's fields
changed since and do not change the graphs' shapes.

Results: `compile-overhaul/baseline/*.json` and `*.log`.

- ratings (frontline 5v5, 16 lanes, 1024 games a batch): build 3.1 s; first batch 747 s;
  second batch 175 s. The stages, compiled during the capture's warm-up ticks:

  | Stage | Compile | Traced ops |
  |---|---|---|
  | `_act` | 529.3 s | 65,516 |
  | `_advance` | 12.6 s | 1,841 |
  | `_resolve` | 21.9 s | 1,295 |
  | `_perceive` | 8.3 s | 628 |
  | the capture itself | ~0.4 s | |

  Inductor's own split, all stages: Dynamo tracing 92 s, AOT dispatch 66 s, lowering 57 s,
  the scheduler 108 s (init) + 120 s (codegen), Triton compilation 14 s; about 70 s of the
  first batch is outside these (most likely the Yemong kernels' first-call autotune). Every
  part but the autotune scales with the ops, so the act stage's 16 unrolled forwards are
  93% of the compile.
- bc (arm2d's config: 2048 environments, L 128, 4 shards, lookahead 128, rating lanes
  on 1024 of them): build 31.7 s; first collect 447 s, first learn 502 s; second collect
  17.8 s, second learn 240.7 s (so a steady update is 258 s, 93% of it the learner, and
  the first update costs about 720 s more). Peak device memory 4.6 GB.

  | Compile | Time | Traced ops |
  |---|---|---|
  | `Rollout._act` | 308.9 s | 18,837 |
  | `Rollout._advance` | 12.6 s | 1,841 |
  | `Rollout._resolve` | 30.7 s | 2,070 |
  | `Rollout._perceive` | 11.6 s | 780 |
  | `YemongController.step` (the bootstrap's own compile) | 65.3 s | 3,881 |
  | `tick` (the shot re-simulation) | 6.4 s | 367 |
  | `Learner._codes` | 12.5 s | 892 |
  | `Learner._loss`, epoch 0 (with the heads' sums) | 104.1 s | 6,759 |
  | `Learner._loss`, later epochs (without) | 70.2 s | 4,416 |

  The compiles sum to 622 s; the first learn's other ~75 s over the steady one is most
  likely the kernels' autotune for the backward's shapes. The act stage's 18.8k ops hold
  the learner's forward, the rating lanes' forwards and the teacher's scripted pass.
- rl (rl-aux-x0's config: PPO with the league's 4 opponent slots and the rating lanes):
  build 28.1 s; first collect 631.5 s, first learn 524.3 s; second collect 24.0 s, second
  learn 225.1 s (a steady update 249 s, 90% the learner; the first costs about 900 s
  more). Peak device memory 4.7 GB. Compiles 839 s in all:

  | Compile | Time | Traced ops |
  |---|---|---|
  | `Rollout._act` | 493.1 s | 42,611 |
  | `Rollout._advance` | 14.0 s | 1,841 |
  | `Rollout._resolve` | 29.4 s | 2,090 |
  | `Rollout._perceive` | 11.4 s | 780 |
  | `YemongController.step` (bootstrap) | 59.6 s | 3,617 |
  | `tick` (shot re-simulation) | 6.8 s | 367 |
  | `Learner._codes` | 54.1 s | 2,023 |
  | `Learner._loss`, epoch 0 | 107.1 s | 6,987 |
  | `Learner._loss`, later epochs | 63.9 s | 3,160 |

**What the baseline says.** Cold, the first update costs 720 s (bc) to 900 s (rl) more than
a steady one, and a ratings batch 572 s more. The act stage, holding one network forward
per weight set, is 53% (bc) to 93% (ratings) of it; the loss's two variants and the
bootstrap's second network compile are most of the rest. Separately, the steady update is
90-93% learner: the rollout is 18-24 s of 250 s.
- realtime: the first tick compiles in 25.8 s (Dynamo 3.0 s; Inductor 22.5 s, of it the
  scheduler 8.3 s, C++ compilation 6.0 s, wrapper codegen 10.6 s); 2,867 ops, 1 graph.

## Log

- 21:22 the first baseline launch: realtime ran; the other cases failed on stale run
  configs (bc-persist-split's policy has a model field this branch lacks; rl-aux-x0's card
  lacks `engine_knee_scale`). Relaunched with arm2d and `drop=ship_card`.
- Item 1: `bnb.game.VERSION = 2`, `FINGERPRINT` beside it, and
  `tests/sections/test_game_version.py` (48 ticks per preset, CPU eager, the state and
  observation rounded to 5 significant digits and hashed; the same at 1 and 4 threads and
  across processes).
- 21:48 the bc case was killed by the OOM killer at MemoryMax 20G during the learner's first
  compile (the trainer's pinned host store plus Inductor's compile workers). Relaunched bc
  and rl at 22:31 with MemoryMax 32G, from a detached worktree at a00c5a47
  (`../bnb-compile-baseline`) so the baseline runs the unchanged code while this tree
  changes. bc's partial log: `baseline/bc.oom-20G.log` (its rollout stages: act 308.3 s at
  18,837 ops, advance 12.7 s, resolve 31.1 s at 2,070 ops, perceive 11.6 s, the bootstrap's
  separate `YemongController.step` 65.3 s at 3,881 ops; collect 1 took 447 s).

## Decisions

- **Item 10 stays.** The Arena counts each environment's first game only and plays until
  the slowest has ended. Counting every game that ends within a fixed time would weight
  short games (quick wins, quick losses) more than long ones, which biases a rating;
  the waiting is the price of an unbiased count. Not changed.
- **Item 16** is not the silent eager fallback the review feared: with `fullgraph=True`
  the ninth `Rollout` in a process raises `FailOnRecompileLimitHit` (tested on a toy class:
  each instance recompiles a bound method, the ninth fails). The bench scripts' calls to
  `torch._dynamo.reset()` between presets work around it. `bnb.cli.compile_limits` raises
  the limit to 64 for the commands that build engines.
- **Item 12:** fused Adam already skips a flagged step in its kernel (`found_inf`, as
  gradient scaling uses it: the parameters and moments are not written and the step
  count is taken back). On the CPU (unfused) the old save-and-restore stays.
- **Item 6, the bootstrap:** `Rollout.bootstrap` ran the learner's step through its own
  `torch.compile`, a second compile of the network (65 s cold) for one call per update.
  It now runs uncompiled (and under `no_grad`, which it lacked).
- **Item 6, the heads:** `_loss(heads=...)` was a Python flag, so epoch 0 (with the heads'
  sums) and later epochs were two compiled graphs (104 s and 70 s in bc). The loss now
  always returns what the sums read (detached), and the sums are their own compiled stage
  `Learner.head_sums`, called on epoch 0 only.
- **Item 7:** the labels are the same in every epoch, so `BehaviourCloning.prepare` makes
  them once per update (eager, chunked to the views a loss micro-batch held), and the loss
  reads `derived["teacher_labels"]`; the teacher's carries are no longer stored for it.
- **Items 14, 15 (codes, sight) postponed:** the code helpers are already vectorised; their
  ops are many calls with per-feature constants. Folding them changes the model's input
  rounding, which every trained policy depends on, for roughly 1,500 ops of one forward,
  and after the regions the forward is compiled once. Not worth the risk now.

## The stacked forward (items 2 and 5)

`bench/lane_probe.py`: `MatchSeat.step` alone, compiled cold, 8 rows a lane (compile time
does not depend on the rows), one Yemong build (arm2d's), on the dev GPU:

| Lanes | Loop: compile, step | `nested_compile_region`: compile, step | vmap: compile, step |
|---|---|---|---|
| 2 | 107.8 s, 3.3 ms | 110.3 s, 3.3 ms | 96.2 s, 3.0 ms |
| 4 | 162.8 s, 5.6 ms | 169.7 s, 5.6 ms | 109.3 s, 3.5 ms |
| 16 | (~600 s by the slope) | 599.1 s, 19.1 ms | **146.2 s, 8.0 ms** |

- The region was tried first (it keeps each lane's arithmetic): Dynamo traced the forward
  once (3,495 ops at 16 lanes, against 65,516 unrolled), but the compile grew with the lanes
  as fast as the loop's (AOT dispatch and Inductor's scheduling still scale), so it bought
  nothing and was dropped.
- vmap holds one batched copy (5,533 ops at any lane count): 16 lanes compile 4.1x faster,
  and a tick's network part runs 2.4x faster, since each lane's forward was latency-bound
  (about 1.15 ms each even at 8 rows).
- Correctness (`tests/sections/test_stacked.py`): each set's policy logits and carry equal a
  forward of its own within the batched products' rounding (float32 1e-4; bf16 2e-2 of each
  tensor's scale), with no resets: a reset draws the position offset, and vmap's batched
  draw (`randomness="different"`) is not the separate forwards' draws. Sampled actions and
  offsets therefore differ from the loop's draw for draw, as any change to the draws would;
  their distribution is the same.
- The plain path's attention (`SpatialBlock._attend`) now casts to the autocast dtype itself:
  under vmap autocast skipped the cast and a float32 query met bf16 values (the league
  section's reason for the loop). The explicit cast is bitwise autocast's on CPU and GPU.
- The training lanes' players have unequal row counts (the live policy plays in more
  matchups): `stacked.padded_rows` pads each to the largest with its first row, and the
  padding's decisions are dropped.
- Left: the duplicate forwards in training (lane LIVE is the learner's weights, lane
  league_k a slot's). They are now batch entries of one forward rather than separate graph
  copies, so they cost rows, not compile.

## After the stacked forward and the learner changes (b87dc04f, d0b3f494)

Same cases, same method, cold (`compile-overhaul/after-1/`).

- ratings: first batch **353 s** (was 747 s); second batch 167.9 s (was 175.0 s).

  | Stage | Before | After |
  |---|---|---|
  | `_act` | 529.3 s, 65,516 ops | 142.9 s, 5,623 ops |
  | `_advance` | 12.6 s | 13.7 s |
  | `_resolve` | 21.9 s | 20.7 s |
  | `_perceive` | 8.3 s | 8.6 s |
  | all four | 572 s | 186 s |

  The steady batch barely moved (4%): the network forwards were not what a ratings tick
  spends its time on (`bench/arena_stages.py` to find what is).
- The first bc run of this failed: the training lanes' padded rows were taken with a
  boolean mask inside the act stage, a data-dependent size Inductor rejects ("u4 possible
  memo disaster"; the CPU tests run uncompiled and missed it). The kept positions are now
  made in `LaneSeat.__init__`, and a slow test compiles a lane seat whole.
- bc (d0b3f494): build 31.3 s; first collect 299.7 s (was 447.0), first learn 395.6 s
  (was 501.9); steady collect 19.0 s (17.8), steady learn 233.0 s (240.7). Compiles
  383.6 s (was 622): act 206.8 s at 10,923 ops (308.9 s at 18,837), advance 12.5,
  resolve 31.5, perceive 13.1, tick 6.3, codes 13.1, the loss 63.5 s at 3,162 ops (two
  variants, 174 s, before; the teacher no longer in it), the heads' sums 36.8 s. The first
  update now costs 475 s over a steady one (was 723 s).
- rl (d0b3f494, with the lanes padded): build 27.0 s; first collect 370.9 s (631.5), first
  learn 453.5 s (524.3); steady collect **28.7 s (24.0: slower)**, steady learn 229.3 s
  (225.1; the lane tests ran beside it). Compiles 513.8 s (was 839): act 270.9 s at 12,891
  ops (493.1 s at 42,611), codes 50.7, the loss 65.3 s (two variants, 171 s, before), the
  heads' sums 62.6 s at 3,825 ops.
- The slower rl collect was the padding: rl-aux-x0's lane matchups give live and the
  exponential average 4.75 units of rows each and each league slot 2.25, so padding every
  player to the most ran about half again as many network rows. bef843eb orders the
  players by row count and runs one stacked forward per run of equal counts, unpadded
  (here two runs: two players, four players); measured in `after-2`.
- The heads' sums in rl (62.6 s, 3,825 ops) are a diagnostic measured on every
  micro-batch of epoch 0. Sampling a few micro-batches would make them cheap at run time
  but not at compile time; compiling them is the cost. Left for the author: running them
  uncompiled on a sample is a change to what the series mean.

## Cold against warm (after-2, bef843eb)

`bench/compile_warm.sh`: one case twice in two processes on the same caches, empty then
warm (what a repeat run of the same preset and size pays).

- ratings: cold first batch 352.9 s (stages 142.5 + 14.1 + 21.2 + 8.7 = 186 s); **warm
  first batch 185.3 s** (stages 13.7 + 2.0 + 1.8 + 0.8 = 18 s); steady 166.7 s and 166.9 s.
  A repeat ratings run now pays about 20 s of compile over its play. Before the stacked
  forward the warm act stage would still have traced 65k ops (Dynamo alone took 92 s
  cold), which is why the backlog saw "5-10 minutes even with the cache warm".
- So each new preset or fleet size is a cold compile (about 3 min for ratings), and each
  repeat is about 20 s; the play itself (167 s a batch of 1024 frontline games) is now
  the cost of an evaluation.

## The training lanes' grouping (bef843eb, 8b301089)

rl-aux-x0 trains over two modes, so the lanes are split per mode and the lane players'
row counts are 454 (live), 454 (exponential average), 218, 218, 214, 214 (the league's
slots). Three ways to stack them, measured cold (rl):

| Grouping | Act stage compile | Steady collect |
|---|---|---|
| loop (baseline) | 493.1 s, 42,611 ops | 24.0 s |
| one group, all padded to 454 | 270.9 s, 12,891 ops | 28.7 s |
| a group per exact count (three) | 384.5 s, 20,891 ops | 24.0 s |
| within 10% (two; 8 padded rows) | 322.5 s, 16,964 ops | 24.1 s |

Padding everyone to the most ran 1.54x the network rows (+4.7 s a collect, about 2% of an
update, every update); a group per exact count paid a network copy in the graph for each
count, and counts differ by a few rows from the split's rounding. `RUN_SLACK` (10%)
groups players within a tenth of each other's rows.

RL with the 10% grouping (8b301089, `after-3/`), cold then warm:

- cold: {'build': 26.9, 'collect_1': 417.0, 'learn_1': 448.6, 'collect_2': 24.1, 'learn_2': 222.4}; compiles 563.7 s: [('Rollout._act', 322.5), ('Rollout._advance', 14.2), ('Rollout._resolve', 32.3), ('Rollout._perceive', 11.2), ('tick', 6.5), ('Learner._codes', 51.0), ('Learner._loss', 65.0), ('Learner._head_sums', 61.1)]
- warm: {'build': 4.6, 'collect_1': 63.2, 'learn_1': 228.3, 'collect_2': 24.1, 'learn_2': 217.5}; compiles 49.5 s: [('Rollout._act', 32.3), ('Rollout._advance', 1.9), ('Rollout._resolve', 3.8), ('Rollout._perceive', 1.0), ('tick', 0.5), ('Learner._codes', 1.9), ('Learner._loss', 4.1), ('Learner._head_sums', 3.9)]

A steady rl update is now 24.1 + 222.4 s (baseline 24.0 + 225.1).
The first update costs, over a steady one, 646 s cold (baseline about 900 s) and
54 s warm.
- The trainer's build (27-32 s cold) is 4.6 s warm: it is a compile too (the scripted
  agent's characteristics fly a compiled flight tick, `runtime/flights.py`), and the
  on-disk caches already cover it.

## What a steady ratings batch spends (bench/arena_stages.py, bench/network_kernels.py)

`arena_stages.py` (frontline 5v5, 16 lanes, 1024 games, warm): a captured tick is 18.0 ms;
outside the graph the act stage is 20.6 ms of which the network group's stacked forward is
18.5 ms and the scripted pass 0.34 ms; advance 0.40, resolve 0.63, perceive 0.23 ms. A
32-tick segment is 580 ms against 575 ms of graph replays, so the host's work between
segments (the store writes, the tally read, the finished check) is about 1%.

- **Item 11 (a lean Arena) is not worth doing**: the records, the store and the analytics
  are under 2 ms of an 18 ms tick together with the physics. The batch is the network.
- One forward of the controller over 2048 views (no lanes) takes 11.8 ms
  (`network_kernels.py`, arm2d's build: 2.63 M parameters, width 128): about 36% in the
  matrix products and the Triton kernels, and about half in large fused elementwise
  kernels of the inputs' and the belief's codes (the model's input design; not this
  work's to change).
- So the 16 stacked lanes cost 1.6x one forward over the same rows (19.5 against 11.8 ms
  at 2048 rows, `lane_probe.py block_environments=128`): vmap has no batching rule for
  the Triton kernels (`torch.library.triton_op`), so it calls each once per lane. Their
  plain twins batch (as batched products) but run slower: all plain 24.8 ms a step
  (compile 65 s instead of 133 s: the kernels' compile and autotune are about 60 s of a
  cold stage), plain but the attention 22.5 ms. The kernels stay.
- Candidate, not done: batched-weight variants of the gated-MLP and head kernels (a lane
  axis on the weights, one launch for every lane) and a vmap rule for them, which would
  bring a ratings tick toward the 11.8 ms of one forward (about a third off an
  evaluation's play time). Kernel work; left for a session of its own.
- Tried and dropped: a vmap rule for the attention kernel (the sets folded into its views,
  one launch). On a quiet machine the seat step at 16 x 128 rows was 20.2 / 21.2 ms without
  it and 21.3 / 20.2 ms with it: no difference beyond noise, so the stacked lanes' extra
  cost over one forward is not the attention's per-set launches.

## Integration (integrate/next-runs, 2026-11-11 03:00-04:00 KST)

The author's next fresh BC and RL runs fly the real-scale ships (`real_scale_*` presets).
`integrate/next-runs` = perf/compile-overhaul (42a45e2e) + a squash of origin/perf/gpu-box
(3497f549) without results/gpu-box (81d1b146) + feat/live-tuning-panel (154be767) +
`--preset real_scale_mixed` (876bfee6). perf/compile-overhaul is frozen as the record;
work continues on integrate/next-runs.

- 7 conflicts, all kept both sides (see 81d1b146's message); found and fixed where the
  branches met: the deathmatch target under live tuning (the play shell crashed), the
  fleet presets listed as play maps, reuse across game versions, the fingerprint test.
- Fleet-size presets now exist for every base game, real-scale too (`FLEET_PRESETS`).
- A bug of mine caught by the GPU tests after the merge: the stacked scalars' bounds were
  built from host numbers inside the step (a host copy an uncompiled CUDA-graph capture
  refuses). Fixed with buffers (47113b20 on the overhaul branch, carried into the squash).
- The merged game computes the same numbers on the five base games: VERSION stays 2.

Next (agreed with the author): a short real-scale BC run; re-baseline compile and steady
speed on real_scale_mixed; then measure before acting: the learner's micro-batch size
(160/320/640 rows), the heads' sums' run-time share, the Arena's wasted ticks after first
games, benching against a compile per fleet size; and continue the graph shrink (the delta
code's per-component symlog decodes: about 740 of a step's 3,226 ops).

## Next runs: real-scale BC, graph shrink 2 (2026-11-11 04:00- KST)

- Short BC run on `--preset real_scale_mixed` (budget 1.5M steps, before the shrink below;
  power profile performance): runs end to end. Update 1 rollout 332.7 s, learn 528.0 s
  (compile included); update 2 rollout 17.8 s, learn 232.9 s; bc kl 2.88 then 1.55. The
  first update's overhead, about 610 s, covers both presets of the mix.
- Graph shrink, counted with Dynamo and an op-counting backend on the CPU (no Inductor) on
  one controller step of a fresh Yemong in the arm's settings (`next_state=delta`,
  persistence, ellipse spreads; frontline; the count does not depend on the rows): **3,708
  -> 2,625 ops (-29%)**, the maths unchanged within rounding.
  - The delta code by family (3,708 -> 2,982): its components fall into runs of one layout
    (position along: 175 bins; the fine 51; the coarse 31), each now one (..., k, width)
    tensor, one softmax run of k groups, one encode and one decode with per-component
    knees and spacings as tensors (`codes.symlog_*_at`; the one-component functions are
    wrappers over them and the test's reference). Constants are float32 buffers made at
    init (the median's index too: an index made from a list in the step is a host copy).
  - Position codes: both axes in one circular encode and decode (2,982 -> 2,774; bitwise
    equal to before, eager, float32 and float64).
  - The ellipse inputs: eight two-hot codes in one call (2,774 -> 2,625).
  - The default build (`next_state=absolute`, what `--preset real_scale_mixed` trains
    unless set): 2,940 -> 2,533 ops (-14%; the position codes; it has no delta code).
  - Left: `circular_encode`'s level loop (sequential remainders, kept for float32: see its
    docstring), the scalar codes' moment matching, the rotary embedding (168).
- The vmap fallback's warning for the `yemong::` kernels is silenced in `stacked.py` (it
  printed on every trace; the per-set loop is measured and kept, above).

### Real-scale baseline, cold (after 205c22f6; `compile-overhaul/real-scale/`)

`bench/compile_baseline.sh`, empty caches per case, power profile performance, quiet
machine. BC and RL: the short BC run's TrainConfig (`--preset real_scale_mixed`, default
Yemong: `next_state=absolute`), RL the same with objective ppo. Ratings: the scripted agent
and that run's two policies on `real_scale_frontline`, the default ArenaConfig.

| Case | Build | First collect | First learn | Steady collect | Steady learn | Overhead |
|---|---|---|---|---|---|---|
| BC | 112.0 s | 312.2 s | 425.2 s | 17.7 s | 240.5 s | ~479 s |
| RL | 108.6 s | 307.9 s | 412.4 s | 17.7 s | 228.9 s | ~474 s |

Ratings: build 84.7 s, first batch 499.7 s, steady batch 319.6 s (compile overhead ~180 s).
Overhead: the first collect and learn less the steady ones. The short BC run before the
shrink (caches not guaranteed cold) took 332.7 s and 528.0 s for its first collect and
learn, about 610 s of overhead. Its steady learn was 232.9 s against 240.5 s here: to check
warm A/B (noise, or a cost of the stacked position codes).

A real-scale ratings batch takes 320 s against 167 s on standard frontline: the games are
longer, which makes the batch's tail (`bench/arena_utilisation.py`) worth measuring there.

### The Arena's tail (`bench/arena_utilisation.py`; `compile-overhaul/arena/`)

`real_scale_frontline`, the scripted agent and the short BC run's two policies (two updates
old), the default ArenaConfig (8 blocks x 128 games), three steady batches:

- Every batch ran 18,016 ticks, the game clock (10 min) plus the segment it ended in. Only
  **44% of the row-ticks were in counted games**.
- The tail is not a spread of lengths: blocks 3 and 5 (the two BC policies against each
  other) averaged about 17,900 ticks, nearly every game a time-out; the other six blocks
  averaged about 4,600 ticks (at most 7,936). One slow matchup holds the batch to its
  clock.
- Simulated from these lengths (each block's games i.i.d. from its measured ones), "every
  game started before T, each to its end": T = 1x the median game (4,800 ticks) 1.49x the
  games per batch tick, 2x 2.27x, 4x 2.03x (the batch then runs past one clock). The rule
  is a stopping rule, so wins / games stays consistent (Wald's identity).
- The policies here barely play, so the time-outs overstate it; trained policies will
  differ. The structure is general: the batch waits for its slowest block. Proposal for the
  author (changes how a tournament counts its games, so not done): the started-before-T
  rule at T about 2x the median, or a T from the batch's own game lengths.
- **Author's decision (2026-11-11 10:00 KST): adopt the started-before-T rule.** The
  measure is games per unit of time, not per batch. The simulated "games per batch tick"
  is that measure: every row ticks together, so a batch's time is its ticks times a fixed
  steady tick. T is to be picked on games per second. Each later batch keeps reallocating
  its blocks to the most informative pairs, as the adaptive tournament does now. Not built
  yet.

### Steady speed after the shrink (warm A/B, `compile_baseline.py train ... steps=4`)

The short real-scale BC run's config, one process per tree on the shared default caches,
updates 2-4 steady:

| Tree | Learn (s) | Collect (s) |
|---|---|---|
| 552dc851 (before) | 240.7, 241.8, 241.0 | 17.8, 17.9, 17.9 |
| 205c22f6+ (after) | 239.4, 240.0, 239.8 | 17.8, 17.8, 17.8 |

No cost: 0.6% faster, at noise. The 232.9 s of the short run's second update was that run's
own variation, not the old tree's speed.

### Learner micro-batch and memory (`bnb train --preset real_scale_mixed --probe`, PPO)

| Micro-batch rows | Allocator | Allocated | Reserved | |
|---|---|---|---|---|
| 160 | default | 4.71 GB | 6.16 GB | fits |
| 160 | expandable_segments | 4.69 GB | 5.43 GB | fits |
| 320 | expandable_segments | | | OOM (7.08 GB allocated, 218 MB short) |
| 640 | expandable_segments | | | OOM |

On the 8 GB card 160 rows is the most, even with the better allocator, so the speed of
larger micro-batches is a measurement for the 24 GB card. expandable_segments works with
the graph capture and frees 0.73 GB of reserved memory at 160 rows.

expandable_segments, warm, the real-scale BC config (`steps=3`): steady learn 236.9 and
237.2 s against 239.4-240.0 s without, collect 17.8 s both. It is now `bnb`'s default
(`cli.main`, `PYTORCH_CUDA_ALLOC_CONF`; an explicit setting wins). A warm first update
(collect 50.8 s, learn 246.8 s) is about 42 s over a steady one.

Pre-existing failures seen on the way (not this work's): `tests/contract/test_yemong_export.py
::test_yemong_step_exports_to_onnx[accelerations]` fails on a00c5a47 too (onnxruntime has
no `Where(16)` for a node of the accelerations variant); the installation example fails in
a worktree (`uv run` finds no `.venv` there). Fixed: the bench scripts' usage lines broke
the strict docs build (mkdocs read `[preset=frontline]` as a cross-reference), and
`bnb.cache` (from the play-shell merge) had no reference page.

### A size sweep: an engine per size, or one engine with benching (`bench/fleet_sizes.py`)

Frontline's `_arena` fleet presets (one map radius at every size, so a benched game is the
smaller preset's game; the `_density` ones grow the map and cannot be benched to), the
scripted agent against the short BC run's policy, 8 blocks x 32 games, warm-ish caches, one
process. Tick: a captured block's replay over its ticks.

| Ships per team | Own engine: first segment | Own: tick | Benched (40's engine): tick |
|---|---|---|---|
| 10 | 155.3 s | 4.73 ms | 14.96 ms |
| 20 | 152.9 s | 7.75 ms | 14.99 ms |
| 40 | 163.3 s | 14.97 ms | 15.00 ms |

The benched engine's own first segment took 18.9 s (the 40's graph in the cache), and
switching its sizes nothing. A benched ship still costs its full tick: a benched 10v10 runs
at the 40v40's price, 3.2x its own engine's. An extra size's compile (about 155 s here,
about 20 s warm) is repaid after about 15,000 ticks at 10 ships and 21,000 at 20 (256
games; at a ratings run's 1,024 games the tick costs about 4x as much, so 4x sooner): about
one batch. So: an engine per size for size sweeps; benching for what needs unequal teams
in one batch (the crossover, its purpose). Benching costs nothing at full size (15.00 against
14.97 ms).

### The heads' sums' share (`compile_baseline.py train ... heads=0`)

The real-scale RL config, warm, expandable_segments, steady updates 2-3: learn 229.0 and
228.2 s with the heads' sums (every update's first epoch), 219.0 and 219.5 s without: **9.4 s
an update, 4.1% of the learn, 3.8% of the update** (collect 17.8 s both). Measuring them
every 10th update (as the cadence diagnostics are) would save about 3.4%; it is the
author's call (it thins the heads' series).

**Author's decision: on the cadence.** The trainer now measures the heads' sums on the one
diagnostics cadence (`metrics.due`, every 10th update and the first), with the gradient and
representation groups; `Learner.update(heads=...)`, the trainer passing its cadence. The
heads' losses stay every update's.

### Where the steady learn goes (`bench/learn_profile.py`; `compile-overhaul/learn/`)

The real-scale RL config, warm, two profiled minibatch steps of a steady update (1,024
steps, 222.6 s; three micro-batches of 160 rows a step): **227 ms wall, 220 ms of kernels**,
so the learn is GPU-bound, not launch-bound. By kind: Triton reductions 30%, Triton
pointwise 28%, GEMMs 19%, the model's own kernels (the heads', the gated feedforward's,
the scan's) 16%, attention 4%, copies 3% (the micro-batches' pinned uploads), the fused
Adam 0.3%. No kernel holds more than 5%. The largest (11.7 ms a step) is the shot
trajectory head's cross-entropy gradient: 37.8M float32 elements a micro-batch, 154 code
columns per age query. The loss tails of the auxiliary heads' codes, in float32, are a
large part of the pointwise and reduction time.

The spatial-only ablation and the nearest-neighbour MLP both pass the frontline probe
(4.24 and 3.93 GB allocated).

### Learner speed (`perf/learn-speed`, 2026-11-11 11:15-13:00 KST)

The steady PPO learn of the real-scale RL config (`real_scale_mixed`, the default
`TrainConfig.yemong`: `temporal_first`, `coarsest`, `absolute`), power profile
`performance` throughout, every timing warm. Two benches: `bench/learn_micro.py` captures
one real micro-batch (16 views x 128 ticks x 10 ships, 18 bullet ages: 368,640 shot
queries, a 154-wide code each, 56.8M elements) and times its step as `Learner.gradient`
runs it (the label codes, the loss, the backward) with CUDA events, median of 30
(`compile-overhaul/learn/micro.json`); `compile_baseline.py train ... steps=4 heads=0`
times whole updates (the heads' sums off, as on 9 updates in 10), old tree against new,
one process each on the shared caches (`steady-before.json`, `steady-after.json`).

**The real update: steady learn 221.6 -> 191.0 s (-13.8%)**, collect unchanged:

| Tree | Learn 2, 3, 4 (s) | Collect 2, 3, 4 (s) |
|---|---|---|
| b137fc5b (before) | 219.2, 220.6, 224.9 | 17.9, 17.9, 18.0 |
| 07a05b14 (after) | 190.6, 191.6, 190.9 | 18.0, 18.0, 18.1 |

The new tree's first update was cold for the changed graphs: the rollout's 8-tick capture
took 151.8 s (its Inductor and Triton caches invalidated by the head kernel's new source;
50.5 s warm before) and the first learn 266.3 s. Both are one-time cache fills.

**The micro-batch step: 61.8 -> 51.6 ms (-16.6%)**, kept in this order:

| Change | Step (ms) | Commit |
|---|---|---|
| before | 61.79 | b137fc5b |
| the shot trajectory code made inside the loss (the label carries the 18 x 5 snapshots, not the 18 x 154 code) | 60.67 | 27147fcf |
| `codes.grouped_cross_entropy` as lse(z) sum(t) - t.z (an `autograd.Function`: no code-wide log-softmax kept for the backward) | 60.30 | 27147fcf |
| the learner's compile without Inductor's comprehensive padding | 56.62 | 27147fcf |
| the shot cross-entropy from the code's 46 coordinates (`shots.trajectory_coordinates`; the code, two-hot in every group, made of them in the loss) | 54.38 | 8a5edea7 |
| the staged head kernel: the whole `ShotTrajectoryHead.query` in `_head_forward` / `_head_backward` | 51.56 | 07a05b14 |

- **Padding.** Inductor padded the row stride of the shot logits' gradient (L, R, N,
  ages, 154) to 128 bytes, so the head kernel's and the weight gradient's (rows, 154)
  views of it each needed a copy (two 113 MB passes a micro-batch). The learner's
  `compiler` turns `comprehensive_padding` off for its stages (the loss, the label codes,
  the shot tick, the heads' sums); the rollout keeps its own compile. Alone, on the old
  tree: 61.79 -> 58.65 ms (-5.1%).
- **Coordinates.** The shot labels' trajectory code was a 227 MB float32 tensor written by
  the label stage, read by the loss and kept for its backward. Now the loss makes it from
  46 numbers per query (`trajectory_coordinates`: colour and fraction per level, lower bin
  and fraction per scalar), bit for bit the encoders' code (a test). The compiler still
  keeps one code-wide float32 tensor for the backward, softmax(z) sum(t) - t, which it
  computes in the forward: AOTAutograd's partitioner will not recompute a chain further
  than `max_dist_from_bw` (3) nodes from the backward.
- **Staged head.** `x = gelu(first_norm(latent + age))` (368,640 x 128) was Inductor's,
  and its backward five more passes over it (the gelu and norm gradients, the norm scale's
  column sums, the ages' and rows' sums). The head kernel now makes x in registers from a
  latent row and an age row (the forward and the recomputing backward), and the backward
  forms the input norm's gradient there too, writing the gradient of latent + age (summed
  over ages and rows by Inductor) and x for the first weight's gradient. It applies where
  every row shares the ages (`age_features` (ages, F): the learner's); a per-row query
  takes the plain path. latent + age, its norm and gelu run in float32 and x is rounded
  once, where the plain bf16 path rounds the sum and the norm too; the kernel tests hold it
  to the plain query's error against float64, and to bitwise repeatability.
  `fast-yemong-notes.md`'s kernel table names it.

**Its compile cost.** An `autograd.Function` in the loss graph bypasses AOTAutograd's
cache ("Unsupported call_function target autograd_function_apply"), so a process
re-traces and re-partitions the loss on warm caches (Inductor's own graph cache still
hits): `Learner._loss` 3.5 -> 12.2 s warm in `compile_baseline.py`, the micro-batch's first
step 8.2 -> 12.4 s. The plain form of the same mathematics (lse(z) sum(t) - t.z by
autograd, the target made of the coordinates in the graph) caches but is slower: 53.29
against 51.68 ms a step, measured back to back (about 6 s an update). The `Function` stays:
it repays its 9 s within the first update.

Loss values: identical on the captured micro-batch at every kept step (the loss and every
term within 1e-7 relative); the parameters' gradient norms within 1.2e-4 relative of the
old tree's (the bf16 trunk; the padding switch alone moves them 8e-5).

Tried and dropped:

| Attempt | Step (ms) | Why dropped |
|---|---|---|
| the target encoded from the snapshots inside the `Function` (forward and backward) | 57.89 (against 57.08) | the partitioner still moved softmax sum(t) - t into the forward and kept it; the remainder chain ran twice |
| bf16 logits into the `Function` (the upcast inside) | 57.80 | the same choice by the partitioner |
| g target built in the backward (`encode(source, g)`, so nothing code-wide is free of the gradient) | 56.15 (against 54.38) | the partitioner did recompute, but building the two-hot code is ALU-bound: slower than reading the kept float32 |
| the shot logits flattened to (rows, 154) for the loss | 51.46 (against 51.56) | noise: the gradient is still written in two layouts (one for the head kernel, one for the weight GEMM) |
| more head-kernel autotune configurations (a one-off sweep of the old kernel at the shot shape: BLOCK_M 32-128, BLOCK_K 32-128, BLOCK_O 32-256, 4 or 8 warps, 1-3 stages; `head-configs.json`) | best forward 0.96 against the current configurations' best 1.16 ms, backward 1.67 against 1.71 | under 0.5% of a step, for more autotuning |

Where the step goes now (`ages=1`, a diagnostic that queries the shot head at one age):
36.3 ms of the 51.6 are the rest of the model and its heads with one age, 15.3 ms the
shot head's other 17 ages (17.8 ms before the staged kernel). The rest is spread thin: no kernel over 2.7 ms (the gated MLP's backward), GEMMs
9.0 ms, the heads' kernel 3.5, attention 2.8, RMSNorm backwards about 2.2.

Proposals (not done):

1. **The grouped cross-entropy's kept factor in bf16.** Compiled, the backward's
   softmax(z) sum(t) - t is computed in the forward and kept in float32 (for the shot code
   227 MB written and read a micro-batch); the gradient it makes is rounded to bf16 for the
   head kernel anyway. Computing that factor in the forward and keeping it in bf16 (every
   grouped cross-entropy: next state, correction, shots), measured on the captured
   micro-batch: step 51.07 -> 50.17 ms (-1.8%); the loss and every term unchanged (the
   forward stays float32); the parameters' gradient norms within 5.5e-4 relative (the
   next-state head's last layer and the shot head's latent move most), against 1.2e-4 for
   everything kept here. A precision decision for the author; not landed.
2. **The coordinate remainders.** `circular_coordinates`' level-by-level remainders are
   recomputed by every level's element of the stacked output (36 remainders where 8 would
   do; 1.2 ms a micro-batch). Direct remainders per level would be the same mathematics but
   not the same float32 bits as the chain (the chain is there for exactly that), so it is
   left.
3. **Queries that no loss reads.** The shot head runs at every (tick, row, age); the loss
   reads only the labelled shots' valid ages: 64.5% of the captured micro-batch's queries
   (64.8% of its rows are labelled: dead ships and shots whose flight leaves the window
   are not). Querying only those would cut about a third of the head's 16 ms, but their
   count varies by micro-batch: a fixed capacity either drops queries when it overflows
   (a different loss) or sizes for the worst case (no saving), and a varying one
   recompiles. Left as a design question.
