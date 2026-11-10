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
| 2 | Ratings lanes: 16 serial Yemong forwards per tick in one graph (`MatchSeat._network`) | region written (`runtime/regions.py`), to measure |
| 3 | `default_lanes` gives one network group all 16 slots; at most 8 can play | done (capped at `blocks x min(players, 2)`) |
| 4 | `bnb ratings` builds a new Arena and compiles everything per preset (3x) | |
| 5 | Training act stage: 11 forwards, about 5 duplicates (LIVE = learner, league_k = slot k) | region applied to both loops, to measure |
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
