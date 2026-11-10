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
| 2 | Ratings lanes: 16 serial Yemong forwards per tick in one graph (`MatchSeat._network`) | |
| 3 | `default_lanes` gives one network group all 16 slots; at most 8 can play | |
| 4 | `bnb ratings` builds a new Arena and compiles everything per preset (3x) | |
| 5 | Training act stage: 11 forwards, about 5 duplicates (LIVE = learner, league_k = slot k) | |
| 6 | Learner loss compiled in several variants (heads flag, micro-batch remainders, cadence, a second learner step for bootstrap) | |
| 7 | BC teacher labels computed inside the loss, every epoch | |
| 8 | Several scripted passes per act stage; the frontline layer on every row | |
| 9 | Triton autotune keyed on exact shapes | |
| 10 | Arena counts only first games but plays until the slowest ends | |
| 11 | Arena runs the training bookkeeping (records, store, analytics) and syncs 3x per segment | |
| 12 | Non-finite step guard clones every parameter and Adam state per step | |
| 13 | Diagnostics over their 2% cap; cadence OOM at 4096 environments on 8 GB | |
| 14 | Yemong code helpers: ~2,100 of 3,663 ops from Python loops | |
| 15 | `sight.sighted` unrolls one copy per field | |
| 16 | Dynamo's recompile limit (8 per code object): a process building more than 8 Rollouts may fall back to eager | |

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
