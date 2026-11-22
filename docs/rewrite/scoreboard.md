# Scoreboard (core-brief.md section 3)

First numbers, 2026-11-23, laptop CPU, one thread, uncompiled, platform profile
"performance". Operation counts: compute operations dispatched per call (the rule of
docs/rewrite/sections/simplification/regions.py: every aten op, views and metadata apart);
they do not depend on the batch size. Times: E = 256 environments, 5v5, median of 30 calls.
Script: docs/rewrite/sections/scoreboard/game_ops.py.

## Game and scripted opponent

| call | compute ops | view ops | CPU ms per call (E = 256, 5v5, one thread) |
|---|---|---|---|
| step, v2 config (no fields) | 1339 | 116 | 11.16 |
| observe (one team), v2 config (no fields) | 81 | 7 | 0.31 |
| reset_done, v2 config (no fields) | 266 | 77 | 1.21 |
| step, v2 config, 10 fixed fields | 1675 | 124 | 22.25 |
| observe (one team), v2 config, 10 fixed fields | 165 | 9 | 0.75 |
| reset_done, v2 config, 10 fixed fields | 350 | 79 | 1.71 |
| step, core config (core flight and damage, ghosts, 10 drawn fields) | 1692 | 124 | 23.38 |
| observe (one team), core config (core flight and damage, ghosts, 10 drawn fields) | 165 | 9 | 0.77 |
| reset_done, core config (core flight and damage, ghosts, 10 drawn fields) | 401 | 100 | 1.94 |
| scripted act (one team), core config state | 172 | 31 | 0.65 |

### After agreed fix 11 (the medium cached, 2026-11-23 07:05)

| call | compute ops | view ops | CPU ms per call |
|---|---|---|---|
| step, v2 config (no fields) | 1342 | 124 | 10.54 |
| step, v2 config, 10 fixed fields | 1510 | 128 | 15.08 |
| step, core config | 1527 | 128 | 13.67 |
| observe (one team), with fields | 80 | 8 | 0.33 |
| reset_done, core config | 405 | 102 | 1.97 |

Fields now cost about 170 operations instead of 340, and the core config's step takes
13.7 ms instead of 25.1 at E = 256.

## Against the tripwires

| Tripwire | Value | Now | |
|---|---|---|---|
| Operations per game tick, uncompiled | 1,300 | 1,342 (v2 config, no fields); 1,527 (core config, 10 fields; 1,692 before fix 11) | over |
| Operations per scripted call | 170 | 172 | at |
| Operations per policy forward | 650 | 509 compute (+294 view) per `step`, new model | under |
| Traced operations across every compiled graph | 10,000 | pending (GPU) | |

## Reading

- Fields cost about 340 operations per step (1,339 to 1,675): the medium is evaluated four
  times per tick (ships twice, bullets twice). Agreed fix 11 caches the end-of-tick values
  and halves that.
- `reset_done` is 266 to 401 operations, about a fifth to a quarter of a step, every tick
  (agreed fixes 1, 2 and 9 target it).
- The core config costs 17 operations more than the v2 config with fields (core flight's
  branches, ghost bullets and outcomes); the time difference (25.1 against 21.3 ms) includes
  ten drawn fields per environment against ten shared ones.

## Policy (after the model port, 2026-11-23)

1,997,681 parameters. Per `step` forward: 509 compute and 294 view operations; `sequence`
at T = 8: 552 and 328. CPU, E = 256, N = 10, F = 10: 144 ms on one thread. GPU (bf16, one
short run): 3.10 ms eager, 1.38 ms compiled, first compile 10.9 s. Training at the default
size: 2,800 steps/s with 8 minibatches (4.8 GB), 3,150 with 16 (2.5 GB); 4 minibatches run
out of memory.

## Not yet measured

Policy forward operations and time (after the model port), compiled against uncompiled step
time, compile time, and the cost of a new team size: these need the GPU and the new model.
