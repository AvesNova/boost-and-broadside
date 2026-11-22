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

### After agreed fix 11 (the medium cached, 2026-11-23 03:10)

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

## GPU: uncompiled against compiled, compile time, new team sizes (2026-11-23 03:36)

Laptop RTX 4070 (8 GB), committed code at 25478e10 (a clean worktree), E = 2048, one fresh
process per row with an empty inductor cache; policy under bf16 autocast. Script:
docs/rewrite/sections/scoreboard/gpu_compile.py. "First call" includes the compile; the sizes
run in order 5v5, 10v10, 20v20, 50v50 in one process.

| | 5v5 | 10v10 | 20v20 | 50v50 |
|---|---|---|---|---|
| Game step (core config), uncompiled: steady ms | 9.38 | 22.04 | 62.29 | out of memory |
| Game step, compiled (default): first call s / steady ms | 18.38 / 0.95 | 15.90 / 3.85 | 0.01 / 9.22 | 0.04 / 35.75 |
| Game step, compiled dynamic=True: first call s / steady ms | 27.49 / 1.25 | 27.97 / 3.82 | 0.01 / 9.12 | 0.03 / 33.51 |
| Policy step, uncompiled: steady ms | 28.93 | 62.58 | 129.05 | 317.17 |
| Policy step, compiled (default): first call s / steady ms | 12.04 / 14.36 | 14.97 / 31.24 | 0.06 / 63.90 | 0.16 / 159.20 |
| Policy step, compiled dynamic=True: first call s / steady ms | 18.59 / 16.07 | 17.70 / 31.39 | 0.06 / 64.20 | 0.16 / 159.44 |

- Compiling makes the game step about 10x faster at 5v5 and the policy step about 2x.
- The policy dominates a rollout tick: two policy steps (one per team) are about 29 ms
  against 0.95 ms for the game step at 5v5.
- The uncompiled game step runs out of 8 GB at 50v50, E = 2048 (the dense (E, N K, N) hit
  test's temporaries); compiled it fits.
- A new team size: the first new size recompiles once (about 16 s for the game, 15 s for the
  policy), then no more. `dynamic=True` recompiles at the second size as well, so something in
  both graphs specialises on the ship count: to find (it decides what "a new size costs").
- Compile time, cold: game 18 s, policy 12 s.

## Not yet measured

The compiled backward pass and the learner's update; traced operations across every graph.
