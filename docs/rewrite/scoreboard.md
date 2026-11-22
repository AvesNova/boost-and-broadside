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
  policy), then no more. `dynamic=True` recompiled too. Cause (TORCH_LOGS=recompiles): "duck
  sizing" tied the ship-slot count to the field count because both were 10 at 5v5
  (`state.constants['field_x'].size()[2] == state.bullet_x.size()[1]`; for the policy
  `obs.field.size()[1] == obs.ship.size()[1]`); at 10v10 the guess fails.
- **With duck sizing off** (`torch.fx.experimental._config.use_duck_shape = False`) and
  `dynamic=True`: one compile (game 30.4 s, policy 18.4 s at E = 256), then 10v10, 20v20 and
  50v50 cost nothing (first calls 0.00 to 0.02 s). Steady at E = 256: game 0.49, 0.67, 1.32,
  5.14 ms; policy 1.60, 2.62, 5.72, 16.91 ms. This meets "a new team size adds at most 1 min"
  with nothing added; proposed as the compile setting for training and evaluation (D5).
- Compile time, cold: game 18 s, policy 12 s.

## Play latency on the CPU (gate G2, 2026-11-23 03:43)

One play tick at batch size 1: two observations, two policy steps (one per team), one game
step; core config, the new model, one thread, CPU of the laptop (GPU unused). 1,000 ticks after
50 warm-up ticks (300 for the warm-cache row). Committed code at 25478e10. **Provisional: the
machine was loaded (load average 1.7 to 4.5) by two agents' test runs.** Script:
docs/rewrite/sections/scoreboard/play_latency.py.

| | first tick | p50 ms | p99 ms | worst ms | against p99 16 ms, worst 33 ms |
|---|---|---|---|---|---|
| uncompiled, 5v5 | | 6.98 | 7.78 | 10.50 | meets |
| uncompiled, 20v20 | | 9.66 | 10.84 | 17.07 | meets |
| uncompiled, 50v50 | | 17.35 | 19.28 | 23.41 | p99 over |
| compiled (dynamic), 5v5, cold cache | 31.5 s | 3.30 | 3.99 | 4.23 | meets |
| compiled (dynamic), 50v50, cold cache | 34.1 s | 11.36 | 12.58 | 17.61 | meets |
| compiled, 20v20, cold cache | 33.0 s (38.4 s from launch) | 5.58 | 6.10 | 6.24 | meets |
| compiled, 20v20, warm cache | 12.4 s (16.8 s from launch) | 5.12 | 5.98 | 7.10 | meets |

- Compiled, every size to 50v50 meets the frame budget on the CPU; uncompiled meets it to
  20v20.
- Launch: a compile takes 31 to 34 s cold and still 12 s warm, against "launch to a playable
  match at most 10 s". So play starts uncompiled and switches to the compiled step when a
  background child process has built it (v2's play shell did this with AOT builds), or the
  compiled artefact ships prebuilt.

## Not yet measured

The compiled backward pass and the learner's update; traced operations across every graph.
