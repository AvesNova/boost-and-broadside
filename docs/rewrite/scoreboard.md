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
| Traced operations across every compiled graph | 10,000 | 5,793 (4 graphs, step K; "Compile time" below) | under |
| Cold start (compile_check.py: process start to the end of its RL update, quiet laptop) | 260 s (budget, step K) | 221.3 s | under |

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

## Training on the GPU (2026-11-23 05:25 to 05:55)

Laptop RTX 4070 (8 GB), platform profile "performance", load average 1.5 to 2.1 (no other GPU
job). Code: branch core rebased on 681eb492, commits 8c3deab4 and 94706d2d (the second only
changes the eager snapshot path; the fp32 and eager rows ran on it). Script:
docs/rewrite/sections/scoreboard/train_gpu.py, one fresh process per row, 3 updates each:
the first update includes the compile, "steady" is the mean of updates 2 and 3. The loop is
lean/train.py's: collect (128 ticks), update (4 epochs, advantages and value targets
recomputed from the critic each epoch), league addition. The league holds all 8 snapshots
from the start (copies of the initial weights), as in a run's steady state; half the
environments self-play, so the update trains 1.5 seats per environment. Steps are
environment ticks (E x 128 per update). Policy forward passes under bf16 autocast unless
marked fp32. Peak memory: torch's peak allocated / reserved over the run.

| E, game | | collect steps/s | update steps/s | loop steps/s | peak GiB | first update (compile) |
|---|---|---|---|---|---|---|
| 256, v2 | eager | 11,377 | 2,590 | 2,110 | 5.16 / 5.62 | 3.1 + 12.7 s |
| 256, v2 | compiled | 25,974 | 6,439 | **5,160** | 3.67 / 4.35 | 103.7 + 93.9 s cold; 42.3 + 27.9 s warm |
| 256, core | eager | 8,251 | 2,226 | 1,753 | 5.56 / 6.10 | 4.0 + 14.8 s |
| 256, core | compiled | 22,038 | 5,877 | 4,640 | 3.97 / 4.76 | 132.9 + 100.3 s (cold for the core game's graphs) |
| 1024, v2 (32 minibatches) | eager | 12,106 | 2,594 | 2,136 | 5.54 / 6.20 | 11.0 + 50.6 s |
| 1024, v2 (32 minibatches) | compiled | 23,243 | 6,464 | 5,057 | 4.05 / 4.67 | warm |
| 1024, core (32 minibatches) | compiled | 20,662 | 5,899 | 4,589 | 4.37 / 5.27 | warm |
| 256, v2, fp32 | eager (16 minibatches) | 9,296 | 2,357 | 1,880 | 3.76 / 4.05 | |
| 256, v2, fp32 | compiled | 17,029 | 3,630 | 2,992 | 5.80 / 6.44 | |

Variants, compiled, 256, v2:

| | collect | update | loop |
|---|---|---|---|
| default (above) | 25,974 | 6,439 | 5,160 |
| no snapshots in the league | 87,345 | 6,447 | 6,003 |
| targets once per update from the stored values (v2's rule; `targets_every_epoch=False`) | 25,655 | 8,628 | 6,457 |

- **Start-up.** Process start to the end of the first update: 199 s with a cold inductor
  cache, 72 s warm (dynamo still traces every graph; the cache saves the inductor builds).
  Eager starts in 17 s. The compile saves 9.2 s an update, so it is repaid after about 20
  updates cold, 6 warm; not within the first update.
- **Graph breaks and recompiles:** none in any row (TORCH_LOGS=graph_breaks,recompiles),
  across team sizes, environment counts and minibatch sizes.
- **Memory.** 8 minibatches (48 seats of 128 ticks) fit 8 GB at E = 256 with about 3 GB to
  spare compiled, 2 GB eager; 4 minibatches run out of memory (bf16, compiled; 6.5 GB
  allocated at the failure). At E = 1024, 32 minibatches keep the same minibatch size.
- **The update is now the cost:** at 256, 5.1 s of the 6.3 s loop. Of that (timed with
  synchronised wrappers before the rebase on 681eb492, same update code): the critic's
  target pass (one forward over the rollout per epoch) about 1.3 s, the minibatch forwards
  1.4 s, the backward passes 2.3 s; Adam, clipping and GAE together under 0.1 s.
- **Snapshots** cost 3.7 ms a tick at 256 (collect 26k against 87k steps/s without them):
  every snapshot slot runs on every league environment in one vmapped call (8 x 128 rows).
  The earlier form (each member present on just its environments, grouped with a per-tick
  read-back of the member counts) ran at 20,546 steps/s; the read-back itself cost 5% of
  collect (21,577 with the counts read once per rollout), the eight small model calls the rest.
- **bf16** nearly doubles the update (3,630 to 6,439 steps/s compiled) and lowers its memory
  (5.8 to 3.7 GiB).

**Against v2's floor of about 5,000 steady environment steps per second on the laptop:** met
with the v2 game (the training default) at 256 environments, 5,160, and at 1,024, 5,057; by
3% and 1%. Not met with the core game (fields and fog): 4,640 and 4,589. Computing the
targets once per update from the stored values, as v2 does, would give 6,457 (v2 game); that
is the author's decision (lean/learn/NOTES.md).

## Update at parity with v2: streaming and micro-batches (2026-11-23 10:32 to 11:14)

Laptop RTX 4070 (8 GB), platform profile "performance", no other GPU job (nvidia-smi showed
no compute process before the first row), each row inside `systemd-run --user --scope -p
MemoryMax=8G`. Script: docs/rewrite/sections/scoreboard/train_gpu.py (one fresh process per
row; the loop is lean/train.py's; the league full with 8 snapshots from the start; compiled,
bf16). The update is v2's size: 2,048 environments x 4 segments x 128 ticks = 1,048,576
environment ticks (12,288 views with half self-play), 32 minibatches of 384 views, 4 epochs,
micro-batches of 64 views; the rollout streamed to pinned host memory (lean.learn.store).
"Steady" is the mean of the updates after the first (2 updates in the 3-update rows, 1
otherwise); the first update includes the compile (inductor cache not controlled). Copy
waits: the host's seconds waiting for copies (the store's `ready` and the staging
buffers), the host's seconds gathering micro-batches (overlapped with the GPU), and the
compute stream's seconds waiting for a micro-batch copy (CUDA events), each per update.

| Row | collect steps/s | update steps/s | loop steps/s | compute stall / host wait / host gather per update | peak GiB allocated / reserved | first update collect + update (start to end) |
|---|---|---|---|---|---|---|
| v2 game, targets once (default) | 22,032 | 8,482 | **6,124** | 0.002 / 0.004 / 2.0 s (update 124 s) | 5.18 / 5.78 | 124.9 + 183.8 s (310 s) |
| core game, targets once | 19,736 | 7,791 | **5,586** | 0.003 / 0.004 / 2.1 s | 5.58 / 7.27 | 177.8 + 201.7 s (383 s) |
| v2 game, targets every epoch | 21,999 | 6,239 | 4,860 | 0.004 / 0.007 / 3.0 s | 5.18 / 5.81 | 87.9 + 191.2 s (281 s) |
| v2 game, everything on the device | out of memory (7.0 GiB allocated) | | | | | |

Streaming against everything on the device, at a size where both fit (1,024 environments x
2 segments, 32 minibatches of 96 views, so each minibatch is one micro-batch of 64 and one
padded to 64; 3 updates each):

| | collect | update | loop | compute stall / host wait / host gather | peak GiB |
|---|---|---|---|---|---|
| streamed | 23,356 | 6,356 | 4,997 | 0.003 / 0.001 / 0.51 s | 4.84 / 5.42 |
| on the device | 23,290 | 6,365 | 4,999 | 0 | 5.33 / 5.92 |

The difference is within run-to-run noise: the copies are hidden. (This update is slower per
step than the parity one because a quarter of its micro-batch work is padding.)

Bytes per environment tick of the stored rollout (1.5 seats per environment, 4v4, 8 slots;
the initial carries, 432 bytes, included in every column; CPU, from the arrays' sizes):

| Game | full precision (the old stacked rollout) | compact, fields every tick | compact, fields once per episode (the store) |
|---|---|---|---|
| v2 (no fields) | 3,249 | 2,100 | 2,094 |
| core (10 fields) | 3,564 | 2,325 | 2,097 |

At parity the store holds 2.2 GB of pinned host memory (ship rows 0.73 GB, carries 0.45 GB,
values and rewards 0.25 GB each, next-state targets 0.18 GB).

- **Against the earlier rows** ("Training on the GPU", 256 environments x 1 segment, 8
  minibatches, older code): targets once per update gave collect 25,655, update 8,628, loop
  6,457 there; at parity this code gives 22,032, 8,482 and 6,124. Not a like-for-like
  comparison (different sizes and code); the collect difference was not investigated.
- **Memory** follows the micro-batch, not the minibatch: 5.2 GiB allocated at parity. The
  core game reserves 7.27 GiB at peak (fragmentation; allocated 5.58), the closest to the
  8 GB limit; expandable segments were not tried.
- **Compact storage against full precision** (lean/learn/tests/test_store.py, CPU float32,
  initial weights): the largest change of a sampled action's log-probability 1.9e-6, mean
  KL 4e-9 to 8e-9; PPO's clip is 0.2.
- **v2's floor (about 5,000 steady environment steps per second on the laptop):** met at
  parity with both games (6,124 and 5,586).

## Compile time (step K, 2026-11-24 01:42 to 02:30)

Laptop RTX 4070, profile "performance", load average 0.7 to 3.2 (nothing else running; the
first attempt, 2026-11-23, ran beside another agent's CPU jobs and its times are not used).
One fresh process per row, an empty inductor cache each (TORCHINDUCTOR_CACHE_DIR), each
commit's own docs/rewrite/sections/scoreboard/train_gpu.py at one size: 256 environments, 1
segment of 128 ticks, 2 updates, 8 minibatches, micro-batches of 64, 8 snapshots, the core
game at every point (the v2 game went at G1; D5 and streaming ran their `--game core`).
torch.compile was wrapped to count each Dynamo graph's traced operations (FX call nodes;
scratch harness, the same counting as compile_check.py). "Cold" is process start to the end
of the first update; at this size the update's own work is about 10 s, the rest is compile.
Steady: the second update, steps per second.

| Point | Graphs | Traced operations: act / league / advance / loss = total | Cold | Steady collect / update / loop |
|---|---|---|---|---|
| D5 8d1d6603 | 4 | 1,630 / 812 / 2,557 / 983 = 5,982 | 215.3 s | 21,164 / 7,951 / 5,780 |
| streaming 8231e8a0 | 4 | 1,346 / 812 / 2,139 / 963 = 5,260 | 203.4 s | 23,518 / 5,841 / 4,679 |
| G1 31358e2a | 4 | 1,317 / 808 / 2,153 / 961 = 5,239 | 201.9 s | 22,893 / 5,629 / 4,518 |
| G2 585dca98 | 4 | 1,340 / 816 / 2,158 / 967 = 5,281 | 204.0 s | 22,958 / 5,630 / 4,521 |
| M 49a1f3ee | 4 | 1,411 / 841 / 2,151 / 1,051 = 5,454 | 215.9 s | 22,089 / 5,486 / 4,395 |
| L1 ab205a41 | 4 | 1,414 / 836 / 2,129 / 1,080 = 5,459 | 215.3 s | 22,681 / 5,485 / 4,417 |
| L2, vmap routing b69242f6 | 4 | 1,414 / 881 / 2,215 / 1,080 = 5,590 | 219.8 s | 50,094 / 5,684 / 5,105 |
| L2 as merged (per-slot routing) 484c5d10 | **5** | 1,414 / 746 x 2 / 2,215 / 1,080 = 6,201 | 254.4 s | 13,756 / 5,448 / 3,902 |
| E2, core 245e537f | **5** | 1,746 / 746 x 2 / 2,220 / 1,081 = 6,539 | 258.3 s | 11,451 / 5,463 / 3,698 |
| step K fixes on 245e537f | 4 | 1,746 / 746 / 2,220 / 1,081 = 5,793 | 237.2 s | 12,412 / 5,461 / 3,792 |
| the same, warm cache (second run, same cache) | 4 | | 93.2 s | 13,312 / 5,460 / 3,872 |

The league column is the snapshot move (snapshot_actions, network_actions, then slot_actions).
At D5 the targets ran with `--targets-once` (that commit's default was every epoch). D5's
act and advance are larger (waste fixes 6, 7 and 9 came after it).

- **The doubling was not one.** 199 s (D5) was measured at 256 environments, 403 s (M) at
  parity, 2,048 x 4 x 128, where the first update also trains for about 170 s (the steady
  update takes 166 s). At one size the cold start moved 215 -> 203 -> 216 s to L1, 254 s at
  L2 and 258 s at E2.
- **L2: one recompile, +35 s.** The per-slot routing compiled `slot_actions` twice (and
  `advance` a second time at the first league network, BC to RL; not in this measurement,
  which starts with the league full). Guard (TORCH_LOGS=recompiles_verbose): "'NoneType'
  object has no attribute 'stride'". The scripted rows of the league side were a slice (a
  view) of the scripted actions; Dynamo guards a view's base, and the next slot's input, a
  graph's output, has none. Fixed by copying the rows (`rollout.league_rows`); a slow test
  counts 2 graphs where there were 4. Cold 258.3 -> 237.2 s. Within L2, the vmap form (one
  graph, 219.8 s) had no recompile; the merged per-slot form had it from the start.
- **E2 (scripted BFM): +332 traced operations in act** (1,414 -> 1,746, +6% of the total),
  +3.9 s cold (254.4 -> 258.3 s, within run-to-run noise). Not a regression at this budget.
- **What is left above L1** (237.2 against 215.3 s): E2's operations, the per-slot routing's
  first run (ten slots a tick) and the routing graph's own compile.
- Steady speed at this size is lower from L2 on because the per-slot routing is launch-bound
  at 256 environments (collect 22,681 at L1, 12,412 now); at parity it is not (below).

Where the operations are (CPU trace of the same graphs, step K, attributed by call stack):
act = the policy step 647 + the scripted call 394 (E2 since) + two observations 312;
advance = the game step 1,741 of 2,245, re-deal 180, rewards 175; the loss = the policy's
sequence 735. The four spatial sublayers trace 292 operations in each policy graph, the two
temporal ones 215 in a step and 304 in the sequence. Inductor builds take 110 to 145 s of a cold start, Dynamo's tracing 35 to 50 s, the
autotuner's first runs about 30 s.

Levers tried or weighed (same size, same machine):

| Lever | Cold | Steady loop | Verdict |
|---|---|---|---|
| Fix the recompile (done) | 258.3 -> 237.2 s | 3,698 -> 3,792 | kept |
| Warm cache (the default cache directory survives between runs until /tmp is wiped) | 93.2 s | 3,872 | the default already; tracing (44 s) is what a warm start still pays |
| Regional compilation (torch.compiler.nested_compile_region on the spatial and temporal sublayers; and on the policy's forward) | fails to trace: "Freevar has no source: item" (PyTorch 2.15, dynamic=True) | | not available here; the most it could save is three of four spatial and one of two temporal sublayers per policy graph, about 330 to 370 operations in each of the three, about 1,000 of 5,793 |
| One CUDA graph for every league slot's call | not timed cold | parity 6,320 against 6,301; collect at 256 10,700 -> 18,325 | reverted (no gain at parity, +0.28 GiB reserved) |
| act and slot_actions both trace the policy step | the routing graph costs 17.5 s of inductor build plus its tracing | | merging them means one batched call for the live policy and the league networks (stacked weights): not cheap, not tried |
| advance partly eager | | | no: the game step is 78% of it and is about ten times slower uncompiled (scoreboard "GPU"); the rest (re-deal, rewards, about 500 operations) eager would add hundreds of launches a tick |

Parity, one short run each (2 updates, 2,048 x 4 x 128; cold times not used: another agent's
CPU jobs were running): steady collect / update / loop 46,648 / 7,286 / 6,302 before the fix,
46,328 / 7,293 / 6,301 after.

**Tripwire.** docs/rewrite/sections/scoreboard/compile_check.py runs the two stages in
miniature (64 environments, one 16-tick segment, a BC update with no league network, two
snapshots, a segment with the routing, an RL update) from an empty cache, so it also sees
the BC to RL recompile. Against compile_baseline.json it fails on any graph break, a new
graph or recompile, a function's traced operations more than 10% over the baseline, or a
cold time over the budget; it warns when the cold time grows by more than 25%. Time is
compared only on the baseline's GPU and power profile, and should be taken on a quiet
machine. Baseline (2026-11-24 01:57, quiet): 4 graphs, act 1,746, advance 2,220,
minibatch_loss 1,009 (the miniature's BC and RL loss), slot_actions 746 = 5,721 traced
operations; cold 221.3 s. **Budget: 260 s** (proposed: 18% over today, about where L2's
recompile put the full measurement). Run it at the end of every step:

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py

## Not yet measured

**After waste fixes 6, 7 and 9 (2026-11-23, one run, compiled, 256 environments, v2 game,
performance profile):** collect 28,620 (was 25,974), update 6,440 (6,439), loop 5,257 (5,160);
peak 3.68 / 4.35 GiB. Operations per tick (eager CPU, 5v5): v2 game 2,618 -> 2,345, core game
3,273 -> 2,791. The core game was not re-run on the GPU.

## Evaluation games (2026-11-24 02:51-02:58, rough, one run each)

Laptop RTX 4070, performance profile, 4v4 core game, standard spawn, compiled (dynamic), no
rollout storage. Scripted vs scripted: game length mean 1,326 ticks (std 744); network
(untrained) vs scripted: 778 (std 299); both biased short (censored at the run's end).

| E | scripted vs scripted ticks/s, games/s, memory | network vs scripted ticks/s, games/s, memory |
|---|---|---|
| 1,024 | 430k, ~324, 0.08 GB | 122k, ~157, 0.81 GB |
| 4,096 | 881k, ~665, 0.16 GB | 125k, ~161, 1.34 GB |
| 16,384 | 993k, ~749, 0.60 GB | 116k, ~149, 3.58 GB |
| 32,768 | 990k, ~747, 1.24 GB | out of memory |
| 65,536 | 984k, ~742, 2.50 GB | not run |

First call 72 s cold (scripted tick), 54 s more for the network tick; warm 33 s and 31 s; new E
values 0.01-0.15 s. Scripted parameters as persona tensors: no recompile on new values. One
yardstick float equal to 1.0 is baked into the graph (rollout.py:221 resume) and recompiles
when changed. The benchmark's own wrapper had graph breaks (28 graphs; core.built_once's get
hit the recompile limit and ran eager), so the scripted numbers may be low.
