# Seed isolation: progress

Branch `feat/seed-isolation` (from `origin/integrate/next-runs` df8294f3). Goal: arms of a
pairwise experiment share their random streams wherever they are the same (common random
numbers), so the arm-to-arm difference within one seed is tightly coupled. No GPU work
until rl-v20-seed1 ends (~01:10); CPU tests and micro-benchmarks only.

## Baseline (re-run of the audit scripts, 18:30)

- Init: `yemong.gate` on shifts 61 of 134 shared parameters; `blocks+1` shifts 24;
  `geometry_head=False` shifts none (it is built after every other draw).
- Maps: any model option re-rolls every map and action (0/18 episodes alike); `arena_scale`
  off re-rolls every map; perturbed weights keep the maps but no action; same config
  reproduces exactly.

## Milestones

- 18:45 `bnb.streams`: one keyed-hash utility (`keyed_hash`, `keyed_uniform`,
  `keyed_normal`, `stream_key`, `stream_generator`, `inverse_cdf`); the league swap and the
  learner's generators moved onto it. Next: keyed model initialisation. ETA 19:30.
- 18:50 Keyed initialisation (`bnb.models.initialise`): every shared parameter bit-identical
  across gate / geometry head / blocks+1 / error head (audit script: 0 shifted of 134).
  Found: `stream_generator`'s old `seed << 32 | crc` seeding is ignored above 32 bits by the
  CPU generator, so the learner's minibatch order never depended on the seed. Fixed (32-bit
  stream key).
- 19:00 Rollout game draws keyed by (environment, episode[, clock]); arena slot always
  drawn; seats by episode; stagger by environment; `restart(batch)`; arena/tournament/CLI
  seeds. Audit re-run: every model option keeps all 17 episodes' maps.
- 19:15 Decision keys (`Controller.step(..., decision_key)`): inverse-CDF joint sampler
  (power, stick | power, shoot | both), factored heads, scripted agent (actions, switches,
  sense, flags), uniform floor, view offsets; BC's teacher re-reads the stored keys
  (`Trajectory.decision_key`). Audit: `gate` on/off now 91/96 ticks with every action
  equal (was 0/96), geometry head 96/96. Default-generator save/restore removed (dead).
  Next: tests pass on the touched files, goldens re-recorded, CPU micro-benchmark, docs.
  ETA 20:30, worst 21:15.
- 19:55 Done on the CPU. Quick tier on every touched test file passes (group runs: 170,
  350 and 707 tests, the last 8 failures fixed and re-run). Goldens re-recorded. New
  tests: tests/sections/test_streams.py, tests/loop/test_keyed_draws.py, the init case in
  test_yemong_model.py. ONNX: the hash exports (dynamo) and onnxruntime matches exactly
  (scratch check; the gpu-marked export test itself not run).

## CPU cost (laptop, performance profile, training run on the GPU alongside: noisy)

| Path | old | new |
| --- | --- | --- |
| tick draws, B 2048, U 140, compiled | 1.31 ms | 1.78 ms |
| tick draws, eager | 2.26 ms | 11.66 ms |
| joint sampler, R 4096 x 5 ships, compiled | 3.46 ms | 1.15 ms |
| joint sampler, eager | 6.21 ms | 3.71 ms |
| rollout tick, eager, B 512 deathmatch, chaser v random (two runs each) | 27.6, 30.6 ms | 29.4, 35.0 ms |

The old tree's CPU-compiled rollout fails in Inductor (KeyError buf55), so there is no
compiled rollout pair on the CPU; the new one runs at 5.98 ms/tick. Compiled, the keyed
draws plus the new sampler cost less than the old pair.

## Pending (after rl-v20-seed1, ~01:10)

- GPU steps per second, old vs new, on the same config (the real check: graph replay).
- The gpu-marked tests on the touched files (graph replays draw fresh, the ONNX export,
  the CUDA league/arena tests).
