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
