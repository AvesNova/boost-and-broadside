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
