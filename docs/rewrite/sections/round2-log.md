# Round 2 log (what I did and why, while the author is away)

The author (2026-11-08 ~01:40): accepted every recommendation O1-O10 in `round2-plan.md`;
revised the arms (2: persistence, one head on every row; 3: persistence, physics on visible
rows + a separate error head; 4: dead reckoning); "keep running experiments, use your best
judgment, fix bugs or anything unexpected, take a note of everything you do".

Entries: time, what, why. Newest last.

- **00:32** C-fixed reached its 100M budget; queued its continuation to 150M as a background
  job (job 3) under the standing rule (an idle GPU continues the champion).
- **00:35** Brief `acceleration-inputs-brief.md`; agent built `feat/acceleration-inputs`.
- **01:05** Verified (ruff, its 27 tests, the slow dashboards test). Fixed a pre-existing
  failure in that slow test: the rotary read-out and delta-head panels log only with their
  switches on; exempted them. Merged as v2 825a44d3.
- **01:07** Queued arm 1 (job 4, BC 30M, `2026-11-08-bc-arm-C-accel`, worktree
  `bnb-run-accel` at 825a44d3) and its tournament against C-fixed's `policy-uniform-30M`
  (job 5). It preempted job 3, which re-queued as a resume.
- **01:10** Brief `baselines-bench-brief.md`; agent running the bench (`bench/baselines`).
- **01:30** Wrote `round2-plan.md` and a mobile page of it.
- **01:40** Arm 1 at update 3: KL 0.7252, identical to C-fixed's at update 3; learn 232 s per
  update (C-fixed 231 s): the inputs cost no speed.
