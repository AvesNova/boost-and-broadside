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
- **02:00** Bench merged (v2 0eeddc43; `baselines-bench.md`). Verified: ruff, its tests and
  `tests/docs` (487 pass). Headline: extrapolation (E) leaves the least for position (1.9 mm
  median vs the rule's 6.6) and velocity (0.12 vs 0.33 m/s); near zero while the control is
  held; it costs on a power change (carries the old thrust). The nose rate: E overshoots (the
  held change decays by 0.41 per tick), persistence is better. Axis correlation, folded by turn
  sign: 0.45-0.63 for constant-velocity and rule forecasts at 0.5-1 s in the nose and track
  frames, 0.2-0.55 in world axes; one tick: below 0.08 everywhere.
- **02:00** **O3 (my call, from the bench):** arm 4's baselines per quantity: position E (the
  arc form; equal to the literal form on the data, exact on a turn); velocity E (accepting the
  power-change cost); heading E; **nose rate persistence** (E overshoots); angle of attack E;
  energy E; shield and countdowns the rule (D2); ln n the gradient baseline ln n + grad . v dt
  (state-deltas measured it better than E in the tail: p99.9 0.013 vs 0.020).
- **02:00** **O8 (my call):** the error head codes 2D quantities on **three projection axes
  (0, 60, 120 degrees, world)**: a hidden ship's uncorrected error is correlated 0.45-0.63 at
  0.5-1 s in every frame, with the sign set by the turn direction, so no fixed frame makes
  per-axis codes independent; three marginals recover the full 2 x 2 covariance. The physics
  head (one tick, correlation < 0.08) stays per axis in the velocity frame.
- **02:00** Arm 1 at update 10: KL 0.4843 vs C-fixed's 0.4890 at the same step (noise level).
- **02:05** Brief `observation-accelerations-brief.md` (O10 then O1); agent building
  `feat/observation-accelerations` (worktree `bnb-observation-accel`). Arm 2's build waits for
  it (arm 2 should train on the final input path).
- **02:08** Drafted `arm2-persistence-brief.md` (launch after the observation change merges).
  My calls in it: round 1's rule baselines are removed (no arm uses them; the aborted
  round-1 policies stop loading); under persistence the countdowns and shield are plain
  changes (the model learns regeneration and the countdown, as the author wanted); their
  belief advance uses the decoded **median** (= most likely when a jump has p < 0.5, D8);
  position along the track is the whole move (median 2.6 m, never near 0), so it gets finer
  bins (<= 5 cm at the median, up to ~201) or the persistence arm would be capped by code
  resolution rather than learning (new confound **C11**); the spread is the full 6 x 6
  covariance with the physical F (risk: double-counting velocity error on hidden rows; log
  coverage).
- **02:50** Observation change merged (O10 + O1). Verified: ruff; its new tests, the
  acceleration tests, the dashboards slow test and the contract tests (94 pass); C-fixed's
  30M policy and an arm-1 checkpoint load on the new tree. The agent showed arm 1 plays bit
  for bit the same on both trees (72,000 actions, 138k rows of inputs). Sim speed unchanged
  within noise (perceive alone 13-18% slower, a small part of a tick). Noted from its report:
  a learner sequence that starts mid-game with a blank carry now gets valid visible
  accelerations where arm 1's did not (the observation knows t-1); live play is identical.
