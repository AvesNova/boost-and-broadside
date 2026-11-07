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
- **02:45** Observation change merged (O10 + O1). Verified: ruff; its new tests, the
  acceleration tests, the dashboards slow test and the contract tests (94 pass); C-fixed's
  30M policy and an arm-1 checkpoint load on the new tree. The agent showed arm 1 plays bit
  for bit the same on both trees (72,000 actions, 138k rows of inputs). Sim speed unchanged
  within noise (perceive alone 13-18% slower, a small part of a tick). Noted from its report:
  a learner sequence that starts mid-game with a blank carry now gets valid visible
  accelerations where arm 1's did not (the observation knows t-1); live play is identical.
- **02:47** Arm-2 build agent launched (worktree `bnb-arm2`, branch `feat/arm2-persistence`, from f5be4390).
- **02:56** Arm-2 agent's plan OK; told it the ellipse inputs replace the scalar spreads in the token (the brief's wording), not sit beside them. Its ETA: code 06:00, done 08:30 (worst 10:00).
- **03:07** Corrected my 02:56 note: the agent pointed out that C-fixed blurs the position and velocity codes by sigma (telling the first layer which levels to trust), and 'replacing' had made them sharp. Told it to keep the blur (per world axis from the covariance's marginals) and let the ellipse features replace only the explicit spread slots.
- **03:12** Arm-2 agent: today's token has no explicit spread slots (the spread enters only through the blur), so the ellipse code is added at the end of the ship token; position blurred per axis by sqrt(P_xx), sqrt(P_yy); velocity blurred by its full block as today. Agreed.
- **03:26** Arm 1 finished 30M: final KL 0.5011 (C-fixed 0.500 at 30M): level. Tournament (job 5) next.
- **03:30** Arm 1 against C-fixed at the same 30.4M step (metrics.jsonl): KL 0.501 vs 0.491,
  agreement 0.668 vs 0.672 (level); outcome Brier skill lower (5-15 s 0.33 vs 0.56; 30-60 s
  0.24 vs 0.45), but round 0 showed Brier moving 0.2 by seed alone, so not yet evidence;
  belief position within-1-sigma similar at short ages, lower at 2-30 s. The tournament
  (job 5, started 03:26) decides.
- **03:50** Arm-2 measurements: loss coefficient 0.0033 (delta's trunk share was 1.24x the absolute head's at C-fixed's 0.0041; verifying); 175-bin position-along costs nothing measurable (update 33.0 vs 33.1 s on CPU, +~40 MB per micro-batch at production): taken. Coverage: position within 1 sigma 0.52 at 0-0.1 s, 0.05-0.14 later (sigma too small, not double-counted; the absolute head was 0.00-0.03). Heading sigma passes pi by 0.5 s: asked the agent to compare with the real heading error and fix over-inflation if that is what it is.
- **03:40** Arm 1 tournament: frontline done in one batch (largest standard error 20.5); deathmatch running.
- **03:51** **Arm 1 tournament** (scripted = 1000; out `runs-v2/ratings/2026-11-08-c-accel`):
  frontline C-fixed 736 +- 18, arm 1 702 +- 21 (-34); deathmatch arm 1 676 +- 22, C-fixed
  666 +- 20 (+10); mean of the two modes arm 1 689 vs 701 (-12). Inside the noise (the bar is
  about +50 frontline, +40 mean): **no winner; C-fixed stays champion.** The accelerations
  neither help nor hurt play at 30M; KL and agreement level; Brier lower (seed-noisy). As the
  author decided (C8), arms 2-4 keep the inputs; arm 2 is compared with arm 1 (the plan) and
  with C-fixed. The GPU goes back to the champion continuation (job 3).
- **03:55** Heading spread bug (arm-2 branch, found by my check): real heading error 0.18/0.37/0.55/0.88 rad at 0.1-0.5/0.5-1/1-2/2-5 s vs implied sigma ~1/3/12 rad. Cause: F's heading += nose-rate dt counted the turn twice (the persistence heading change already holds it) and treated the damped nose rate as a random walk. Fixed by the agent: F couples position to velocity only; heading variance capped at pi^2/3 (the uniform circle). Position coupling kept: its coverage showed sigma too small, not too large.
