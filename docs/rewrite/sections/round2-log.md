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
- **04:13** Arm-2: coefficient verified (0.0033 gives trunk norm 0.0144 vs the absolute head's 0.0143 at 0.0041). After the heading fix: heading within 1 sigma 0.74/0.49/0.37/0.38/0.29 by age (a little narrow); nose-rate sigma was also a random walk (100% coverage), now capped at a uniform over the code range. Position still too narrow (8/4/6% within 1 sigma at 0.1-2 s) on the 20-update CPU run: accepted for this arm (its belief is open-loop; the error head in arm 3 is the fix by design); watch it in the GPU run.
- **04:23** **Arm 2 merged** (v2 122bd6bc). Verified: ruff; the delta, acceleration and
  observation-record tests (78 pass); C-fixed's and arm 1's policies load. Agent's report:
  quick tier 1506, slow dashboards 2 pass. Codes: position along asymmetric 175 bins (4.9 cm
  at the 2.6 m median), the rest 51/31; 624 outputs (absolute 490); round trip < 2e-4 bins
  (needed a mean-inverse table: the plain decode was biased up to 0.37 bins at the padded
  ends); learner.next_state 0.0033. Changes from the brief: F couples only position to
  velocity (heading and nose rate decoupled and capped, see 03:55); ellipse code appended
  (168 inputs) since the token had no spread columns; targets masked to ships alive at t and
  t+1. Open: position sigma overconfident on the short run (arm 3's error head is the
  designed fix).
- **04:24** Queued arm 2 (job 6, `2026-11-08-bc-arm2-persistence`, worktree `bnb-run-arm2` at
  122bd6bc; it preempted the champion continuation, job 3) and its tournament (job 7: arm 2,
  arm 1 and C-fixed 30M uniform averages, frontline + deathmatch, out
  `runs-v2/ratings/2026-11-08-arm2`). I will check the first updates against round 1's
  early numbers (belief error and coverage by age, the head's error against the bench
  medians, the gradient share).
- **04:30** Brief `arms34-brief.md`; my calls in it: r = each quantity's one-tick physical bound (the belief moves toward the truth no faster than the ship could move); the error target's range is the full reachable error (position +-26 km); the error head's coefficient starts equal per row to the physics head's and changes only if its share is >1.5x off. Agent launched (worktree `bnb-arms34`).
- **04:50** Arms 3/4 agent found arm 3 as briefed unstable on a short CPU run: the error head's decoded mean is tail-dominated (broad code to +-26 km), predicting 40-400 m for every hidden row (realised 5 m at 0-0.1 s), so the step saturated at r for 100% of rows and the 1-2 s belief error grew to 355-459 m (arm 2: 20-58 m). Its fix (agreed): step toward e's **median** (per projection, then least squares), still capped at r; sigma stays Var e + (E e - step)^2. A median of a near-marginal (untrained) head is about 0, so an untrained error head does not move the belief. Rerunning.
- **05:05** Arm 2 at update 3 (3.1M) vs arm 1 at the same step: KL 0.712 vs 0.725; next-state one-tick error, visible ships: position 0.44 m (persistence 2.69; arm 1's absolute head 2.70), velocity 1.42 m/s (persistence 1.72; arm 1 5.77); belief within 1 sigma, position 0.43/0.13/0.08/0.06 at 0-0.1/0.1-0.5/0.5-1/1-2 s (arm 1: 0.00), heading 0.58/0.43/0.37/0.34 (arm 1: 0.22/0.18/0.18/0.17). Healthy start; position still has to learn v dt (round 1's arc baseline was at 0.022 m by update 4, by construction).
- **05:10** Arm 3's median step is unstable too (target past r on 85-100% of hidden rows by update 4-5; 1-2 s belief error 35 -> 172 m): a closed loop, the replay one update stale, gain >> 1. The agent added a confidence gain |m|^2/(|m|^2+sigma^2); I asked it to add a fixed per-tick rate on top, step = eta * gain * median(e) then the cap r, eta = 0.1 (time constant ~0.33 s; the author's 'rate-limited pseudo gradient descent'), 0.03 if still unstable, as a config field.
- **05:15** Arm 3 with eta 0.1 + gain: stable over 20 CPU updates. Belief position error 1.2/7.1/21/54/147 m by age (arm 2 1.0/6.5/20/46/143: the correction does little yet); sigma now too WIDE (error head's position sigma 450-680 m vs realised 3-130 m; within 1 sigma 0.74-1.00; arm 2 0.04-0.42). Step 6% of r, 1.4% capped. Trunk shares: physics 0.0146, error head 0.0065 (0.78x per row): coefficient stays 0.0033. Watch in the GPU run whether the error head's sigma sharpens (a wide sigma blurs the hidden inputs). Arm 4 code in; 45 arm-3/4 tests pass incl. equality with the bench's E.
- **05:32** Arm 3 committed on its branch (a711cd58; quick tier 1550, dashboards slow 4, docs 480 pass). Arm 4 knees re-measured (position 0.26/1.2 mm, velocity 0.019/0.073 m/s; 500 outputs). Arm 4's short run diverged by update 8 (velocity error 1039 m/s at 2-5 s): for hidden rows the rates were the belief's own change, which includes the head's decoded change, so each tick's change became the next tick's acceleration and compounded. Agent's fix: hidden rows hold their last sighting's rates (as the bench's iterated E). I asked for a decay of the held rates with hidden age (default time constant 0.5 s) and/or a top-speed clamp, since a held thrust integrates without bound.
- **05:48** Arm 4 with held rates decaying at tau 0.5 s: stable, no speed clamp needed. By age (0-0.1 ... 5-10 s): position 0.04/1.9/10.8/32/141/394 m (arm 2 1.0/6.5/20/46/143/377), velocity about equal to arm 2, heading worse (0.25/0.63/1.07 rad at 0.1-2 s vs arm 2 0.17/0.37/0.51): the held nose rate keeps a hidden ship turning. Asked the agent to decay the nose rate feeding hidden headings the same way (one short run; keep only if it helps).
