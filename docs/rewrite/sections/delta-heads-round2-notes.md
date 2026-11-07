# Notes: the delta heads, round 2 (discussion of 2026-11-07, not yet decided)

> Superseded as the plan by `round2-plan.md` (2026-11-08); kept for the round-1 findings.

The first delta arm (`2026-11-07-bc-arm-C-delta-aborted`) was stopped at update ~5 to fix
problems found in its first updates. Proposals below are under discussion with the author;
nothing here is decided until the author says so.

## Findings from the first arm (updates 1-4)

- Position: arc baseline 0.031-0.045 m vs persistence 2.5-2.7 m one-tick error; the head
  reached 0.022 m by update 4. Belief far better than C-fixed at short ages (2 vs 21 m at
  0.1-0.5 s hidden; 14 vs 63 m at 0.5-1 s), coverage 60% at 0.1-0.5 s, then 22/11/7/5%.
- Velocity: the rotate-by-omega-dt baseline was worse than persistence on the mean (2.15 vs
  1.72 m/s), better on the median (report: 0.33 vs 0.64). Cause: the game turns velocity by
  omega dt in the drift but alpha changes in the kicks (the nose oscillator), so the true
  turn is omega dt - d alpha (the flight-path rate); rotating by the nose's swing
  over-rotates in transients (|v| |d alpha| ~ 9 m/s at p99.9).
- Recharge wait: the advance uses the expected value p*delay + (1-p)*countdown; for a hidden
  ship (p ~ 0.02/tick) it settles near 4.4 s, so the shield baseline never regenerates
  hidden shields (frozen). The gun cooldown is an exact code mixture (fine); only its
  diagnostic (expected value) looks bad.
- Spread: the advance propagates only block-diagonal covariance (no Cov(position, velocity),
  no Cov(heading, nose rate)): position variance grows ~n^2 instead of ~n^3, ~22% of the
  true sigma at 1 s. Per-tick changes are also treated as independent though controls are
  correlated in time.
- Open loop: hidden-ship target is truth(t+1) - baseline(truth(t)), so the head learns the
  process noise Q and expected drift, never a correction toward the truth.

## Proposed restart (pending the author's go)

1. Events: advance by the most likely outcome (countdown unless p > 0.5); diagnostics the
   same. Known limit: misses hits on hidden ships in fights (p 5-20% < 0.5). Later: the wait
   and shield as a joint distribution (an exact per-tick histogram, or particles).
2. Position: arc dead reckoning (the arc removes omega-determined curvature); residual in the
   velocity frame. The residual ~ Arc (v_mid - v_t) dt ~ dt/4 x the velocity change: mostly
   redundant with the velocity target (a consistency signal; its loss adds weight to dv).
3. Velocity: persistence (no baseline), in the velocity frame (along: speed change; across:
   |v| x path turn). Case study: drag, lift and refraction align with the velocity frame,
   thrust with the nose; each frame isolates one control and leaks the other by sin alpha;
   the velocity frame does not swing with the nose oscillator. Log a warning below 1 m/s.
4. Heading: dead reckoning (+ omega dt). Shield and countdowns: rule baselines and event
   bins, as now (the author: linear regeneration is useful knowledge; keep).
5. Spread: the full 6 x 6 covariance of (x, y, vx, vy, heading, omega) in the predict step
   (P' = F P F^T + Q, Q = the head's change variance rotated to world), carried in the
   rollout's recurrent state; inputs read marginal spreads (position: the ellipse's larger
   axis, or an average: open).
6. No correction head in this arm (noted limit).

## Next challenger: physics plus a capped correction

One head, two outputs: the physics change (every row, truth to truth, velocity frame) and,
on hidden rows only, a correction toward the truth: target r tanh(|e| / r) e/|e|, e =
truth(t+1) - forecast(belief), in world coordinates, a symlog code (cheap even for km
ranges: ~67 bins for +-26 km at a 2 mm knee); applied to position (and maybe heading), never
to the velocity belief. Its variance is calibrated against real belief errors (the update
the filter lacks). The three regimes (just hidden, small drift, large drift) fall out of
physics + capped correction.

## Later

- Negative information at inference: drop the part of the belief inside ally-visible regions
  (exact and trivial with particles; a crescent for a Gaussian). Needs the visibility
  geometry from what the model sees.
- Particles: one shared head conditioned on each particle's state and an index embedding;
  winner-take-all (responsibility-weighted) corrections, or all particles collapse onto the
  truth; joint particles for wait and shield.
- Check whether in-flight bullets reach the model (the model sees which ships fire; the
  shot-trajectory head can predict paths).
