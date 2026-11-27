# Look-ahead personas, round 3: progress

Brief (author 2026-11-28 ~03:00): map the top of the skill hill fully (round 2's summit was
truncated in seven rows) and build a persona sampler: a Gaussian on the robust centre with
covariance c x the inverse negative Hessian of the Elo surface. Understanding, not
optimisation; about 1.5 h of GPU. Branch `personas-3` from core 902d3811; laptop RTX 4070,
performance profile. Data: `~/avesnova/boost-and-broadside/data/lookahead-personas-3/`.

- 02:55 step 1 one-at-a-time: 98 agents against the round-2 robust centre (12 rows at box
  offsets -2, -1.25, -0.5, +0.5, +1.25, +2; 6 rows at +-0.75; 5 crowd ratios at +-1, team sizes
  only; 4 hard ends: collision 0, temperature 0.001, horizon 0.1 s, fire tolerance 20). Games
  per agent 256 / 256 / 128 (1v1 / 4v4 / 16v16). Code bounds: every row takes any positive
  value (no clamps); the look-ahead points are fixed at 0.25-2 s and the power look at 1-5 s,
  so a horizon far above 2 s (power horizon above 5 s) only flattens the time weights.
  ETA 03:20 (worst 03:35).
- 03:19 step 1 one-at-a-time done (oat-3.json, oat-3-summary.json; 1v1 / 4v4 / 16v16 Elo
  against the round-2 centre, SE about 25 / 25 / 35 per agent).
  - Tried: 12 rows to +-2 box units, 6 rows +-0.75, 5 crowd ratios +-1, 4 hard ends.
  - Bounded now (falls off both sides): horizon (0.24 s: -231/-248/-303; best near -0.5 to
    -1.25 at 16v16, +69/+83), speed weight (+-1.25: -120 to -200 at every size), temperature
    upward only, angles, preferred speed (1v1 and 16v16 want opposite sides), caution at
    16v16.
  - Plateaus to the hard end: collision weight (lower is better at every size, collision 0:
    +81/+51/+5; no ramming signature: hit range 0.13 vs 0.17 at 1v1, same at team sizes),
    temperature downward (0.001: -11/-29/-60; argmax play is not degenerate), fire tolerance
    upward (20: shots x2.7-2.8, hits unchanged; -15/-31/-131), threat weight, power
    temperature and power horizon flat over +-2.
  - Degenerate: horizon 0.1 s (shots a tenth, Elo -936/-1231/-1126); preferred speed -2 (slow,
    speed 0.54-0.66 of corner). Round 2's stall rule revisited: round-2 stalls were all at
    stickiness 0.17-0.29; ratio 8.6 at stickiness 0.106 was normal, and temperature 0.001
    (ratio 44, stickiness 0.044) is normal: the stall is absolute stickiness, not the ratio.
    New rejection: stickiness > 0.15, horizon < 0.35 s.
- 03:22 Sobol design: 96 points over the enlarged top (design-3-ranges.json, offsets from the
  round-2 centre in box units), 256 / 256 / 128 games each against the centre.
  ETA 03:40 (worst 03:55).
- 03:43 design done (96 points; 69% beat the round-2 centre, mean over sizes; best +178:
  +46/+185/+303). Surface refit (surface-r3.json): 711 agents, 2,207 pairings, 748,416 games;
  held-out deviance 75.9% of coin-to-saturated (round 2: 82.7% on the narrower range: a
  quadratic fits a wider range less well). Predicts the step-1 sweep with r 0.91 (RMSE 78 Elo)
  and the design with r 0.95 (RMSE 30).
- 03:45 summit (top-r3.json): the robust centre moved far (collision weight 1.63 -> 0.36,
  caution 1.66 -> 7.85, fire tolerance 1.85 -> 4.89, power horizon 5.1 -> 9.3 s, horizon
  0.81 -> 0.58 s, ...). The peak still sits at the (now wider) data edge in the plateau rows
  (threat, collision, caution, power horizon / temperature, disengaged energy, fire
  tolerance, stickiness); the steep directions are interior.
- 03:52 sampler (sampler-3.json): Gaussian on the round-3 centre, covariance c x (-H)^-1 of
  the mean surface, curvatures floored at 10 Elo/box^2, sd capped at 0.75 box; c 7.4 / 11.5 /
  21.8 for predicted drops 25 / 50 / 100; box at equal drop = 1.55 x round 2's half-widths.
  Playing 108 agents (16 + 32 + 16 Gaussian, 32 box, 12 along flat directions) against the
  round-3 centre, 128 / 128 / 64 games. ETA 04:06 (worst 04:20); then label blur (no
  games), then the ladder.
