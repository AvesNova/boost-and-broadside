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
