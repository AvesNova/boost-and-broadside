# HL-Gauss codes: progress

## Resume note (2026-11-15 20:53, WIP, laptop shutdown)

- **Switched (code done, ruff clean):** `codes.hl_gauss_scalar_encode/decode/at` (new; padding 2 a
  side; total sigma = sqrt(0.75^2 + spread^2) bins; folded HL-Gauss mixed with HL-Gauss bumps at the
  outer bins by a closed-form 2x2 solve so mean and variance are exact, then one `match_moments`
  pass; target variance capped at VARIANCE_REACH = 0.8 of c(L - c); decode subtracts
  0.75^2 + 1/12). Used by: state code velocity (3x103), nose rate (45), the five scalars (25 each),
  ellipse features (8x25), shot speed (45) and log index (25); `decode.shot_path`. Trajectory
  coordinates are now 44 numbers. Measured round trip: mean and spread exact to ~1e-14 (float64),
  ~1e-5 bins (float32), at every coordinate incl. edges; cost about 1.2x the old encoder (CPU).
- **Bug found and fixed in passing:** the old `scalar_encode` missed the mean by up to 7 bins near
  an edge with a large spread (fold + one match pass); the new encoder is exact there.
- **Kept, with comments:** colour codes (`codes` circular section), cooldown (`_cooldown_sharp`),
  near-miss (`scalar_encode`, `shots.outcome_code`). Old `scalar_decode` and
  `scalar_sharp_coordinates` removed; `tests/reference/yemong/yemong_arm.py` keeps v1's decode /
  velocity / trajectory compositions so the goldens pass (43 passed).
- **Tests:** test_yemong_codes.py updated (width 580, shot width 162, 44 coordinates) but NOT yet
  rerun; line 146 still expects shape (1, 154) -> change to 162. Not yet run: spread_fit, delta
  (ellipse width 8*21 -> 8*25 at lines 400-419), error_head (544 at line 501 -> 580),
  extrapolation, inputs, position_codes, learner_shots, diagnostics_heads, yemong_model, export.
- **Next:** fix those tests; add tests (round trip incl. edges and s = 0, spread round trip, kept
  codes unchanged); check consumers of spreads (belief, forecast feedback, head_statistics,
  play/predictions.py all go through `decode`); docs (docs/explanation/yemong.md, learner.md);
  merge origin/feat/imugi-boundary (it ignores .venv); never stage .venv.
