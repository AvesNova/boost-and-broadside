# HL-Gauss codes: progress

Decision (author, 2026-11-15): HL-Gauss, sigma 0.75 bins, for every binned target and input
unless there is a specific reason not to, with padding bins where needed.

## Done (2026-11-15)

- **Encoder** `codes.hl_gauss_scalar_encode/at/decode` (+ `hl_gauss_scalar_coordinate`):
  2 padding bins a side; total sigma `sqrt(0.75^2 + spread^2)` bins; the folded HL-Gauss
  mixed with the HL-Gauss at each outer bin by a closed-form 2 x 2 solve so the mean and the
  variance are exact, then one `match_moments` pass; the variance capped at
  `VARIANCE_REACH` = 0.8 of `c (L - c)`. Decode clips the mean to the real range and
  subtracts `0.75^2 + 1/12` bins^2 (spread 0 below 64 eps). Round trip: mean and spread to
  ~1e-14 bins in float64, ~1e-5 in float32, at every coordinate incl. the ends. About 1.2x
  the old encoder's cost (CPU).
- **Switched:** state code velocity (3 x 99 real + 4 = 3 x 103), nose rate (41 + 4 = 45),
  the five scalars (21 + 4 = 25 each); ellipse inputs (8 x 25 = 200); shot trajectory speed
  (41 + 4 = 45) and log index (21 + 4 = 25), 44 coordinates, code 162 wide. State code 580
  wide (544 before).
- **Kept, comment at each:** circular colour codes (position, heading, shot displacement
  and direction), the gun cooldown (`_cooldown_sharp`, two-hot), the near-miss code
  (`codes.scalar_encode`, `shots.outcome_code`).
- **Bug fixed in passing:** the old `scalar_encode` missed the mean by up to 7 bins near an
  edge with a large spread (fold + one match pass); the new encoder is exact there.
- **Removed:** `codes.scalar_decode`, `codes.scalar_sharp_coordinates` (nothing uses them).
- **v1 reference goldens:** `tests/reference/yemong/yemong_arm.py` (the adapter, which the
  suite's README lets the section write) keeps v1's linear decode, velocity code and
  two-hot shot speed / log index, so the goldens still pin what bnb shares with v1 (the
  velocity compression and projections, the colour codes, `scalar_encode`).
- **Consumers of decoded spreads** (belief, forecast feedback, `decode.ship_forecast`,
  `head_statistics`, `play/predictions.py`, figures) all go through `ShipStateCode.decode`
  or `decode.shot_path`, in physical units: no change needed.
- **Tests:** widths updated (codes, delta, error_head, model, fourier inputs); the arm
  goldens (`tests/sections/golden/yemong_arms_steps.pt`, `yemong_v1_spread_steps.pt`)
  re-recorded with `arm_goldens.py --record`, since the switch changes every arm's inputs.
  New in `test_yemong_codes.py`: each switched code's mean and spread round-trip across the
  range incl. the ends and spread 0 (float64, float32), an interior exact value is the plain
  HL-Gauss, a value past the range codes as its end, the velocity at its reach, the kept
  colour codes and two-hot cooldown.
- **Docs:** `docs/explanation/yemong.md` (binned codes, ellipse inputs, widths),
  `docs/explanation/learner.md` (next-state target, shot codes).
