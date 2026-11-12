# Position codes: progress

Branch `feat/position-codes` (from `integrate/next-runs`). Brief: the coordinator's
message of 2026-11-12 (rotary ladder r = 3.405 from 48 m with a per-band turn, heading
harmonics 1, 2, 4, colour code 36 m to 236 196 m, arena-radius input, pairwise geometry
head replacing the rotary read-out).

## Step A: aliasing sweep (2026-11-12, 13:05-13:35 KST)

**Tried.** `bench/rotary_aliasing.py`. A code is 8 bands, each a period P_k and an axis
angle; S(s) = the mean over bands and both axes of cos(2 pi s.u / P_k), the normalised dot
product of the codes of x and x + s (heading left out). Exact extrema over 2D shifts by
branch and bound (each pair's phase over a cell is an interval; S(s) = S(-s)):

- confusion C(d, R) = max S over d <= |s| <= R (1 = an exact alias);
- phase margin M(d, R) = min over d <= |s| <= R of the RMS phase error, per pair, that
  makes s tie with the truth (degrees). "equal": every band read equally well; "coarse":
  band k's noise sigma x sqrt(2)^k toward the coarse end (the coarse bands read less
  precisely), in units of the finest band's sigma: how small a coarse-band phase error makes
  a far shift look like the truth.

Records: `rotary_aliasing.jsonl` (the named candidates, every d and R),
`rotary_aliasing_search.jsonl` (ratios 3.3-3.5 x 6 structured + 16 random turn sets),
`rotary_aliasing_turns.jsonl` (the one-parameter family band k turned k x t, t = 0..45 in
1.25 degree steps; t and t + 90, and t and -t, are the same code), and
`rotary_aliasing_profile.{png,json}` (M and C against d at R = 130 km).

**Table** (d = 1 km; value @ where the worst shift is, km):

| code | C, R 2 km | C, 20 km | C, 130 km | M equal, 130 km | M coarse, 20 km | M coarse, 130 km | C, 500 km |
|---|---|---|---|---|---|---|---|
| today: 8192 m / 2^k, world axes | 0.794 @ 1.0 | **1.000 @ 16.4** | **1.000** | **0.04°** | **0.07** | **0.03** | 1.000 |
| r 3.405, world axes | 0.901 @ 1.6 | 0.901 @ 1.6 | 0.904 @ **64.6** | 13.6° @ 64.6 | 2.50 @ 19.0 | **1.78 @ 64.6** | 0.959 @ 220 |
| r 3.405, golden angle 137.5° | 0.859 @ 1.6 | 0.859 | 0.859 @ 1.6 | 15.9° | 4.59 @ 6.9 | 3.73 @ 114 | 0.885 @ 488 |
| r 3.405, golden fraction of 90° (55.6°) | 0.844 @ 1.9 | 0.844 | 0.844 | 17.1° | 3.95 @ 19.5 | 3.63 @ 74.5 | 0.905 @ 383 |
| r 3.405, 45° steps | 0.886 | 0.886 | 0.886 | 14.2° | 4.40 | 3.15 @ 58.9 | 0.941 |
| r 3.405, 11.25° steps | 0.866 | 0.866 | 0.866 | 15.9° | 4.11 @ 19.2 | 4.11 @ 19.2 | 0.906 |
| r 3.405, random fixed set | 0.867 | 0.877 @ 6.0 | 0.878 @ 20.1 | 15.2° | 5.06 @ 6.0 | 3.80 @ 72.0 | 0.883 |
| **r 3.405, turn 35° (chosen)** | **0.841 @ 1.5** | **0.841** | **0.841** | **17.1°** | **4.86 @ 9.8** | **3.88 @ 59.5** | 0.921 @ 384 |

Nearby ratios with turns (`_search`, `_turns`): 3.3-3.5 give the same picture; the
spread among turned sets is about ±10% on every summary, the turned-versus-unturned gap
2x.

**Worked / found.**
- Today's ladder aliases exactly at 8192 m (and its multiples): S = 1, margin 0.
- r = 3.405 on the world axes removes the exact alias but keeps a far ghost: a 64.6 km
  shift is as similar as a 1.6 km one (0.90), and with the coarse bands noisier 1.8
  finest-band sigmas of phase error make it the best match. The cause is the axes: a shift
  along x leaves all 8 y pairs at cos 1, so S >= 0.5 along each axis, and the bands' x
  lattices line up along it.
- Every turned set removes the ghost within 130 km (the worst match stays near the 1 km
  edge) and doubles the coarse-noise margin (3.4-4.3).
- The golden angle 137.5° is only 47.5° mod 90 (the two axes repeat every 90°), so its
  bands pair up: 0/5°, 47.5/52.5°, ... It still scores in the middle of the turned sets.
- The landscape over the turn is jagged at 1.25° (one turn's worst shift is not its
  neighbour's), so a single best point is noise. **Chose 35°**: its neighbourhood
  33.75-38.75° is good on every summary at r = 3.405 (equal margin 16.5-17.1°, coarse
  margin 3.7-3.95 at 130 km and 4.45-4.86 at 20 km, similarity 0.84-0.85), and stays
  reasonable at r 3.35 and 3.45. The golden angle's neighbourhood holds a weak turn
  (43.75°: 2.64 at 130 km).
- Beyond 130 km (to 500 km, information only) every code has a ghost near 2-3 coarsest
  half-periods; it is past twice the largest map.

**Didn't / limits.** The similarity is the uniform-weight code (an untrained query); a
trained query weights bands, which the "coarse" profile only approximates. The margins at
the d edge are main-lobe tails, not ghosts (the "@" distance tells which).

## Step B: build (2026-11-12)

YemongConfig build options (defaults are the record's, so older builds load unchanged;
`TrainConfig.yemong` sets the new ones):
- `rotary_ratio` (2.0), `rotary_turn` (0.0) beside `rotary_period` / `rotary_octaves`;
  new runs: 254 719 m coarsest, 8 bands, ratio 3.405, turn 35°, `heading_harmonics`
  (1, 2, 4): 19 pairs, 38 dims. `layers.rotary_tables` takes per-pair axes and periods.
- `position_period` (26 244 m): the colour code's coarsest, the offset's range; new runs
  236 196 m (36 m finest). Threaded through the state code, the global/field/zone codes,
  the decode helpers and the belief statistics.
- `arena_code` (False): the global token reads ln(R / 7800 m) as an HL-Gauss code, 21 bins
  a quarter octave apart (0.25x-8x) plus 2 padding bins per side.
- `geometry_head` (False): `bnb.models.yemong.geometry`; the learner's `geometry` term
  (nats per target over pairs), COUNTS "pairs", `geometry/cross_entropy/*` series.
- Removed: the rotary read-out (model, trunk `retable`, labels, loss, statistics,
  dashboards, tests). `YemongConfig.RETIRED` / `LearnerConfig.RETIRED`
  (`bnb.runs.serialise`): a record with `rotary_readout` off still reads; one with it on
  (arm D's policies) no longer loads.
- The error head's position bound stays 26 244 m (a constant of that code).

Tests: `tests/sections/test_position_codes.py`, `tests/sections/test_learner_geometry.py`
(new); `test_yemong_order.py`, `test_dashboards.py`, the reference arm's rotary adapter
(edited); `test_learner_rotary_readout.py` removed.

## Step C: the geometry head's weight

(below)
