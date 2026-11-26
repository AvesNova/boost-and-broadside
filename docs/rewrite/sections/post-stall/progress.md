# Post-stall regime: progress

Branch `post-stall` (from `core`). Rough first version for the author to fly; charts, docs and
the card retune after.

## 2026-11-26, step 1: the regime in the flight model

**Tried / built** (lean/game/core.py, card.py, tuning.py):

- `GameConfig.post_stall = True` (off: full stick on the top line, the old parabola tail, no
  post-stall drag; the middle stick's new shape and the nose damping stay either way).
- Card terms (design inputs): `deep_stall_angle` 45 deg, `deep_stall_band` 0.15 of corner
  speed, `post_stall_drag_ratio` 6, `nose_damping` 0.7 -> 0.59.
- Shared constants (card.py): `POST_STALL_LIFT_FRACTION` 0.75 (flat plate 1.2 over C_Lmax 1.6),
  `POST_STALL_DIP_WIDTH` 0.2 rad, `POST_STALL_DRAG_ANGLE` 45 deg.
- Stick levels as tables by stick index: `stick_cap_table` (middle 2/3 of the g limit) and
  `deep_stall_table` (full stick). Middle = sqrt(softmin(C v^2, 2/3 a_max)^2 - held^2); gradual
  unchanged; full: the commanded angle opens from the top line's angle to the deep angle by a
  smootherstep over 1.0 -> 0.85 x corner speed (`_commanded_angle`).
- Lift past the stall: C sin(2a) + (C_max - C sin 2s - 2C cos 2s (a - s)) exp(-((a - s)/w)^2):
  max and zero slope at the stall, dip to 0.708 of max near 30 deg, 0.75 at 45, 0 at 90.
  Measured lift over max: 20 deg 0.922, 25 0.776, 30 0.708, 35 0.715, 45 0.750, 60 0.650.
- Drag past the stall: post_stall_drag x 2u^2/(1+u^2), u = (cos s - cos a)/(cos s - cos 45):
  smooth, bounded (twice its 45 deg value). Measured drag over drag at the stall: 30 deg 1.83,
  45 deg 6.00, 60 deg 8.82, 90 deg 10.37.
- Past the stall the held vertical comes out of the lift (`_path_lift`, a smootherstep share
  over the dip width): without it a deep stall at 0.4 x corner turned faster than the middle
  stick (28.7 vs 18.7 deg/s) because the 45 deg lift went all into the turn and the 1 g was free.
- Nose: total damping ratio = the card's nose_damping at every angle (the nose's own damping
  is 2 z omega - kappa), so no runaway past the stall. Today's total was 0.7 + kappa / 2 omega,
  about 1.10 at corner (no overshoot).
- Derived characteristics and the card derivation unchanged: `steady_rates` defaults to the
  top line (cap 1, no deep). Card and every derived value bit-identical except nose_damping.

**Worked** (docs/rewrite/sections/post-stall/flight_check.py, CPU, float32 flight tick):

- Throw overshoot 9-13% (deep pulls 12-13%, the commanded angle moving as the ship slows);
  release overshoot past centre 9.3-11.9%.
- A hard pull at 1.1 x corner peaks at 1.030 x the g limit (3% transient); at corner 0.953.
- Full above corner identical on and off. No jump at corner: within 2% of corner the largest
  step per 0.001 x corner is 0.083 deg/s (rate) and 0.11 deg (commanded angle).
- Middle turns faster than full from 0.4 to 0.934 x corner (quasi-steady).

**Didn't / open:**

- From 0.934 x corner to corner full beats middle: the middle level's g cap (2/3) meets the
  maximum-lift line at 0.816 x corner, so near corner middle sits on its 2/3 g line while full
  is still near the maximum lift. The steady full-stick rate dips to 45.9 deg/s at 0.92 x corner
  and jumps back to 67.5 at 0.96 (smooth but steep, the lift dip near 30 deg crossed in the band).
- Deep stall bleeds hard: full from 0.8 x corner (205 m/s) loses 42 m/s in 0.5 s, 82 in 1 s
  (literature suggested 10-20% per snap). Held 10 s from 1.05 x corner it settles at 75 m/s,
  below the 90 m/s stall speed, and the store sinks 0.50 -> 0.33.
- The scripted opponent was tuned with full stick = top line: it now deep-stalls below corner.
  Its `middle_stick_speed` reference (corner / sqrt(2/3)) no longer means what it says.
- The doghouse charts and flown checks (lean/tool/flown.py) fly with post_stall off and draw
  the middle line as 2/3 of the top line; the new middle shape and the deep line are not charted.
- Drag past the stall still counts the held vertical's induced drag on top of the wing's whole
  lift (double counting, small next to the post-stall drag).

**Next:** compile check, sandbox, then the author flies.
