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

## 2026-11-26, step 2: compile check and sandbox

**Compile check** (laptop 4070, performance profile, load below 1.5, cold cache; baseline file
not updated): `advance` 2545 traced operations against core's 2300 measured the same hour
(+245, +10.7%) and the recorded baseline's 2220 (+15%: the check FAILs at its 10% tolerance;
core itself is already +3.6% over the file). First version 2605; one lift-and-slope evaluation
and the middle angle's post-stall drag shared brought it to 2545. act, minibatch_loss and
slot_actions equal core's. Cold 227.4 s (core 228.6 s), 0 graph breaks, 4 graphs.

**Sandbox:** flies the regime by default (`World` with the default GameConfig);
`BNB_POST_STALL=0` in the server's environment turns it off. Headless, 150 ticks of full stick
from the spawn: speed 53.5 m/s, AoA 45.0 deg, 8.1 deg/s (off: 149.5 m/s, 11.5 deg, 39.5 deg/s).
The Ships tab's sliders include the deep angle, band, drag ratio and nose damping.

**Next:** the author flies; then charts (new middle shape, the deep line), the scripted
opponent's stick use, and cutting ops if the 10% tripwire matters.

## 2026-11-26, step 3: after the author's first flight

Author's rule for this round: no compile check, speed tests or quick tier until the feel is
settled; measurements only (docs/rewrite/sections/post-stall/slow_checks.py).

**Built:**

1. Deep angle fades near stall speed (`core._commanded_angle`): the extra angle (deep less the
   lift's angle) times smootherstep(spare / card.DEEP_STALL_SPARE_LIFT), spare = 1 - (stall
   speed / speed)^4 (the lift beyond 1 g, squared, over the whole, below corner); 0.8 opens it
   fully at about 1.5 x stall speed. Chosen over a speed band: same cost, and it is the spare
   lift itself. At stall speed and below full stick asks for the lift's angle only (0 with no
   spare lift).
2. Firing never takes the ship below stall speed (`_fire`, `_fly`, `GameConfig.fire_store_band`
   0.1): the thrust a shot withholds is cut from the speed above 1.1 x stall speed, paid from
   the store (thrust kept, its power x tick from the store) at stall speed and below, a
   smootherstep between. Below 1.1 x stall speed an empty store blocks the shot (the gun stays
   ready). The scripted opponent only pulls the trigger (it reads the cooldown, counts no
   shots), so the game decides; nothing changed there.
3. Sandbox without fields: `BNB_FIELDS=0` in the server's environment.

**Measured** (World, one ship, no fields, before -> after):

- Nose, full stick from straight (the ship accelerates meanwhile, so the angle still opens as the
  spare lift grows). Peak AoA / peak AoA rate: 1.0 x stall 51.0 deg (0.93 s) / 106.6 deg/s ->
  20.1 deg (1.00 s, still rising with speed) / 32.7 deg/s; 1.2 x 51.7 / 115.1 -> 43.6 (0.73 s) /
  98.3; 1.5 x 50.9 / 139.8 -> 50.7 / 139.8. The nose frequency is nose_frequency x speed
  (floor card.NOSE_FREQUENCY_FLOOR 0.50 rad/s, never reached here): 3.97, 4.77 and 5.96 rad/s
  at 1.0, 1.2, 1.5 x stall (90.4 m/s), unchanged.
- 5 s of continuous fire from 1.2 x stall (108.5 m/s, store 0.500), speed / store each second:
  straight before 105.7/0.500, 102.1/0.500, 98.2/0.500, 94.2/0.499, 90.5/0.496;
  after 105.7/0.500, 102.1/0.500, 98.2/0.499, 96.6/0.492, 96.5/0.480.
  Full stick before 80.9/0.497, 68.0/0.466, 65.1/0.430, 64.1/0.392, 63.9/0.354 (with step 1 only:
  93.6/0.500, 88.9/0.495, 86.7/0.487, 85.7/0.476, 85.2/0.464); after both 95.0/0.495,
  95.7/0.471, 95.4/0.446, 95.4/0.422, 95.4/0.397. 149 shots in every case.
  Empty store, straight, 5 s of fire from 1.2 x stall: 134 of 149 shots, speed held at 99.8 m/s.
- Hard turning alone (no fire), 10 s of full stick from 1.05 x corner: lowest 99.8 m/s, end
  104.3 m/s, store 0.499 (was 75.1 m/s and 0.330). It no longer goes below stall speed.
- Quasi-steady at 0.4 x corner (1.13 x stall): full 10.7 deg/s, +5.4 m/s^2 (was 7.6, -34.5);
  middle 18.7 unchanged. Middle still turns faster than full from 0.4 to 0.934 x corner.

**Didn't / open:** in the first pull from 0.6 x corner the commanded angle at 1 s is now 33 deg
(the ship has slowed to about 106 m/s, where the deep angle is fading) while the nose is still
at 44.6, so it falls back. Compile and test tiers deferred to the branch's end.
