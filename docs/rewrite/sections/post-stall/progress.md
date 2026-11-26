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

## 2026-11-26, step 4: the doghouse from the game's own levels

Author's report: the sandbox flew full held 104 m/s, 5 deg/s, AoA about 30 deg, and middle held
149 m/s, 39 deg/s, 11 deg; the doghouse disagreed. Cause: the chart side. Its middle line was
2/3 of the top line and its full line the top line, and the flown checks flew with post_stall off.
Units were not the problem: the readout and the chart both show game m/s at the chosen time
scale (2x by default), converted by the same lean.tool.units factors.

**Built:**

- `tuning.level_lines`: each stick level's line from the stick tables (core.stick_level_shape)
  through core.steady_rates under the given GameConfig, plus where holding the level settles
  (Ps falls through 0). `tuning.LEVEL_INDICES` names the levels' stick indices from
  card.STICK_LEVELS.
- doghouse.py: the full, middle, gradual, dive and climb lines come from level_lines. The corner
  marker is the full line's peak. New markers show where each level settles when held. The
  shading stops at the most turn any level flies.
- One GameConfig for the whole tool: `lean.tool.flight_config()` / `CONFIG_SETTINGS`
  (BNB_POST_STALL, BNB_FIELDS) feeds the sandbox, the flown checks and the charts. The flown
  checks no longer force post_stall off.
- design.py's stall speed is now the game's derived constant, not a re-derived sqrt(g / C).
- Physics fix found by the chart: below the stall, the deep angle's opening near stall speed
  gave lift the 1 g already uses (full turned 28.4 deg/s at 97.5 m/s against middle's 14.8).
  `_path_lift` now never gives more than sqrt((C_max v^2)^2 - vertical^2). After the fix:
  97.5 m/s full 16.4, middle 14.8. The rest is thrust tilted by the larger angle.

**Matched** (quasi-steady line / the real tick in World, 30 s held from the spawn, coasting):

- Middle held: chart 148.4 m/s, 39.0 deg/s; flown 148.4 m/s, 39.0 deg/s, AoA 11.4 deg.
- Full held: chart 104.3 m/s, 5.1 deg/s; flown 104.3 m/s, 5.1 deg/s, AoA 28.9 deg, store 0.497.
- Gradual held: chart 332.1 m/s, 17.9 deg/s.
- Full line peak (the corner as flown): 249.1 m/s, 68.4 deg/s on a 0.1 m/s grid. Flown corner
  check: 252.7 m/s, 66.6 deg/s.
- Best sustained with any stick (the envelope, Ps = 0): chart 169 m/s, 40 deg/s; flown 169.0
  m/s, 39.9 deg/s at stick 0.88 of the top line. It needs a stick between the levels, so no
  single level holds it.

**Derived numbers, unchanged:** corner 255.8 m/s / 67.2 deg/s (the top line, where the g limit
meets the maximum lift: it sets the store and the deep band) and best sustained 167.7 m/s /
39.9 deg/s (any stick). The chart's corner is now the full line's peak: 249.1 / 68.4.

**Duplication left, bigger than this round:**

- The Ps shading, the sustained curve and the best and tightest sustained points use a
  continuous stick fraction of the top line, which the game's three levels do not offer.
- flown.critical_points flies that continuous stick by overwriting the constants' middle-stick
  table columns, a measurement override.
- tuning.stick_levels() still returns the card's fractions (tests only).
- The flown checks fly from the spawn speed, so lowest speeds near stall (full 97.8 m/s)
  depend on the bleed search.

## 2026-11-26, step 5: no cliffs in the full-stick line

Author: the step-4 doghouse's full line had two near-jumps, near 97-104 m/s and 230-248 m/s.
They were continuous but very steep: the commanded angle crossed the lift dip past the stall at
the ends of two narrow schedules.

**Changed:**

- Lift past the stall, monotonic with no dip: from the maximum at the stall down to the flat
  plate's 0.75 at 45 deg by a smootherstep in the angle (zero slope at both ends), then
  0.75 C_max sin 2a (0 at 90 deg). Lift over the maximum: 20 deg 0.993, 25 0.952, 30 0.879,
  35 0.805, 45 0.750, 60 0.650.
- Deep band below corner: `deep_stall_band` 0.15 -> 0.35 (opens from corner down to
  0.65 x corner).
- Opening near stall speed: a smootherstep in speed from stall speed to
  `card.DEEP_STALL_OPEN_SPEED` 1.8 x stall speed, replacing the spare-lift form. Over the same
  grid it gave the gentler line: the spare-lift form at 0.8 gave a low-end slope of -3.07 deg/s
  per m/s; the speed form at 1.5 / 1.8 / 2.2 / 2.6 gave 1.52 / 1.01 / 1.00 / 1.00. 1.8 keeps a
  real deep stall at moderate speed; at 2.6 full held at 137.7 m/s, 36.2 deg/s, close to middle.
- `card.POST_STALL_DIP_WIDTH` renamed `POST_STALL_SHARE_WIDTH` (it is now only the width over
  which the held vertical comes out of the lift).

**Steepest slope of the full line** (quasi-steady, 0.1 m/s grid, 95-470 m/s, deg/s per m/s):
before coast -27.90 at 102.6 m/s, climb -28.17 at 102.6, dive +3.02 at 241.7; after coast +1.01,
climb +1.01, dive +0.78. For comparison the middle line's steepest is +1.00. One gentle hump
remains: full rises to 30.2 deg/s at 120 m/s, falls to 28.9 at 140 (steepest fall -0.59), then
climbs to the corner.

**Held points, quasi-steady = flown in World:** middle 148.4 m/s, 39.0 deg/s, AoA 11.4 deg
(unchanged); full 122.5 m/s, 31.4 deg/s, AoA 23.3 deg (was 104.3 m/s, 5.1 deg/s, 28.9 deg).
Full-line peak: 255.6 m/s, 67.2 deg/s (the top line's corner again).

**Snaps** (full stick from straight; peak AoA, speed lost in 0.5 s; before -> after):
0.6 x corner 50.8 deg, 16.4 m/s -> 48.2 deg, 16.2 m/s; 0.8 x 50.7, 42.2 -> 46.0, 25.7;
0.95 x 47.0, 14.0 -> 21.8, 9.6. At 0.6 x corner the commanded angle falls from 45 to 19 deg as
the ship slows (the opening closes toward stall speed), so the nose peaks and comes back.

Chart: docs/rewrite/sections/post-stall/doghouse-p51d.png (game units, 2x).
