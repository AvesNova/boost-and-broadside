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

## 2026-11-26, step 6: envelope on top, middle capped at 6 g, monotonic toward the stall

**Causes of a better turn near the stall** (step 5, full stick):

- The near-stall ramp closed the deep angle back toward the maximum lift: the turn rose from
  14.3 deg/s at 109 m/s to 18.5 at 102.
- The thrust's sideways part at a deep angle (thrust x sin(angle) / speed) went into the turn
  whole: at 57.5 m/s, 45 deg, it alone gave about 17 deg/s (a short-lived version gave
  9.9 deg/s there).
- A softness floor in g, not in the held vertical, gave about 4 m/s^2 of lateral below stall.
- The step-4 cap used the wing's maximum lift, not the envelope's soft minimum: full went
  0.37 deg/s over the envelope at 0.85 x corner, 1.7 deg/s at 0.4 x corner (thrust).

**Changed:**

- No near-stall angle ramp: below the band full stick asks for the deep angle at every speed.
  Closing it toward the stall always brings the angle back to the maximum lift, so the turn
  rises; the "slow nose" variant was tried and dropped (with a weak spring the nose coasted to
  78 deg as the ship slowed through stall speed).
- `_path_lift`: the thrust's normal part turns the path in the same share as the lift (lateral
  over C v^2), so with no lift to spare there is no turn. The whole turn is never more than the
  envelope's at that speed (its lateral lift plus its thrust at its own angle). The square's
  smooth floor scales with what is taken (0 in ordinary flight). The same share scales the
  nose's kappa. `card.PATH_LIFT_SOFTNESS` 0.25.
- Middle stick: the maximum-lift line against `middle_g_fraction` 0.75 of the g limit (card
  slider, about 6 of 8 g), the same soft minimum, then that g level. card.MIDDLE_STICK = 1 (all
  of its own line); derived `middle_stick_speed` = corner x sqrt(0.75) = 221.5 m/s.
- Full stick: the envelope from `deep_stall_start` 0.85 x corner up (card slider); below, the
  deep angle opens over `deep_stall_band` 0.30 (open from 0.55 x corner down).
- Doghouse: the top boundary and the Ps shading are the envelope (any stick, maximum lift
  against the g limit); the full (orange), middle (blue) and gradual lines are drawn inside it.

**Checks** (quasi-steady, 0.1 m/s grid from 85 m/s up):

- Envelope minus max(middle, full): never short by more than 0.018 deg/s (at 0.735 x corner),
  never over.
- Largest rise toward the stall: 0.0000 deg/s for gradual, middle, full and the envelope.
  At stall speed: envelope = middle 7.65, gradual 2.55, full 1.77 deg/s. The envelope reaches
  about 0 just below stall speed: 3.19 at 85 m/s, 1.19 at 80, 0.01 at 60. It is not 0 at stall
  speed itself, because the 1 g held is a soft minimum against the budget there.
- Steepest slope (deg/s per m/s): envelope 1.05, middle 1.05, gradual 0.35, full 1.69 (at 105 m/s,
  where the deep angle's lift starts to carry 1 g: stall speed / sqrt(0.75) = 104.4 m/s).
- Derived numbers: card unchanged; best sustained 167.69 m/s (was 167.72, from the smooth floor).

**Held (flown 60 s in World from the spawn):** middle 148.4 m/s, 39.0 deg/s, AoA 11.4 deg; full
57.5 m/s, 0.0 deg/s, AoA 45 deg, store 0.50 -> 0.002 (it sinks the whole store); gradual 317.4 m/s,
18.8 deg/s after 60 s, still accelerating toward its quasi-steady 332.1 m/s, 17.9 deg/s.

**Snaps** (full stick; peak AoA, speed lost in 0.5 s): 0.6 x corner 50.3 deg, 15.7 m/s; 0.8 x
18.2 deg, 4.4 m/s (the band only starts at 0.85); 0.95 x 14.9 deg, 9.2 m/s (the envelope).

**Nose, full stick from straight** (peak AoA rate, time to 30 deg): 1.0 x stall 98.4 deg/s, 0.43 s;
1.2 x 116.6, 0.40; 1.5 x 142.0, 0.33; 2.0 x 108.2, 0.47 (2.0 x stall is 0.71 x corner, still in
the band).

**Open:** "the nose angle you can get shrinks near the stall" is not done. With the deep angle
at every low speed the nose still goes to 45 deg at stall speed. Any shrink of the angle toward
the maximum lift raises the turn again (above). Options: shrink the angle only below stall
speed, or slow the nose's swing with a strong damping.

## 2026-11-26, step 7: full stick keeps the 1 g; one force budget for lift and thrust

Author decisions: full stick never gives up the 1 g (below the band the deep angle closes from
45 deg at about stall speed / sqrt(0.75) to the stall angle at stall speed); thrust's size never
depends on the angle: the normal force is lift plus thrust x sin(angle), it holds the vertical
first like the lift, what is left turns the path; thrust x cos(angle) drives speed as before.
middle_g_fraction 0.75, deep_stall_start 0.85, deep_stall_band 0.30 unchanged.
Measurements: docs/rewrite/sections/post-stall/lines_check.py.

**Built** (lean/game/core.py):

- `_path_lift`: lateral^2 = (lift v^2 + thrust sin a)^2 - share x vertical^2 (share as before,
  0 below the stall, so ordinary flight is unchanged). The step-6 "thrust turns in the lift's
  share" scaling is gone, and so is the lift-only envelope floor on `taken`. Kept: the cap at
  the envelope's turn (its lateral lift plus its thrust at its own angle). The nose's kappa
  share is now turn / normal force.
- `_holding_angle` (full stick's deep angle, `_commanded_angle` opens toward it instead of the
  fixed deep angle): the deep angle down to the speed where its wing lift holds the vertical
  (104.4 m/s coasting), a smootherstep in speed down to the stall angle at the speed where the
  maximum lift holds it (stall speed 90.4 m/s coasting), the stall angle below. The vertical is
  the setting's (at most 1 g), so a dive's span moves down by sqrt(vertical / 1 g).
  `_commanded_angle` takes the setting's vertical (callers in core, flight_check.py updated).
- New constants: `holding_start_speed` (stall speed), `inverse_holding_span`.

**What I decided myself (the author should check):**

- The schedule counts the wing's lift only, not thrust's share. Counting thrust, 45 deg holds
  1 g at stall speed (normal force 1.18 g there, thrust about 0.61 g at these speeds), and
  down to about 80 m/s (1.02 g), so "the deepest angle that holds 1 g" would stay 45 deg below
  stall speed and never shrink. Thrust is in the force budget (the turn), not in the schedule.
- Not the exact deepest angle: the first version solved C(a) v^2 = 1 g by bisection. Its angle
  has an infinite slope in speed at both ends (the lift past the stall is a smootherstep in the
  angle), and the full line's steepest slope was 3.67 deg/s per m/s at 104.4 m/s (a corner).
  The smootherstep in speed over the same span is smooth and cheap. In the span the wing alone
  dips to 0.929 g (at 100.0 m/s); the normal force never falls below 1.145 g.
- The span scales with the held vertical: with the 1 g span the full dive line rose toward the
  stall by 1.21 deg/s in all (largest 0.085 deg/s per 0.1 m/s step at 97.5 m/s); a linear,
  smoothstep or later-ending span (up to 1.3 x 104.4 m/s) did not fix it (0.49 to 1.74 deg/s);
  scaling by the vertical did (0).

**Lines** (quasi-steady, coast unless named, spawn store, 0.1 m/s grid 85-470 m/s; the rise
toward the stall is measured below each line's peak):

- Largest rise toward the stall: none (at most -0.0000 / -0.0001 deg/s per step) for gradual,
  middle, full, the envelope, full diving and full climbing.
- Steepest slope (deg/s per m/s): gradual 0.348 (92.2 m/s), middle 1.054 (92.3), full 1.054
  (92.3; was 1.69), envelope 1.054, full dive 0.753 (179.5), full climb 1.054.
- Envelope minus max(middle, full): largest 0.0176 deg/s (187.8 m/s), never over.
- The thrust question (author's point 2): with the whole normal force the deep turn rises
  toward the stall, because thrust x sin(a) is up to 0.43 g at these speeds and, as the angle
  closes, the share of the vertical taken falls: before the envelope cap, full stick turns
  16.80 deg/s at 100 m/s, 21.73 at 95, 28.29 at 90, 27.09 at 85. The envelope cap holds the
  line to the envelope there: full = envelope = middle from 85 m/s to about 99 m/s (7.25 deg/s
  at 90, 12.42 at 95). So the cap throws away up to 23.9 deg/s (at 85 m/s) of the budget's turn
  near the stall. Above about 100 m/s the cap does not bind (full turns below the envelope,
  e.g. 19.17 at 104, 24.35 at 120, 29.42 at 140, against the envelope's 19.88, 28.46, 36.20).
  "The turn there is 0" from stall speed to 104 m/s is not what it does: thrust's spare normal
  force turns, capped by the envelope.

| v m/s | full commanded deg | wing lift / 1 g | normal force / 1 g | full deg/s | envelope deg/s |
|---|---|---|---|---|---|
| 90 | 16.0 | 0.990 | 1.133 | 7.25 | 7.25 |
| 95 | 23.1 | 1.072 | 1.271 | 12.42 | 12.42 |
| 98 | 33.3 | 0.972 | 1.246 | 15.25 | 15.25 |
| 100 | 39.8 | 0.929 | 1.247 | 16.83 | 16.93 |
| 104 | 45.0 | 0.992 | 1.336 | 19.17 | 19.88 |
| 120 | 45.0 | 1.320 | 1.640 | 24.35 | 28.46 |

**Held** (World, 60 s from the spawn; speed, turn, AoA, store):

- Full: 96.8 m/s, 14.18 deg/s, 29.1 deg, store 0.453 (was 57.5 m/s, 0.0 deg/s, 45 deg,
  0.002). Lowest 91.9 m/s. It no longer goes below stall speed, but the store still drains
  slowly: 0.492 at 10 s, 0.476 at 30 s, 0.453 at 60 s, 0.00073 of the store per second at
  97 m/s. Cause: `_setting`'s held vertical is the soft minimum of 1 g and the wing's maximum
  lift (exponent 16): 0.9939 g at 97 m/s, and the shortfall sinks from the store. Not changed
  (outside this round); counting the thrust's normal part in that hold, or a sharper soft
  minimum, would stop it.
- Middle: 148.4 m/s, 39.04 deg/s, 11.4 deg, store 0.500 (unchanged).
- Gradual: 316.8 m/s, 18.83 deg/s, 1.8 deg, store 0.500, still accelerating at 60 s (quasi-
  steady 332 m/s).

**Snaps** (full stick from straight; peak AoA (time), speed lost in 0.5 s): 0.6 x corner
50.1 deg (0.63 s), 15.7 m/s; 0.8 x 18.2 deg (1.00 s), 4.4 m/s; 0.95 x 14.9 deg (0.37 s),
9.2 m/s. Within 0.2 deg and 0.1 m/s of step 6.

**Nose** (World, full stick from straight, 2 s; commanded at the start speed; peak AoA (time,
speed); commanded at the peak):

- 1.0 x stall (90.4 m/s): 16.2 deg; 34.7 deg (1.07 s, 96.4 m/s); 27.6 deg.
- 1.2 x (108.5 m/s): 45.0 deg; 50.3 deg (0.80 s, 97.5 m/s); 31.6 deg.
- 1.5 x (135.7 m/s): 45.0 deg; 50.7 deg (0.70 s, 115.9 m/s); 45.0 deg.

The command moves with the speed: from 1.0 x stall the ship accelerates into the span and the
angle follows it up; from 1.2 x it slows into the span while the nose is still swinging toward
45 deg, so the peak is 19 deg past the command at that moment and comes back.

Chart: docs/rewrite/sections/post-stall/doghouse-p51d.png (game units, 2x). The full line runs
below the envelope from about 100 to 190 m/s and on it below about 99 m/s; full held marker at
97 m/s, 14 deg/s. Doghouse flown points: corner 252.5 m/s / 66.6 deg/s, best sustained 167.9 m/s.

**Not run this round** (author's rule): tests, compile check, speed.

**Open for the author:** keep the envelope cap near the stall (it throws away thrust's turn,
up to 23.9 deg/s at 85 m/s) or let thrust turn past the envelope there; the store's slow drain
at the full-stick hold (the soft-minimum 1 g).

## 2026-11-26, step 8: Sustain, Max, Snap; the nose spring per level; thrust in the 1 g hold

Author decisions (all of this round): the three turn levels renamed and redefined (stick
indices unchanged), a per-level nose spring, only straight keeps the shield quiet, thrust's
normal part counts in the 1 g hold, no envelope cap near the stall, the doghouse restyled.
Measurements: docs/rewrite/sections/post-stall/lines_check.py (rewritten for this step; the
step-7 version is in git history).

**Built:**

- Levels (core.py `_setting`, `stick_table`, `sustain_stick_table`, `steady_control`;
  card.LEVEL_NAMES):
  - Sustain (old gradual): Max's line times a per-type table over speed (`_sustain_table`,
    built with the derived characteristics from core.quasi_steady: 97 speeds from stall speed
    to cruise speed, the stick fraction whose turn has Ps = 0 coasting at the spawn store, 1
    where Max gains energy; read linearly in the tick by `_sustain_fraction`, no solve per
    tick). Above cruise speed (372.1 m/s) no sustained turn exists: the table ends at 0 there,
    so Sustain flies straight. Diving and climbing use the same coasting fraction of their own
    top line (one table, not one per power).
  - Max (old middle): the envelope's top line at every speed. middle_g_fraction and the 6 g
    cap are gone (card term, slider, stick_cap_table).
  - Snap (old full): Max's line above corner; below it the deep angle opens from corner speed
    over deep_stall_band (0.30, slider kept; deep_stall_start removed), with the step-7 1 g
    holding schedule near the stall.
- Nose spring (`_kick`, card.NOSE_STIFFNESS, `nose_stiffness_table`): going out (target beyond
  the angle on its side, or on the other side of centre once past it) the stiffness is
  1/3, 2/3, 1 x omega^2 for Sustain, Max, Snap; going back always 1. Friction fixed: the total
  damping coefficient 2 nose_damping omega with nose_damping = 1/sqrt(3) (was 0.59), so the
  damping ratios going out are 1.000, 0.707, 0.577. Chosen per tick; no new state.
- Shield: card.QUIET_STICK_MAX = 0 (only straight is quiet).
- Thrust in the 1 g hold (`_setting`): what holds the vertical is softmin(C_max v^2 +
  thrust |sin angle|, g limit), the angle the one flown (the tick: the current angle;
  steady_rates: resolved twice, the second time at the first's commanded angle). The turn's
  budget is unchanged; the wing's share of the hold is softmin(held, budget) so where thrust
  carries the rest the turn fades smoothly (a hard floor there gave a slope of 5.3 deg/s per
  m/s at 85.5 m/s on the first try).
- `_path_lift`: the envelope cap removed. Two things that cap hid, handled at their cause:
  Snap's open deep angle is the wing's whole (share at least the deep opening: closing the
  angle toward the stall near stall speed no longer frees the 1 g; without this the budget
  gave about 28 deg/s at 90 m/s in step 7), and the wing never carries more than its maximum
  lift against the g limit (the step-6 floor on what is taken, restored: without it Snap
  turned up to 1.22 deg/s above Max at 240.5 m/s, just below corner, the opening angle near
  the stall carrying 1 g for free on top of the maximum lift).
- Doghouse: envelope black (the most turn of any level, so Snap's rise near the stall shows);
  Max solid blue #0072b2, Snap dashed vermillion #d55e00, Sustain dotted bluish green #009e73
  (Okabe-Ito; lighter steps in the dark theme), the dashed lines above the solid; the
  separate Ps = 0 contour removed; level names in the legend and on the lines. No "held"
  marker for Sustain (it holds energy at every speed from about 148 m/s up).
- Names in the tool: flown.CONDITIONS (snap, max, sustain, dive, climb), tuning.LEVEL_INDICES,
  slider labels ("Snap: ..."), the sandbox's key help.

**Lines** (quasi-steady, coast, spawn store, 0.1 m/s grid 85-470 m/s; deg/s):

| v m/s | Sustain | Max | Snap | envelope |
|---|---|---|---|---|
| 85 | 0.56 | 0.56 | 9.12 | 9.12 |
| 90 | 6.40 | 6.40 | 13.50 | 13.50 |
| 95 | 12.26 | 12.26 | 18.65 | 18.65 |
| 100 | 16.91 | 16.91 | 16.76 | 16.91 |
| 104 | 19.88 | 19.88 | 19.15 | 19.88 |
| 120 | 28.46 | 28.46 | 24.35 | 28.46 |
| 167 | 39.94 | 44.91 | 35.39 | 44.91 |
| 200 | 37.98 | 54.61 | 42.35 | 54.61 |
| 255 | 31.00 | 67.21 | 67.21 | 67.21 |
| 350 | 13.34 | 51.09 | 51.09 | 51.09 |
| 370 | 3.40 | 48.31 | 48.31 | 48.31 |

- Rise toward the stall (below each line's peak): none for Sustain, Max and Snap diving. Snap
  (coasting and climbing) and so the envelope: 2.37 deg/s in all, 16.63 at 99.4 m/s up to
  18.99 at 95.9 m/s (largest 0.1 m/s step +0.107 at 97.6). Mostly not thrust: in that span the
  closing schedule's wing lift goes from 0.937 g (99.4 m/s) back up to 1.078 g (95.9), while
  thrust's normal part falls from 0.303 g to 0.208 g. Below 95.9 m/s it falls again (13.50 at
  90, 9.12 at 85). Snap turns more than Max from about 99 m/s down (9.12 against 0.56 at
  85 m/s: thrust at the open angle).
- Snap above Max near corner: at most 0.31 deg/s (229.8 m/s), the thrust tilted by Snap's
  larger angle.
- Steepest slope (deg/s per m/s): Sustain -1.675 at 369.2 m/s (to 0 at cruise), Max +1.794 at
  86.8, Snap +1.236 at 93.4, Snap dive +0.900 at 217.9.
- Sustain's table against a fresh bisection of the Ps = 0 stick, 150-365 m/s: within
  0.041 deg/s (largest at 365 m/s).

**Nose going out from straight** (flight tick; damping ratio, peak angle over the commanded
angle at that moment, time to 90% of the commanded angle):

| start | Sustain | Max | Snap |
|---|---|---|---|
| 0.8 x corner (204.7 m/s) | 1.000; 6.2 over 6.2 deg, -0.2% (no overshoot); 0.73 s | 0.707; 14.0 over 13.3, +5.4%; 0.37 s | 0.577; 48.1 over 45.0, +6.9%; 0.30 s |
| 1.5 x stall (135.7 m/s) | 1.000; 10.7 over 11.2, -4.3% (still rising at 1.5 s); 1.20 s | 0.707; 11.3 over 10.9, +3.5%; 0.57 s | 0.577; 50.7 over 45.0, +12.8%; 0.40 s |

Against the linear expectations (0%, 4.3%, 10.8%) Max is 1.1 points over at 0.8 x corner and
0.8 under at 1.5 x stall, Snap 2.0 over at 1.5 x stall (the commanded angle moves with the
speed, and the path's response kappa adds to the spring); going back past the target uses the
full stiffness.

**Reversals** (Snap left held 1.5 s from 0.8 x corner: 45.4 deg, 103.8 m/s at the switch; time
to cross centre; far side):

- Release: 0.60 s; 5.27 deg past centre (11.6% of the angle at the switch) at 0.87 s.
- Sustain right: 0.53 s; 11.9 deg at 0.90 s against 6.7 commanded (+77%).
- Max right: 0.53 s; 12.1 deg at 0.87 s against 6.7 (+82%).
- Snap right: 0.33 s; 49.5 deg at 0.80 s against 21.5 (+130%; the ship is near stall speed by
  then, where Snap's angle closes toward the stall).

The back half is the full spring and crosses centre fast; the soft spring going out on the
far side, with the friction fixed, does not stop that rate, so Sustain and Max overshoot their
small far-side targets by 6-7 degrees.

**Held** (World, 60 s from the spawn; speed, turn, AoA, store; speed/store at 10, 30, 60 s):

- Sustain: 154.1 m/s, 39.50 deg/s, 10.8 deg, 0.5000 (154.1/0.5000 throughout).
- Max: 148.4 m/s, 39.04 deg/s, 11.4 deg, 0.5000.
- Snap: 96.9 m/s, 17.07 deg/s, 32.0 deg, store 0.4975 (98.2/0.4995, 96.6/0.4987,
  96.9/0.4975). The drain is 0.0025 in 60 s (step 7: 0.047). Not zero: the soft minimum in
  the hold still leaves the held vertical at 0.9997 g at 95.9 m/s, 0.9923 g at 90 m/s.

Chart: docs/rewrite/sections/post-stall/doghouse-p51d.png. Flown points: corner 252.3 m/s /
66.3 deg/s, best sustained 168.0 m/s / 39.9 deg/s, tightest 148.4 m/s / 217.8 m.

**The scripted opponent is now out of date** (not changed; lean/scripted/opponent.py):

- Level 1 (its "gradual", the small-error turn) is now Sustain: a size that changes with
  speed, equal to Max below about 148 m/s, straight above cruise.
- Level 2 ("middle") is its capped level: the two-circle fight and, disengaged, the level
  above `middle_stick_speed` ("no turn lost there"). Max is now the envelope; the derived
  `middle_stick_speed` is now corner speed (kept only for the scripted opponent).
- Level 3 ("full") for the break at corner speed and the one-circle fight: Snap now opens its
  deep angle from corner speed (it was 0.85 x corner).
- The `fire_worth` rule counts a shot as the thing that breaks the shield's quiet tick; now
  every turn level breaks it too.
- Its STICK_GATES and personas were tuned with the old levels.

**Not run this round** (author's rule): tests (lean/game/tests/test_flight.py still uses the
removed stick_cap and GRADUAL/MIDDLE/FULL_STICK), compile check, speed. The tick now does a
table lookup (Sustain), the stiffness choice and a second soft minimum.

**Open:** Snap's 2.37 deg/s rise near 96-99 m/s comes from the closing schedule's shape (the
exact 1 g inverse had no rise but a corner of 3.67 deg/s per m/s at 104 m/s, step 7); the
far-side overshoot of a reversal into Sustain or Max; the Snap hold's small store drain.
