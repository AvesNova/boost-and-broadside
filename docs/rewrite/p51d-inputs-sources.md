# P-51D: where each web-tool input comes from

Reference for the web tool's Ships tab (`lean/tool/design.py`), preset `p51d`, as of
2026-11-24 (branch `core`). One entry per input, in the tool's groups and order. 36 inputs.

Units: "game" is the game's 2x time (speeds and rates x2, accelerations x4); "real" is 1x.
Lengths, per-metre coefficients, angles and loads in g are the same in both.

The game's conditions, for every entry: ISA sea level only (rho 1.225, no altitude, no
thinner air), the engine always at war emergency power (67 in Hg, 3,000 rpm), one weight
(4,427 kg / 9,760 lb, no fuel burn), clean, 2x time.

Sources named by short name; full list at the end. "(A)" to "(F)" are the grades of
`aircraft-references.md` (A flight test ... F forum).

## Summary

**From a reference (1).** Top speed 375 mph (the "cruise" slider): AAF flight test of
P-51D 44-15342 (A), sea level, 67 in Hg, 9,760 lb, true airspeed.

**Derived from references (6).** Stall speed, zero-lift drag, the settled turn rate, the
low-speed acceleration, and the drag rise (critical speed and exponent). Each is arithmetic
or a fit on referenced data plus at least one assumption (C_Lmax basis, Oswald e 0.8,
propeller figure of merit 0.75).

**Game choices (7).** g limit 8 (author 2026-11-23); the 1 g dive and zoom trade; the
separation drag 0.1; thrust softness 8; all three flap inputs.

**Copied (20).** Two flight terms from v2's rules, the four stick and nose inputs, eight of
the gun inputs, all seven shield, head-on and vision inputs: from v2's Vizia via
`tuning.GAME_FIELDS`. A few of those trace back to P-51D or M2 gun research (muzzle speed,
fire rate, spread, bullet drag). The rest are v2 game choices or placeholders.

**Guess or builder's default (2).** Stall angle 15.5 deg (v2's estimate; not found in any
document). Gun range 600 m (the default argument of `tuning.handles_of`).

Most important caveats:
- C_D0 (0.0150) and propeller efficiency (0.82) were fitted together to two test points.
  Neither number means anything alone.
- The test's climb used 1,780 hp, but the model uses the level-flight 1,630 hp for both
  points. The fit absorbs the difference.
- The "settled full-stick turn rate" and the best sustained turn behind it have no test
  value. They come from the model.
- The 8 g limit is a forum figure, tied to an 8,000 lb design weight. At the tested
  9,760 lb the forum rule gives about 6.6 g.
- The +15% flaps are a game invention. The manual's stall chart gives about +9% for 20 deg
  and about +12% for 40 deg of real flap (gear down, IAS).

## Behind the sliders (not inputs)

The tool does not show the airframe data. They set the preset's row: the tool's
"Performance targets" and "Airframe" values are that row's numbers.

- Mass 4,427 kg (9,760 lb). AAF memo report TSCEP5E-1908 (A). "Combat take-off weight":
  full ammunition, 21 gal oil, 184 gal wing fuel, 25 of 85 gal fuselage fuel. The "clean"
  configuration carried a bomb rack on each wing.
- Power 1,630 hp at 67 in Hg / 3,000 rpm. The same report, level-flight table at sea level
  (the report's web page). At sea level its climb table lists 1,780 hp at 67 in Hg.
  Military (61 in) is 1,500 hp and normal (46 in, 2,700 rpm) 1,075 hp.
- Wing area 21.67 m^2 (233.2 ft^2). NACA ACR 4K02, p.3 (A). That figure is for a P-51B.
- Aspect ratio 5.83. The October fit's value. NACA gives 5.815 (P-51B) and Loftin 5.86
  (P-51D).
- C_Lmax, clean: 1.6. Back-solved from an AAF turn calculation (see "stall speed").
- Oswald e: 0.8. Assumed. The game row's solve moves it to 0.819 equivalent.
- Propeller: Hamilton Standard Hydromatic, four blades, Dwg K6523A-24 (the report). The
  3.40 m (11 ft 2 in) diameter is the October fit's figure. It is not in the downloaded
  pages.
- Static figure of merit 0.75: assumed (`planes.py`).
- C_D0 and propeller efficiency: fitted (see "zero-lift drag").

There is no climb slider. The best climb, 3,600 ft/min (A), is a target of the preset's
solve. Moving the tool's sliders does not hold it.

## Performance targets

### Cruise (top) speed
- Game 335.28 m/s; real 167.64 m/s = 375.0 mph.
- Origin: **reference.** AAF 44-15342, Fig. 3 "Level flight data" and the summary (A).
  The preset flies it exactly: `fit_point_mass` makes the point mass fly it, and
  `game_row` holds it with the game terms on.
- Reference conditions: sea level, true airspeed, 67 in Hg at 3,000 rpm (war emergency),
  9,760 lb.
  - "Clean" includes two wing bomb racks.
  - Oil and coolant flaps on thermostats (coolant 107 C, oil 70 C).
- Against the game: the same power and weight, at sea level. The game is always at sea
  level, so the altitude speeds (417 mph at 10,000 ft, 442 mph at 26,000 ft) have no
  counterpart.
- Caveats:
  - The label says "cruise", but the value is the top level speed at full power.
  - A published cruise speed is at reduced power and usually at altitude. The report's
    Appendix B gives 323 mph TAS at sea level at 2,700 rpm / 46 in (normal rated), and
    lower figures at 2,150-2,500 rpm. None of these compares with this slider.
  - Other sea-level figures: 363-364 mph at military power (A); 368 mph at war emergency
    from NAA's calculation NA-46-130 at 9,611 lb (B).
  - The report says its speeds differ from the Tech Order data by about 3% on average.
  - In the tool this slider sets the engine power (the handles route solves the thrust
    power to fly it).

### Settled full-stick turn rate
- Game 0.5972 rad/s (34.2 deg/s); real 0.2986 rad/s = 17.11 deg/s.
- It settles at game 120.6 m/s (real 60.3 m/s, 134.8 mph). That is coasting, the store
  half full, flaps out.
- Origin: **derived.** It is not a target: it is what the preset's row flies
  (`handles_of`: `performance`'s `settled_turn_rate`).
- The row was solved to a different turn number: the point mass's best sustained turn,
  18.99 deg/s at 163 mph, radius 220 m. That is the model's output from the airframe data.
  No measured value exists.
- Reference conditions: none. No measured P-51D turn rate, radius or 360-degree time was
  found (`aircraft-references.md`).
  - Relative: TAIC 38 says a Zeke 52 gained on the P-51D within one 360 (A, relative).
  - Simulator: about 15 deg/s at 150 mph IAS, war emergency, 500 ft (Hong 1998, E).
  - v2's research: 15.7 deg/s (low; simulator and community compilations).
- Caveats:
  - This is the settled full-stick turn, not the best sustained turn.
  - It depends on C_Lmax, e, the game terms and the flaps.
  - In the tool it sets the induced drag.

### Low-speed acceleration (match switch off by default)
- Game 14.76 m/s^2; real 3.69 m/s^2 (0.38 g). Straight and level at the stall speed,
  empty store.
- Origin: **derived.**
  - The flown result of the static thrust: momentum theory, (FM^2 P^2 2 rho A)^(1/3),
    with 1,630 hp, a 3.40 m disc and an assumed FM of 0.75.
  - That gives 26.4 kN (5.97 m/s^2 real, per kg); the drag at the stall speed takes the
    rest.
  - With the switch off (the default) the static thrust stays the preset's and this value
    is not matched.
- Reference conditions: none measured.
  - v2's research estimated 2.73 m/s^2: a point mass at 1.1 x stall at military power,
    x 0.88 for propeller efficiency.
  - That is at a different power and speed from this one.
- Caveats:
  - A power-limited ship flies the same whatever its static thrust (`handles_row`
    docstring), so this value barely changes the flight.
  - FM 0.75 is assumed.

### Dive and zoom trade
- Game 39.24 m/s^2; real 9.81 m/s^2 (1 g).
- Origin: **game choice.** The core model trades speed and store at 1 g, "a vertical dive
  or zoom" (`airframe_row`). v2's Vizia had 0.983 g (placeholders scaled from v2's standard
  fighter).
- Reference conditions: not applicable. A real dive or zoom gains or loses speed at
  g sin(angle) plus thrust minus drag.
- Caveat: the tool sets the zoom to minus the dive (one trade).

## Airframe

### Stall speed (1 g)
- Game 90.44 m/s; real 45.22 m/s = 101.2 mph.
- With the flaps out the game's 1 g stall is 94.3 mph.
- Origin: **derived:** sqrt(2 m g / (rho S C_Lmax)), with 4,427 kg, 21.67 m^2,
  rho 1.225 and C_Lmax 1.6.
- Where C_Lmax 1.6 comes from: back-solved from the AAF "Performance Data on Fighter
  Aircraft" (26 Jul 1944, B/C). That gives a radius of 883 ft at 4 g, 10,000 ft density
  altitude, P-51B at 9,000 lb, at its "estimated maximum power-on lift coefficient".
- Reference stall figures and their conditions:
  - Pilot's manual AN 01-60JE-1, Fig. 24, p.25 (B): about 88 mph IAS at 7,400 lb, rising
    to about 100.5 mph IAS at 9,800 lb (read off chart). Flaps 0, gear up, no external
    load; power not stated.
    - The chart warns that the stall reads "approximately 10 mph higher IAS than previous
      P-51 airplanes due to a change in the airspeed system".
  - Loftin, NASA SP-468 (D): 100 mph at 10,100 lb.
  - NA-46-130 (B): 95.4 mph at 9,071 lb, probably flaps and gear down.
- Against the game: the game's 101.2 mph is true airspeed at sea level, where IAS is close
  to TAS apart from the instrument's position error.
- Caveats:
  - C_Lmax depends on the stall-speed basis. Stall-derived values span 1.50-1.70 with the
    weight and the IAS error. The preset was 1.50 until 2026-11-24.
  - A power-on 1 g stall includes slipstream lift. NACA's F6F tests found that the
    accelerated (turning) value is lower than the power-on 1 g value. The value used here
    is the AAF's power-on estimate.
  - In the tool this slider sets the maximum lift coefficient (`g / stall^2`).

### g limit
- 8 g, the same in game and real (game 313.92 m/s^2 maximum acceleration).
- Origin: **game choice** (author, 2026-11-23, core-decisions.md: "P-51D g limit stays 8").
  v2's research had it as "recalled standard figure, not re-checked".
- Reference conditions:
  - Forum only (F): +8 / -4 g limit, 12 g ultimate, at an 8,000 lb design gross weight.
  - Above 8,000 lb the limit reportedly falls as 64,000 / weight: about 6.6 g at the
    tested 9,760 lb, about 6.4 g at 10,000 lb.
  - Sources: ww2aircraft.net thread "P-51B/C/D/K G Limit at 8000lbs"; a DCS forum post
    quoting a manual.
  - No primary document was found. The downloaded manual's limitations (Fig. 23) give no
    g limit.
- Against the game: the game holds 8 g at 9,760 lb, about 20% above the forum's
  weight-scaled limit.
- Caveat: the corner speed follows from it: stall x sqrt(8) = 286 mph.

### Zero-lift drag
- 4.506e-5 per metre (game and real): rho S C_D0 / (2 m). C_D0 equivalent 0.01503
  (game row); the point mass's 0.01504.
- Origin: **derived (a fit).**
  - `fit_point_mass` sets C_D0 and propeller efficiency (0.821) so the point mass flies
    375 mph and 3,600 ft/min at sea level (A).
  - `game_row` then re-solves zero-lift drag, thrust power and induced drag so the row
    with the game terms on still flies them (efficiency 0.820 equivalent, e 0.819).
  - It is re-fitted every time the card is derived.
- Reference conditions: two test points, both at sea level, war emergency, 9,760 lb.
  - Top speed: 375 mph true, 1,630 hp.
  - Climb: 3,600 ft/min at 165 mph IAS (corrected for instrument error), and the climb
    table gives 1,780 hp.
  - The model uses 1,630 hp for both.
- Comparison values:
  - Loftin: C_D0 0.0163 (D). That is a fit too: from 437 mph at 25,000 ft and 1,490 hp,
    with an assumed efficiency and e.
  - NACA ACR 4K02, Figs. 12-13 (A): about 0.021-0.022 measured with the propeller off.
    P-51B, dives from 25,000-30,000 ft, dusty surface, no Meredith thrust.
- Caveats:
  - C_D0 and propeller efficiency are only meaningful together.
  - The fit absorbs cooling drag, exhaust thrust, the Meredith effect and the 150 hp gap
    between the climb and level power.
  - The model's best-climb speed is 144 mph TAS, against the test's 165 mph IAS.
  - The report's Fig. 8 drag curves and Fig. 4 power required (5,000 and 25,000 ft) could
    separate C_D0 from e given a propeller efficiency. They are not used.

### Stall angle
- 0.2705 rad = 15.5 deg (game and real).
- Origin: **guess.** v2's research estimated it: "NACA 45-100 laminar section, clean CLmax
  ~1.4; typical fuselage-reference stall angle 15-17 deg".
- Reference conditions: not found for any type (`aircraft-references.md`: "the stall angle
  of attack was not found").
- Caveat: it shapes the lift curve's peak and the nose's response, not the 1 g stall speed.

## Game terms

### Drag rise: critical speed (switch on)
- Game 524.06 m/s; real 262.03 m/s (586 mph) = Mach 0.77 at the sea-level speed of sound,
  340.3 m/s.
- Origin: **derived** from NACA ACR 4K02 Fig. 12 (A), as read in `p51d-model.md` (not
  re-read for this table).
  - P-51B with the propeller off: C_D flat at about 0.0215 to M 0.68, about 0.030 at
    M 0.735, about 0.036 at M 0.755.
  - The game's zero-lift drag doubles at M 0.77.
- Reference conditions: dives from 25,000-30,000 ft. The Mach number there is at altitude
  temperature.
- Against the game: the game converts Mach with sea-level temperature, so the critical
  speed is a fixed true speed.
- Caveats:
  - Negligible at the game's speeds: +0.013% at the 375 mph top speed (M 0.49).
  - The measured dive (44-14134, `p-51d-dive-27-feb-45.pdf`) reached M 0.85. The manual
    gives a dive limit of 505 mph IAS.

### Drag rise: exponent
- 20 (no units).
- Origin: **derived**, fitted to the same figure. 1 + (M / 0.77)^20 gives +15% at M 0.70,
  +39% at M 0.735 and +68% at M 0.755, against the figure's +40% at 0.735 and +67% at
  0.755.
- v2's 12 would add 13% at M 0.65, where the figure is flat.

### Separation drag at maximum lift (switch on)
- 0.1: the induced drag +10% at the wing's maximum lift.
- Origin: **game choice**, step A1, 2026-11-24.
  - It is the largest value tried that keeps the best sustained turn's speed near the
    point mass's (+8% at 0.3, +22% at v2's 1.0).
  - The author decided to keep the terms on, with the solver compensating. The 0.1 itself
    is not recorded as an author decision.
- Reference: none. It is a game term, not physics data.

### Separation drag: exponent
- 8.
- Origin: **copied** from v2 (`rules.toml` `separation_exponent` 8; `GAME_FIELDS`).

### Lift softness: exponent (switch on)
- 16. The soft minimum of the wing's lift against the g limit.
- Origin: **copied** from v2 (`rules.toml` `lift_softmin_exponent` 16).
- Caveat: it holds full stick below the hard limit near the corner. The corner rate is 8%
  below the point mass's (32.2 against 35.0 deg/s real).

### Thrust softness: exponent (switch on)
- 8. The soft minimum of the thrust against the static thrust.
- Origin: **game choice**, A1, 2026-11-24. v2 had 2, which took 10% of the thrust near the
  best-climb speed.

## Flaps

The P-51D had no combat flap setting: all three values are a **game choice** (the author's
2026-11-24 decision for automatic flaps, scheduled by demand; values set in A1 and
"Charts").

Reference conditions for comparison: the manual (AN 01-60JE-1, B).
- Stall chart (Fig. 24): about +9-10% maximum lift with 20 deg of flap and about +12% with
  40 deg, both with the gear down. Read off the chart, IAS. My reading, from the ratio of
  the stall speeds squared.
- Flap limits (Fig. 23): 10 deg up to 400 mph IAS, 20 deg to 275, 30 deg to 225, 40 deg to
  180, 50 deg to 165.

### Extra lift (switch on)
- 0.15 of the clean maximum (Delta C_L 0.24).
- Caveat: more than the manual's 40-degree landing flap gives (above). The game's flaps are
  "as far out as useful", not a real flap setting.

### Blown back at
- 1.0 x the clean corner speed. Fully out to 0.6 x corner (172 mph real); clean from
  286 mph.
- It is chosen so that the corner and the corner rate stay the clean airframe's.

### Drag of the extra lift (switch on)
- 7.54 in the card's units. That is Delta C_D = 1.5 (Delta C_L)^2 with the flaps fully
  used (0.086 at the full bonus): 1.5 / (rho S / 2m x 66.348).
- How it was set: a sweep on 2026-11-24 so the tightest sustained turn is at least 10%
  slower than the best sustained rate (the author's margin). The ratio came out at 0.852.
- Reference: none.

## Stick and nose

All four are control feel, not physics. Each is **copied** from v2's Vizia (`GAME_FIELDS`)
and not re-derived for the P-51D under the core model.

### Nose frequency at corner speed
- Game 11.23 rad/s; real 5.62 rad/s.
- Origin: Vizia's card. It comes from v2's `nose_reversal_time` handle (0.394 s game):
  "placeholder: the standard fighter's, scaled as the standard fighter is to the P-51D".
- Reference: not found (no pitch-response data was gathered).

### Fine stick
- 0.210 of full.
- Origin: v2's rule `fine_stick_share` 0.3 x the sustained stick (`solver.stick_levels`).

### Sustained stick
- 0.700 of full.
- Origin: Vizia's best sustained turn's lift fraction under v2's flight model
  (`solver.stick_levels`).
- Caveat: not the P-51D's own best sustained turn under the core model.

### Nose damping
- 0.7.
- Origin: v2's rule value (`rules.toml` `nose_damping`), kept.

## Guns

All **copied** from Vizia (`GAME_FIELDS`, via v2's `vizia.toml` notes and
`bnb/reference/real_planes.py`), except the gun range. Vizia's guns were the P-51D's six
Browning AN/M2 .50 cal in the wings, represented as one stream of shots.

### Hit damage
- 21 per hit (no time units).
- Origin: a chain of v2 game choices.
  - It began as the P-51D's weight of fire, 3.68 kg/s for six M2s (v2 research, medium).
  - Then balance and centre-hit rescalings (24, 15, 9.75).
  - Then game v12's lethality: 21, "about 6 core hits kill".
- The real weight of fire is not the 21. No real counterpart.

### Explosive share
- 0 (pure kinetic).
- Origin: v2's default ("proposal").
- Real: the .50 cal's standard loads were ball, AP and API. No explosive-share value was
  gathered.

### Fire rate
- Game 30 shots/s; real 15/s.
- Origin: v2's derived value: "the whole-Hz volley rate dividing 30 nearest its rate per
  gun". A 30 Hz tick divisor.
- Reference: the AN/M2's cyclic rate is 750-850 rounds per minute (12.5-14.2 Hz per gun):
  Wikipedia "M2 Browning" and simple.wikipedia. That is a cyclic rate, not a sustained one.
- Caveat: one game shot stands for a volley of six guns.

### Muzzle speed
- Game 1,712 m/s; real 856 m/s (2,808 ft/s).
- Origin: v2's research, graded medium, "AAF Manual 200-1". That document was not
  downloaded and is not verified here.
- Web figures vary: 2,750 ft/s (MAPS Air Museum PDF), 2,845 ft/s for the aircraft gun
  (hitechcreations wiki), 2,910 ft/s for M33 ball (simple.wikipedia), 2,930 ft/s (War
  Thunder wiki). Those are 838-893 m/s.
- The game adds the ship's speed to the muzzle speed.

### Bullet spread
- Game 3.084 m/s (sideways velocity sigma); real 1.542 m/s.
- Origin: v2's research, graded low: a 6 mil 75% cone, sigma = D75 / (2 x 1.665) x
  muzzle speed.
- Caveat: the guns' spacing and convergence are ignored (all shots from the nose).

### Bullet drag
- 0.0005 per metre (game and real).
- Origin: v2's set floor (`BULLET_DRAG_FLOOR`), noted as "the .50 cal's", medium.
- No ballistic source was found in the docs.

### Gun range
- 600 m.
- Origin: **builder's default.** `tuning.handles_of(row, gun_range=600.0)`; Vizia had
  586.9 m.
- With cruise and muzzle speed it gives a bullet life of 11 ticks: 0.367 s game, 0.733 s
  real. That is the preset's own (`GAME_FIELDS`), so 600 m changes nothing.
- Real: not referenced. P-51D gun harmonisation distances were not gathered.

### Hit radius
- 12 m.
- Origin: game v12's damage profile (game choice). It began as half the span, 5.645 m (the
  span 11.29 m, 37.04 ft, v2 research "good"). It was raised to 10, then to 12.
- Caveat: about twice the real half-span.

### Core radius
- 5 m.
- Origin: game v12 (game choice). Full damage inside 5 m, a smoothstep to zero at the hit
  radius.

## Shield, head-on and vision

All **copied** from Vizia (`GAME_FIELDS`), all game rules. None has a real counterpart.

### Vision range
- 1,000 m (game and real).
- Origin: v2's Vizia handle. `card.VISION_RANGE` 3,072 m x the type's scale 0.3255 =
  1,000 m, "for every type".
- v2's note reads "the standard 1024 m times the map scale (3)". That is the 3,072 m, not
  the 1,000 m. Where the 1,000 m comes from was not found.

### Shield
- 100.
- Origin: v2's "standard 100 x (1 + 0.1 per rating step from the P-51D's 2)". The P-51D's
  survivability rating of 2 is v2's research ("medium": a liquid-cooled Merlin with a
  ventral radiator), so 100.

### Shield recharge rate and wait
- The wait before recharge: game 3.0 s (real 6.0 s). Then empty to full in game 4.2 s
  (real 8.3 s).
- The tool's values: rate 24/s, wait 3 s, game units.
- Any non-quiet input restarts the wait: a stick beyond the fine turns (index above 2).
- Origin: v2's "placeholder: the standard fighter's". The author kept the old timing after
  flying game v12.

### Head-on damage share
- 0: no damage dead ahead.
- Origin: game v12 lethality (game choice).

### Head-on angle
- 0.785 rad = 45 deg: half damage there.
- Origin: game v12 (game choice).

### Head-on sharpness
- 3: "a flatter armoured nose".
- Origin: game v12 (game choice).

## Sources

Downloaded (`~/avesnova/boost-and-broadside/data/aircraft-references/p51d/`, URLs in
`SOURCES.txt`):
- AAF memo report TSCEP5E-1908, P-51D-15 44-15342, 15 Jun 1945 (A).
  - Read here: Fig. 3 (level), Fig. 5 (climb), Fig. 8 (drag curves), Appendix B.
  - The report's page, read for the 1,630 / 1,780 hp, the loading, the racks and the
    thermostat flaps:
    <http://www.wwiiaircraftperformance.org/mustang/p51d-15342.html>.
- Pilot's Flight Operating Instructions AN 01-60JE-1, 5 Apr 1944 (B). Fig. 23
  (limitations, flaps) and Figs. 24-25 (stall) on p.20 and p.25; read here.
- NACA ACR 4K02, Nissen, Gadeberg, Hamilton 1945 (A): wing area, AR, Figs. 12-13. Taken
  from `aircraft-references.md` and `p51d-model.md`; not re-read here.
- NAA NA-46-130 (B), the V-1650-7 low-blower power chart (B), the 44-14134 dive report (A).

Repository: `lean/game/tuning.py` (`P51D`, `P51D_TERMS`, `GAME_FIELDS`),
`docs/rewrite/p51d-model.md`, `docs/rewrite/aircraft-references.md`,
`docs/rewrite/core-decisions.md`, `src/bnb/game/ships/real-scale/vizia.toml` [notes],
`src/bnb/game/ships/real-scale/rules.toml`, `src/bnb/reference/real_planes.py`,
`src/bnb/game/handling/solver.py` (`stick_levels`).

Web (2026-11-24):
- ww2aircraft.net, "P-51B/C/D/K G Limit at 8000lbs":
  <https://ww2aircraft.net/forum/threads/p-51b-c-d-k-g-limit-at-8000lbs.56386/>
- ww2aircraft.net, "WW II airframe G limits":
  <https://ww2aircraft.net/forum/threads/ww-ii-airframe-g-limits.62214/>
- M2 Browning: <https://en.wikipedia.org/wiki/M2_Browning>,
  <https://simple.wikipedia.com/wiki/M2_Browning_machine_gun>
- <https://www.hitechcreations.com/wiki/index.php/.50_calibers>
- <https://mapsairmuseum.org/wp-content/uploads/2024/02/M-2-Machine-Gun.pdf>
- <https://warthunder.fandom.com/wiki/12.7mm_Browning_(.50_cal)>
