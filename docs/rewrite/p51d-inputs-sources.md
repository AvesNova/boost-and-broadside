# P-51D: where each design input comes from

Reference for the ship design inputs (`lean/game/tuning.py` `DESIGN_INPUTS`, the web tool's
Ships tab), preset `p51d`, as of 2026-11-24 (branch `core`, the design system of
core-decisions.md "Ship design"). One entry per input, in the table's groups and order: 36
inputs, one per card field. The tool shows each input's short provenance tag (`source`) beside
its slider. Each slider starts at the reference value given here.

Units: "game" is the game's 2x time (speeds and rates x2, accelerations x4); "real" is 1x.
Lengths, per-metre coefficients, angles and loads in g are the same in both.

The game's conditions, for every entry:
- ISA sea level only (rho 1.225, no altitude, no thinner air);
- the engine always at war emergency power (67 in Hg, 3,000 rpm);
- one weight (4,427 kg / 9,760 lb, no fuel burn);
- clean, at 2x time.

Sources are named by short name; the full list is at the end. "(A)" to "(F)" are the grades of
`aircraft-references.md` (A flight test ... F forum).

## Summary

How each input is set (`p51d-model.md`, "Design inputs"):
- **Closed:** the field is given by a closed form.
- **Solved:** a flown target; `design_row` solves the field it mainly moves so that the row
  flies it.
- **Short circuit:** the field itself.

**From a reference (5).**
- Top speed 375 mph, best climb 3,600 ft/min and the best-climb speed 165 mph: AAF flight
  test of P-51D 44-15342 (A), sea level, 67 in Hg, 9,760 lb.
- Critical Mach and the drag-rise exponent: NACA ACR 4K02 Fig. 12 (A).

**Derived from references (3).** Stall speed (C_Lmax 1.6), static thrust (momentum theory,
FM 0.75 assumed) and the best sustained turn rate (the point mass, Oswald e 0.8 assumed).

**Game choices (6).** g limit 8 (author 2026-11-23), the gap 0.85 and the separation exponent
16 (author 2026-11-24), the 1 g dive and zoom trade, and the three flap inputs (off).

**Copied (20).** The lift softness (v2's 16), the three stick and nose inputs, and eight gun
inputs. All seven shield, head-on and vision inputs too. They come from v2's Vizia, now in
design units (`tuning.P51D_VALUES`). A few trace back to P-51D or M2 gun research (muzzle speed,
fire rate, spread, bullet drag); the rest are v2 game choices or placeholders.

**Guess or builder's default (2).** Stall angle 15.5 deg (v2's estimate; not found in any
document). Gun range 600 m (v2's handles default).

Not inputs, solved or read out:
- **Solved fields:** zero-lift drag, thrust power, induced drag, separation drag and thrust
  softness.
- **Readouts:** the corner speed and rate, every radius, the settled full-stick turn and the
  low-speed acceleration.
- **Derived by the game:** the stick levels and the bullet's damage reference speed.

Most important caveats:
- C_D0 (0.0158) and the propeller efficiency (0.859) are fitted, as equivalents, to the flight
  test's top speed, climb and climb speed with the game terms on. Neither number means
  anything alone.
- The test's climb used 1,780 hp, but the model uses the level-flight 1,630 hp for both
  points. The fit absorbs the difference.
- The best sustained turn rate has no test value. It comes from the model.
- The 8 g limit is a forum figure, tied to an 8,000 lb design weight. At the tested
  9,760 lb the forum rule gives about 6.6 g. The tool warns about this at the default.

## Behind the inputs: the reference airframe (`tuning.P51D`)

The tool does not show the airframe data. They make the reference (`tuning.reference_of`): the
point mass is fitted to the flight test, and the inputs without a documented value take its
numbers. The tool's implied-physics panel holds the mass, wing area, aspect ratio, shaft power
and propeller diameter, and reads every row back as C_Lmax, C_D0, e, propeller efficiency and
figure of merit.

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
- Oswald e: 0.8. Assumed for the point mass. The game row's solve moves it to 0.965
  equivalent.
- Propeller: Hamilton Standard Hydromatic, four blades, Dwg K6523A-24 (the report). The
  3.40 m (11 ft 2 in) diameter is the October fit's figure. It is not in the downloaded
  pages.
- Static figure of merit 0.75: assumed (`planes.py`).
- g limit rated at 8,000 lb (`g_limit_mass`), for the structure's warning.
- C_D0 and propeller efficiency: fitted (see "top speed").

## Performance

### Stall speed (1 g)
- Game 90.44 m/s; real 45.22 m/s = 101.2 mph. Closed: the maximum lift coefficient is
  g / stall^2.
- Origin: **derived:** sqrt(2 m g / (rho S C_Lmax)), with 4,427 kg, 21.67 m^2,
  rho 1.225 and C_Lmax 1.6.
- Where C_Lmax 1.6 comes from: it is back-solved from the AAF "Performance Data on Fighter
  Aircraft" (26 Jul 1944, B/C). That gives a radius of 883 ft at 4 g, 10,000 ft density
  altitude, for a P-51B at 9,000 lb, at its "estimated maximum power-on lift coefficient".
- Reference stall figures and their conditions:
  - Pilot's manual AN 01-60JE-1, Fig. 24, p.25 (B): about 88 mph IAS at 7,400 lb, rising
    to about 100.5 mph IAS at 9,800 lb (read off chart). Flaps 0, gear up, no external
    load; power not stated.
    - The chart warns that the stall reads "approximately 10 mph higher IAS than previous
      P-51 airplanes due to a change in the airspeed system".
  - Loftin, NASA SP-468 (D): 100 mph at 10,100 lb.
  - NA-46-130 (B): 95.4 mph at 9,071 lb, probably flaps and gear down.
- Against the game: the game's 101.2 mph is true airspeed at sea level. There IAS is close
  to TAS, apart from the instrument's position error.
- Caveats:
  - C_Lmax depends on the stall-speed basis. Stall-derived values span 1.50-1.70 with the
    weight and the IAS error. The tool warns above 1.7.
  - A power-on 1 g stall includes slipstream lift. NACA's F6F tests found the accelerated
    (turning) value lower than the power-on 1 g value. The value used here is the AAF's
    power-on estimate.

### g limit
- 8 g, the same in game and real (game 313.92 m/s^2 maximum acceleration). Closed: n g.
- Origin: **game choice** (author, 2026-11-23, core-decisions.md: "P-51D g limit stays 8").
  v2's research had it as "recalled standard figure, not re-checked".
- Reference conditions:
  - Forum only (F): +8 / -4 g limit, 12 g ultimate, at an 8,000 lb design gross weight.
  - Above 8,000 lb the limit reportedly falls as 64,000 / weight: about 6.6 g at the
    tested 9,760 lb, and about 6.4 g at 10,000 lb.
  - Sources: the ww2aircraft.net thread "P-51B/C/D/K G Limit at 8000lbs", and a DCS forum
    post quoting a manual.
  - No primary document was found. The downloaded manual's limitations (Fig. 23) give no
    g limit.
- Against the game: the game holds 8 g at 9,760 lb, 22% above the forum's weight-scaled
  limit. The tool shows this as a warning at the default.
- Caveat: the corner speed follows from it: stall x sqrt(8) = 286 mph (a readout).

### Top level speed
- Game 335.28 m/s; real 167.64 m/s = 375.0 mph. Solved, mainly by the zero-lift drag.
- Origin: **reference.** AAF 44-15342, Fig. 3 "Level flight data" and the summary (A).
- Reference conditions: sea level, true airspeed, 67 in Hg at 3,000 rpm (war emergency),
  9,760 lb.
  - "Clean" includes two wing bomb racks.
  - The oil and coolant flaps are on thermostats (coolant 107 C, oil 70 C).
- Against the game: the same power and weight, at sea level. The game is always at sea
  level, so the altitude speeds (417 mph at 10,000 ft, 442 mph at 26,000 ft) have no
  counterpart.
- Other sea-level figures:
  - 363-364 mph at military power (A);
  - 368 mph at war emergency, from NAA's calculation NA-46-130 at 9,611 lb (B).
- The report says its speeds differ from the Tech Order data by about 3% on average.
- The zero-lift drag: C_D0 0.0158 equivalent (P-51D powered fit 0.0148-0.0163,
  `p51d-drag-check.md`). Loftin's 0.0163 (D) is a fit too. NACA ACR 4K02 Figs. 12-13 (A)
  measured about 0.021-0.022 with the propeller off (dusty surface, no Meredith thrust).

### Best climb rate
- Game 36.58 m/s; real 18.29 m/s = 3,600 ft/min (Ps at 1 g). Solved, mainly by the thrust
  power.
- Origin: **reference.** The same test (A): 3,600 ft/min at 165 mph IAS at sea level, with
  1,780 hp in the climb table (the model uses 1,630 hp).
- Thrust power: a propeller efficiency of 0.859 equivalent at 1,630 hp. The drag check's
  efficiency at top speed is 0.85 (0.82-0.885).

### Best-climb speed
- Game 147.52 m/s; real 73.76 m/s = 165 mph. Solved, mainly by the thrust softness (the
  soft minimum of thrust against the static thrust).
- Origin: **reference:** the test's climb speed, 165 mph IAS corrected for instrument
  error (A; `p51d-drag-check.md` section 1). Test schedules are normally flown near the best
  speed.
- The solve gives a softness of 2.83. The soft limit then takes 5.4% of P / v at 165 mph.
  The tool warns when it takes more than 10%: v2's softness of 2 took about 12%. With the
  old fixed softness of 8 the climb peaked at 140 mph.

### Best sustained turn rate
- Game 0.6630 rad/s; real 18.99 deg/s, at 181.6 mph (r 245 m) in the game row. Solved,
  mainly by the induced drag.
- Origin: **derived:** the point mass's best sustained turn (18.99 deg/s at 163 mph, r 220 m)
  from the airframe data, with e 0.8 assumed. No measured value exists.
- Reference conditions: no measured P-51D turn rate, radius or 360-degree time was found
  (`aircraft-references.md`).
  - Relative: TAIC 38 says a Zeke 52 gained on the P-51D within one 360 (A, relative).
  - Simulator: about 15 deg/s at 150 mph IAS, war emergency, 500 ft (Hong 1998, E).
  - v2's research: 15.7 deg/s (low; simulator and community compilations).
- Caveats:
  - The solve pays for the separation drag with less induced drag: e 0.965 equivalent,
    which is outside WWII fighters' 0.78-0.93 (a warning at the default).
  - The best sustained speed is 11% above the point mass's. The tool warns above 12%.

### The gap
- 0.85: the tightest sustained turn's speed over the best sustained rate's. Solved, mainly by
  the separation drag, at the separation exponent 16 (coefficient 0.41).
- Origin: **game choice** (author 2026-11-24): the one-circle and two-circle fights ask for
  clearly different speeds. The flight test asks for at least a 10% gap (`test_flight.py`),
  and the tool warns above 0.9.
- Reference: none. The point mass has no gap (0.998).

## Airframe

### Static thrust / weight
- 0.609 (26.4 kN at 4,427 kg). Closed: the static thrust per kg is the ratio times g.
- Origin: **derived:** momentum theory, (FM^2 P^2 2 rho A)^(1/3), with 1,630 hp, a 3.40 m
  disc and an assumed FM of 0.75 (author 2026-11-24: static thrust stays physical, not
  a low-speed acceleration target).
- The low-speed acceleration it gives (straight at the stall speed, empty store) is a readout:
  3.26 m/s^2 real. v2's research estimated 2.73 m/s^2, at military power and 1.1 x stall.
- The tool warns if the implied figure of merit reaches 1.

### Stall angle
- 0.2705 rad = 15.5 deg (game and real). Closed: over the card's standard 15 deg.
- Origin: **guess.** v2's research estimated it: "NACA 45-100 laminar section, clean CLmax
  ~1.4; typical fuselage-reference stall angle 15-17 deg".
- Reference conditions: not found for any type (`aircraft-references.md`: "the stall angle
  of attack was not found").
- Caveat: it shapes the lift curve's peak and the nose's response, not the 1 g stall speed.

### Dive and zoom trade
- 1 g (game 39.24 m/s^2; real 9.81 m/s^2). Closed: times g.
- Origin: **game choice.** The core model trades speed and store at 1 g, "a vertical dive
  or zoom" (`airframe_row`). v2's Vizia had 0.983 g (placeholders scaled from v2's standard
  fighter).
- Reference conditions: not applicable. A real dive or zoom gains or loses speed at
  g sin(angle) plus thrust minus drag.

## Flight terms

### Drag rise: critical Mach
- 0.77: game 524.06 m/s, real 262.03 m/s (586 mph) at the sea-level speed of sound of
  340.3 m/s.
- Origin: **derived** from NACA ACR 4K02 Fig. 12 (A), as read in `p51d-model.md` (not
  re-read for this table).
  - P-51B with the propeller off: C_D flat at about 0.0215 to M 0.68, about 0.030 at
    M 0.735, and about 0.036 at M 0.755.
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

### Separation drag: exponent
- 16. The separation drag is the induced drag times 1 + coefficient x (lift fraction)^16.
  Its coefficient is solved by the gap (0.41 at the P-51D).
- Origin: **game choice** (author 2026-11-24): at 16 the coefficient still moves the gap
  (`ship-design-proposal.md` section 5). At the previous 24 it had saturated, and the gap
  stayed at about 0.845 whatever the coefficient.

### Lift softness: exponent
- 16. The soft minimum of the wing's lift against the g limit.
- Origin: **copied** from v2 (`rules.toml` `lift_softmin_exponent` 16).
- Caveat: it rounds the corner. The corner rate is 4% below the point mass's (33.55 against
  35.0 deg/s real).

## Flaps

The P-51D had no combat flap setting. All three values are a **game choice**, and the flaps
are off (author 2026-11-24). The machinery stays, switchable per type. The tool warns if the
flaps make the full-stick rate fall with speed below corner.

Reference conditions for comparison: the manual (AN 01-60JE-1, B).
- Stall chart (Fig. 24): about +9-10% maximum lift with 20 deg of flap, and about +12% with
  40 deg, both with the gear down. Read off the chart, IAS. My reading, from the ratio of
  the stall speeds squared.
- Flap limits (Fig. 23): 10 deg up to 400 mph IAS, 20 deg to 275, 30 deg to 225, 40 deg to
  180, 50 deg to 165.

### Extra lift
- 0 (off). When on, a fraction of the clean maximum: the tests fly 0.15 (Delta C_L 0.24),
  which is more than the manual's 40-degree landing flap gives.

### Blown back at
- 1.0 x the clean corner speed: fully out to 0.6 x corner, and clean from the corner. This
  keeps the corner and the corner rate the clean airframe's.

### Drag of the extra lift
- 7.54 in the card's units. That is Delta C_D = 1.5 (Delta C_L)^2 with the flaps fully
  used: 1.5 / (rho S / 2m x 66.348). It was set by a sweep on 2026-11-24, when the flaps
  were on.

## Stick and nose

These are control feel, not physics. The stick levels themselves are derived by the game
(`core.stick_table`): full stick is the envelope's top line, and the sustained level holds
energy at the best sustained turn (P-51D 0.881).

### Nose damping
- 0.7.
- Origin: v2's rule value (`rules.toml` `nose_damping`), kept.

### Nose frequency at corner speed
- Game 11.23 rad/s; real 5.62 rad/s.
- Origin: Vizia's card. It comes from v2's `nose_reversal_time` handle (0.394 s game):
  "placeholder: the standard fighter's, scaled as the standard fighter is to the P-51D".
- Reference: not found (no pitch-response data was gathered).

### Fine stick
- 0.5 of the sustained level (0.440 of full).
- Origin: **game choice** (author 2026-11-24).

## Guns

All **copied** from Vizia (`P51D_VALUES`, via v2's `vizia.toml` notes and
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
- Origin: **builder's default** (v2's handles default); Vizia had 586.9 m. Closed: the
  bullet life in whole ticks to fly it when fired at the top speed.
- With the top and muzzle speeds it gives a bullet life of 11 ticks: 0.367 s game, 0.733 s
  real, as before. The tool reads a row's gun range back as the whole life's (637 m).
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

All **copied** from Vizia (`P51D_VALUES`), all game rules. None has a real counterpart.

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

Repository: `lean/game/tuning.py` (`P51D`, `P51D_VALUES`, `DESIGN_INPUTS`),
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
