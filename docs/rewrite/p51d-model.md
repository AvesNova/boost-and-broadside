# The P-51D under the core flight model

The P-51D is the core game's ship. Its row is built from real data through the tuning route
(`lean/game/tuning.py`), with the game's flight terms on and the low-level coefficients solved
so it still flies the real aircraft's headline numbers (core-decisions.md, "Flight model
direction and shared code", 2026-11-24). History: the author's request of 2026-11-23 (a P-51D at
the game's 2x time, comparable with real charts) first produced its pure point mass, the
game terms off; step A1 (2026-11-24) rebuilt it as described here.

Everything here is quasi-steady under `GameConfig()` (coasting, mid-store, no fields), from the
flight code's own functions (`core.steady_rates`: `_wing`, `_setting`, `_lift_coefficient`,
`_rates`), as the doghouse charts and the web tool are. Code: `lean/game/tuning.py`; tests:
`lean/game/tests/test_tuning.py` and `test_flight.py`; charts:
`docs/rewrite/sections/turns/doghouse.py`.

## Time scale

The game plays at 2x real time (v2's `real_scale.EQUIVALENT_FACTOR`). Lengths are real
metres, so in game units speeds are x2, accelerations x4 (gravity 39.24 = 4 x 9.81, normal
gravity), thrust power per kg x8 and turn rates x2. Turn radii, loads in g and the card's
per-metre lift and drag coefficients do not change. Real units below are game values
converted back: speeds, turn rates and Ps halved.

## What the tuning code does (`lean/game/tuning.py`)

Routes to a card row. None changes `core.py`.

- **Airframe** (`airframe_row`): a direct conversion with no fitting. Lift and zero-lift drag
  per metre are rho S C / (2 m); the induced-drag factor is k / (rho S / 2 m) with
  k = 1 / (pi e AR); thrust power is shaft power x propeller efficiency / m; static thrust
  comes from momentum theory, (FM^2 P^2 2 rho A)^(1/3) / m with FM 0.75; the g limit times
  39.24; dive and climb trade at 1 g (a vertical dive or zoom). This is the October fit's
  conversion (`physics-checks/planes.py` `from_real`), at 2x time. With the game terms off
  (`POINT_MASS_TERMS`) the row is the aircraft's **point mass**.
- **Point-mass fit** (`fit_point_mass`): C_D0 and the propeller efficiency set so the point
  mass flies a tested top speed and best climb (the October procedure,
  `physics-checks/real_fit.py`). Neither is measured data: a fit absorbs cooling drag, exhaust
  thrust and the Meredith effect into the two (aircraft-references.md, "CD0").
- **Game row** (`game_row`): the airframe converted with a type's game terms on, then its
  zero-lift drag, thrust power and induced drag (`GAME_SOLVED`) solved so it flies its point
  mass's top speed, best climb and best sustained turn rate (`GAME_TARGETS`). Newton in log
  parameters with finite differences (`_newton`, shared with the handles route), first with
  the flaps off and then with them from there: the flaps' part of the sustained turn is
  lift-limited, so no drag coefficient moves it, and a start whose clean best turn lies below
  it sees no way to the target. Kept as the airframe's: the lift and stall speed, the g limit
  and so the corner speed, the stall angle, the dive and climb trade, and the static thrust
  (no target sees it). The best sustained turn's speed is not solved; the terms are chosen to
  keep it near the point mass's. The P-51D's card row is `game_row(P51D, P51D_TERMS,
  GAME_FIELDS)`; a test holds the card equal to it.
- **Handles** (`handles_row`): v2's handles, as v2's ship files hold them (x2 units). The
  direct fields use v2's closed forms (`solver._flight_card`'s direct ones and
  `weapons.combat_card`). C_max = g / stall^2; a_max = g_limit g; the bullet life is whole
  ticks over the gun range; the reference bullet speed, the spread, drag, shield and armour
  scales, and the shot's cost are also closed form. Three flown handles still mean something
  under the core model and are solved quasi-steadily: cruise speed sets the engine power;
  low-speed acceleration (straight, at the stall speed, empty store) sets the static thrust;
  the settled full-stick coasting turn sets the induced drag. The web tool's sliders go
  through this route (with its fast path: Broyden updates, a coarser speed grid); the base
  row's flight terms, flaps included, are kept.
- `performance(row, plain=False)` gives a row's quasi-steady numbers: stall, corner (closed
  form, and flown: the peak full-stick rate), top speed, best climb (Ps), best sustained
  turn, minimum sustained radius, settled full-stick turn and low-speed acceleration.
  `plain=True` switches off the game terms (any row). `envelope` gives the (stick, speed)
  grid the doghouse and the web tool draw.

How v2's flight model maps to the core model:

| v2 | core model |
|---|---|
| Two engine powers: `engine_power_full_store` drives speed, and the surplus (`engine_power` minus it) charges the store, fading above the engine knee | No charge. Thrust uses the full-store power only. The climb time, engine surplus, coast exchange and knee are inert. |
| Separate dive and climb exchange rates | Their mean. 1 g from an airframe. |
| Store = g x ceiling; spawn at 40%; shots cost store | Two vertical loops at corner speed; spawn at half; shots are free. |
| Nose reversal, stick levels | Control feel, not physics: from `GAME_FIELDS` (v2's Vizia's). Nose damping 0.7 (v2's) kept. |
| Weight paid from the store above 2% | Weight always on the wing. |
| Flight terms shared across types | Per-type card columns (`card.FLIGHT_TERMS`), flaps included. |

## The P-51D's data (`tuning.P51D`)

| Quantity | Value | Source |
|---|---|---|
| Mass | 4,427 kg (9,760 lb), as tested | AAF TSCEP5E-1908 (A) |
| Power | 1,630 hp at 67 in Hg, war emergency rating (decided: the game's one power setting is set for an all-out fight) | the same test |
| Wing area | 21.67 m^2 (233.2 ft^2) | NACA ACR 4K02 (A); was 21.83 |
| Aspect ratio | 5.83 | the October fit (NACA gives 5.815 for the P-51B) |
| C_Lmax, clean | 1.6 | the AAF's power-on value behind its turn calculation; stall-derived values span 1.50-1.70 (aircraft-references.md); was 1.50 |
| Oswald e | 0.8 | assumed (two test points cannot separate e from propeller efficiency) |
| Propeller | Hamilton Standard, 3.40 m | the test report |
| g limit | 8 (decided) | v2's research (forum figure, at 8,000 lb) |
| Stall angle | 15.5 deg | v2's research |
| C_D0, propeller efficiency | 0.01504, 0.821 | `fit_point_mass` to 375 mph and 3,600 ft/min at sea level (A) |

The point mass: stall 101.2 mph, corner speed 286 mph (stall x sqrt 8), and, as a model, a best
sustained turn of 18.99 deg/s at 163.3 mph with radius 220 m.

## The game terms and the solve (`tuning.P51D_TERMS`)

| Term | P-51D | Why |
|---|---|---|
| Drag rise | the zero-lift drag doubles at Mach 0.77 (262 m/s real, `critical_speed` 524.06 game), exponent 20 | NACA ACR 4K02 Fig. 12 (P-51B, propeller off): C_D flat at about 0.0215 to M 0.68, then about 0.030 at M 0.735 and 0.036 at M 0.755 (read off the chart); 1 + (M / 0.77)^20 gives +15% at 0.70, +39% at 0.735, +68% at 0.755 (v2's exponent 12 at the same point would add 13% at M 0.65, where the figure is flat). At the game's speeds it is negligible: +0.013% at the sea-level top speed (M 0.49) |
| Separation drag | 0.1, exponent 8 (the induced drag +10% at the wing's maximum lift) | larger values move the best sustained turn's speed off the point mass's: with the flaps off, +4% at 0.2, +8% at 0.3, +22% at v2's 1.0, which also needs e 1.28 |
| Soft minimum, lift against the g limit | exponent 16 | v2's |
| Soft minimum, thrust against static thrust | exponent 8 | much less soft than v2's 2 (which took 10% of the thrust near the best-climb speed) |
| Automatic flaps | +15% maximum lift below 1.3 x stall, fading to clean at 1.6 x stall; flap drag 0.2 (Delta C_L)^2 | next section |
| Nose damping | 0.7 | v2's, kept |

Solved (`GAME_SOLVED`), as airframe equivalents:

| | point mass | game row |
|---|---|---|
| C_D0 | 0.01504 | 0.01489 |
| propeller efficiency | 0.821 | 0.808 |
| Oswald e | 0.8 | 0.900 |

The soft minimum of lift and the separation drag both lower the turn near the lift limit (the
soft minimum alone takes 1.8% of the full-stick rate at 150 m/s game, 0.04% at 110 and 3.4% at
200), so the induced drag falls (e 0.900) to hold the best sustained rate. The other two move
by 1 to 2%.

## Automatic flaps (`core._wing`)

The wing's maximum lift coefficient rises at low speed as if the flaps were exactly as far out
as useful. No control: it is part of what full stick means at low speed.

- **Lift.** The maximum lift coefficient is the clean one times 1 + `flap_lift` below
  `flap_full_speed_ratio` x the clean stall speed, fading by a smootherstep in speed (no kink:
  the slope is continuous) to the clean one at `flap_clear_speed_ratio` x the stall speed.
  Everything that reads the wing's maximum reads it at the speed: the lift budget, the lift
  curve (its peak at the stall angle), the nose's lift slope, and the separation drag (which
  rises toward the flapped maximum).
- **Drag.** The flaps' drag is `flap_drag_scale` x the induced-drag factor times the square of
  the lift coefficient beyond the clean maximum. It is zero while the clean wing suffices
  (straight flight, gentle turns), so the flaps cost only when their lift is used.
- **Cost in the step.** One smootherstep of speed per half-kick (the flaps are held through
  the half-kick at its starting speed), a reciprocal, and in each drag evaluation a square
  root, a subtraction, a clamp and a square: the step's operations below.
- **Per type.** `flap_lift`, `flap_full_speed_ratio`, `flap_clear_speed_ratio` (multiples of
  the clean stall speed, so they carry across time scales and the web tool's stall slider),
  `flap_drag_scale`. Off: `flap_lift` 0.

**The P-51D's values are a game choice:** the P-51D had no combat flap setting. They are kept
modest: +15% (Delta C_L 0.24, about what 10-15 degrees of a plain flap gives) below 1.3 x stall
(131.5 mph), fading to clean by 1.6 x stall (161.9 mph), with Delta C_D = 0.2 (Delta C_L)^2
(0.0115 at the full bonus). The fade ends below the best sustained turn's speed, so the flaps
change the tightest turn and not the best rate. The 1 g stall with the flaps out is 94.3 mph
(clean 101.2).

What they do to the P-51D (the same solved row, flaps off against on):
- tightest sustained turn 202 m at 136.7 mph, against 223 m at the best rate's speed (165.5
  mph) without them: 9.5% tighter at a speed 17% lower;
- the best sustained rate and its speed do not change;
- the maximum instantaneous rate is nearly flat from 1.4 to 1.5 x stall as the flaps retract
  (17.76 deg/s at 128 m/s game, 17.66 at 136, rising again from 138);
- the flap drag costs speed in a slow full-stick turn (at 1.35 x stall, d speed / dt 3.33
  against 3.76 m/s^2 game without it) but not the sustained radius: the P-51D has power to
  spare there, so its slow sustained turn is lift-limited.

Gentler fades fail the separation: with +15% fading from 1.0 to 1.6 x stall the radius is flat
(224 m at 115 and at 145 m/s game) with no clear minimum; +10% fading from 1.3 to 1.6 gives a
radius 4.6% tighter. A fade ending above the best rate's speed (1.8 x stall) moves the best
rate itself onto the flaps' hump.

## Comparison with real figures (real units)

"Before": the card row until 2026-11-24 (the October point mass: 21.83 m^2, C_Lmax 1.50, the
game terms off). "Point mass": the referenced airframe above, game terms off. "Game row": the
card now. Sustained figures are coasting, total energy rate >= 0. The radii at multiples of
the stall speed use each row's own clean stall speed (sustained; full stick where different).

| Quantity | Published figure (source) | Before | Point mass | Game row |
|---|---|---|---|---|
| Stall speed, clean, 1 g | 100 mph at ~10,100 lb (Loftin); 101-102 mph at 9,000-9,500 lb (author's set 1); about 100.5 mph IAS at 9,800 lb (AN 01-60JE-1) | 104.1 mph | 101.2 mph | 101.2 mph (94.3 with the flaps) |
| Top speed, sea level | 375 mph at 67 in Hg, 4,427 kg (flight test) | 375.1 mph | 375.0 | 375.0 |
| Best climb, sea level | 3,600 ft/min at 67 in Hg (flight test); 3,410 (NAA calculation) | 3,599 ft/min at 142 mph | 3,600 at 142 | 3,600 at 142 |
| Corner speed | none published; stall x sqrt(g limit) (derived) | 294.4 mph | 286.1 | 286.1 (the flown peak at 286) |
| Turn rate at corner | derived only | 34.0 deg/s | 35.0 | 32.2 |
| Best sustained turn | no test value. Models: the October fit's 18.6 deg/s at 167 mph, r 231 m; author's set 2 (E-M derived) 360 in 18.2 s, r 215 m (19.8 deg/s); simulator compilations 15.7 deg/s | 18.63 deg/s at 169.4 mph, r 233 m | 18.99 at 163.3, r 220 | 18.99 at 165.5, r 223 |
| Tightest sustained turn | none found | 233 m at 169.3 mph | 220 m at 163.2 | 202 m at 136.7 |
| Tightest / best-rate speed | | 0.998 | 0.998 | 0.826 |
| Radius at 1.2 x stall | | 292 m (125 mph) | 276 m (121 mph) | 218 m (121 mph) |
| Radius at 1.5 x stall | | 239 m (156 mph) | 226 m (152 mph) | 220 m (152 mph) |
| Radius at 2 x stall | AAF calculation: 883 ft (269 m) at 4 g, 10,000 ft, P-51B (not comparable at sea level) | 331 m sustained, 225 m full stick (208 mph) | 316 m, 212 m (202 mph) | 300 m, 219 m (202 mph) |
| Level acceleration at the stall speed | 2.73 m/s^2 (v2 research, estimated) | 3.74 m/s^2 | 3.83 | 3.73 |

Reading:
- Top speed, climb and best sustained rate are the solve's targets and match the point mass
  to 1e-6. The best sustained speed is 1.3% above the point mass's (not solved), between it
  and the October and set-2 models' 166-167 mph.
- The corner speed is exact (closed form from C_Lmax and the g limit). The corner rate is 8%
  below the point mass's: the soft minimum of lift against the g limit (16) holds the full
  stick below the hard limit near the corner. Matching it would need a nominal g limit above
  8 (the soft minimum reaches the nominal value at high speed), so it is left (morning notes).
- With the flaps, the radius at 1.2 x stall is 21% tighter than the point mass's, and the
  tightest sustained turn sits 17% slower than the best rate: a one-circle fight now asks for
  a different speed than a two-circle fight.

## Charts

The P-51D's doghouse, its card row (game terms on, flaps), drawn from `tuning.envelope`. In real
units speeds, turn rates and Ps are halved; radii and loads are unchanged. Marks: the corner
(circle), the best sustained turn (square), the tightest sustained turn (triangle).

| Real units (1x) | Game units (2x) |
|---|---|
| ![P-51D, core, real units](doghouse-p51d-core-real.png) | ![P-51D, core, game units](doghouse-p51d-core.png) |

Dark variants: `doghouse-p51d-core-dark.png`, `doghouse-p51d-core-real-dark.png`; the
comparison panels (rate, radius, sustained 360 time against speed):
`doghouse-comparison-core[-dark].png`.

    for theme in light dark; do THEME=$theme OMP_NUM_THREADS=1 PYTHONPATH=.:src \
        uv run --no-sync python docs/rewrite/sections/turns/doghouse.py; done

The vizia_plain and Vizia charts (`doghouse-vizia-plain-core*.png`, `doghouse-vizia-core*.png`)
and `sections/turns/doghouse-p51d-vizia-core*.png` are records drawn before G1 (2026-11-23),
when Vizia left the card; they show the P-51D's earlier point mass.

## Tests

`lean/game/tests/test_tuning.py`:
- the card's row is `game_row(P51D, P51D_TERMS, GAME_FIELDS)` (the solved coefficients to 1e-9);
- the point mass flies the flight test (375 mph and 3,600 ft/min to 1e-6), and `fit_point_mass`
  returns P51D's C_D0 and propeller efficiency;
- the game row flies its point mass's top speed, best climb and best sustained rate (1e-6), its
  best sustained speed within 1.5%, the same stall and corner speeds (the flown peak within
  0.1%), and differs from its plain version (the terms are on);
- the handles route recovers the row from its flown handles; another time scale flies the
  same aircraft; the scripted agent's references are the flown numbers (P-51D: quarter turn
  2.37 s, store turn 2.56 rad).

`lean/game/tests/test_flight.py`:
- every card type's tightest sustained turn is at least 10% slower than its best sustained rate,
  which is below its corner (P-51D: 122.25, 147.95 and 255.7 m/s game, ratio 0.826);
- the flaps: the flapped maximum below the full speed, clean above the clear speed, never
  rising with speed between, with a continuous slope;
- the flaps' drag only for lift beyond the clean wing (straight flight slow pays none; full
  stick slow bleeds more with it than without).

The web tool (`lean/tool`) reads the same functions (`tuning.envelope`, `performance`,
`handles_row`); its tests pass, and a live server answered a stall-speed change (80 m/s) in
1.2 s with the flaps following the stall speed.

## Record: Vizia's climb modifications (2026-11-23)

Kept as the record of the request of 2026-11-23; Vizia and vizia_plain left the card at the
game sign-off (G1).

v2 changed the real-scale cards so the store would refill faster. The sources are v2's builder
(`bnb.game.real_scale` before b4dbbdee), `docs/explanation/game/real-scale-pacing.md`, the ship
files and the solver. Each modification, where it is set, its unmodified value and what the core
model does with it:

| Modification | Where set | Vizia's value (x2 units) | Unmodified (physical) | Under the core model |
|---|---|---|---|---|
| Climb pacing x4, set-wide: every researched climb rate times 4 | `real_scale_balance.json` "_rules" `climb_multiplier`, applied in `real_scale.target_handles` (pacing doc, "The choice") | the P-51D's 15.2 m/s became 60.8 m/s; its climb time 2,081 -> 520 s | 15.2 m/s (researched, estimated; climb time 2,081 s x2, 4,163 s real) | inert: no store charging |
| Vizia's own climb time | `vizia.toml` [handles] `climb_time`, the author's design (96d16d61, 2026-11-12): 600 s x1, 300 s x2 | 300 s: an initial charge of g x 105.5 m/s real, 6.9 times the researched climb | 2,081 s | inert |
| The engine surplus that pays for the charge | the solver: `engine_power = engine_power_full_store + m a_0 u_cruise` (`solver._flight_card`; direct charge since game v8) | 10,292.6 - 2,038.5 = 8,254 W/kg (Yongma 7,423, Jujak 9,178) | g x 15.2 m/s = 1,193 W/kg (x2) | dropped: thrust uses the full-store power only. vizia_plain sets `engine_power = engine_power_full_store` |
| Coast exchange (the store's charge while coasting) | `coast_exchange_scale` 23.27 on v2's card (rules `coast_exchange` 1.09) | 25.4 m/s^2 | the researched climb's | not in lean's card; inert under direct charge |
| Dive and climb exchange rates | the solver, from the dive and zoom acceleration handles (placeholders: the standard fighter's, scaled to the P-51D) | 4.41 / 3.46 (mean 0.983 g) | 1 g each (a vertical dive or zoom; also the handles' mean, 39.24 m/s^2) | used, as the mean: vizia_plain sets 1 g |
| Engine knee: the charge fades above half the store | rules.toml `flight.engine_knee` 0.5, `engine_knee_scale` 1 | 0.5 | not a climb boost: the pacing study left it at 0.5 | inert |
| Store size | g x ceiling (researched 12,680 m, not paced) | 497,563 J/kg | the same | replaced: two loops at corner speed |
| Spawn energy | the standard fighter's 40% of the store (placeholder) | 199,025 J/kg | | replaced: half the store |
| The shot's cost, tied to the climb charge | `shots_per_store`: continuous fire costs 14.5 x the researched initial climb charge | 575 J/kg a shot | | replaced: shots are free |
| Store fade band | rules.toml `flight.store_fade_band` 0.02 | 2% | | replaced: 5% smootherstep |

**Finding: under the core model, Vizia's climb modifications are already inert.** The core model
uses only the full-store power, which v2's solver set to hold the cruise handle; under core,
Vizia's top speed is 364 mph, exactly the researched value. The surplus, the climb time, the
coast exchange, the knee, the store, the spawn store and the shot's cost are all unused or
replaced. The only change that reaches the flight is the dive and climb trade: 0.983 g becomes
1 g. vizia_plain coasts exactly as Vizia does (`test_vizia_plain_flies_as_vizia_when_coasting`),
and its doghouse is Vizia's apart from the dive and climb lines.

Two Vizia changes are not climb-related, but they still separate it from the P-51D: the g limit
9 against the researched 8, and the sustained-turn handle, raised from 15.7 to 19.29 deg/s for
flight realism (game v8 to v11). Under the core model that handle is no longer met: Vizia's
settled full-stick turn is 12.5 deg/s, because the core model keeps the weight on the wing.
Re-solving Vizia's handles under core (`handles_row`) to fly 19.29 deg/s would need an induced
drag scale of 0.155, an e AR near 10: not physical.

