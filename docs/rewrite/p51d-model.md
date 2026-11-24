# The P-51D under the core flight model

The P-51D is the core game's ship. Its row is built from real data through the tuning route
(`lean/game/tuning.py`), with the game's flight terms on and the low-level coefficients solved
so it still flies the real aircraft's headline numbers (core-decisions.md, "Flight model
direction and shared code", 2026-11-24). History: the author's request of 2026-11-23 (a P-51D at
the game's 2x time, comparable with real charts) first produced its pure point mass, the
game terms off; step A1 (2026-11-24) rebuilt it with the game terms on; later on 2026-11-24
the flaps became demand-scheduled and the row derived at load (author, "Charts"); last on
2026-11-24 the stick levels became fractions of the most turn the ship can give, the flaps
were switched off and the separation drag raised (author; "The stick", "Separation drag").
The ship design system (option C, core-decisions.md "Ship design") then made the row a
reference ship plus design inputs, with the best-climb speed and the gap among the solved
targets (section "Design inputs").

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

Ship design follows option C of `ship-design-proposal.md` (core-decisions.md, "Ship design",
2026-11-24). There are two layers: a reference ship built from the documents, then design
inputs that start at the reference's values. One function, `design_row`, turns the design
inputs into the card row, for the card and for the web tool. None of it changes `core.py`'s
step.

**The card is derived, not stored.** `lean/game/card.py` holds no numbers for a real
aircraft. On first use of `card.CARD`, `tuning.load_card` makes every type's row from its
aircraft (`REAL_AIRCRAFT`: the data, the flight test and the documented design values) through
`real_row`: `design_row` of its reference under its profile (`reference_in`; the P-51D's is
"airframe and engine", below). The row is cached in memory and on disk
(`~/.cache/bnb-card/<type>-<key>.json`, or `$BNB_CARD_CACHE`). The key is a hash of the
aircraft and of the flight model's source (`core.py`, `card.py`, `tuning.py`, `contract.py`),
so a physics change or a change to the design-input table re-solves by itself. A cold solve
takes about 5 s on the laptop; a cached load about 1 ms.

**Layer 1, the reference** (`reference_of`, `reference(name)`):
- `airframe_row`: an aircraft's point mass, a direct conversion with no fitting. Lift and
  zero-lift drag per metre are rho S C / (2 m). The induced-drag factor is k / (rho S / 2 m)
  with k = 1 / (pi e AR). Thrust power is shaft power x propeller efficiency / m. Static
  thrust comes from momentum theory, (FM^2 P^2 2 rho A)^(1/3) / m with FM 0.75. The g limit
  is times 39.24, and dive and climb trade at 1 g. The game terms are off (`POINT_MASS_TERMS`).
  This is the October fit's conversion (`physics-checks/planes.py` `from_real`), at 2x time.
- `fit_point_mass`: C_D0 and the propeller efficiency are set so that the point mass flies
  the tested top speed and best climb (the October procedure, `physics-checks/real_fit.py`).
  Neither number is measured data. The fit absorbs cooling drag, exhaust thrust and the
  Meredith effect into the two (aircraft-references.md, "CD0").
- Each design input's reference value is its documented one (`RealAircraft.values`: the
  flight test's numbers, the gap, the flight terms, the game choices). Any other input takes
  what the point mass has or flies (its `measure`): the stall speed, the g limit, the static
  thrust, the stall angle, the trade and the best sustained rate.

**Layer 2, the design** (`DESIGN_INPUTS`, `design_row`, `design_of`):
- `DESIGN_INPUTS` is one ordered table, and everything else is generated from it. Each entry
  holds:
  - the input's name, label and group;
  - the card field it mainly sets;
  - `measure(row, performance)`: the value a row has or flies;
  - `apply(value, design)`: the closed form for a closed input, or None for a solved one;
  - a `first_guess`, the reference's provenance (`source`), the unit kind and the slider
    range.
- Every card field is set exactly once (`check_inputs`).
- `design_row(reference, design, start=None, fast=False)` works in this order:
  1. the closed inputs, each by its `apply`;
  2. every solved input, met by one Newton solve in the logs of the fields they set
     (`_newton`: a finite-difference Jacobian, pseudo-inverse steps clamped to 0.5, to 1e-7
     in log). The solve starts from `start` (a row), or from the reference's point mass with
     the table's first guesses where the point mass has no usable value.
- The web tool's fast path uses Broyden updates, to 1e-5, warm from the preset's last solve
  (its row and Jacobian, `Solved.jacobian`; the clean stage then skipped), else from the
  preset's row; finite differences again after a step that does not reduce the residual.
- A design with flaps is solved clean first, then with them.
- `design_row` returns the row with the solve's residual, iterations, condition number and Jacobian
  (`Solved`). A solve that does not converge raises: that is the one refusal.
- `design_of(row)` is the inverse: every input's measure.

**The flown measures** (`performance(row, plain=False)`) are the row's quasi-steady numbers:
- stall;
- corner (closed form, and flown: the peak full-stick rate);
- top speed;
- best climb and its speed;
- best sustained turn;
- tightest sustained turn and the gap;
- settled full-stick turn;
- low-speed acceleration;
- how far the full-stick rate falls below corner.

Every extremum is refined below the speed grid: a 33-point local grid, then a parabola, with
the value flown at its vertex. The tightest turn is also checked exactly at the kink where the
sustained turn leaves full stick. So the numbers agree between a 0.05 and a 1 m/s grid to
1e-6, and the solve's finite differences see slopes, not grid steps. `plain=True` switches off
the game terms (any row). `envelope` gives the (stick, speed) grid that the doghouse and the
web tool draw.

**Plausibility** (`implied`, `PLAUSIBILITY`, `warnings`): the row is read back as the
reference airframe's physical quantities. Mass, wing area, aspect ratio, shaft power and
propeller diameter are held. That gives C_Lmax, C_D0, Oswald e, the propeller efficiency, the
static figure of merit, static thrust / weight, the g limit over the structure's at this mass,
and the held wing and power loadings. Each check is data (quantity, bounds, "limit" or
"range", reason, the input it points at), from section 7 of the proposal. Warnings never
refuse.

### Changing the design inputs

To add, remove or swap an input, edit `DESIGN_INPUTS` only:
- a closed input gives `apply`; a solved input gives a `measure` that `performance()` reports
  or that can be computed from the row, and the field it mainly moves;
- every card field must still be set once.

`design_row`, the reference, the card and the tool's sliders follow from the table. A
documented reference value goes in the aircraft's `values`; without one, the input takes its
point mass's measure. The table's version (`inputs_version`) changes with it, so saved tool
presets from another table load as fixed ships. `test_an_input_is_swapped_by_editing_the_table_only`
puts the corner speed (solved for the maximum acceleration) in place of the g limit on a copy
of the table, and the solve meets it.

How v2's flight model maps to the core model:

| v2 | core model |
|---|---|
| Two engine powers: `engine_power_full_store` drives speed, and the surplus (`engine_power` minus it) charges the store, fading above the engine knee | No charge. Thrust uses the full-store power only. The climb time, engine surplus, coast exchange and knee are inert. |
| Separate dive and climb exchange rates | One trade, 1 g from an airframe. |
| Store = g x ceiling; spawn at 40%; shots cost store | Two vertical loops at corner speed; spawn at half; shots are free. |
| Nose reversal, stick levels | Nose frequency: control feel, v2's Vizia's; nose damping 0.7 (v2's) kept. Stick levels: derived from the flight (section "The stick"); only the fine level's fraction of the sustained one is a field (`fine_stick_fraction`, 0.5). |
| Weight paid from the store above 2% | Weight always on the wing. |
| Flight terms shared across types | Per-type card columns (`card.FLIGHT_TERMS`), flaps included. |
| A stored reference bullet speed | Derived by the game (`core._derived_characteristics`): a bullet fired at cruise, 300 m out. |

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
| g limit | 8 (decided), rated at 8,000 lb: 6.6 g at the tested weight | v2's research (forum figure, F) |
| Stall angle | 15.5 deg | v2's research |
| C_D0, propeller efficiency | 0.01504, 0.821 | `fit_point_mass` to 375 mph and 3,600 ft/min at sea level (A), run when the reference is made (`P51D` holds its start) |

The point mass: stall 101.2 mph, corner speed 286 mph (stall x sqrt 8), and, as a model, a best
sustained turn of 18.99 deg/s at 163.3 mph with radius 220 m.

## The card: the airframe with a 2,200 hp engine (author 2026-11-24 night)

The P-51D's card is the documented airframe with a 2,200 hp engine (late-war and post-war
engines; the P-51H had about 2,200 hp), under the "airframe and engine" profile (C_D0 and the
shaft power closed inputs, nothing solved; `P51D_VALUES["shaft_power"]`). Its reference is the
documents' row (the airframe fitted to the 1,630 hp flight test below: C_D0 0.0160), then the
aircraft's own shaft power; the propeller efficiency is held at the documents' (0.875). The
documents profile and its solve are unchanged and stay the base. Measured on the card (real
units):

| | documents (1,630 hp) | card (2,200 hp) |
|---|---|---|
| top speed | 375 mph | 416 mph |
| best climb | 3,600 ft/min at 168 mph | 4,882 ft/min at 191 mph |
| best sustained | 17.85 deg/s at 173 mph | 19.97 deg/s at 188 mph |
| corner | 286 mph, 33.5 deg/s | 286 mph, 33.6 deg/s |

The one warning on the card is the structure (the 8 g limit, 1.22 x its 6.6 g at 9,760 lb).

## Design inputs (`tuning.DESIGN_INPUTS`, the P-51D's `P51D_VALUES`)

Performance inputs. All are game units in the code; the real values are listed here.

| Input | Sets | How | P-51D | Source |
|---|---|---|---|---|
| stall speed (1 g) | `max_lift_coefficient` | closed, g / v^2 | 101.2 mph | the airframe, C_Lmax 1.6 (B) |
| g limit | `max_acceleration` | closed, n g | 8 | forum (F); the author's choice |
| top level speed | `zero_lift_drag` | solved | 375 mph | flight test (A) |
| best climb rate | `engine_power` | solved | 3,600 ft/min | flight test (A) |
| best-climb speed | `thrust_softmin_exponent` | solved | 165 mph | flight test (A), `p51d-drag-check.md` |
| best sustained turn rate | `induced_drag_scale` | solved | 18.99 deg/s | the point mass (model, e 0.8 assumed) |
| the gap: tightest over best-rate sustained speed | `separation_drag` (at exponent 16) | solved | 0.85 | design (author 2026-11-24) |

The other inputs are closed forms or short circuits:
- **airframe:** static thrust / weight 0.609 (momentum theory, FM 0.75); stall angle
  15.5 deg; the dive and zoom trade, 1 g;
- **flight terms:** critical Mach 0.77 and drag-rise exponent 20 (NACA ACR 4K02 Fig. 12);
  separation exponent 16; lift softness 16 (v2's);
- **flaps:** off (extra lift 0, blown back at 1 x the clean corner speed, drag 1.5
  (Delta C_L)^2 when on);
- **stick and nose:** v2's Vizia's, with the fine stick at half of the sustained level;
- **guns, shield, head-on, vision:** v2's Vizia's, in design units (gun range 600 m from the
  top speed, vision 1,000 m, shield recharge 24 /s after 3 s, head-on angle 45 deg).

Not inputs, read out:
- corner speed and rate, and all radii;
- the best sustained speed and the settled full-stick turn;
- the low-speed acceleration (the static thrust stays physical);
- the stick levels and the store.

The solve at the P-51D, before (the row until 2026-11-24's design system: separation drag
0.5 at exponent 24, thrust softness 8, three targets) and after, real units. Measured with
today's `performance`. Coefficients as airframe equivalents (`implied`):

| | before | after |
|---|---|---|
| top speed | 375.0 mph | 375.0 mph |
| best climb, at | 3,600 ft/min at 139.9 mph | 3,600 ft/min at 165.0 mph |
| corner | 286.0 mph, 33.54 deg/s | 286.0 mph, 33.55 deg/s |
| best sustained | 18.99 deg/s at 179.1 mph, r 241.5 m | 18.99 deg/s at 181.6 mph, r 244.9 m |
| tightest sustained | 226.5 m at 151.8 mph | 224.4 m at 154.4 mph |
| the gap | 0.848 | 0.850 |
| settled full-stick turn | 17.06 deg/s | 17.62 deg/s |
| low-speed acceleration | 3.70 m/s^2 | 3.26 m/s^2 |
| C_D0 | 0.0148 | 0.0158 |
| propeller efficiency | 0.802 | 0.859 |
| Oswald e | 0.952 | 0.965 |
| thrust softness | 8 | 2.83 |
| separation drag | 0.5 at exponent 24 | 0.41 at exponent 16 |

At 2.83 the soft thrust limit takes 5.4% of P / v at the best-climb speed. The drag check's
3.7 gave 160 mph against the test's 165. Two warnings fire at the default, the same two as
before:
- Oswald e 0.965, outside 0.78-0.93;
- the 8 g limit, 1.22 x the structure's 6.6 g at 9,760 lb.

## The stick (`core._setting`, `core.stick_table`)

The author's design (2026-11-24):

- **Full stick is the most turn the ship can give at that speed**: exactly the envelope's top
  line. The lift budget is the wing's lift against the g limit, one soft minimum (exponent 16:
  the true corner, rounded); after the 1 g held, the most lateral acceleration left is
  sqrt(budget^2 - held^2), and full stick takes all of it. Before, the stick's command was
  itself soft-minimised against what was left (a soft minimum of two nearly equal values),
  which cost a factor 2^(-1/16) = 0.958 along the whole g-limit and lift-limit lines.
- **Every level is a fraction of that same maximum** at the current speed: the same shape as
  full stick (lift-limited below corner, g-limited above, the same rounded corner). With the
  flaps on, the maximum is the full-flap wing's.
- **The sustained level is derived**, per type, with no stored copy: the stick that holds
  energy at the best sustained turn (`core._best_sustained_turn`, at every build of the
  constants), so the sustained level passes exactly through the best sustained point.
- **The fine level** is `fine_stick_fraction` of the sustained one: a per-type field, default
  0.5 (a rule the author may change).

The P-51D, before (the flapped row, 0.1 separation drag at exponent 8, stick levels 0.210 /
0.700 of the lift command) and after (flaps off, 0.5 at exponent 24, levels 0.448 / 0.895 of
the top line):

| | Before | After |
|---|---|---|
| Full stick at 1.5 x corner speed | 7.630 g (of sqrt(63) = 7.937) | 7.937 g |
| Full stick at 0.6 x corner speed | 3.083 g (flaps out) | 2.701 g (of 2.701, clean) |
| Corner rate | 32.24 deg/s | 33.54 deg/s |
| Sustained, fine level | 0.700, 0.210 | 0.895, 0.448 (derived; half) |
| Best sustained | 18.99 deg/s at 158.4 mph, r 214 m | 18.99 deg/s at 179.1 mph, r 242 m |
| Tightest sustained | 202 m at 135.0 mph | 226 m at 151.8 mph |
| Tightest / best-rate speed | 0.852 | 0.848 |
| Settled full-stick turn | 17.11 deg/s at 134.8 mph | 17.06 deg/s at 150.9 mph |
| Solved C_D0, propeller efficiency, e | 0.01503, 0.820, 0.819 | 0.01480, 0.802, 0.952 |
| Scripted references: sustained radius, store's turn, sustained stick speed (game) | 213.6 m, 2.30 rad, 305.7 m/s | 241.5 m, 2.40 rad, 270.4 m/s |

The corner rate rises 4% (the old tax is gone); the slow turns lose the flaps (full-stick
radius at 1.2 x stall 276 m, was 218 m).

The sustained level above corner speed is now its fraction of the g limit (0.895 x 8 g), not
the g limit: before, any level reached the g limit above corner / sqrt(level). The scripted
agent's `sustained_stick_speed` keeps its formula (corner / sqrt(sustained level), now
270.4 m/s game) as the reference of its level cap, but above it the sustained level no longer
turns as hard as full stick: the cap now costs some turn (BFM-CMA2's values were searched on
the old stick).

**Separation drag capped at the wing's maximum.** Flying steadily the load never passes the
wing's maximum lift, but in a transient the nose's lagging lift and the vertical held can ask
for up to twice it; with exponent 24 that multiplied the separation drag by up to 2^12 and
stopped a ship dead in one tick (found by the scripted competence test). The separation term
now stops at the maximum (`(load / max^2).clamp_max(1)`); the quasi-steady numbers are
unchanged (checked: identical to the digit).

## Separation drag

The author (2026-11-24): the separation drag was too small. With the flaps off and the new
stick, 0.1 at exponent 8 leaves the best sustained turn at full stick, the same point as the
tightest (speed ratio 1.000). Each row below is the P-51D re-solved (`real_row`: the three
targets held, so every row flies 375 mph, 3,600 ft/min and 18.99 deg/s best sustained), real
units, coasting at the spawn store; the propeller efficiency and Oswald e are the solved
coefficients as airframe equivalents. Speed ratio: the tightest sustained turn's speed over
the best sustained rate's (the test asks at most 0.9).

| Separation drag | Exponent | Flaps | Best sustained: speed (mph), rate (deg/s), radius (m) | Tightest sustained: speed (mph), radius (m) | Speed ratio | Sustained level | C_D0 | Propeller efficiency | Oswald e |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | off | 176.2, 18.99, 238 | 162.7, 221 | 0.923 | 0.913 | 0.01452 | 0.778 | 1.256 |
| 0.1 | 8 | off | 163.3, 18.99, 220 | 163.3, 220 | 1.000 | 1.000 | 0.01489 | 0.809 | 0.895 |
| 0.15 | 8 | off | 164.5, 18.99, 222 | 163.3, 220 | 0.993 | 0.992 | 0.01482 | 0.803 | 0.943 |
| 0.2 | 8 | off | 169.4, 18.99, 229 | 163.0, 220 | 0.962 | 0.956 | 0.01476 | 0.798 | 0.984 |
| 0.25 | 8 | off | 173.4, 18.99, 234 | 162.4, 221 | 0.937 | 0.930 | 0.01471 | 0.795 | 1.018 |
| 0.3 | 8 | off | 176.6, 18.99, 238 | 161.7, 221 | 0.915 | 0.910 | 0.01468 | 0.792 | 1.047 |
| 0.35 | 8 | off | 179.4, 18.99, 242 | 160.8, 221 | 0.897 | 0.894 | 0.01465 | 0.789 | 1.072 |
| 0.5 | 8 | off | 185.9, 18.99, 251 | 158.2, 223 | 0.851 | 0.858 | 0.01458 | 0.784 | 1.136 |
| 0.75 | 8 | off | 193.4, 18.99, 261 | 153.7, 225 | 0.794 | 0.821 | 0.01450 | 0.777 | 1.215 |
| 1 | 8 | off | 198.9, 18.99, 268 | 149.5, 228 | 0.751 | 0.795 | 0.01445 | 0.773 | 1.278 |
| 1.5 | 8 | off | 206.7, 18.99, 279 | 142.4, 234 | 0.689 | 0.762 | 0.01437 | 0.767 | 1.376 |
| 2 | 8 | off | 212.5, 18.99, 287 | 143.3, 240 | 0.674 | 0.740 | 0.01432 | 0.763 | 1.454 |
| 0.3 | 16 | off | 177.6, 18.99, 239 | 158.4, 222 | 0.892 | 0.904 | 0.01478 | 0.800 | 0.968 |
| 0.4 | 16 | off | 180.4, 18.99, 243 | 155.7, 224 | 0.864 | 0.888 | 0.01475 | 0.797 | 0.990 |
| 0.5 | 16 | off | 182.6, 18.99, 246 | 153.2, 225 | 0.839 | 0.876 | 0.01472 | 0.795 | 1.008 |
| 1 | 16 | off | 189.5, 18.99, 256 | 150.2, 233 | 0.792 | 0.839 | 0.01465 | 0.789 | 1.068 |
| 0.5 | 24 | off | 179.1, 18.99, 242 | 151.8, 226 | 0.848 | 0.895 | 0.01480 | 0.802 | 0.952 |
| 1 | 24 | off | 183.8, 18.99, 248 | 155.1, 232 | 0.844 | 0.869 | 0.01475 | 0.797 | 0.989 |
| 1 | 32 | off | 180.1, 18.99, 243 | 157.6, 231 | 0.875 | 0.889 | 0.01481 | 0.802 | 0.947 |
| 2 | 32 | off | 183.7, 18.99, 248 | 160.3, 236 | 0.873 | 0.870 | 0.01477 | 0.799 | 0.974 |
| 0.1 | 8 | +15% | 158.4, 18.99, 214 | 135.0, 202 | 0.852 | 0.888 | 0.01503 | 0.820 | 0.819 |
| 0.25 | 8 | +15% | 160.6, 18.99, 217 | 135.9, 204 | 0.846 | 0.874 | 0.01492 | 0.811 | 0.878 |

Reading:
- At exponent 8 (the only one asked for) no coefficient gives both: a 10% gap needs about
  0.35, where the solve already needs e 1.07; e stays below 1 only up to about 0.22, where the
  gap is about 5%. The solver pays for the separation drag with less induced drag.
- A steeper exponent concentrates the drag near the maximum lift, where the tightest turn
  flies, and leaves the best-rate turn (at 0.9 of the lift) nearly alone: at 24, 0.5 gives a
  ratio of 0.848 with e 0.952; at 16 and 32 the margins are smaller.
- **Then: 0.5 at exponent 24** (judgement; superseded by the design system, section "Design
  inputs": the gap is now an input that solves the coefficient at exponent 16). Cost: the best sustained speed moves 9.7% above the point mass's (179.1
  against 163.3 mph), so `test_card_rows_meet_their_targets` now allows 12% (was 5%) and
  holds e <= 1 and the propeller efficiency in 0.7-0.9.
- For comparison, the flaps on (+15%) with 0.1 at exponent 8 under the new stick: ratio 0.852
  with e 0.819, the most plausible coefficients in the table.

## Automatic flaps (`core._wing`)

Scheduled by stick demand, as an F-16's automatic flaps are: the flaps come out when the stick
asks for more lift than the clean wing gives, at any speed up to the blow-back speed. No
control: they are part of what the stick means. Author's rules (core-decisions.md, "Charts"):
smooth to the curvature (C2) in speed and in demand, and the turn rate never falls with speed
below corner (the speed-scheduled flaps of before, which retracted between 1.3 and 1.6 x stall,
made a shelf there in the doghouse).

- **The bonus and its blow-back.** The full-flap maximum is the clean maximum times
  1 + `flap_lift` up to (1 - `card.FLAP_BLOWBACK_BAND`) x the blow-back speed, falling by a
  smootherstep in speed to the clean maximum at the blow-back speed itself,
  `flap_blowback_ratio` x the clean corner speed. The band is 40% (shared): the bonus's fall,
  steeper with a narrower band, must not turn the full-stick rate down.
- **The stick.** A stick level is a fraction of the most lateral acceleration the full-flap
  wing gives at that speed (section "The stick"); full stick asks for all of it.
- **Deployment.** The demanded lift coefficient: 1 g held (whatever the power) and the stick's
  fraction of the full-flap maximum across it, sqrt((g / v^2)^2 + (stick x full)^2). The flaps
  deploy by a smootherstep from `1 - card.FLAP_DEMAND_BAND` (90%) of the clean maximum to all of
  it, so a demand the clean wing gives flies clean (straight above the stall speed, gentle and
  sustained turns at speed) and one it cannot give has the flaps out as far as the speed allows.
  Everything that reads the wing's maximum reads this one: the lift budget, the lift curve (its
  peak at the stall angle), the nose's lift slope and the separation drag. Straight and slow,
  the 1 g alone deploys them: the 1 g stall with the flaps out is the clean one over
  sqrt(1.15).
- **Drag.** `flap_drag_scale` x the induced-drag factor x (extra lift)^3 / (the full bonus),
  on the lift coefficient beyond the clean maximum: zero while the clean wing suffices, equal
  to the old square law with the flaps fully used, and the cube keeps the curvature continuous
  where it starts.
- **Smoothness.** Smootherstep in speed and in demand (value, slope and curvature continuous);
  the cube in the drag. Tested by refinement (`test_flaps_are_smooth_in_speed_and_in_demand`):
  on a grid four times finer the largest neighbour-to-neighbour change of the finite-difference
  slope and curvature shrinks about fourfold, which a kink or a curvature step would not do.
- **Cost in the step.** Per half-kick (the flaps held through it at its starting speed and
  stick): one smootherstep of speed, a square root of the demand and one smootherstep of it, a
  few multiplies; in each drag evaluation a square root, a clamp and a cube.
- **Per type.** `flap_lift`, `flap_blowback_ratio` (a multiple of the clean corner speed, so it
  carries across time scales and the web tool's sliders), `flap_drag_scale`. Off:
  `flap_lift` 0. Shared: the two bands. A large bonus with the 40% band could turn the
  full-stick rate down in the blow-back (the test holds every card type).

**Off for the P-51D** (`flap_lift` 0, author 2026-11-24); the machinery and its tests stay
(the tests fly the P-51D with +15%), switchable per type. What follows is the record of the
flaps when they were on, under the stick of before.

**The P-51D's values were a game choice:** the P-51D had no combat flap setting. +15%
(Delta C_L 0.24), blown back by the clean corner speed (fully out to 0.6 x corner, 172 mph), so
the corner and the corner rate are the airframe's; Delta C_D = 1.5 (Delta C_L)^2 with the flaps
fully used (0.086 at the full bonus). The drag is set so the tightest sustained turn stays
clearly slower than the best sustained rate (the author's 10% margin): a sweep of the drag
(2026-11-24, the row re-solved each time) gave the tightest turn's speed over the best rate's
as 0.958 at 0.2 (the old value, which fails the margin), 0.921 at 0.5, 0.882 at 1.0, 0.852 at
1.5 and 0.865 at 2.0; +25% with 1.0 gave 0.815. The 1 g stall with the flaps out is 94.3 mph
(clean 101.2).

What they do to the P-51D (the same solved row, flaps off against on):
- tightest sustained turn 202 m at 135.0 mph, against 224 m at 161.2 mph without them: 10%
  tighter at a speed 16% lower;
- the best sustained rate rises from 18.42 to 18.99 deg/s (the solve's target) and its speed
  falls from 161.3 to 158.4 mph: the flaps reach the sustained turn now;
- the full-stick rate rises with speed all the way to the corner (14.3 deg/s at 1.2 x stall,
  18.3 at 1.4, 21.6 at 1.6, 26.9 at 2.0, 32.2 at 2.8; no shelf);
- the flap drag costs speed in a slow full-stick turn (at 1.35 x stall, d speed / dt -0.49
  against 2.89 m/s^2 game without it).

## Comparison with real figures (real units)

This comparison predates the design system; its "Game row" is the "before" column of
section "Design inputs", which gives the current row's numbers.

"Before": the card row until 2026-11-24 (the October point mass: 21.83 m^2, C_Lmax 1.50, the
game terms off). "Point mass": the referenced airframe above, game terms off. "Game row": the
card now (the stick as fractions of the top line, flaps off, separation drag 0.5 at exponent
24); the flapped row of earlier on 2026-11-24 is in section "The stick". Sustained figures are coasting, total energy rate >= 0. The radii at multiples of
the stall speed use each row's own clean stall speed (sustained; full stick where different).

| Quantity | Published figure (source) | Before | Point mass | Game row |
|---|---|---|---|---|
| Stall speed, clean, 1 g | 100 mph at ~10,100 lb (Loftin); 101-102 mph at 9,000-9,500 lb (author's set 1); about 100.5 mph IAS at 9,800 lb (AN 01-60JE-1) | 104.1 mph | 101.2 mph | 101.2 mph |
| Top speed, sea level | 375 mph at 67 in Hg, 4,427 kg (flight test) | 375.1 mph | 375.0 | 375.0 |
| Best climb, sea level | 3,600 ft/min at 67 in Hg (flight test); 3,410 (NAA calculation) | 3,599 ft/min at 142 mph | 3,600 at 142 | 3,600 at 140 |
| Corner speed | none published; stall x sqrt(g limit) (derived) | 294.4 mph | 286.1 | 286.1 (the flown peak at 286.0) |
| Turn rate at corner | derived only | 34.0 deg/s | 35.0 | 33.5 |
| Best sustained turn | no test value. Models: the October fit's 18.6 deg/s at 167 mph, r 231 m; author's set 2 (E-M derived) 360 in 18.2 s, r 215 m (19.8 deg/s); simulator compilations 15.7 deg/s | 18.63 deg/s at 169.4 mph, r 233 m | 18.99 at 163.3, r 220 | 18.99 at 179.1, r 242 |
| Tightest sustained turn | none found | 233 m at 169.3 mph | 220 m at 163.2 | 226 m at 151.8 |
| Tightest / best-rate speed | | 0.998 | 0.998 | 0.848 |
| Radius at 1.2 x stall | | 292 m (125 mph) | 276 m (121 mph) | 276 m (121 mph) |
| Radius at 1.5 x stall | | 239 m (156 mph) | 226 m (152 mph) | 226 m (152 mph) |
| Radius at 2 x stall | AAF calculation: 883 ft (269 m) at 4 g, 10,000 ft, P-51B (not comparable at sea level) | 331 m sustained, 225 m full stick (208 mph) | 316 m, 212 m (202 mph) | 290 m, 212 m (202 mph) |
| Level acceleration at the stall speed | 2.73 m/s^2 (v2 research, estimated) | 3.74 m/s^2 | 3.83 | 3.70 |

The game row with the speed-scheduled flaps (A1, earlier on 2026-11-24): best sustained 18.99
deg/s at 165.5 mph, r 223 m; tightest sustained 202 m at 136.7 mph (ratio 0.826); corner
32.2 deg/s at 286 mph.

Reading:
- Top speed, climb and best sustained rate are the solve's targets and match the flight test
  and the point mass to 1e-6. The best sustained speed is 9.7% above the point mass's (not
  solved; the separation drag that sets the tightest turn apart moves it up).
- The corner speed is exact (closed form from C_Lmax and the g limit); the corner rate is 4%
  below the point mass's (33.5 against 35.0 deg/s): only the corner's own soft minimum
  (exponent 16) rounds it now; the stick's second soft minimum cost 8% before.
- Without the flaps the slow turns are the airframe's (the radius at 1.2 x stall is the point
  mass's 276 m); the tightest sustained turn sits 15% slower than the best rate, 3% wider than
  the point mass's: a one-circle fight asks for a different speed than a two-circle fight.

## Charts

The P-51D's doghouse, its card row (game terms on, flaps off), drawn from `tuning.envelope`. In real
units speeds, turn rates and Ps are halved; radii and loads are unchanged. Marks: the corner
(circle), the best sustained turn (square), the tightest sustained turn (triangle), each with a
small hollow marker where the game flies it (`lean.tool.flown`). Each line starts at the
lowest speed the game flies it (game m/s: full stick 115.5, sustained 115.3, fine 103.4, dive
131.2, climb 115.5; the stall speed is 90.4), nothing below; how they were found:
`lean/tool/NOTES.md`.

| Real units (1x) | Game units (2x) |
|---|---|
| ![P-51D, core, real units](doghouse-p51d-core-real.png) | ![P-51D, core, game units](doghouse-p51d-core.png) |

Dark variants: `doghouse-p51d-core-dark.png`, `doghouse-p51d-core-real-dark.png`; the
comparison panels (rate, radius, sustained 360 time against speed):
`doghouse-comparison-core[-dark].png`.

    for theme in light dark; do THEME=$theme OMP_NUM_THREADS=4 PYTHONPATH=.:src \
        uv run --no-sync python docs/rewrite/sections/turns/doghouse.py; done

The vizia_plain and Vizia charts (`doghouse-vizia-plain-core*.png`, `doghouse-vizia-core*.png`)
and `sections/turns/doghouse-p51d-vizia-core*.png` are records drawn before G1 (2026-11-23),
when Vizia left the card; they show the P-51D's earlier point mass.

## Tests

`lean/game/tests/test_tuning.py`:
- the point mass, fitted when the card is derived, flies the flight test (375 mph and 3,600
  ft/min to 1e-6);
- every card row flies its reference design under its profile (every input to 1e-6: the
  documented airframe's C_D0 and 2,200 hp), its best sustained speed within 12% of its own
  point mass's, the same stall and corner speeds, the terms on; the solver's tests (moved
  designs, warm starts, the point-mass guess, the hand-made implausible row) use the
  documents profile;
- derived rows are cached by their aircraft and the source (a second load reads the disk
  without solving; another target or another source is another key);
- the design-input table sets every card field once; moved designs (faster, slow turner,
  late climber, flaps) and a cold start fly every input after the solve (1e-6; the fast
  path 1e-4); an input swapped by editing a copy of the table is met; the measures agree
  across grids and their finite differences are smooth; a hand-made implausible row warns on
  e, propeller efficiency, C_D0, C_Lmax and the thrust loss, and the P-51D on e and the
  structure; the damage reference speed is derived; another time scale flies the same
  aircraft; the scripted agent's references are the flown numbers (P-51D: quarter turn
  2.25 s, store turn 2.09 rad).

`lean/game/tests/test_flight.py`:
- every card type's tightest sustained turn is at least 10% slower than its best sustained rate,
  which is below its corner (P-51D: 135.75, 160.10 and 255.75 m/s game, ratio 0.848);
- the turn rate never falls with speed from half the stall speed to the corner, at full stick
  and at the sustained stick, for every card type, with its flaps off and on;
- full stick is the envelope's maximum: away from the corner it is all the lateral acceleration
  left after 1 g, to 0.1% (above corner sqrt(a_max^2 - g^2), below sqrt((C v^2)^2 - g^2)), and
  no stick level turns more; every level is its fraction of full stick at every speed;
- the sustained level passes through the best sustained point (its rate within 0.1%, the
  energy rate about zero), and the fine level is fine_stick_fraction of it;
- the flaps (flown on a row with +15%, since the P-51D's are off) deploy by demand (clean flying straight above the stall speed, the full bonus at
  full stick below the blow-back band, clean from the blow-back speed, out below the 1 g speed
  flying straight), are C2 in speed and in the stick (by refinement, a check that itself fails
  the old square law's curvature step), and cost drag only for lift beyond the clean wing.

The web tool (`lean/tool`) reads the same functions (`tuning.envelope`, `performance`,
`design_row`, `implied`, `warnings`) and its presets read the derived card (`tuning.row_of`).

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

