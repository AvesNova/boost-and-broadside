# Ship design: inputs, outputs and the solve (proposal)

Proposal, 2026-11-24, branch `core`. Nothing in the code changed. It follows the author's
model of 2026-11-24:
- **outputs** are the sim's own coefficients;
- **design inputs** are whatever is most convenient to design with;
- the system is square, and short circuits are allowed;
- an optional two-layer route: documents, then a reference model, then design sliders;
- warnings, not refusals, for non-physical regimes.

Units: game units (2x time) unless marked real. Every number below is either cited, or
computed with the repo's code at the P-51D's row. The harness that computed them is in the
session scratchpad, not the repo: `measures.py`, `jacobian.py`, `conditioning.py`,
`solve.py`, `second_point.py`, `implied.py`.

## 1. The outputs: what the sim reads per ship

The card's 37 columns are `card.CARD_FIELDS` (`lean/game/card.py:48-77`). Some numbers are
derived and never stored:
- the store: two loops at corner speed (`core.py:146-150`);
- the stick levels (`core.stick_table`, `core.py:218`; `_derived_characteristics`,
  `core.py:228`).

**Flight, aerodynamic (17).**

| Output | What it controls |
|---|---|
| `max_lift_coefficient` | the lift limit (`_setting`, `core.py:594`); the stall and corner speeds; the store size; the nose frequency scaling (`core.py:183`) |
| `max_acceleration` | the g limit: one soft minimum against lift (`core.py:594`) |
| `zero_lift_drag` | parasite drag (`_rates`, `core.py:640`) |
| `induced_drag_scale` | induced drag, times `INDUCED_DRAG_FACTOR` (`card.py:92`) |
| `engine_power` | thrust P / v (`core.py:620`) |
| `static_thrust` | the low-speed thrust cap (soft minimum, `core.py:620`) |
| `exchange_acceleration` | the dive and zoom trade (the action tables) |
| `stall_angle_scale` | the angle of attack at maximum lift: the lift curve (`core.py:652`), the nose's lift slope, the thrust's sin(angle) |
| `critical_speed`, `drag_rise_exponent` | the drag rise (`core.py:640`) |
| `separation_drag`, `separation_exponent` | extra induced drag near maximum lift (`core.py:640`) |
| `lift_softmin_exponent` | how round the corner is |
| `thrust_softmin_exponent` | how soft the static-thrust cap is |
| `flap_lift`, `flap_blowback_ratio`, `flap_drag_scale` | the automatic flaps (`_wing`, `core.py:546`; off for the P-51D) |

**Pure game choices, no aerodynamic meaning (20).**
- **Control feel:** `corner_nose_frequency`, `nose_damping` (the nose kick,
  `core.py:711`), and `fine_stick_fraction`. The stick rule itself has no column: full stick
  is the top line and the sustained level is derived.
- **Guns:** `hit_damage`, `explosive_share`, `muzzle_speed`, `fire_rate`, `bullet_life`,
  `bullet_spread_scale`, `bullet_drag_scale` and `reference_bullet_speed`.
- **Hit geometry:** `hit_radius` and `core_radius`.
- **Shield:** `max_shield`, `shield_recharge_rate_scale` and
  `shield_recharge_delay_scale`.
- **Head-on:** `head_on_fraction_scale`, `head_on_angle_scale` and `head_on_sharpness`.
- **Vision:** `vision_range_scale`.

Two of the gun columns are not free:
- `reference_bullet_speed` follows from cruise, muzzle speed and bullet drag through a rule
  (`tuning.py:395`, `DAMAGE_REFERENCE_DISTANCE`). No input maps to it alone.
- `bullet_life` is the gun range rounded to whole ticks (`tuning.py:386`). It is
  one-to-one with the gun range, but quantised.

The game-choice block already has a clean, closed-form input set: the handles of
`tuning.direct_fields` (`tuning.py:378`). The rest of this note is about the 17 flight
outputs.

## 2. Today's routes (for contrast)

There are two routes to a flight row. They solve different things.

- **The card** (`game_row`, `tuning.py:255`) solves zero-lift drag, induced drag and power
  so that the row flies its top speed, best climb and best sustained rate
  (`GAME_TARGETS`, `tuning.py:251`). Everything else comes from the airframe or from
  `P51D_TERMS`.
- **The tool** (`handles_row`, `tuning.py:422`, called from `design.py:216`) works
  differently:
  - it solves power, static thrust and induced drag for cruise, low-speed acceleration and
    the *settled full-stick* turn rate;
  - it takes zero-lift drag as a direct slider;
  - it does not hold the climb (`p51d-inputs-sources.md`, "There is no climb slider").

So the preset row and the slider row answer different questions. The handles also keep v2's
names: `cruise_speed` is the top speed, and `sustained_turn_rate` is the settled turn. They
keep v2-only checks too (`half_lift_drag`, a dive/zoom pair averaged into one trade). This is
the coupling to remove.

`p51d-inputs-sources.md` is stale against the code. It still gives:
- separation 0.1 at exponent 8 (the code has 0.5 at 24, `tuning.py:191-201`);
- flaps +15% (the code has them off);
- stick levels 0.700 and 0.210 (the code derives 0.895 and 0.448, `p51d-model.md`, "The
  stick").

## 3. Measured: sensitivities at the P-51D

Method:
- d log(measure) / d log(coefficient), by central differences of +-1% (checked against
  +-2%) on the game's own `core.quasi_steady`;
- speeds on a 0.1 m/s grid, with each extremum refined by a parabola. The repo's
  `performance()` takes a grid argmax, which is quantised and too coarse for finite
  differences on speeds.

The reference values reproduce `p51d-model.md` (game units):

| Measure | P-51D row |
|---|---|
| stall speed | 90.44 m/s |
| corner speed (flown) | 255.7 m/s |
| corner rate | 1.171 rad/s |
| top speed | 335.28 m/s |
| best climb | 36.58 m/s at 125.1 m/s |
| best sustained | 0.663 rad/s at 160.1 m/s, r 241.5 m |
| tightest sustained | r 226.5 m at 135.7 m/s |
| sustained leaves full stick | 134.8 m/s |
| rate at 1.5 x best sustained speed / best | 0.714 |
| low-speed acceleration | 14.79 m/s^2 |

**The author's guess is right.** The tightest sustained point is where the sustained line
leaves the full-stick line: 135.7 against 134.8 m/s, within 1 m/s.

What each measure mainly responds to (largest log sensitivities):
- **stall speed:** lift only (-0.5). It is closed form.
- **corner speed:** lift and the g limit (-0.5, +0.5). It is closed form.
- **corner rate:** lift and the g limit (+0.50, +0.51), plus lift softness (+0.044).
- **top speed:** power and zero-lift drag (+0.357, -0.339).
- **best climb:** power (+1.21) and induced drag (-0.16).
- **best sustained rate:** power +0.43, lift +0.39, induced drag -0.37.
- **tightest / best speed ratio ("the gap"):** the separation exponent +0.116, induced drag
  +0.147, power -0.191. The separation coefficient gives **0.000**.
- **low-speed acceleration:** power +1.08, static thrust +0.21.
- **best-climb speed:** thrust softness -0.086, static thrust -0.163.

The separation coefficient is inert at 0.5 and exponent 24: the term has saturated. At 0.35
and exponent 8 the coefficient becomes the active handle (gap -0.149) and the exponent's sign
flips (-0.054). This agrees with the sweep in `p51d-model.md` ("Separation drag"):
- at exponent 24, 0.5 gives 0.848 and 1.0 gives 0.844;
- at 0.5, exponent 8 gives 0.851, 16 gives 0.839 and 24 gives 0.848.

So the gap is not monotone in the exponent, and it saturates in the coefficient at high
exponents.

## 4. Candidates that duplicate or over-constrain

Rank and condition number are for the square log-log Jacobian.

- **Stall speed, corner speed and g limit**, all three: rank 5 of 6. corner = stall x
  sqrt(n) exactly, so any two set the third.
- **Corner speed, corner rate and corner radius:** the radius is speed / rate. Condition
  number 2.4e5.
- **Best sustained speed, rate and radius:** an identity (radius = speed / rate). Condition
  number 1.6e14.
- **Any radius together with stall speed:** near duplicates. Every radius is mostly the wing
  loading. Cosine similarity with the stall speed:
  - corner radius 0.999;
  - best sustained radius 0.986;
  - tightest radius 0.975.

  Rate and radius at the best sustained turn, with no stall speed, give condition number
  1,050.
- **Corner rate**, once stall speed and g limit are set: only the corner's softness moves
  it (+0.044). As an input it solves `lift_softmin_exponent`, with condition number 47.5:
  workable but weak. It is a natural output.
- **The departure speed** (where the sustained line leaves the full-stick line) is the
  tightest speed. Cosine similarity with the best sustained speed is 0.984. Use one of them.
- **The Ps bubble's fall-off:** with top speed, climb and sustained rate set, it is fixed
  by the drag polar under constant power. Adding it with any thrust term gives condition
  number about 4,000. It over-constrains. It is an output.
- **The shape of the max-lift line:** below corner the line is sqrt((C v^2)^2 - g^2) / v
  (`_setting`). Its dip near stall is the 1 g held. With C set, only the flaps (`flap_lift`,
  `flap_blowback_ratio`) change the shape, and only the lift softness rounds the corner. It is
  not a separate input.
- **The best-climb speed:** a usable input through the thrust softness (condition number
  22). It is A-grade data: 165 mph IAS (`p51d-drag-check.md`, section 1).

## 5. Three input sets

Each set maps one-to-one onto the 17 flight outputs. The game-choice block is the same in all
three (section 1). In the notes below, "data" means the value is available for WWII aircraft
(grade from `p51d-inputs-sources.md`), and "design" means it is a game choice.

### A: performance points only

Every flight output is set by a flown number.

| Input | Sets mainly |
|---|---|
| stall speed | lift |
| g limit | max acceleration |
| top speed | power / zero-lift drag |
| best climb | power |
| best sustained rate | induced drag |
| gap (tightest / best speed) | separation term |
| best sustained speed | the other separation term |
| low-speed acceleration | static thrust |
| corner rate | lift softness |
| best-climb speed | thrust softness |
| stall angle | itself (short circuit) |

The remaining 6 (exchange, drag rise x2, flaps x3) are short circuits.

- **Data:** top speed and best climb are A-grade (flight test). Stall speed is B-grade
  (manual). The best-climb speed is A-grade. The g limit is F-grade (forum). There is none
  for the sustained turn, the gap or the corner rate.
- **Conditioning, all 11:** rank 11, condition number 146. The weakest direction is thrust
  softness against static thrust: two low-speed thrust terms seen only by the low-speed
  acceleration and the climb speed. With both softnesses as short circuits, the condition
  number is 107, and the weakest direction is the separation coefficient.
- **Verdict:** square but fragile. Two separation terms from two speeds is the weak part.

### B: physical sliders (two layers, no solve)

The sliders are layer 1: mass, wing area, aspect ratio, C_Lmax, C_D0, Oswald e, shaft power,
propeller efficiency, propeller diameter with FM, g limit and stall angle, plus the game terms.
The row is `airframe_row` (`tuning.py:209`), closed form. Performance is shown, not set.

- **Data:** every slider has a provenance line (`p51d-inputs-sources.md`).
- **Not square.** Mass, wing area, C_Lmax and C_D0 enter only through rho S / 2m
  (`Airframe.per_metre`, `tuning.py:97`). e and AR enter only as a product. Power, efficiency
  and mass enter only as P eta / m. About 12 sliders drive 8 coefficients, so many slider
  states give the same ship.
- **Conditioning:** the forward map is trivial (diagonal in logs). But designing "a 5% better
  sustained turn" means searching by hand.
- **Verdict:** a good layer 1, but a poor set of sliders.

### C (hybrid): reference model, then performance sliders (recommended)

Layer 1 is today's documents plus fit:
- `Airframe` and the flight test go through `fit_point_mass` (`tuning.py:230`) into the
  reference point mass;
- that fit also produces the documented `P51D_TERMS` choices.

Layer 2 is the sliders. Each slider starts at the reference's flown value, so a ship with no
adjustments is the documented aircraft.

| Input | Sets | Method | Data or design |
|---|---|---|---|
| stall speed | `max_lift_coefficient` | closed form, g / v^2 | B: manual; C_Lmax 1.50-1.70 |
| g limit | `max_acceleration` | closed form, n g | F: forum; decided 8 |
| top speed | `engine_power` (with drag) | Newton | A: flight test |
| best climb | `engine_power` | Newton | A: flight test |
| best sustained rate | `induced_drag_scale` | Newton | none; model |
| gap | `separation_drag` | Newton | design |
| low-speed acceleration | `static_thrust` | Newton | none; momentum theory |
| separation exponent | itself | short circuit | design |
| lift softness | itself | short circuit | design (v2's 16) |
| thrust softness | itself | short circuit | design (8) |
| critical Mach, rise exponent | the drag rise | closed form | A: NACA ACR 4K02 Fig. 12 |
| dive and zoom trade | `exchange_acceleration` | short circuit | design (1 g) |
| stall angle | `stall_angle_scale` | short circuit | guess (15.5 deg) |
| flap lift, blow-back, drag | the flaps | short circuit | design (off) |

Corner speed is shown as an output. The g limit and the corner speed are interchangeable as
the input (open question 1).

**Conditioning (6x6 block: stall speed, g limit, top speed, climb, sustained rate, gap):**

| Gap solved by | P-51D (0.5 at 24) | 0.35 at 8 |
|---|---|---|
| separation coefficient | 260 | 11.7 |
| separation exponent | 12.1 | 24.1 |

For comparison:
- today's 3x3 `game_row` solve: 5.0;
- the 5x5 block with no gap: 6.2;
- adding the low-speed acceleration to static thrust: 16;
- adding the best-climb speed to thrust softness: 22.

**The gap needs a separation term that is still live.** At the current 0.5 and exponent 24,
the coefficient is saturated, so the exponent would have to carry the gap. The exponent
folds at about 16 (`p51d-model.md` sweep), so it is not a safe handle either.

The fix is to choose the shape (the exponent) as a short circuit where the coefficient still
moves the gap, and to solve the coefficient. At exponent 16 the sweep gives:

| Coefficient | Gap | Oswald e |
|---|---|---|
| 0.3 | 0.892 | 0.968 |
| 0.4 | 0.864 | 0.990 |
| 0.5 | 0.839 | 1.008 |

At exponent 24, 0.5 gives a better e (0.952), but the gap cannot be moved from there. This
is an author choice (open question 2).

**Solve check** (scratch harness, with the gap solved by the exponent at the current
point):
- **The P-51D's own targets:** converges.
- **Faster, climbs more, turns better:** stall -3%, 7.5 g, top +3%, climb +5%, sustained rate
  +5%, gap +1%.
- **Slow turner:** stall -10%, 7 g, top -7%, sustained rate +10%, gap +2%.

Each case converges in 2-3 Newton or Broyden iterations, with 7-11 flight evaluations, in
0.4-0.7 s on the CPU. The point-mass first guess (below) is within 1-13% of the solved
coefficients at the P-51D.

What the two variants imply (layer-1 mass, wing area, AR and shaft power held):

| Variant | Oswald e | Propeller efficiency | C_D0 |
|---|---|---|---|
| faster | 0.957 | 0.832 | about 0.014 |
| slow turner | 0.993 | 0.806 | about 0.018 |

## 6. Recommendation: set C, and how to solve it

One route builds every row: `design_row(reference, adjustments)`. The card's presets are
`design_row(reference, {})` plus any stored adjustments. The tool's sliders are the same call
with the slider values.

**Closed form, in order:**
1. C = g / v_stall^2.
2. a_max = n g. Corner speed and the store follow from these two.
3. Every game-choice field, through today's `direct_fields`.
4. The short circuits, copied.
5. A first guess for zero-lift drag, induced drag and power, from the point mass with hard
   limits:
   - top speed: P = D0 V^3 + k g^2 / V;
   - best climb: the maximum of (P - D0 v^3 - k g^2 / v) / g;
   - best sustained rate: the maximum over v of g sqrt(n^2 - 1) / v, with
     k (n g)^2 / v^2 = P / v - D0 v^2, capped by the lift limit (the crossing found
     exactly);
   - the three are solved by a cheap 3x3 Newton on these formulas.
6. Static thrust from the low-speed acceleration plus the drag at the stall speed, as
   `handles_row` does now (`tuning.py:466-470`).

**Iterative:**
- Newton in logs on (zero-lift drag, induced drag, power, separation term, static thrust)
  against (top speed, best climb, best sustained rate, gap, low-speed acceleration), with the
  game terms on. This is `_newton` (`tuning.py:481`) as it stands: pseudo-inverse, steps
  clamped to 0.5.
- Cold, it starts from the point-mass guess. In the tool it starts from the last row with
  Broyden updates, which is today's `fast` path.

**One change the solve needs:** the speed-type targets (the gap) must be smooth in the
coefficients. `performance()` takes argmax and argmin on a 0.05 m/s grid (`tuning.py:595`,
`tuning.py:601`), so a 1e-5 finite-difference step sees steps, not slopes. Two ways to fix
it:
- a parabola through the three grid points around each extremum (what the harness does);
- the two-pass grid of `core._best_sustained_turn` (`core.py:340`).

**What stays derived and is never an input:**
- corner speed and rate, all radii, the best sustained speed, the departure speed;
- the Ps fall-off and the store;
- the stick levels;
- the implied physical numbers (section 7).

## 7. Plausibility checks and what the tool shows

The implied physical numbers are found by holding layer 1's "size" numbers (mass, wing
area, AR, shaft power, propeller diameter). Each coefficient is then read back as C_Lmax,
C_D0, e, efficiency or FM (the inverse of `airframe_row`). Open question 5 is which numbers
to hold.

| Quantity | Warn when | Source or reason |
|---|---|---|
| Oswald e | above 1.0 | the elliptic wing is ideal; the test holds e <= 1 (`test_tuning.py:27`) |
| Oswald e | outside 0.78-0.93 (soft) | Fw 190 0.78-0.84; Bf 109 0.93, optimistic (`aircraft-references.md`); NACA P-51B 0.88-0.91 (`p51d-drag-check.md` 5) |
| Propeller efficiency | outside 0.7-0.9 | the card test (`test_tuning.py:27`); P-47D measured 0.78-0.80 |
| Propeller efficiency | 1.0 or above (hard) | energy |
| C_D0 | 0 or below (hard) | physics |
| C_D0 | outside 0.0148-0.034 (soft) | P-51D powered fit 0.0148-0.0163 (`p51d-drag-check.md`); Fw 190 A-8 0.0265 / 0.0340, cooling open |
| Clean C_Lmax | above 1.7 | top of the P-51D stall-derived span 1.50-1.70; the accelerated values are lower (F6F about 1.4, Spitfire VA 1.21) |
| Static FM | 1.0 or above (hard) | momentum theory; 0.75 assumed (`tuning.py:59`) |
| g limit | above the aircraft's weight-scaled limit | P-51D: 8 g at 8,000 lb, 64,000 / W, so 6.6 g at 9,760 lb (forum, F) |
| Gap | above 0.9 | the flight test's margin (`test_flight.py`) |
| Best sustained speed | over 12% from the reference | the card test's tolerance |
| Full-stick rate falls with speed below corner | any | `test_flight.py` (flaps can cause it) |
| Thrust softness | low (v2's 2) | took 10% of the thrust near best climb (`p51d-inputs-sources.md`) |

A solve that does not converge is the one refusal: the tool keeps the last good row and says
so. Everything else is a warning.

The tool has three panels in place of today's eight groups:
- **Design:** the layer-2 sliders, each with the reference value marked and a warning badge
  when out of range. Warnings are also listed at the top, one line each, with the reason
  ("e 1.07: above the elliptic wing's 1.0").
- **Implied physical:** C_Lmax, C_D0, e, propeller efficiency (at top speed), FM, wing
  loading (204 kg/m^2 at the P-51D), shaft power loading (275 W/kg) and static
  thrust / weight (0.61).
- **Solved coefficients:** the raw card row, with the solve's residual, its iterations and
  its condition number.

## 8. Build plan

Small steps. One source of truth throughout.

1. **Smooth measures.** Add sub-grid refinement to `performance()` for the sustained, tightest
   and climb extrema. Tests: the existing values are unchanged to the grid, and the finite
   differences are smooth.
2. **`DESIGN_INPUTS` in `tuning.py`.** One ordered table: name, label, unit, the output it
   sets, and its kind (closed, solved or short circuit). Today the labels are duplicated in
   `design.py` `INFO` (`design.py:46`); they move here.
3. **`design_row`, `design_of` and `implied`, `warnings`.**
   - `design_row(reference, adjustments)` and its inverse `design_of(row)`.
   - `implied(row, airframe)` and `warnings(row, airframe)`, with the section 7 table as data.
   - Round-trip test: `design_row(design_of(row)) == row` within rounding.
4. **Presets through `design_row`.** `real_row` becomes `fit_point_mass`, then the reference
   design, then `design_row`. `RealAircraft` gains `adjustments` (empty for the P-51D, or the
   chosen gap), and `row_key` hashes them. The P-51D row must stay within the card test's
   tolerances.
5. **Delete** `game_row`, `GAME_TARGETS` and `GAME_SOLVED`, and the handles route:
   `handles_row`, `handles_of`, `_flown`, `SOLVED`, the `half_lift_drag` check, the
   dive/zoom pair, and the settled-turn target. `direct_fields` is kept, renamed, for the
   game-choice block. In `design.py`, `FIXED_HANDLES`, `MATCH` and the handles merge in
   `build_row` go too.
6. **Rework the tool.** `inputs()` reads `DESIGN_INPUTS`, and the tool adds the three panels
   (section 7). There is no physics in the tool.
7. **Docs.** Rewrite `p51d-inputs-sources.md` against the code (it is stale, section 2), and
   add a "Design inputs" section to `p51d-model.md`.
8. **Optional.** Move `reference_bullet_speed` out of the card and into
   `core._per_type` as a derived value (it is a rule, not a choice).

`card.py` keeps its columns: the outputs are the physics' own (the author's rule).

## Open questions for the author

1. **Corner input:** g limit or corner speed? They are exactly interchangeable once the stall
   speed is set. Never offer both, with stall speed, as inputs.
2. **The gap:** should it be an input, solving the separation coefficient? If so, at what
   exponent?
   - At 24 (now) the coefficient is saturated, and the gap sits at about 0.845.
   - At 16 it is live: 0.89-0.84 for e 0.97-1.01.
   - The alternative: the separation terms stay as direct inputs, and the gap is an output
     checked at 0.9 or below.
3. **The P-51D's default gap:** its point mass has none (0.998). Today's row gives 0.848.
   Which is the design value?
4. **The best-climb speed:** add it now, to solve the thrust softness? This is the
   `p51d-drag-check.md` proposal (efficiency 0.85 as a prior, softness 8 to 3.7, climb speed
   144 to 160 mph against the test's 165).
5. **Implied numbers:** which layer-1 numbers are held when sliders move? The proposal holds
   mass, wing area, AR, shaft power and propeller diameter, so C_Lmax, C_D0, e, efficiency
   and FM are implied.
6. **Low-speed acceleration:** a solved input, or static thrust kept physical (FM and
   diameter, short circuit)? The P-51D is power-limited: sensitivity 0.21.
7. **The settled full-stick turn:** drop it as a target, in favour of the best sustained
   rate that the card already uses?
8. **`reference_bullet_speed`:** move it out of the card (build step 8)?
