---
covers: [src/bnb/game/handling/, src/bnb/game/ship_types.py]
reviewed: 2026-11-03 09:45
---

# The ship card

<!-- generated: section-status -->

The [ship card](../glossary.md) is the per-type parameter table,
`bnb.game.ship_types.ShipTypes`. The package `bnb.game.handling` connects it to
**handles**: the quantities a designer sets for a ship type, such as its stall speed, its
cruise or its sustained turn rate. `bnb.game.handling.measure` flies a card and reads its
handles; `bnb.game.handling.solve` finds the card that meets a set of handles. Both fly
the game's own flight code, `bnb.game.physics.flight.flight_tick`, so a handle is what a
ship does in the game, not what a formula predicts.

## What the card holds

Every field of `ShipTypes` is a tensor with one entry per type, or one row per environment
for a sweep. The defaults and their units are in [Physics](physics.md#ships-and-their-card).
Three kinds of field:

- **Physical values** that were per type from the start: the lift and g limit, the drag,
  the engine, the store, the nose frequency, the gun and the shield.
- **The bleed exponent** `e` (`bleed_exponent`, 2 by default). Induced drag is
  `k (C^2 + (z/u^2)^2)^(e/2)`, with `C` the lift coefficient, `z` the vertical lift and `u`
  the speed. A higher `e` makes a hard turn cost more drag relative to a gentle one.
  `bnb.game.physics.flight.bleed` computes the power as `s * s^(e/2 - 1)`, which is
  exactly `s` at `e = 2`, so the default card's drag is the formula without the exponent.
- **Scales of shared values.** A parameter that is a shared value in `Rules` or
  `FlightConstants` is per type as a scale of that value: a type's stall angle is
  `FlightConstants.stall_angle` times its `stall_angle_scale`. `bnb.game.ship_types.SCALED`
  lists every scale and the value it multiplies. Every scale is 1 on the default card.
  Setting a shared value still applies to every type, as the race reference tests assume.
  Code sized to the card reads the largest per-type value instead: Yemong's coding ranges
  span the widest stall angle and the longest recharge delay of the card it is built with.

Whose value applies: the flight and shield values are the ship's own; the bullet spread and
drag are the shooter's; the head-on fraction is the target's; the vision range is the
observer's. Perception takes no card, so the observers' ranges are kept in the state,
`Match.vision_range`, written by every reset and every tick.

## The handles

`bnb.game.handling.Handles` holds every handle per type: the flight handles first, then
the combat handles. **Direct** handles follow from the card in closed form; **flown**
handles are read off flights at the standard field index (no fields).

| Handle | Measured as | Sets |
|---|---|---|
| `stall_speed` | `sqrt(g / C_max)` | the lift coefficient |
| `g_limit` | `a_max / g` | the g limit |
| `cruise_speed` | flown: the speed the flight tick leaves unchanged, coasting straight at a full store | the engine power at a full store |
| `climb_time` | flown: from empty to the store less its fade band (`store_fade_band` in `bnb.game.rules.FlightConstants`), coasting straight from cruise | the coast exchange |
| `ceiling` | the store size | the store size |
| `low_speed_acceleration` | flown: the start rate of speed at the stall speed, store empty | the static-thrust cap |
| `sustained_turn_rate` | flown: the settled full-stick coasting turn, store held below the engine knee | the induced-drag factor |
| `half_lift_drag` | `0.5^e`: the induced drag at half lift over that at full lift | the bleed exponent |
| `nose_reversal_time` | flown: from a settled full-stick turn at corner speed, the time to 90% of the reversed steady angle of attack | the nose frequency |
| `dive_acceleration` | flown: the start rate of speed in a boost from cruise, store half full | the dive exchange |
| `zoom_acceleration` | flown: the start rate of speed in a reverse from cruise, store half full | the climb exchange |
| `stall_angle` | the angle-of-attack limit | the stall angle |
| `vision_range` | the observer's range | the vision range |
| `zero_lift_drag` | the zero-lift drag coefficient | itself; the solver holds it |

The sustained turn and the half-lift drag set the induced drag's size and shape. The
sustained turn flies at full lift (the lift fraction about 0.99), where the shape does not
matter, so it sets the drag at full lift; the half-lift drag sets how fast the drag falls
below it. A low half-lift drag (a high exponent) is a ship that turns hard but bleeds fast
in hard turns and keeps its energy in gentle ones; a high one spreads the cost.

**Why the shape is a direct handle (author, 2026-11-09).** It used to be flown: the bleed
rate, the specific energy rate at corner speed and three-quarter stick, set the exponent.
That turn flies at a lift fraction of about 0.76, so the two handles measured the drag
curve at nearly the same point: the exponent's leverage between them was ln(0.99/0.76),
about 0.26, and holding the bleed rate while the sustained turn moved 10% swung the
exponent from 2 to about 0.8 or 3.2 (at a third more turn no exponent held it). With the
shape set directly the default card's condition number fell from 16-17 to about 9 and 7.
The bleed rate is still shown: the flights measure it (`Flown.bleed_rate`), and
`bnb.game.handling.estimate.bleed_rate_now` gives it in closed form within 1%.

**Why the zero-lift drag is held.** Under the cruise fix (below), the engine's full-store
power is the drag's power at cruise. Whatever the zero-lift drag, the full-store power that
pays for it meets the cruise, so the cruise does not determine the drag. The drag is
therefore a handle of its own, held like the store size, and the cruise sets the
full-store power instead.

The combat handles are the damage, the fire rate in shots per second, the muzzle speed, the
spread, the gun range (the distance a bullet fired at cruise along the flight path travels
in its 0.4 s life), the shots a full store pays for, the shield capacity, recharge rate and
delay, the head-on fraction, the hit radius and the spawn energy. They map to the card in
closed form (`bnb.game.handling.weapons`).

## Measuring

`bnb.game.handling.manoeuvres.fly` flies every manoeuvre of every type of every card in one
batch: one environment per card, one ship per manoeuvre and type. It runs in the card's
dtype; the solver uses float64 on the CPU.

- A **start rate** is the slope at the start of a quartic fitted over the manoeuvre's first
  0.2 s (`FIT_WINDOW`).
- The **climb's time** interpolates linearly between ticks. The **nose reversal**, four to
  eight ticks long, interpolates through the cubic on the four ticks around the crossing.
- The **sustained turn** flies 20 s and takes the Aitken limit of the path's turn rate
  over its last three seconds (at the standard card's scales; see below).
- The **turns** fly at a quarter-full store: below the engine knee, so the store does not
  change the engine, and clear of the store's fade bands.

The windows (the fit window, the settling turn and its samples, the longest reversal and
climb) are written in seconds at the standard card's scales and flown stretched by each
card's own, so a card of any size flies them for the same length in its own units. A
type's time scale is `V_s / g`, with `V_s = sqrt(g / C_max)` its stall speed; the climb's
is `E_max / (m g V_s)`, the store over the specific power scale `g V_s`. A batch flies its
largest scales over the standard heavy's (`manoeuvres.card_stretch`, the reference
`manoeuvres.REFERENCE`, 1.30 s and 14.8 s): the climb's window by the climb scale, the rest
by the time scale. The standard card flies exactly the seconds written (20 s, 2 s, 900 s);
a P-51-scale card, about 3.6 times as slow with a 125 kJ store, flies its climb for up to
about 17,000 s. The shown quantities follow the same rule (`bnb.game.handling.shown`: the
settled turns by the time scale, the boost and the firing endurance by the climb's).

`bnb.game.handling.manoeuvres.cruise_speed` finds the straight equilibrium at any store
by bisection on one-tick flights, from the fastest speed at which the tick's speed gain
changes from positive to negative.

## Solving

`solve` sets the direct handles' parameters in closed form and the combat handles' with
`bnb.game.handling.weapons.combat_card`. The fire rate and the recharge delay must be whole
numbers of ticks at the rules' tick, or the solve fails with the reason: the fighter's
15 Hz is 2 ticks at 30 Hz, 4 at 60 Hz, and no whole number at 25 Hz. The game refuses such
a card at reset too (`bnb.game.ship_types.ticks_between_shots`, in exact fractions). The
seven flown handles set seven parameters
(`bnb.game.handling.solver.FLOWN_PARAMETERS`{.nolink}):

1. **The cruise fix.** Cruise does not depend on the store: the engine power below the
   knee is the full-store power plus the charging power at cruise,
   `P = P_min + m a_0 u_cruise`. The solver derives `P` this way, and solves the full-store power so that
   the flown cruise meets its handle.
2. **First guess.** At the exponent the half-lift drag sets, every parameter inverts the
   game's instantaneous equations (`setting` and `rates` in `bnb.game.physics.flight`) at
   its manoeuvre's state. The nose frequency comes from the reversal
   time with the damping lift adds to the nose: at corner speed, where the reversal flies,
   the angle error obeys `e'' + (kappa + 2 zeta omega_c) e' + omega_c^2 e = 0`, `kappa` the
   path slope, so its damping ratio is `zeta + kappa / (2 omega_c)`, about 0.9 rather than
   `zeta` = 0.7. Without that term the guess was a third low and flew the reversal 70-100%
   slow; with it, the first guess flies every flown handle within a few percent
   (`tests/sections/test_ship_card.py::test_first_guess_flies_close_to_its_handles`).
3. **Levenberg-Marquardt** on the relative errors of the flown handles, in the logs of the
   parameters, with a finite-difference Jacobian. Each iteration flies the card and its
   seven perturbations in one batch. The induced drag is solved as its value at full lift,
   `k C_max^(e-2)`, so the exponent reshapes the curve below full lift without rescaling
   it. A half-lift drag outside `EXPONENT_BOUNDS`' range is infeasible before any flight.
4. **Stop** when every flown handle is within `TOLERANCE` of its target, or after
   `ITERATIONS` iterations.

A card that ends further than `ACCEPTANCE` from a handle fails with its reason: the handle it
misses, the condition number of its Jacobian and the two handles its weakest direction
involves. `solve` raises `bnb.game.handling.Infeasible` by default; with `strict=False` it
returns the reasons in `Solution.failures`. Handle sets that cannot be met at all, such as a
g limit at or below 1, a cruise below the stall speed or a sustained turn above the
instantaneous peak, fail before any flight. So does a nose reversal or a climb longer than
its manoeuvre flies (`REVERSAL_LIMIT` and `CLIMB_LIMIT` in
`bnb.game.handling.manoeuvres`, stretched by the card's scales: `solver.windows`), with the
window named. The relative errors have floors for handles near zero (`ERROR_FLOOR`, set at
the standard card's scales and taken in each type's own units). A condition number above
`CONDITION_WARNING` raises a `bnb.game.handling.RedundantHandles` warning.

For live tuning, `solve` also takes `start` (a card to start from instead of the first
guess: near the answer it converges in two or three iterations), `progress` (called after
every iteration with the best card so far and the flown handles it flies,
`bnb.game.handling.solver.Progress`), `cancel` (asked before every iteration; True returns
the best card so far) and `tick_function` (the flight tick to fly with; `measure` takes it
too). The
play shell passes a compiled flight tick (`bnb.runtime.flights`): a measurement then takes
about 1 s instead of 12.5 s and a warm-started solve about 4 s instead of 40 (on the
development CPU), after a one-time compile per batch size that the on-disk cache keeps.

## Estimating

A solve flies the game's code every iteration, tens of seconds on the CPU, too slow for a
slider. `bnb.game.handling.estimate.Estimate` answers at once, from closed forms only,
anchored on an exact card and its handles: each field of the estimated card is the
anchor's times the ratio between the closed forms (`closed_card`) at the new handles and
at the anchor's. At the anchor it is the anchor exactly; away from it the closed forms'
systematic errors mostly cancel. `closed_card` is the solver's first guess.

Measured on the default card (fighter and heavy), moving one flown handle by 10%, the
estimated card flies the moved handle within 0.8% and every other flown handle within
1.5%, except the sustained turn rate, which moves the low-speed acceleration 3-4%: both
fly at full lift, so the drag at full lift the turn sets is also the drag at the stall
speed, and the static-thrust cap's closed form is least accurate when it has to absorb a
large change (a quarter less sustained turn moves the low-speed acceleration 29%). The
direct handles move the flown ones at most 3.2% (the stall speed and the g limit set the
corner speed every turn is flown at). Moves of 25-33% stay within about 7% but that one. The
play shell's tuning panel applies the estimate at once and the exact solve, started from
it, once the handles settle (`bnb.play.tuning`).
## How independent the handles are

On the default card (measured on the development laptop's CPU, the cases in
`tests/sections/ship_card_cases.py`), re-solving after moving one flight handle by 5% holds
every other flight handle within the acceptance band for all 28 moves
(`tests/sections/test_ship_card.py::test_independence`), and the default card's condition
numbers are about 9 (fighter) and 7 (heavy). The details from when the bleed rate was a
handle are in `docs/rewrite/sections/ship-card-notes.md`.

## The reference chart

`bnb ship-card` prints every handle, every card field (the scales and the values they
make) and the shown quantities of a card, and writes three charts: the specific excess
power against speed per stick level, the turn-rate chart with the stall boundary, the g
limit, the sustained turns and corner speed, and the ship's ratios against a P-51D
(`bnb.game.handling.shown.P51D`). The shown quantities are flown: corner speed, the
instantaneous peak turn and the minimum radius, the settled turn per stick level with the
gun's offset, top speed, firing endurance in four flight conditions, and damage per
second. [Designing a ship](../../guides/designing-a-ship.md) shows how to use it.
