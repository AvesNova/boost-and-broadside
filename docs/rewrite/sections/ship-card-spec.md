# Ship card spec (section 1b): handles to physics parameters

Status: spec, 2026-11-01. Design: `ship-card.md` (concluded 2026-10-31). Physics: game
spec sections 1.1 and 3. **(proposal)** marks a choice the spec makes, implemented as
written and listed for the author.

Designers set **handles**, the quantities a fighter's designers would specify. A solver
finds the physics parameters that make the simulated ship meet them, by flying standard
manoeuvres in the game's own code.

## 1. What becomes per type

Today's card (`bnb/game/ship_types.py`) holds 15 per-type values; the rest are shared
constants in `Rules` and `FlightConstants`. This section moves to the card every
parameter a handle sets:
- **Flight:** the coast exchange `a_0`, the induced-drag factor `k`, the new bleed exponent
  `e`, the dive and climb exchanges `a_d` and `a_c`, the stall angle `α_s`;
- **Sight:** the vision range. The observer's type sets its range;
- **Combat:** the bullet spread, the bullet drag, the shield recharge rate and delay, and
  the head-on fraction.

**Defaults equal today's shared values,** so the game does not change: the game's tests
and fixtures (`tests/reference/`) pass untouched. The global shape rules stay shared:
`m`, `g`, `e_k`, `β`, the soft-min exponent, `ζ`, `ω_min`, `ε`, and the bullet life
(0.4 s, 12 moves at 30 Hz).

## 2. Physics changes

- **The bleed exponent:** induced drag becomes `k (C² + (z ι)²)^(e/2)`, with `e` per type
  and 2 by default.
  - At `e = 2` it must equal today's formula within the fixtures' tolerances; prefer an
    implementation that is exact there.
  - The cost must not be measurable at the bench sizes **(measure)**.
- **The cruise fix** (author, 2026-10-31): cruise does not depend on the store.
  - The solver imposes `P_min` = the null power (drag × speed at cruise) and
    `P = P_min + m a_0 u_cruise`, so `P` and `P_min` are derived, not set.
  - **The low-speed thrust cap (proposal):** keep it in the drive term, and let the
    solver's measurement include it.
    - The acceptance measures how much cruise varies across the store for the default
      card and for the solved test cards.
    - If any varies by more than 0.5%, report it. The alternative, moving the cap out of
      the drive term, is then the author's decision. Do not change the formula yourself.
- **The default card's numbers stay as they are** in this section. The report gives the
  default card's measured handles, and what the solver would make of them under the
  cruise fix. Adopting re-solved defaults is a later step, because it changes the game's
  goldens.

## 3. Handles

Thirteen flight handles per type, one per free flight parameter. All are measured at the
standard field index (`n = 1`, no fields) and below the engine knee unless stated.
**Direct** handles follow from the setting's definitions in closed form; **flown**
handles are measured by flying the game's `flight_tick` in a batched simulation.

| # | handle | measured as | sets |
|---|---|---|---|
| 1 | stall speed `u_s` | direct: `sqrt(g / C_max)`, the speed where the lift reach is 1 g | `C_max` |
| 2 | g limit `n` | direct: `a_max / g` | `a_max` |
| 3 | cruise | flown: the terminal speed coasting straight | `C_D0` |
| 4 | climb | flown: the time to fill the store from empty to `(1 − β) E_max`, coasting straight from cruise | `a_0` |
| 5 | ceiling | direct: the store size `E_max` | `E_max` |
| 6 | low-speed acceleration | flown: `du/dt` coasting straight at the stall speed, store empty | `F_s` |
| 7 | sustained turn rate | flown: the steady flight-path turn rate holding full stick while coasting, once the speed has settled | `k` |
| 8 | bleed rate | flown: the specific-energy rate `u du/dt + (dE/dt)/m` at corner speed, full stick, coasting, with the nose at its steady angle of attack | `e` |
| 8b | nose reversal time | flown: from a settled full-stick turn one way at corner speed, the time after reversing the stick until the angle of attack reaches 90% of its new steady value | `ω_c` |
| 9 | dive acceleration | flown: `du/dt` at the start of a boost from cruise, the store half full | `a_d` |
| 10 | zoom | flown: `du/dt` at the start of a reverse from cruise, the store half full | `a_c` |
| 11 | AoA limit | direct: `α_s` | `α_s` |
| 12 | vision range | direct | vision range |

- **Shown, not set:**
  - corner speed `u_s sqrt(n)`, and the minimum radius;
  - the instantaneous peak turn;
  - the gun's offset per stick level;
  - the turn rate and speed per stick level at the coast datum;
  - top speed;
  - firing endurance coasting, climbing, diving and in a hard turn.
- **Sub-tick timing:** times (4, 8b) interpolate linearly between ticks, so that they are
  continuous in the parameters and comparable across tick rates.
- **Combat handles:**
  - damage and fire interval (DPS shown);
  - muzzle speed;
  - spread;
  - range: the distance a bullet fired at cruise along the flight path travels in its
    life. It sets the bullet drag in closed form (`s = ln(1 + k_b w₀ t)/k_b` at `n = 1`,
    solved for `k_b` by bisection), checked against a bullet flown in the game's code;
  - shots from a full store, which sets the energy per shot;
  - shield capacity, recharge rate and delay;
  - head-on fraction;
  - hit radius.

## 4. The solver

- **`bnb/game/handling.py`:** `measure(card, rules) → Handles` and
  `solve(handles, rules) → card`, per type, batched: every candidate flies at once, one
  per environment, through the per-environment card (`ShipTypes` with a batch axis).
- **The steps:**
  1. **First guess:** a closed form in log space, where handles are near products of
     powers of the parameters.
  2. **Correction:** Newton on the flown handles against the measured ones, with the
     Jacobian by finite differences in log space. The base and the 8 perturbations fly
     in one batch per iteration.
  3. **Stop** at 0.5% on every handle (soft; report the achieved accuracy).
- **Independence:** the handles are coordinates. Changing one re-solves with every other
  held, whatever the parameters must do.
- **Conditioning:** the solver reports the Jacobian's condition number and warns when
  two handles are nearly redundant.
- **Infeasible sets fail with a reason,** never a silent nearest fit (for example a g
  limit below 1, a cruise below the stall speed, a sustained turn above the
  instantaneous peak, or no convergence within the iteration budget).
- **Precision:** the solver runs in float64 on the CPU if the game's flight code runs in
  float64 unchanged; otherwise in float32, with the achieved accuracy reported. The
  resulting card is stored in float32.

## 5. Tick rates

- **Time-native rules:**
  - every per-type time (fire interval, recharge delay) is stated in seconds, and
    converted to whole ticks at construction;
  - a tick rate at which any is not a whole number of ticks is refused with the reason
    (today's 1/15 s fire interval is 2 ticks at 30 Hz and 4 at 60 Hz, but not whole at
    25 Hz).
- **Supported rates:** 25, 30 and 60 Hz for flight. Weapon cadence is validated per rate.

## 6. The reference chart (tool)

- **`bnb ship-card [--card PATH] [--out DIR]`** prints every handle, parameter and shown
  quantity per type. It writes the charts:
  - the Ps curve;
  - the E-M "doghouse": turn rate against speed, with the stall boundary, the g line, the
    sustained curve per stick level and corner speed;
  - the similarity ratios against the P-51D set in `ship-card.md` ("Similarity ratios"),
    with that table's inputs.
- **Plotting (proposal):** matplotlib as an optional `tools` extra, so the core install
  stays lean.

## 7. Acceptance

1. **The game is unchanged** at the default card: every existing test and fixture passes
   untouched. Report the bench cells before and after (the bleed exponent's cost).
2. **Round trip:** handles → parameters → measured handles, within 0.5%, for the default
   card's own handles and for at least 8 varied cards per type spanning the design
   envelope.
3. **Independence:** perturbing each handle by ±5% and re-solving moves that handle by the
   perturbation, while every other measured handle stays within 0.5%.
4. **Tick-rate robustness:**
   - the same parameters give handles that agree within 0.5% at 25, 30 and 60 Hz;
   - energy conservation holds at each rate: drift with drag, engine and fire off within
     the game's existing bound;
   - an unsupported rate for a card's weapon cadence is refused with its reason.
5. **The cruise fix:** for solved cards, cruise varies by at most 0.5% across the store.
   The default card's variation is reported.
6. **Infeasible sets** fail with a reason, one test per listed reason.
7. **The chart tool** runs on the default card and writes its charts. Its printed shown
   quantities match an independent measurement.
8. **Each flown handle** matches an independent flight of the game for one card: a hand
   computation for the direct ones, a separate simulation script for the flown ones.
