# Section 1b (ship card): notes

Branch `section/ship-card`. The spec (`ship-card-spec.md`) is implemented in `bnb.game`: the
per-type card and the bleed exponent in the physics, and the new package
`bnb.game.handling` (handles, manoeuvres, solver, weapons, shown quantities, chart).
`bnb ship-card` prints and plots a card. See "Progress" at the end for the test status.

## Design

### The card (`game/ship_types.py`)

- **New per-type fields** (spec 1): the bleed exponent `e` (2 by default), and a per-type
  **scale** (1 by default) for each value that was shared: the induced-drag factor `k`, the
  stall angle `alpha_s`, the coast, dive and climb exchanges `a_0`, `a_d`, `a_c`, the vision
  range, the bullet spread and drag, the shield recharge rate and delay, and the head-on
  fraction.
  - A type's value is the shared value times its scale: `alpha_s = flight.stall_angle *
    stall_angle_scale`. `ship_types.SCALED` maps each scale to its shared value.
  - Why scales and not absolute per-type values: see decision 1. The default card is the
    old game bit for bit (`x * 1.0` is exact, and the shared constants keep their values).
- `ship_types(dtype=...)` builds a float64 card for the offline tools; the game's card stays
  float32.
- **Whose value:** the flight values are the flying ship's; the bullet spread and drag are
  the shooter's; the head-on fraction is the target's (frontal armour); the shield values
  and the vision range are the ship's own.

### Physics changes (spec 2)

- **The bleed exponent** (`physics/flight.py`, `bleed`): induced drag is
  `k (C^2 + (z/u^2)^2)^(e/2)`, written `s * s^(e/2 - 1)` with `s` clamped away from 0 inside
  the power. At `e = 2` the power is `pow(s, 0) = 1` exactly, so the drag equals the old
  formula bit for bit (tested on 10 000 values, zero included).
- **The cruise fix** is imposed by the solver (`P = P_min + m a_0 u_cruise`, `P_min` the
  power whose flown full-store cruise meets the handle). The default card's numbers are
  unchanged.
- **Plumbing, not physics:**
  - **Vision range** is the observer's type's. `perceive(rules, state)` takes no card, so
    the range lives in the state: `Match.vision_range` (B, N), set by `reset` and by every
    `resolve` from the card. `blank` fills it with the shared range. `visibility` no longer
    takes the rules.
  - **Bullet drag** is the shooter's: `move_bullets` takes the card and the owners' types;
    `bullet_motion(rules, bullets, fields, drag_scale=1.0)`.
  - **Bullet life is time-native:** `rules_of` sets `bullet_moves` from a 0.4 s life
    (`BULLET_LIFE`): 10, 12 and 24 moves at 25, 30 and 60 Hz.

### Handles (`handling/handles.py`)

`Handles` holds every handle, flight then combat, as tensors shaped like the card. Fourteen
flight handles: the spec's thirteen plus the zero-lift drag (decision 2).

| handle | how | sets |
|---|---|---|
| stall speed | `sqrt(g / C_max)` | `C_max` |
| g limit | `a_max / g` | `a_max` |
| cruise | flown: the straight coasting equilibrium at a full store | `P_min` |
| climb | flown: empty to 98% of the store, coasting straight from cruise | `a_0` |
| ceiling | `E_max` | `E_max` |
| low-speed acceleration | flown: `du/dt` at the stall speed, store empty, coasting | `F_s` |
| sustained turn rate | flown: the settled full-stick coasting turn | `k` |
| bleed rate | flown: `u du/dt + dE/dt / m` at corner speed, full stick, nose at its steady angle | `e` |
| nose reversal time | flown: from a settled full-stick turn at corner speed, to 90% of the reversed steady angle | `omega_c` |
| dive acceleration | flown: `du/dt` at the start of a boost from cruise, store half full | `a_d` |
| zoom acceleration | flown: `du/dt` at the start of a reverse from cruise, store half full | `a_c` |
| AoA limit | `alpha_s` | `alpha_s` |
| vision range | direct | vision |
| zero-lift drag | direct (held) | `C_D0` |

Combat handles: damage, fire interval (s), muzzle speed, spread, gun range, shots per full
store, shield capacity, recharge rate and delay (s), head-on fraction, hit radius, spawn
energy.

### Manoeuvres (`handling/manoeuvres.py`)

- **One batch:** every manoeuvre of every type of every card flies at once, one environment
  per card and one ship per (manoeuvre, type), through `airframe_of` and `flight_tick`
  unchanged, in the card's dtype (float64 for the solver), at `n = 1` (one field 10^9 m
  away).
- **Cruise** is the fixed point of the flight tick, coasting straight at a full store (the
  terminal state: the store no longer moves). It is bracketed on a log grid of one-tick
  flights from the stall speed to 30 times it (the last sign change of the speed's gain,
  so the stable root above the low-speed one) and bisected. A 200 s flight agrees to 1e-6.
- **The turns** (sustained, bleed, reversal) fly at a quarter-full store: below the engine
  knee, so the store does not change the engine, and clear of both fade bands. At an empty
  store, a hard turn's negative exchange (the lift budget's shortfall, `(1 - q) a_d`) is
  clipped by the fade, which changes the settled speed.
- **Start rates** (low-speed, bleed, dive, zoom) are the slope at `t = 0` of a
  least-squares quartic over the first 0.2 s. A cubic left the bleed rate 0.9% short of the
  instantaneous rate (a one-step difference at 3000 Hz: -4412 W/kg for the default
  fighter, against -4373 by the cubic at every rate); the quartic gives -4403 to -4407 at
  25, 30, 60 and 3000 Hz. Found by the independent-flight test.
- **The sustained turn's store is held** at a quarter (reset every tick), and so is the
  coast datum's in the chart tool: a long turn otherwise charges the store past the engine
  knee (the heavy's did in the independent test's 60 s flight, 0.19% off). The bleed has
  its own manoeuvre, free.
- **Times:** the climb interpolates linearly between ticks; the nose reversal through the
  cubic on the four ticks around the crossing (decision 4).
- **The sustained turn** starts at corner speed with the nose at its steady angle (so its
  first 0.2 s is also the bleed manoeuvre), flies 20 s, and takes the Aitken limit of the
  path's turn rate over the last three seconds.
- **The steady angle of attack** is the nose oscillator's rest point: spec 3.5 step 7's
  command at the speed, with the nose rate equal to the path's turn rate.

### Solver (`handling/solver.py`, `weapons.py`)

- `measure(card, rules) -> Handles` and `solve(handles, rules) -> Solution(types, error,
  condition, redundant, iterations)`, for one card `(T,)` or a batch `(B, T)`.
- **Direct** handles set their parameters in closed form; the **combat** handles too
  (`weapons.combat_card`), with times converted to whole ticks or refused (spec 5) and the
  gun's drag from the range by bisection on `ln(1 + k_b w0 t) / k_b`.
- **Flown:** Newton's method (Levenberg-Marquardt) on the eight flown handles in eight
  log parameters (`P_min`, `a_0`, `F_s`, `k C_max^(e-2)`, `e`, `omega_c`, `a_d`, `a_c`),
  relative residuals (with a floor for handles near zero), a finite-difference Jacobian
  (step 1e-5). Each iteration flies the base card and its eight perturbations in one batch.
  - **First guess:** at each of seven bleed exponents (0.75 to 4), every other parameter
    inverts the game's own instantaneous equations (`setting`, `rates`) at its manoeuvre's
    state: the climb integral of `1 / (1 - phi)` for `a_0`, the null power at cruise for
    `P_min`, the inverse soft minimum for `F_s`, the induced drag where the sustained turn
    settles for `k`, a damped oscillator's 95% rise for `omega_c`. The seven fly in one
    batch, and each card and type starts from its best.
  - **Steps:** Levenberg-Marquardt from each card's best point (Gauss-Newton while it
    improves, toward scaled gradient descent while it does not), at most 0.5 in any log
    parameter, the bleed exponent kept in [0.5, 5].
  - **Stop:** every flown handle within 1e-4 (the spec's bar is 5e-3), at most 16
    iterations; a card beyond 0.5% fails with its reason: the handle it misses, its
    condition number, its two weakest handles, and the exponent bound if it is at one.
  - **Conditioning:** the condition number of each card's 8 x 8 Jacobian, and a
    `RedundantHandles` warning naming the two handles that dominate its weakest left
    singular vector when it exceeds 1e4.
- **Infeasible** sets raise `Infeasible` with the reason: a g limit at or below 1, a cruise
  at or below the stall speed, a sustained turn at or above `sqrt(a_max^2 - g^2) / u_corner`,
  no convergence within the budget; also non-positive handles, an AoA limit above 45
  degrees, a dive or zoom on the wrong side of the coast exchange, a range beyond the
  drag-free reach, a fire interval or recharge delay that is not whole ticks.
- **Precision:** the flight code runs unchanged in float64 on the CPU, so the solver does;
  the card is stored in float32.

### Chart tool (`handling/shown.py`, `chart.py`, `bnb ship-card`)

- `bnb ship-card [--card PATH] [--out DIR]`: the card file is JSON of ship-card fields
  (per-type lists, missing fields default). Prints every handle, every card field and the
  scaled values in physical units, the shown quantities and the similarity table; writes
  `ps_curve.png`, `doghouse.png` and `similarity.png`.
- **Shown** (all flown): corner speed; the instantaneous peak turn and its speed (the
  maximum over 160 speeds of the start turn rate at full stick, nose steady) and the minimum
  radius; per stick level (0.4, 0.75, 1) the settled coast datum's speed, turn rate and gun
  offset (the angle of attack); top speed (the peak of a straight boost from cruise with a
  full store); firing endurance coasting, climbing, diving and in a hard turn (from cruise
  with a full store, firing at every chance through the game's `fire`, until a shot is
  refused); damage per second.
- **Charts:** the Ps curve per stick level; the doghouse with the stall boundary, the g
  line, the turn rate per stick level, the sustained (coasting) curve over twelve stick
  levels and corner speed; the ten ship-card.md ratios against the P-51D, with that
  table's inputs.

## Decisions and deviations

1. **Per-type values are scales of the shared values** (deviation; spec 1 says the
   parameters move to the card).
   - Why: fixed code outside this section reads or sets the shared values, and must keep
     working at the default card:
     - the race adapter (`tests/reference/race_arm.py`, fixed) turns induced drag off
       through `FlightConstants(induced_drag_factor=0.0)` in the energy-conservation test,
       and calls `bullet_motion(rules, bullets, fields)` without a card;
     - Yemong (`models/yemong/state_code.py`, outside my boundary) codes the angle of attack
       over `2 * rules.flight.stall_angle` and the recharge wait over
       `rules.shield_recharge_delay`.
   - With scales, every shared value keeps its meaning (setting it applies to every type,
     as the adapter assumes) and nothing is read and silently ignored. The alternative,
     absolute per-type values with the shared ones kept as defaults, breaks the energy test
     and leaves the shared fields as values that no longer act.
   - The chart tool prints both the scales and the physical values.
   - Request 2 removes the scales.
2. **The zero-lift drag is a held handle; the cruise sets `P_min`** (deviation; spec 3 says
   cruise sets `C_D0`).
   - Why: the spec's system is one handle short. The flight has fourteen parameters per
     type (`C_max`, `a_max`, `C_D0`, `P`, `P_min`, `F_s`, `E_max`, `a_0`, `k`, `e`,
     `omega_c`, `a_d`, `a_c`, `alpha_s`). The cruise fix removes one,
     `P = P_min + m a_0 u_cruise`; its other half, "`P_min` = the null power", is the
     definition of the cruise speed, not a second constraint. That leaves thirteen flight
     parameters for twelve flight handles. With `P_min` computed from the target cruise, the
     cruise handle no longer depends on `C_D0` (the drag and the power that meets it move
     together), which is the missing equation showing up as a near-zero row.
   - Measured at the re-solved default card: the cruise's sensitivity to `log C_D0` is
     -0.010 (fighter) and -0.012 (heavy) with the spec's parametrization, against 0.37
     to `log P_min` with mine. The Jacobian's condition number is 9.8e3 and 1.7e4 against
     about 35.
   - So `C_D0` is a direct handle (held, like the store size), and `P_min` is solved so that
     the flown cruise meets its handle (`P_min` is still the null power, cap included).
     Request 1 offers other choices for the fourteenth handle.
3. **Vision range in the state** (`Match.vision_range`). `perceive(rules, state)` takes no
   card, and its callers (the runtime, the fixed adapter) are outside this section. `reset`
   and every `resolve` write the observers' ranges from the card; `blank` writes the shared
   range, so the adapter's `unflat` keeps working. Request 3.
4. **The nose reversal's time interpolates through a cubic**, not linearly (deviation; spec
   3 says linearly). The reversal is only four to eight ticks long. Linearly, the default
   heavy's time differs by 0.52% between 25 and 60 Hz; through the cubic on the four ticks
   around the crossing, by 0.15%. The climb's time stays linear (its rates agree to 2e-7).
5. **The solver's parameters:** `P_min`, `a_0`, `F_s`, `k C_max^(e-2)` (the induced drag at
   full lift, over `C_max^2`), `e`, `omega_c`, `a_d`, `a_c`, in logs.
   - `k` alone has units m^(e-1), and in (log k, e) the bleed exponent mostly rescales the
     whole polar: the default card's condition numbers were 1.1e3 and 1.4e3. With the
     induced drag at full lift, e reshapes the polar below full lift, and they are 36 and
     34. The physics is unchanged: the card stores `k`.
6. **Levenberg-Marquardt, from a scan of the bleed exponent.**
   - The first guess solves the closed forms at each of seven bleed exponents (0.75 to 4),
     flies them in one batch, and starts each card from its best.
   - A Newton step from e = 2 can fall into a basin of low e and huge `P_min` that misses
     the climb: there, the straight-flight induced drag `(z/u^2)^e` is large, the engine
     power is far above the static-thrust cap, and the cruise depends on the store again.
     One of the varied cards (bleed exponent 2.78) failed this way; with the scan and
     Levenberg-Marquardt it recovers its own parameters to 1e-14.
   - The bleed exponent is kept in [0.5, 5]; a failure at a bound says so.
7. **Batch solves fail card by card.** `solve(..., strict=True)` (the default) raises
   `Infeasible` with every failing card's reason; `strict=False` returns them in
   `Solution.failures` and `Solution.converged`. The acceptance tests solve 37 cards in one
   batch this way.
8. **Cruise is measured as the flight tick's fixed point** (coasting straight at a full
   store), found by bisection on one-tick flights, rather than by a long flight. It is the
   terminal state, and the fixed point of the discrete tick equals the equilibrium at any
   tick rate. A 200 s flight agrees to 1e-6 (test).
9. **The turns fly at a quarter-full store**, and the start rates are a cubic's slope over
   0.2 s (the spec leaves both open). See the design section.
10. **The head-on fraction is the target's** (frontal armour). The game spec's 7.1 gives the
    damage the shooter's value and says nothing of the fraction.
11. **The bullet life is time-native** (0.4 s at any tick rate), so the gun range handle is
    the same at 25, 30 and 60 Hz.
12. **`bnb ship-card` is dispatched by its first word**; `bnb` with options still trains, so
    the README, the docs and their command checks are unchanged.
13. **Solver tolerance 1e-4** on every flown handle, so that a re-solve leaves every other
    handle well inside the independence test's 0.5%.

## Measurements

### The default card's handles (30 Hz) and its re-solve under the cruise fix

| handle | fighter | heavy |
|---|---:|---:|
| stall speed | 31.62 m/s | 35.03 m/s |
| g limit | 8.00 | 5.93 |
| cruise | 103.72 m/s | 105.99 m/s |
| climb (empty to 98%) | 159.9 s | 218.4 s |
| ceiling | 10 000 J | 14 000 J |
| low-speed acceleration | 9.585 m/s^2 | 14.46 m/s^2 |
| sustained turn | 69.86 deg/s | 72.46 deg/s |
| bleed rate | -4407 W/kg | -2062 W/kg |
| nose reversal | 0.1967 s | 0.2887 s |
| dive acceleration | 30.01 m/s^2 | 30.46 m/s^2 |
| zoom acceleration | -23.99 m/s^2 | -23.53 m/s^2 |
| AoA limit, vision, zero-lift drag | 15 deg, 1024 m, 6.961e-4 1/m | 15 deg, 1024 m, 8.5e-4 1/m |
| gun range (at cruise) | 220.8 m | 204.7 m |
| shots per full store | 33.3 | 38.9 |

The climb is mostly the engine knee's taper: below the knee the store fills in about 16 s
(fighter); above it the charge rate falls as `1 - phi`, and the last 48% takes 144 s.

Measured at commit `845f2db5`{.plain} on the development laptop's CPU, float64. Re-solved from
these handles with the cruise fix (to 1e-13; the other card fields are as today):

| parameter | fighter | heavy | today |
|---|---:|---:|---:|
| `P` | 1179.6 W | 1441.2 W | 1200, 1500 |
| `P_min` | 867.8 W | 1121.7 W | 863.2, 1100 |
| `F_s` | 19.99 N | 23.95 N | 20, 24 |
| `k` | 10.37 | 7.13 | 11.4 |
| `e` | 1.975 | 1.883 | 2 |
| `omega_c` | 21.99 | 15.95 | 22, 16 |
| `a_0`, `a_d`, `a_c` | 3.006, 27.22, 26.78 | 3.014, 27.68, 26.32 | 3, 27, 27 |

- The fighter barely moves: its cruise already varies by only 0.17% across the store.
  Its engine below the knee loses 20 W (today's `P - P_min` is 337 W against `a_0 u_cruise`
  = 311 W), so `k` falls 9% to keep the sustained turn.
- The heavy moves more: today its cruise falls from 107.6 to 106.0 m/s as the store fills
  (1.5%). Under the fix it loses 59 W below the knee, and `k` falls 37% and `e` to 1.88 to
  keep the sustained turn and the bleed rate.
- Adopting these changes the goldens (spec 2: a later step).

### Round trip, independence, conditioning

One batch of 37 cards (`tests/sections/ship_card_cases.py`): the default card's handles,
8 varied cards (both types; factors 0.7-1.4 on every parameter, `e` 1.5 to 3, `alpha_s` 10
to 25 degrees, each satisfying the cruise fix), and the 28 perturbations of the default's
fourteen flight handles by -5% and +5%. The batch used the whole budget (16 iterations
after the scan, because of the failing card), 37 minutes on the loaded laptop CPU (load
average above 30 from other agents), at commit `845f2db5`{.plain}.

- **Round trip:** all 9 cards converge; every handle within 7e-7 (the float32 card), the
  flown ones within 7e-14 in float64.
- **Independence:** 27 of 28 perturbations meet the bar (the moved handle at its new
  value to 4e-7, every other within 0.22%; worst: the stall speed +5%, the heavy's
  sustained turn).
  - **The sustained turn +5% fails** (both types): the solve drives the bleed exponent to
    its bounds' neighbourhood and still misses the climb by 8.2% (fighter) and 29% (heavy). It fails with that reason (`Infeasible`, or `Solution.failures`).
  - **The stall speed +5% converges at an extreme card:** `e` 0.81 and 0.69, `P_min` 6000
    and 3920 W (7 and 3.6 times today's). The stall speed -5% needs `e` 3.3 and 3.7.
  - The cause is the pair of handles that set `k` and `e`: the sustained turn and the bleed
    rate are both full-stick turns with the lift near 0.9 `C_max`, so their lifts `S`
    differ by about 13%, and `e` changes their ratio only by `1.13^(de/2)`. Any change that
    moves one against the other (the stall speed moves both states' lift) needs a large
    change of `e`, and a low `e` raises the straight-flight induced drag `(z/u^2)^e` by
    orders of magnitude, which `P_min` must pay. Request 4.
- **Condition numbers** (the 8 x 8 Jacobian in relative handles and log parameters): 36
  and 34 at the default card; 19 to 79 over the varied cards; 6.8e5 and 2.3e4 at the stall
  speed +5% (the warning fires: sustained turn with bleed rate, and with low-speed
  acceleration).
  The spec's parametrization (cruise sets `C_D0`, `P_min` the null power at the target)
  gives 9.8e3 and 1.7e4 at the re-solved default.

### Tick rates (25, 30, 60 Hz; the default card re-solved and the 8 varied cards)

| handle | largest spread |
|---|---:|
| cruise, climb, dive, zoom, sustained turn | below 2e-5 |
| low-speed acceleration | 0.018% |
| bleed rate | 0.12% |
| nose reversal | 0.44% |

- The nose reversal is the closest to the bar, at the fastest noses (0.15-0.17 s: under
  four ticks at 25 Hz). Against a 600 Hz flight the 25 Hz time differs by up to 0.54% on
  the varied generator cards and 30 Hz by 0.40%: the game's nose integrator at 25 Hz, not
  the crossing's interpolation. (Before the bleed got its own manoeuvre and the quartic
  fit, one solved card reached 0.61%.)
- Energy conservation (drag, engine and fire off): drift below 1e-4 over 2 s at each rate,
  for bleed exponents 2 and 3 (test).
- Weapon cadence: the default card's 1/15 s is refused at 25 Hz ("fire interval: 0.0667 s
  is not a whole number of 0.04 s ticks (at 25 Hz)"); 60 Hz gives 4 and 6 ticks.

### The cruise fix: cruise across the store (empty to full, 51 stores)

- **Default card** (not re-solved): 0.17% (fighter), 1.53% (heavy), as the spec measured.
- **Re-solved default:** 0.93% and 0.80%. `P_min` meets the cruise at a full store; at an
  empty store the cap `softmin_4(P/u, F_s)` shaves the larger power more, so the empty
  store cruises slower.
- **Solved cards over 2%** (reported, spec 2): varied card 2 (3.4%, 2.1%), the stall speed
  +5% (18.9%, 10.4%: its engine power is several times what the static-thrust cap passes at
  low speed), the g limit -5% (2.13%, fighter). All other converged cards: 0.1% to 1.6%.

### The bleed exponent's cost

**Provisional:** every timing below was taken on the shared laptop while other agents ran
(CPU load average 20-45, the GPU shared). The coordinator re-measures at verification.

- **The flight tick alone** (compiled, RTX 4070 laptop, bench lock, the three bench cells'
  sizes, the power against the old induced drag alternated 40 times in one process, best
  of 40): 90.3 against 89.8 us, 240.0 against 239.0 us, 208.4 against 209.2 us. Not
  measurable.
- **The game step** (move and resolve, compiled, best of five runs per cell):
  1399 / 640 / 1531 us against the base's 1361 / 621 / 1497 us, about 3% slower. That is the
  per-type card's plumbing (the vision range written every tick, the head-on fraction
  gathered per bullet, the per-owner drag, the shields' per-ship values), not the bleed
  exponent.
- **`bench.run`** (seed 20261031), base then this branch, back to back:

  | cell | base rollout env-ticks/s | this branch |
  |---|---:|---:|
  | 114 ships, 22 fields, 360 envs | 84,969 | 84,820 |
  | 40 ships, 8 fields, 1024 envs | 374,301 | 370,765 |
  | 26 ships, 54 fields, 1576 envs | 392,742 | 404,124 |
  | realtime, CPU, 10 ships (p50 / p99) | 0.26 / 0.29 ms | 0.29 / 0.31 ms |

  The rollouts agree within run-to-run noise. The realtime tick is about 0.03 ms slower;
  the machine was loaded (other agents' tests), so this is an upper bound.
- The solver's own cost: eager float64 on the CPU, about 1-3 ms per flight tick per batch;
  the climb (up to 320 s of flight) dominates. One card: about 25 s per measurement on the
  loaded machine, 2-3 minutes per solve.

### Test failures found on the way, and what was wrong

- **The bleed rate against an independent flight** (0.9% off): the code. A cubic over 0.2 s
  missed the bleed's curvature; the start rates are now a quartic's slope.
- **The heavy's sustained turn against an independent flight** (0.19% off): the test. Its
  60 s turn charged the store past the engine knee. Both the manoeuvre and the test now
  hold the store below the knee, and so does the chart's coast datum (whose 20 s, unheld,
  was 0.7% off a longer flight: the code).
- **The chart's peak turn against an independent flight** (0.27% off, bar 0.2%): the test.
  Its one-tick difference at 3000 Hz read 0.16% high; it now uses a two-tick Richardson
  estimate, and the bar is the spec's 0.5% (the tool flies 30 Hz, 0.12% under the
  fine-tick limit).

## Changes outside `bnb.game`

- `src/bnb/cli.py`: `bnb ship-card ...` (the first word dispatches); `bnb ...` trains as
  before.
- `docs/` (coordinator's request, after merging v2's docs site): the explanation page
  `docs/explanation/game/ship-card.md` replaces the in-progress page; the guide
  `docs/guides/designing-a-ship.md`; the physics and perception pages follow the per-type
  card; `docs/roadmap.yml` gives the section the status `built`, which
  `docs/hooks/generated.py` now accepts ("is built on its branch and not merged yet").
- `pyproject.toml`: the optional extra `tools = ["matplotlib>=3.10"]`; `uv.lock` follows.
- `tests/sections/`: `test_ship_card_game.py` (23 tests, fast), `test_ship_card.py`
  (acceptance, `slow`), `ship_card_cases.py` (their cases), and in
  `test_game_section.py` the defaults test knows the new card fields.
- Not changed: `bnb.contracts` (the card and the state are game-owned), `bench/`,
  `tests/reference/`, the runtime, the models, the docs other than this file.

## Requests and open questions

1. **The fourteenth handle** (decision 2). The spec's handles are one short. I made the
   zero-lift drag a held handle. Designer-facing alternatives: the top speed (the boost's
   peak from cruise, now shown), or the time the speed takes to return to cruise after a
   disturbance (the drag slope at cruise).
2. **Absolute per-type values** (decision 1). Once the race adapter sets the card's values
   (its `k_ind` override through the card, `bullet_motion` with a drag) and Yemong reads its
   coding ranges from the card (the largest `alpha_s` and recharge delay over the types),
   the scales can become absolute values and the shared fields can go.
3. **`perceive` takes the card** (decision 3), so `Match.vision_range` can go. It touches the
   runtime and the adapter.
4. **The bleed rate's state** (independence). The sustained turn and the bleed rate are
   both full-stick turns at nearly the same lift, so `k` and `e` are nearly redundant
   through them: the sustained turn cannot move 5% alone, and the stall speed moving 5%
   needs the bleed exponent from 0.7 to 3.7. The author's framing ("turns hard but bleeds
   fast" against "turns gently but keeps energy") points at a gentle turn: measuring the
   bleed rate at stick 0.4 or 0.75 (lift near 0.4 or 0.6 `C_max`) would give `e` six to
   seven times the leverage. Not measured; a spec change.
5. **Write the bleed exponent around full lift** (proposal): `k C_max^2 (S / C_max^2)^(e/2)`
   instead of `k S^(e/2)`. Then `k` keeps its units (m) and its meaning (the induced drag
   at full lift) at any `e`. The solver already works in this form. It is equal at `e = 2`
   algebraically but not bit for bit, so it changes nothing today except the goldens'
   tolerance.
6. **The nose reversal's tick-rate margin** is thin (0.44% against 0.5%) and set by the
   game's nose integrator at 25 Hz; a faster nose than the varied cards' would cross it.
7. **Cards over 2% cruise variation** (spec 2 asks for a report): see the measurements.
   Solving `P` too so that cruise is equal at an empty and a full store (as the spec's
   measurement did) would remove most of it; it is a different reading of the fix.
8. **Yemong's coding ranges** use the shared `alpha_s` (times 2) and recharge delay; a card
   with a larger per-type value is clipped there (the Yemong section).
9. **The docs' command check** (`tests/docs/resolve.py`) parses every `bnb` command against
   `Train`'s options, so the docs show `bnb ship-card` without its options (they appear as
   plain spans). Teaching the check the `ship-card` command would let them be checked.
10. **`bench/results.jsonl`:** my `bench.run` lines are not committed (I may not change
    `bench/`); they are in the measurements above.
11. **The chart test needs the tools extra** (`uv sync --extra tools`); without matplotlib
    it skips with that reason. Everything else runs on the core install.
12. **The climb time** is mostly the engine knee's taper (90% of the default fighter's 160 s
    is above the knee). As a designer handle, "time to 98%" may read longer than intended;
    the time to the knee would read more directly.

## Size

- `src/bnb/game/handling`: 1280 lines: `solver.py` 365, `manoeuvres.py` 295,
  `shown.py` 237, `chart.py` 183, `weapons.py` 99, `handles.py` 79, `__init__.py` 13,
  `errors.py` 9.
- The rest of `bnb.game`: 181 lines added, 57 removed (the card, the bleed exponent and the
  per-type plumbing in flight, bullets, combat, sight, step, state and rules), and
  `cli.py`.
- Tests: `test_ship_card.py` 400, `test_ship_card_game.py` 235, `ship_card_cases.py` 116.

