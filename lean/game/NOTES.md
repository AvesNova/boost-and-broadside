# lean/game notes

What the game package is today and what is open. Decisions: docs/rewrite/core-decisions.md.
History (v2 parity, the waste fixes, fog, the flight changes): docs/rewrite/core-worklog.md.

## What it is

- `card.py`: the ship card (one row per type, `TYPE_NAMES`; today the P-51D alone) and the
  shared rule constants. Every row is made by `tuning.py` (a test holds it equal).
- `core.py`, pure functions of tensors:
  - `GameConfig()` is the core game: elimination, no shrink, arena 4,200 m, boundary ramp
    300 m, clock 5,400 ticks, "mixed" spawn, lineup ("p51d",), ten fields drawn per
    environment at each deal, ghost bullets on, fog on, the flight model's store of two loops
    at corner speed with 5% edge bands, firing cuts thrust to `fire_thrust_fraction` (0).
    `field_count=0`, `ghost_bullets=False`, `fog=False` are ablations.
  - Spawns (`deal`), the rule a setting chosen at launch. "mixed": every ship uniform over the
    spawn disc (0.85 R), heading uniform. "sides": a random axis per environment; each team
    uniform over a half disc of radius 0.85 R - 0.05 R whose flat edge is 0.05 R from the
    dividing line (so at least 420 m between the teams at 4,200 m), noses along the axis
    toward the other side within 15 degrees. Edge rule, both: a ship within 1,000 m of the
    boundary whose heading points outward has it mirrored in the tangent.
  - `training_spawn` (lean.train turns it on; lean.eval forces it off; play and the sandbox
    leave it off): per ship a proper speed uniform from the stall speed (sqrt(g / lift),
    90.4 m/s for the P-51D: the clean wing's) to the top level speed (`cruise_speed`, 335 m/s), shield and store
    each uniform 0 to full; `standard_spawn_share` (1/8) of environments, drawn per
    environment at the deal, take the standard spawn (full shield, half store, 145.3 m/s over
    the map).
  - The one-tick action delay: `State.pending` (E, N, 3) is the action the next tick
    applies; `step` applies it and stores the action it is given. Neutral (straight, coast,
    no fire) after a deal. Every agent goes through it (network, scripted, chaser, the
    sandbox's player).
  - `ship_card` / `ship_constants`: the run's constants (`contract.Constants`), built once per
    flight settings, lineup and arena. A constant equal for every type of the lineup is a
    Python number; one that differs is a row of a per-slot table (E, C, N) gathered at the
    deal. Derived per type: cruise speed, stall speed, corner speed, bullet launch speed, gun range,
    and the scripted agent's references (the best sustained turn's rate, speed and radius, the
    time to turn 90 degrees at it, the corner rate, the store's turn; `quasi_steady`). The
    arena radius is a constant too. The scripted opponent reads every number it uses from these
    constants by name (it holds no ship characteristics of its own).
  - `flight_tick`: half a kick, a drift, half a kick (`_setting`, `_rates`, `_kick`); the
    medium of the fields bends paths, with a no-fields path for F = 0. The wing's maximum lift
    is a function of speed (`_wing`, the automatic flaps: per type, `flap_lift` below
    `flap_full_speed_ratio` x the clean stall speed, a smootherstep to clean at
    `flap_clear_speed_ratio` x it), held through each half-kick at its starting speed; the
    flaps' drag is on the lift beyond the clean maximum (`flap_drag_scale`). `steady_rates` /
    `quasi_steady`: the same functions with the nose settled, the one quasi-steady evaluation
    (the derived characteristics, tuning's performance and envelope, the web tool).
  - `step`, in stages: `_fire` (a living ship with the trigger held and the gun ready fires;
    no store cost; with ghosts every living ship writes a bullet each tick), `_fly`,
    `_move_bullets`, `_hit_test` (swept closest approach, every bullet against every ship,
    first contact), `_damage` (kinetic x closeness x aspect, the head-on term in the hit
    angle's cosine), `_boundary_damage`, `_shields` (the quiet rule: coast, at most the fine
    stick, no trigger), `_end_check` (a wipe; at the clock more living ships, then more
    total shield over the living ships, else a draw), `_bullet_ends` (moves left, each
    bullet's outcome), then the fog's `perceive`.
  - `deal` / `reset` / `take_deal`: fresh states; lean.learn draws deals once
    per segment and takes them as environments finish.
  - Fog: `perceive` (sight by v2's rule, or a shot or a real bullet's hit that tick),
    `enemy_view` (every ship as its enemy team last saw it), `team_view` (own ships live,
    enemies from `enemy_view`). With fog off both return the state.
  - `observe`: one team's `Obs` (contract.py), built under fog from `team_view` only. The
    ship row has no position scalars (Obs.pos carries position, for the position code and
    rotary). `Obs.pending`: own living ships' joint pending action, `NO_PENDING` (42) for
    enemies (not observable; their shot shows as the fired flag), dead and absent rows.
- `tuning.py` (analysis, not imported by the game): a real airframe into its point mass
  (`airframe_row`; `fit_point_mass` sets C_D0 and the propeller efficiency to a tested top
  speed and climb), or into a game row with its game terms on and the zero-lift drag, thrust
  power and induced drag solved to fly the point mass's top speed, climb and best sustained
  rate (`game_row`: the P-51D's row, with `P51D_TERMS`); a design's handles (`handles_row`)
  into a card row; `performance` and `envelope` (quasi-steady numbers, through
  `core.steady_rates`); `time_scale` (the card at K times real time, for the sandbox's
  `--time-scale`). docs/rewrite/p51d-model.md.
- `StepOut` (contract.py): damage taken, deaths, done and result, each bullet's outcome and
  firing tick (the hit head's labels), the attacker-by-target damage matrix, boundary damage
  and shield restored (the rewards' ledger). The shot flag is `State.fired`.

## Checks

- Tests `lean/game/tests`: flight rules (test_flight.py), damage (head-on term, the damage
  matrix), fields, fog (8 rules), ghost bullets, invariants (finite, inert padding, team
  swap), the observation in a field, tuning (card rows, the P-51D's point mass flies its
  flight test and its game row its point mass, the handles route, time scales). Flight:
  every type's tightest sustained turn at least 10% slower than its best sustained rate,
  below the corner (P-51D 0.826); the flaps smooth and their drag only beyond the clean wing.
- Rules tests (test_rules.py): each clock tie-break case and the wipe, no edge-facing spawn
  under either rule, the "sides" spread and gap, the training spawn's ranges, uniformity and
  one-eighth standard share, the delay (an action takes effect exactly one tick later) and
  the pending action in own rows only.
- Operations (docs/rewrite/sections/scoreboard/game_ops.py, uncompiled, CPU, one thread,
  E = 256, 5v5; compute + view, 2026-11-23 after G2; the step re-measured 2026-11-24 after
  A1): step 1,625 + 210 (no fields 1,287 + 162; before A1's flaps 1,557 and 1,219: +68 for
  the wing's speed dependence and the flaps' drag), observe 102 + 54 (no fields 97 + 53),
  reset_done 265 + 82 (training spawn 277 + 82, "sides" 289 + 82; no fields 105 + 35),
  scripted act for both teams 439 + 109 (2026-11-24, after E2's BFM; lean/scripted/NOTES.md).
  The CPU times beside them vary by about 25% between runs on the laptop; the counts are
  the measure.

## Feature scales

The ship scalars' fixed scales (the same for every type), from their ranges in scripted
play: docs/rewrite/sections/scoreboard/feature_ranges.py, both teams the scripted yardstick,
the core game 4v4 with the training spawn, 256 environments x 3,000 ticks, every living ship
every third tick (2026-11-23). Percentiles are of the raw quantity.

| Feature | Transform | Scale | 1st / 99th percentile (raw) |
|---|---|---|---|
| world velocity x, y | symlog(x / s) | 200 m/s | -327 / 327 m/s (extremes +-770) |
| world speed | symlog(x / s) | 200 m/s | 77 / 457 m/s (extremes 48, 776) |
| nose rate | symlog(x / s) | 1 rad/s | -1.72 / 1.71 rad/s (extremes +-5.5) |
| angle of attack | linear | 0.25 rad | -0.245 / 0.244 rad |
| angle of attack cos, sin | as is | 1 | sin -0.243 / 0.242 |
| store fraction | linear | full store | 0.002 / 0.68 |
| shield fraction | linear | full shield | 0 / 1 |
| shield | linear | 100 | 0 / 100 |
| shield wait | linear, clipped to [0, 1] | 3 s | 3 / 3 s (min 0) |
| gun cooldown | whole ticks | 1 tick | 0 / 0 (always 0 at one shot a tick) |
| inside the boundary | linear, clipped to [-1, 1] | 1,000 m | 752 / 4,049 m |
| outward direction, heading | unit vectors | 1 | |
| real bullet share | linear | 11 slots | 0 / 0.91 (85% zeros) |

## Open

- Inputs that carry little: the angle of attack's cos (at least 0.945 at the measured
  extreme angle) and sin (within 2% of the angle there) repeat the angle; the gun cooldown
  is always 0 at the P-51D's one shot a tick; the shield wait is at its 3 s maximum at both
  percentiles. Kept for other types and other play; a model step may drop them.
- The clock (5,400 ticks) waits for the scripted wipe-time measurement (step X).
- Engine power and the ship roster wait for the WWII research (core-decisions.md).
- On the GPU: the dense (E, N K, N) hit test's memory at large team sizes.
