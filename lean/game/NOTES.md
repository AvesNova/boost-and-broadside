# lean/game notes

What the game package is today and what is open. Decisions: docs/rewrite/core-decisions.md.
History (v2 parity, the waste fixes, fog, the flight changes): docs/rewrite/core-worklog.md.

## What it is

- `card.py`: the ship card (one row per type, `TYPE_NAMES`; today the P-51D alone) and the
  shared rule constants. Every row is made by `tuning.py` (a test holds it equal).
- `core.py`, pure functions of tensors:
  - `GameConfig()` is the core game: elimination, no shrink, arena 4,200 m, boundary ramp
    300 m, clock 5,400 ticks, "sides" spawn, lineup ("p51d",), ten fields drawn per
    environment at each deal, ghost bullets on, fog on, the flight model's store of two loops
    at corner speed with 5% edge bands, firing cuts thrust to `fire_thrust_fraction` (0).
    `field_count=0`, `ghost_bullets=False`, `fog=False` are ablations.
  - `ship_card` / `ship_constants`: the run's constants (`contract.Constants`), built once per
    flight settings, lineup and arena. A constant equal for every type of the lineup is a
    Python number; one that differs is a row of a per-slot table (E, C, N) gathered at the
    deal. Derived per type: cruise speed, corner speed, bullet launch speed, gun range. The
    arena radius is a constant too, read by the scripted opponent (its own fallback numbers
    go in step S).
  - `flight_tick`: half a kick, a drift, half a kick (`_setting`, `_rates`, `_kick`); the
    medium of the fields bends paths, with a no-fields path for F = 0.
  - `step`, in stages: `_fire` (a living ship with the trigger held and the gun ready fires;
    no store cost; with ghosts every living ship writes a bullet each tick), `_fly`,
    `_move_bullets`, `_hit_test` (swept closest approach, every bullet against every ship,
    first contact), `_damage` (kinetic x closeness x aspect, the head-on term in the hit
    angle's cosine), `_boundary_damage`, `_shields` (the quiet rule: coast, at most the fine
    stick, no trigger), `_end_check` (a wipe, or at the clock v2's health rule), `_bullet_ends`
    (moves left, each bullet's outcome), then the fog's `perceive`.
  - `deal` / `reset` / `take_deal` / `reset_done`: fresh states; lean.learn draws deals once
    per segment and takes them as environments finish.
  - Fog: `perceive` (sight by v2's rule, or a shot or a real bullet's hit that tick),
    `enemy_view` (every ship as its enemy team last saw it), `team_view` (own ships live,
    enemies from `enemy_view`). With fog off both return the state.
  - `observe`: one team's `Obs` (contract.py), built under fog from `team_view` only.
- `tuning.py` (analysis, not imported by the game): a real airframe (`airframe_row`) or a
  design's handles (`handles_row`) into a card row; `performance` (quasi-steady numbers);
  `time_scale` (the card at K times real time, for the sandbox's `--time-scale`).
- `StepOut` (contract.py): damage taken, deaths, done and result, each bullet's outcome and
  firing tick (the hit head's labels), the attacker-by-target damage matrix, boundary damage
  and shield restored (the rewards' ledger). The shot flag is `State.fired`.

## Checks

- Tests `lean/game/tests`: flight rules (test_flight.py), damage (head-on term, the damage
  matrix), fields, fog (8 rules), ghost bullets, invariants (finite, inert padding, team
  swap), the observation in a field, tuning (card rows, the P-51D's October fit, the handles
  route, time scales).
- Operations (docs/rewrite/sections/scoreboard/game_ops.py, uncompiled, CPU, one thread,
  E = 256, 5v5; compute + view, 2026-11-23 after the v2 models left): step 1,552 + 210 (no
  fields 1,214 + 162), observe 94 + 53, reset_done 280 + 81, scripted act for both teams
  186 + 96.

## Open

- G2 (decided, not built): the clock tie-break (ships alive, then shield, then a draw);
  the spawn rule a setting with "mixed" as the default and no edge-facing spawns; training
  spawns uniform in speed, shield and store with one eighth standard; the one-tick action
  delay with allies' pending actions observed; x/R and y/R dropped; feature scales from
  measured ranges.
- Engine power and the ship roster wait for the WWII research (core-decisions.md).
- On the GPU: the dense (E, N K, N) hit test's memory at large team sizes.
