# Lean core: work log

Newest last. Each entry: time (KST), item from core-backlog.md, what was done, how it was
checked, the commit. Numbers are measured unless marked otherwise.

## 2026-11-23

- 02:37 Overnight session starts (author asleep until 09:00). Plan: section A (flight
  physics behind a switch), then B1, then section C when blocked. Machine: laptop,
  platform profile "performance", GPU idle.
- 03:20 A1-A8, A11: the core flight model behind `GameConfig.flight_model` ("v2" default,
  "core" new) and `GameConfig.core()` (core model, Vizia only).
  - What changed (lean/game/core.py): in `_setting`, the vertical demand is held first
    (`held = softmin(demand, budget)`, no `1 - sink_room` exemption) and the vertical lift is
    that held value (re-deriving it through a smooth minimum lost about 4% between near-equal
    values and showed up as sinking at full stick); no charge term (the engine never feeds
    the store); thrust from the card's full-store power (unchanged, constant) times a firing
    scale; smootherstep edges over `store_edge_band` (5%). In `ship_constants`: store =
    `store_loops` (2) x g x 2 x corner radius (corner radius with 1 g on the wing), spawn at
    half; dive and climb rates both the mean of the card's two (38.6 m/s^2 for Vizia); bullet
    cost 0. In `step`: firing scales thrust by `fire_thrust_fraction` (0.0) on the ticks it
    fires. Vizia's store: 32,169 J/kg (v2: 497,563, about 15x more).
  - Engine power choice: the card's full-store power (Vizia 2.04 kW/kg), the power v2 already
    uses for thrust at every store; the 7-9 kW/kg "surplus" that only charged the store is
    gone. So top speeds are unchanged.
  - Checked: tests/test_core_flight.py, 9 assumption tests (store size and spawn; coasting
    straight holds altitude; full stick above the 1 g speed holds altitude; the engine never
    raises the store; below the 1 g speed the ship sinks; dive and climb equal and opposite;
    no dive below empty or climb above full; the edge taper never reverses or overshoots;
    firing costs speed and no store). `pytest lean`: 57 passed, 1 xfailed (the recorded
    "mixed" spawn), 36 s, CPU one thread. v2's parity tests unchanged (v2 is the default).
  - Finding for tuning (A12): at full climb rate the store moves about 36% of the band per
    second at 300 m/s, so the 5% edge band is crossed in about 4 ticks and the taper acts
    almost like a wall. Options for the author: a lower exchange cap, a wider band, or a
    larger store.
  - Sandbox: `--core` flies GameConfig.core() (Vizia by default); README updated.
- 03:45 A9: measurements in docs/rewrite/core-flight-measurements.md (script
  docs/rewrite/sections/turns/measure_core.py). Key: coasting holds altitude (50.0% after
  120 s; v2 climbs to 95%); full stick settles at 111 m/s, 25 deg/s (v2 139, 38.6); the store
  is crossed in about 1.5 s per half (a tuning question); firing costs 6 to 7 m/s per second.
  A10 (core charts) handed to a background agent (chart scripts take FLIGHT_MODEL).
- 03:55 B1: the observation's speed and velocity are in the world frame (proper speed over
  the refractive index, as v2's observe.py), per the author. Test
  tests/test_observe.py (a ship inside a strong field). `pytest lean`: all pass.
- 04:35 B3 and the game half of C4: fields drawn per environment at every reset, and field
  rows in the observation.
  - `GameConfig.field_count` (0: the fixed `fields` as before; `GameConfig.core()` uses 10)
    and v2's real-scale elimination map rules (src/bnb/game/physics/spawning.py `new_map`,
    values from `api.rules_of(PRESETS["real_scale_elimination"])`): point-symmetric pairs
    plus a centre field for an odd count; centres uniform in 0.85 of the arena; radius
    log-normal, median 400 m, sigma 0.6, clamped to [120, 1200] m, times the map scale
    (arena / 2,600 m: 1.615 at 4,200 m, so median 646 m); band 160 m; level -2, -1, +1, +2
    half-octaves, a quarter each. Stored per environment as (E, 1, F) constants;
    `reset_done` redraws them where a game ended; `medium` broadcasts over environments.
  - Contract: `Obs.field` (E, F, 3: radius / 1000 m, band / 40 m, level / 2, v2's input
    scales), `Obs.field_pos` (E, F, 2), `Obs.field_mask` (E, F). The model does not read
    them yet (C4's model half).
  - Sandbox: draws the first environment's drawn fields; Tab keeps the fields.
  - Checked: tests/test_fields.py (5 tests: v2's bounds and symmetry; per-environment fields
    and medium; redraw only where done; observation rows; a 60-tick core game with fields
    stays finite). `pytest lean`: 63 passed, 1 xfailed.
- 05:05 A10 and a fix found by it.
  - A10: the chart scripts draw either model (`FLIGHT_MODEL=core`, files suffixed `-core`);
    v2 charts re-rendered pixel-identical. Core charts: doghouse per ship and comparison,
    who-wins, acceleration, dive-climb, bleed, circles. Store levels for core: 97.5% (inside
    the top band), 50% (spawn), 2.5% (inside the bottom band).
  - Fix: inside an edge band, dive and climb faded only the trade, not their effect on the
    wing, so diving on an empty store still freed lift for the turn (a free turn boost on
    the deck) against the decision "at the edges dive or climb acts as coast". Now the band
    taper fades dive and climb's vertical demand back toward 1 g as well as the trade, and
    a vertical shortfall (sinking) is the only part scaled by the bottom band. A first
    version faded the trade twice (band crossing 3.5 s instead of 1.6 s); corrected, the
    measurements are unchanged from 03:45. New test: at the edges dive and climb act as
    coast on the wing. `pytest lean/game`: 32 passed, 1 xfailed.
  - Core numbers (charts, coast, spawn store; v2 in brackets): best sustained Vizia 160 m/s,
    32 deg/s [260, 54]; Jujak 160, 39 [248, 64]; Yongma 164, 26 [262, 44]. Corner and top
    speed unchanged. Minimum full-stick radius now away from the stall: Vizia 213 m at
    224 m/s. So the three speeds separate: minimum radius about 220 m/s, best sustained
    about 160 m/s, corner 268 m/s (v2: all three at about the corner).
