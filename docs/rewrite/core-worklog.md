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
