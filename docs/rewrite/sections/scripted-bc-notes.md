# Sections 4 and 7 (the scripted agent and BC): notes

Branch `section/scripted`. The spec is `scripted-bc-spec.md`.

## Progress (working state)

- **Done:** part A code: `bnb.game.handling` (characteristics), `bnb.controllers.scripted`
  (`Scripted`, `distribution`, `marginals`) with `scripted_parameters`, `scripted_combat`,
  `scripted_frontline`; `bnb.runtime.evaluate` (fresh games to their end; `win_rate` uses
  it). Tests: `tests/sections/test_scripted.py`, `test_handling.py`. First scores (1024
  games, fog on, starting parameters): team deathmatch 0.997 against the chaser, 1.0
  against random; Frontline 0.94 against the chaser (512 games), 1.0 against random.
- **In flight:** the capped parameter search (`/tmp/bnb-scratch-scripted/search.py`).
- **Next:** the acceptance runs (2048 games against random, 1024 against the chaser, per
  mode), the speed measurement, part A docs; then part B (BC).
