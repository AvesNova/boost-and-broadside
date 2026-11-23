# Fresh-eyes review of lean (2026-11-23)

Read-only review of `lean/` at cda84858 by an agent that had not worked on lean. Nothing was
changed. Measured by the reviewer: `pytest lean -m "not slow"` 131 passed, 1 xfailed (47.5 s);
`ruff check lean` 378 findings (372 long lines, 2 unused imports, 3 import order, 1 E402);
5,969 source lines (4,884 without the sandbox and tuning.py), 3,649 test lines (725 in the two
v2-comparison files), 1,746 Markdown lines inside lean/. The sandbox runs headless in both
configs. Line-count savings below are the reviewer's estimates, not measured.

## Verdict

Lean does not repeat v2's scale: game, model and learner are compact and the core maths is
clean (packed state, built-once constants, one scripted call, batched league, no host reads in
the tick). It does repeat v2's pattern in small form, five ways:

1. Switches kept side by side after a decision (v2/core flight and damage, fog off/on, ghosts
   off/on, fixed/drawn fields, two spawn rules, `advantage_normaliser`, `positionless_rotary`,
   `left_is_even`).
2. A compatibility layer for test stand-ins (module injection, `carry_axis`, `is_dataclass`
   checks, a second `StepOut` to keep in sync).
3. Duplicated helpers and numbers (action split/join, offset draw, arena radius, ship
   characteristics, config values held twice).
4. Analysis tooling inside the runtime package (`tuning.py`; two analysis ship types widen the
   model's input).
5. Notes that pile up history and contradict the code.

Estimate: 300 to 400 source lines and about 900 test lines go or shrink with no loss of
behaviour; about 1,200 Markdown lines.

## Findings, ranked

| # | Finding | Decision needed |
|---|---|---|
| 1 | Two flight and two damage models in `game/core.py`, plus fixed fields, `core()`, inert card columns, `train.py --game`, sandbox `--core`, transition tests (`test_against_v2.py`, `test_parity_v2.py`, v2 parts of the core tests and `test_tuning.py`). About −80 source and −750 to −850 test lines | core sign-off |
| 2 | **Scripted:** target hysteresis (decided 0.8) is off everywhere: collect, eval and sandbox call `act` without memory. The action hold is a random switch (decided against). **Power names wrong under core:** "boost when slow or far" and "reverse when close" mean dive and climb in core, so it climbs when close and dives when slow. Personas carried as an 8-key dict of constants at spread 0; temperature per ship, decided per environment. Weakest-target term always runs | power semantics, persona spread, wire memory |
| 3 | Test stand-ins (`learn/tests/standins.py`, 191 lines) bend production: package injection in collect and eval, `carry_axis`, `is_dataclass` guard, generic `map_fields`/`seat_view`, erased types, `importorskip` on lean's own packages. Fix: test against the real packages at tiny sizes | no |
| 4 | Duplicates: action split/join in collect and contract; offset draw in policy and collect; `scripted_uniforms`. **Bug:** the scripted agent's arena radius is fixed at 4,200 m (`ScriptedConfig`); the game never provides it, so other arena sizes steer wrongly. Dead ship-characteristic fallbacks and a renaming shim in `ScriptedConfig`. Config values held twice (`ModelConfig.actions/levels`, `Opponent` shares, League floor/decay) | no |
| 5 | Defaults contradict decisions: `train.py` and `GameConfig()` default to the v2 game with fog, ghosts and drawn fields off (recorded as v2 until sign-off). Update size not at parity (minibatches 8, `--envs 256`). Stale "not decided" comment on `targets_every_epoch`; questions.md 12(b, c) stale | core as default before sign-off |
| 6 | Dead outputs and test-only API: `StepOut.damage_dealt/friendly_damage/killer/fired` unread (killer costs two `scatter_reduce`). Rollout counters read only by tests. Unused parameters and imports in `rewards.py`. `advantage_normaliser="percentile"` is a half-port of v2's rule. `positionless_rotary="zero"`, `left_is_even`, the "mixed" spawn, `head_to_head`, `crossover(match=)` test-only. Analysis types p51d and vizia_plain in the model's one-hot. `Obs.field_mask` always true. Twin dataclasses `ShipCard`/`Constants`; `team_view`/`enemy_view` near-duplicates | percentile normaliser: delete or port |
| 7 | `game/tuning.py` (457 lines) is analysis code in the runtime package and patches `card` module globals in place; `per_type_constants` exists only for it; its tests check v2's TOML files | where analysis tools live |
| 8 | Importing `lean.learn.compiled` sets `use_duck_shape = False` process-wide; `importlib` workaround because the function `collect` shadows the module | no |
| 9 | Speed (unmeasured): eval is eager and re-deals every tick through `reset_done`, reads back every 32 ticks; `observe` runs once per team; GAE is a Python loop over 128 ticks every epoch; snapshots compute empty slots; fields stored every tick (about a sixth of observation floats at 4v4, 10 fields, by shape arithmetic: matters for streaming); host reads per epoch (`support_usage`, boolean index); the scripted lead ignores the medium's refractive index | no |
| 10 | `core.step` 247 lines, 131 statements; `collect.py` 597 lines; reward weights as five fields read by `getattr`; a few short names | no |
| 11 | Stale docs: learn/NOTES.md (662 lines, superseded sections), game, model and scripted NOTES contradict the code; CONTRACT.md describes the unpacked State and no fog; README keeps first-cut tables; LAPTOP_PROMPT.md is a closed one-off. Fix: one short current NOTES per package, history to the worklog | no |
| 12 | Tests not earning their place: transition tests, `test_tuning.py`, stand-in tests, a test restating a config value, tests of unused memory/hold/spread paths. Missing fast tests: compiled collect and update (slow only), crossover with the real game, `abandon` with real packages, scripted at another arena size, scripted power under core, `train.main`, the stored position offset reaching the update | no |
| 13 | Builder defaults never decided: BC gate input and BC inside PPO; clip 0.2, lr 3e-4 after 10-update warm-up, entropy 0.003, value 0.05, next-state 0.1, hit 0.05, BC 2.0 with gate 0.45, gradient clip 1.0, Adam eps 1e-5; advantage normalisation per epoch by std; no KL stop; no weight averaging; snapshot every 10 updates with 8 slots; team sizes 4-4; persona spread 0; range-warning thresholds; value supports ±7/±8/±1/±1/±12 (objective ±1 a placeholder). Game: boundary ramp 300 m, `fire_thrust_fraction` 0, lineup cycle, "mixed" spawn, fog choices 1 to 5. Model: choices 1 to 14 in model/NOTES.md beyond the decided Griffin init, gate and rotary | yes: recipe |
| 14 | Lint: 372 long lines (reformat to 100 or give lean its own setting); 2 unused imports | no |
