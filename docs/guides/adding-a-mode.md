---
covers: [src/bnb/game/modes/, src/bnb/contracts/config.py, src/bnb/contracts/obs.py, src/bnb/game/physics/step.py, src/bnb/models/yemong/inputs.py, tests/sections/modes_cases.py]
reviewed: 2026-11-01
---

# Adding a mode

A mode is a set of rule components: a clock, a win rule, an objective, spawn placement,
spawn protection and its part of the observation. Each component is a hook that
dispatches on the mode's name in Python, so a compiled graph holds only the rules of the
modes in its batch. [Modes](../explanation/game/modes.md) explains the existing modes; this
page lists every place a new one plugs in.

## 1. Name it and give it a clock

- Add the name to `bnb.contracts.config.MODES`. `GameConfig` rejects any other name.
- Add a constant for it beside `bnb.game.modes.rules.TEAM_DEATHMATCH`{.nolink},
  `bnb.game.modes.rules.FRONTLINE`{.nolink} and `bnb.game.modes.rules.SANDBOX`{.nolink}.
- Add its default episode length and respawn wait to `bnb.game.modes.rules.CLOCKS`. Both
  must be whole multiples of the tick.

## 2. Its static rules

`bnb.game.modes.rules.ModeRules` holds every mode's constants as plain Python numbers, so
compiled graphs specialize on them. Add the new mode's constants as fields with defaults,
and copy any that come from `GameConfig` in `bnb.game.modes.rules.mode_rules_of`.

If the mode draws randomness for its spawns or its map, extend
`bnb.game.modes.rules.ModeRules.uniform_count`: the number of uniforms it reads per
environment per tick, after the game's own. The tick's uniform width is the widest mode's,
and each environment reads its own mode's prefix.

## 3. Its state

The mode's part of the match is `bnb.game.state.ModeState`. Every environment of every
mode carries the same leaves and the same `bnb.game.modes.rules.ZONE_SLOTS` zone slots, so
a mixed batch is one set of tensors. A mode that needs new state adds leaves there; other
modes leave them zero and pass them through unchanged.

## 4. Its hooks

Each hook takes the mode's `ModeRules` and runs on that mode's slice of the batch only
(`bnb.game.layout.per_mode`). Add a branch for the new name to each that applies:

| Hook | What it decides |
|---|---|
| `bnb.game.modes.phase.mode_phase` | the mode state after the tick, the two objective events, whether the match is won and the result |
| `bnb.game.modes.phase.clock_result` | the result when the clock ends the match |
| `bnb.game.modes.phase.new_match` | a new match's mode state and map |
| `bnb.game.modes.phase.protected` | which ships take no damage from hits |
| `bnb.game.modes.spawns.placement` | where and facing which way a spawning ship appears |
| `bnb.game.modes.observation.zone_rows` | the zone rows of each view |
| `bnb.game.modes.observation.mode_globals` | the mode's global columns of each view |

Rules every hook keeps:

- **Pure:** tensors in, tensors out; no host copies, so the tick stays capturable in a
  CUDA graph.
- **Seat-blind:** nothing may depend on the team index or the sim index in a way the
  observation reveals. A side-specific role is drawn per episode, as Frontline draws its
  handedness.
- **Zero-sum objectives:** `objective_progress` and `objective` each sum to zero over every
  environment on every tick. The rewards depend on it.

`bnb.game.physics.step.resolve` calls `protected` only when some pool plays Frontline.
A new mode with spawn protection extends that condition in `src/bnb/game/physics/step.py`.

## 5. The observation

A new mode one-hot or a new global column changes `bnb.contracts.obs.GLOBAL_COLUMNS`, and
so the observation contract:

- bump `bnb.contracts.obs.Obs.SCHEMA_VERSION`;
- extend Yemong's global token in `src/bnb/models/yemong/inputs.py` (`GLOBAL_INPUTS` and
  the columns it composes);
- add a scale for the column to the skeleton MLP if it needs one (`bnb.models.mlp.GLOBAL_SCALE`).

Zone columns are the same for every mode. A mode without zones leaves its slots absent:
`present` 0 and every other column 0.

## 6. Presets and the training mix

Add a game preset to `bnb.game.modes.rules.PRESETS` if the mode is to be trained, and a
share to `bnb.runtime.train.TRAINING_MIX` if it joins the default mix. Every preset in one
batch must have the same ships per team and fields.

## 7. Tests

- Add a case to `tests.sections.modes_cases.MODES` (in `tests/sections/modes_cases.py`). The tests parametrized over it
  then cover the new mode: compiled against eager
  (`tests/sections/test_modes.py::test_compiled_tick_matches_eager_in_every_mode`), CUDA-graph
  capture (`tests/sections/test_modes.py::test_eager_cuda_graph_capture_in_every_mode`),
  zero-sum objective events
  (`tests/sections/test_modes.py::test_objective_events_are_zero_sum_and_results_only_at_the_end`)
  and zero-sum rewards with the ledger
  (`tests/sections/test_rewards.py::test_levels_are_zero_sum_and_the_ledger_holds_every_deficit`).
- Write rule tests for the mode's own components: win, draw and clock cases, its spawns,
  its objective.
- Check the seat: extend
  `tests/sections/test_modes.py::test_spawn_layout_does_not_reveal_the_seat` and
  `tests/sections/test_modes.py::test_mode_rows_do_not_reveal_the_seat`.
- Check it in a mixed batch: add the case's name to `tests.sections.test_mixed.GAMES`,
  whose equivalence tests
  (`tests/sections/test_mixed.py::test_a_mixed_batch_equals_each_game_run_alone`) run each
  game alone and in one batch and compare them.

## 8. Docs

Describe the mode in [Modes](../explanation/game/modes.md). The generated tables (clocks,
presets, mode rules, global columns) update themselves.
