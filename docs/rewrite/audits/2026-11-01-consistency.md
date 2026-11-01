# Consistency audit, 2026-11-01

An audit of merged v2 at `49e5e2a9` (after section 3c), report only, by an agent. Tags:
**[fact]** checked against the code (some by running it); **[judgement]** a style or
design opinion. Ruff was clean. The non-slow suite gave 276 passed and 8 skipped, run on
the CPU on a loaded machine. This is the work list for the cleanup pass after sections 6
and 1b merge. Items already handed to the learner agent are marked *(sent to 6)*.

## The five highest-value fixes

1. **Split the overloaded `done`** *(sent to 6)*.
   - `Controller.step(done)` (`contracts/controller.py`) means the environment reset
     *before* this observation.
   - `Trajectory.done` and `Events.done` mean it reset *after* this step.
   - Yemong calls the first `reset`. Use `reset` for before and `done` for after,
     everywhere.
2. **Rewrite or retire `STYLE_GUIDE.md`.** It is v1's guide: it cites
   `boost_and_broadside/profiles/`, `TensorState`, `FeatureCoordinator`, `constants.py` and
   Python 3.13. Its symbols (`T` time, `K` reward components) contradict contracts.md (`T`
   teams, `K` bullet slots). Its "no defaults for active hyperparameters" rule contradicts
   every config dataclass.
3. **The game state is not opaque in practice.** `state.py` and `api.py` say the runtime
   never reads a state field, but `runtime/rollout.py` reads and writes
   `state.match.ticks_since_seen` and imports `game.layout.environment`. Give the API a
   perception-merge function and an `episode_ticks` function.
4. **Process-global settings and compiles in library code** (scaffold rules 2 and 3)
   *(sent to 6)*:
   - `runtime/train.py:121` calls `torch.manual_seed`;
   - `learn/ppo.py:48` calls `torch.compile`;
   - `tests/sections/yemong_cases.py:27` reseeds the global RNG for every test that runs
     after it.
5. **Doc claims that contradict the code:**
   - `Privileged.ships` is not "the observation with fog off": a dead enemy shows
     `visible` 1 and its position there;
   - `cli.py:1` documents a `bnb train` subcommand that does not exist;
   - contracts.md says the ship card is a "required argument of every game function";
   - contracts.md uses `K` for two things;
   - contracts.md's skeleton notes are stale.

## 1. Naming

- **[fact] `K` has three meanings:** bullet slots, reward levels and spawn candidates.
  **`T`** is teams in the conventions but targets in `learn/rewards.py`.
- **[fact]** `gx, gy` at `flight.py:175`, which scaffold rule 10 names as forbidden.
- **[fact]** The contract field `logp` (`Outputs`, `Trajectory`) should be
  `log_probability`.
- **[fact] Abbreviations:**
  - `ppo.py` (replaced by section 6);
  - `rollout.py`: `st`, `ev`, `buf`, `ctrl`, `t_idx`, `dev`, `e`, `self.T/S/N/B`;
  - `scripted.py:54-68` (replaced by section 4);
  - also `train.py`, `lineup.py`, `realtime.py`, `cli.py` and `bench/`.
- **[fact] Unit suffixes:** `episode_s`/`respawn_s` against `capture_seconds`/
  `tick_seconds`.
- **[fact] "tick" names three things:**
  - the tick length (`Rules.tick_length`, `Yemong.tick`, `ShipStateCode.tick`);
  - `TickUniforms`;
  - the episode tick as a float (`observe.py:84`).
- **[fact] Stale:** `team_rel` at `mlp.py:8` and `scripted.py:53`.
- **[fact] "direction":** `Capture.direction` (the majority's push) against the
  observed zone `direction` (the meter's movement).
- **[fact] `_inside`** means zone membership (`phase.py`) and a disc clip (`spawns.py`).
- **[judgement]**
  - `env` and `environment` are mixed;
  - `compile` shadows the builtin;
  - `Hits.taken` is damage before protection.

## 2. Dead or leftover code

- **[fact]** `state_code.MEAN` is unused.
- **[fact]** `frontline.capture_step`'s `alive` parameter and its `Capture.direction`
  output exist only for the rewards fixture. Adapt `rewards_arm.py` and drop both.
- **[fact]** `Hits.taken` is read only by a test.
- **[fact]** `Encoder(zone_inputs=None)` (`network.py`) is a test-compatibility path for
  the Yemong reference arm.
- **[fact]** `inputs.py:177` handles `obs.zones is None`, which only tests reach. Require
  zones (Z = 5 since section 3c).
- **[fact]** The E501 exemption for `perception/sight.py` is dead, and the comment on the
  `physics/*` exemption is stale.
- **[fact] No production caller yet** (wire them in section 6, or delete):
  `ReturnScale`, `outcome_targets`, `HexGrid`, `poisson_terms`, `trajectory_code`,
  `outcome_code`, `grouped_cross_entropy`, `velocity_sharp`.

## 3. Docstrings and comments against the code

- **[fact]** `cli.py:1`: `bnb train` does not exist.
- **[fact]** `Privileged.ships` (`obs.py:75`) is not the observation with fog off.
- **[fact]** The opaque-state claims in `state.py` and `api.py` (see fix 3).
- **[fact]** `train.py:121` sets the seed in library code, while `cli.py:1` claims the
  CLI is the one place that does.
- **[fact]** `obs.py:4`: only ship rows carry `team_relative`.
- **[fact]** `config.py:5` and `test_contracts.py:19`: "255 is reserved for none", but
  ship indices use int32 −1 or int64.
- **[fact]** `model.py:264`: the memory shape is `(..., M + Z, Dm)`.
- **[fact]** `api.py:13`: `perceive`, `rules_of`, `spread`, `uniforms` and
  `start_clocks` take no ship card.
- **[fact] Docstrings that hard-code config defaults:** `combat.py:90`,
  `spawns.py:89,159,181`, `spawning.py:84`, `layout.py:25`, `scripted.py:24`.
- **[fact] Counts written into prose** (correct today, but they will drift):
  `action.py:34`, `observe.py:39,67`, `rewards.py:171`, `heads.py:13`, `mlp.py:50`,
  `inputs.py:60,67,99`, `model.py:112,130,359`, `belief.py:18-20`, `policy.py:1,10`.
- **[fact]** The README Status section stops at section 1 (the docs site rewrites the
  README).

## 4. contracts.md "Current schema" against the code

These match the code:
- every column list;
- the schema versions;
- `REWARD_LEVELS`, `Ledger`, `Pool`, `PoolRules`, `Environment`;
- Z = 5, the multiple of 64, `TRAINING_MIX`;
- the `Outputs` and `Trajectory` additions.

These do not:
- **[fact]** "required argument of every game function";
- **[fact]** the Game API list leaves out `ship_types`, `start_clocks` and `sweep_like`;
- **[fact]** "Mode features come with the modes section" is contradicted later in the
  same section;
- **[fact]** `values (R, S, K)` uses the conventions' `K` (bullet slots);
- **[fact]** `GameConfig.num_teams` duplicates `rules.TEAMS`;
- **[fact] Stale skeleton notes:** `last_seen` is now `ticks_since_seen`; the seat-layout
  test is no longer an xfail; the constants are no longer in `presets.json`; the ledger
  exists; ship indices are int32 −1, not "uint8, 255 = none";
- **[judgement]** rule 8: nothing checks the dtypes of the `Events`, `Obs` and
  `Privileged` that the game produces.

## 5. Layering

- **[fact]** `test_imports.py` ignores relative imports (`node.level == 0`), and most of
  `game/` imports with `from ..`. Resolved by hand, none goes upward.
- **[fact]** The training presets (`PRESETS`) live in `game/modes/rules.py`; they belong
  in `runtime/`.
- **[fact]** Game code is shaped by fixtures (`capture_step`, `Encoder`), and the runtime
  reaches into game internals.
- **[fact]** `test_game_section.py:23-25` imports from the temporary `race_arm` adapter.
- **[judgement]** `ReturnScale` is learner state in `models/`. The model step draws from
  the default RNG.

## 6. Duplication [fact]

1. **A ship's team** (`arange(N) // S`): `observe.py` twice, `phase.py`, `spawning.py`,
   `rollout.py`.
2. **The enemy flip:** re-derived in `spawns.py` (twice) and `phase.py`.
3. **Flight-path velocity:** `bullets.py:58-61` and `observe.py:44-45`.
4. **Disc sampling:** `spawning.py` and `spawns.py`.
5. **Magic numbers:**
   - 0.85 (`spawning.py`, `frontline.py`, `group_centre_fraction`);
   - the zone count 5, defined four times;
   - 2600 (`MAP_SCALE_RADIUS` and `inputs.ARENA_SCALE`);
   - the 40 m band width (`Rules`, `FIELD_BAND_SCALE`, `GRADIENT_SCALE`).
6. **Mode-name strings:** `config.py` and `modes/rules.py`.
7. **Two-hot encoding:** `critic.two_hot` and `codes._two_hot_at`.
8. **Action sizes `[3, 7, 2]`** hard-coded in `bench/cell.py` and two tests; the
   joint-action formula hard-coded in `test_yemong_contract.py` instead of `to_joint`.
9. **The batch multiple 8** hard-coded at `bench/sizes.py:32`.
10. **`INFINITY`** in `combat.py` and `spawns.py`.

## 7. Test hygiene

- **[fact] Slow tests not marked `slow`** (minutes on the loaded machine):
  - `test_unroll_reproduces_the_stored_rollout`: 221 s;
  - the `segment` fixture: 190 s;
  - `test_unroll_reproduces_a_stored_mixed_rollout`: 149 s;
  - `test_a_mixed_batch_equals_each_game_run_alone`: 145 s;
  - `test_absent_zone_columns_change_no_output`: 119 s;
  - `test_unroll_equals_steps_float64`: 104 s;
  - about eight more at 20–90 s.
- **[fact] GPU use not marked `gpu`:**
  - `test_mixed_yemong.py:155` is marked only `slow` and `skipif`;
  - `test_yemong_model.py:104-108` uses CUDA whenever it is visible, so
    `-m "not gpu"` still takes the GPU.
- **[fact] Shared state:**
  - `yemong_cases.py:27` and `test_learning.py:18` reseed the global RNG, and several
    tests draw without a seed (`test_perception.py:66, 93`), so their inputs depend on
    test order;
  - `test_game_section.py:318` resets dynamo and reads its process-wide counters;
  - `REFERENCE_ARM` redirects all reference suites at once.
- **[fact] Cannot fail:**
  - the learner equivalence suite skips until section 6;
  - the second assert at `test_perception.py:74` is redundant.
- **[fact] Gaps:**
  - `test_no_controller_takes_privileged` leaves out `YemongController`, which a
    separate test covers;
  - `test_yemong_model.py:33` uses `globals[..., 2]` instead of `GLOBAL["fogged"]`.
- **[judgement]** Statistical tests run at one fixed seed. The tolerances look sound
  (beyond 4 sigma), but a marginal change would go unnoticed.
