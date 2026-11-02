# Cleanup pass: notes

Branch `section/cleanup`, from v2 at `f51d89a8`. Spec: `cleanup-spec.md`; brief:
`cleanup-brief.md`.

## Progress

The order of the brief. Each group is one or more commits on `section/cleanup`, pushed.

- [x] 1.0 the formatter
- [x] 1.1 contract renames
- [x] 1.2-1.3 names
- [x] 2 dead code and the opaque state
- [x] 3 the sweep's tests
- [x] 4 docs (the stale-page review is left for the end)
- [x] 5 layering and duplication
- [ ] 6 test hygiene: the lock fixture, the CUDA and generator guards, clean skips, the
  recompile test, the gaps are done (`51c4f56f`). Left: mark every test over 10 s `slow`
  and every CUDA test `gpu` from a full run's `--durations=0` (a full run was waiting on
  the shared benchmark lock at hand-over), and report both tiers' times with CUDA hidden.
- [ ] 8 additions: 8.1 game version, 8.2 resume drift, 8.4 reflection test done. Left:
  8.3 (the model's settings and the ship card in `TrainConfig`; replaces
  `TrainConfig.relational_hidden`, which tests pass in about 15 places; the ship card needs
  `Rollout` and `Trainer` to take the card instead of `api.ship_types(device)`) and 8.5's
  measurements.
- [ ] End: the full suite under the lock, the bench after (the rollout script and the
  realtime cell; see Measurements), the `PossiblyStalePage` review
  (`uv run --no-sync python -m tests.docs.staleness`), the size table.

## Items

### 1. Names

- **1.0 the formatter:** done, `aa313fd2` (in `.git-blame-ignore-revs`).
  `tests/contract/test_formatting.py` runs `ruff format --check` and `ruff check`.
  Deviation: the files the open branches own (`bnb/models/yemong/kernels/`,
  `bnb/runtime/store.py`, `bnb/play/`, 30 files) are not formatted, and
  `[tool.ruff.format] exclude` in `pyproject.toml` leaves them out of the check, so this
  pass stays out of those branches' diffs (brief). Request below.
- **1.1 contract renames:** done.
  - `logp` is `log_probability` everywhere (`Outputs`, `Trajectory`, the store's field,
    the losses' arguments and locals). `Outputs` and `Trajectory` carry no
    `SCHEMA_VERSION` (neither is saved), so no version rose.
  - `GameConfig` and `Pool`: `episode_seconds`, `respawn_seconds`. No other `_s`, `_m` or
    `_ms` field exists in a contract or config (see 1.3 for metric keys).
  - `done`/`reset`: `Controller.step(carry, obs, reset)` in every controller, test
    controller and caller. `Events.done` and `Trajectory.done` already meant "after this
    step", and Yemong already said `reset`; nothing else was left.
- **1.2 shape letters:** done, with one deviation.
  - `Z` zone slots and `C` spawn candidates are in the conventions (`contracts.md`) and the
    code style's table; `spawns.py` and `modes/rules.py` write `C`, and in code the
    dimension is `candidates`.
  - `learn/rewards.py`: the attacker and target axes are ships, `(..., N, N)`; the
    ledger's rows are `rows`.
  - **Deviation (proposal):** the spec proposes `L` for reward levels, but `L` is the
    steps of a segment in every learner shape (`Trajectory (L, R, ...)`, 126 uses with
    `R`), and `values (L, R, S, K)` would become `(L, R, S, L)`. The reward levels are
    written `levels` instead (`values (R, S, levels)`), and the code style's table now
    lists `L` (a segment's steps) and `R` (views); it said `R` was "rollout steps", which
    no code used. Request below.
  - `game/handling/` used `T` for ship types, `G` for a grid, `S` and `C` for stick
    levels and flight conditions: they are `types`, `grid` (`points` in
    `cruise_speed`, whose grid is a separate variable), `sticks` and `conditions` in
    shapes, and `type_count`, `stick_count`, `conditions` in code.
- **1.3 abbreviations:** done in every file the audit lists, then swept over `src/` and
  `bench/` with an AST scan for names of three letters or fewer (outside the off-limits
  files):
  - `flight.py`: `gx, gy` are `gradient_x, gradient_y`;
  - `rollout.py`: `state`, `events`, `controller`, `outputs`, `config`; the carry's
    `ctrl` and `t_idx` are `controllers` and `segment_tick`; `self.T/S/N/B` are
    `teams`, `ships_per_team`, `ships`, `environments`; `_assign(target, source)`;
  - `train.py`: the update's timers, the league copy's loop names; the update's metrics
    `rollout_s`, `learn_s` and the pool metric `reward_per_s` are `rollout_seconds`,
    `learn_seconds`, `reward_per_second`;
  - `lineup.py`: `Lineup.env` is `Lineup.environments`;
  - `realtime.py`: `config`;
  - `cli.py`: the parsed options are `options` (`given_options` for the set of names
    given);
  - `bench/`: `cell.py`, `run.py`, `sizes.py`, `lock.py` (`HELD_ENV` is
    `HELD_VARIABLE`, `envs_for` is `environments_for`);
  - the Chaser (`scripted.py`): `to_x`, `to_y`, `enemy`, `distance`, ...
  - kept: matrix entries in `flight.py` (`m11`, `a12`: the spec's notation for a 2x2
    exponential), `obs` (see open questions), real words (`out` became `outputs` where it
    was a controller's outputs; `own`, `low`, `row`, `key` are words).
- **`environment`, not `env`:** `num_envs` is `environments` (`TrainConfig`, `Rollout`,
  `Lineup`, `rules_of`, `environments_per_pool`, the CLI's `--environments`);
  `Learner.env_steps` and the update's metric `env_steps` are `environment_steps`.
  Kept: `bench/results.jsonl`'s keys (`envs`, `compile_s`, `rollout_ms`, ...), so its
  history compares; `bench.cell` reads the cell's `envs` key and writes those keys, with
  a comment saying why.
- **"tick":** `Yemong.tick` and `ShipStateCode.tick` are `tick_length` (the build
  records already said `tick_length`); `inputs.compose` takes `tick_length`; the float in
  `observe.py` is `episode_progress` (the global column's own name). `episode_tick`
  and `TickUniforms` unchanged.
- **`_inside`:** `_in_zones` (`phase.py`) and `_clip_to_disc` (`spawns.py`).
- **`compile` as a parameter:** `compiled` (`Rollout`, `TrainConfig`, `first_results`,
  `teacher_distributions`, the test helpers). Not `Realtime`: the play shell calls
  `Realtime(config, compile=...)` and `bnb/play/` is off limits (request below).
- **`team_rel`:** `team_relative` in `mlp.py` and `scripted.py`.

### 2. Dead code, fixture-shaped code, the opaque state

- **Deleted:** `state_code.MEAN`; `Hits.taken` and its assert in
  `test_game_section.py`; both E501 exemptions: after the formatter, no line in
  `game/physics/` or `perception/sight.py` is over 100 characters, so the `physics/*`
  exemption was dead too (its comment was the stale part; nothing is left to describe).
- **`capture_step`:** takes no `alive` and returns no `direction`; its docstring says
  `inside` is masked to the living ships. `rewards_arm.py` drops the fixture's `alive`
  and computes the majority's push from the same inputs (`_majority`), so the fixture
  still checks it against v1.
- **`Encoder`:** `zone_inputs` and the zone arguments are required; the Yemong reference
  adapter builds the encoder with the zone kind, zeroes its weights, passes every zone
  absent (present 0) and compares the field rows of the memory (the fixture predates
  zones).
- **`inputs.py`:** zones are required (`obs.zones is None` is gone). The tests that built
  views without zones (`test_yemong_model.py`, `test_yemong_contract.py`,
  `test_yemong_export.py`, `yemong_cases.step_loop`) now pass the game's zones; the
  ONNX export of `Yemong.step` takes `zones` as an input.
- **No production caller** (verdicts after sections 6, 1b, 4 and 7 merged):

  | Name | Verdict |
  |---|---|
  | `ReturnScale` | kept: the learner (`ppo.py`) and policy files (`checkpoint.py`) use it |
  | `outcome_targets` | kept: `ppo.py` calls it |
  | `HexGrid` | kept: `ppo.py` and `labels.py` |
  | `poisson_terms` | kept: `ppo.py` |
  | `trajectory_code`, `outcome_code` | kept: `ppo.py` (the shot codes) |
  | `grouped_cross_entropy` | kept: `ppo.py` and `losses.py` |
  | `velocity_sharp` | deleted: no caller and no pending spec. The Yemong reference adapter composes it from `codes`' pieces (`scalar_sharp` of the projected, compressed velocity), so the fixture still checks them |

- **The opaque state:** `bnb.game.api` gains `perceived(state)` (the part `perceive`
  changes, a tuple the runtime copies leaf by leaf), `episode_ticks(rules, batch,
  device)`, `launch_capture(state, sweep)` (the shot capture's launch state, move and
  alive flags), `ship_team(ships_per_team, device)` and re-exports `view_order`. The
  rollout imports only `bnb.game.api`. `tests/contract/test_layering.py` walks
  `runtime/` and fails on `.match` or `.bullets` anywhere, on `.ships` or `.mode` of a
  state expression (a name containing `state`, or `x["state"]`: `Obs.ships` and
  `GameConfig.mode` are other things with the same names), and on an import from
  `bnb.game` other than the API and the handling tools. Checked by putting the old
  `ticks_since_seen` copy back: it fails.
- **Process-wide settings:** already out of library code after section 6, except
  `runtime.train.set_precision` (a library function setting TF32), which moves to
  `bnb.cli.set_precision`. `test_layering.py` fails on a `torch.manual_seed`, matmul
  precision, thread count or deterministic-algorithms call, or an assignment under
  `torch.backends`, `torch._dynamo.config` or `torch._inductor.config`, outside
  `bnb.cli`, a `__main__` module or a `main` function (`bnb.play.benchmark.main`
  sets threads and seeds as an entry point); and on `torch.compile` outside
  `bnb.runtime` and those entry points. Checked with a seed and a compile added to
  `ppo.py`. Kept: `checkpoint.restore_default_generators` sets the default generators'
  state; the entry point calls it on resume (its docstring says so), and the spec's list
  does not name it.

### 3. The sweep's tests

Written on `section/cleanup-sweep` (merged, `43fc89a2`..`674d50c3`), in new files so they
stay out of the other groups' diffs. Each mutation was applied by hand in `src/`, the test
failed, and the change was reverted (item 2 re-checked on the merged tree). No test found
a real bug, so there is no `xfail`. All 39 run on the CPU in about 5 s; none is `slow` or
`gpu`.

| Item | Test | Mutation that made it fail |
|---|---|---|
| 1 | `test_physics_rules.py::test_a_dead_ship_with_a_cool_gun_and_energy_fires_nothing` | `ships.alive &` dropped from `fire` |
| 2 | `test_physics_rules.py::test_a_ship_with_exactly_the_shot_cost_in_energy_fires` | `>=` to `>` |
| 3 | `test_perception_rules.py::test_without_fog_no_ship_goes_unseen_not_even_a_dead_one` | `visible \| unfogged` to `visible` |
| 4 | `test_perception_rules.py::test_the_reset_flag_is_one_on_an_episodes_first_observation_only` | `episode_tick == 0` to `== 1` |
| 5 | `test_perception_rules.py::test_the_privileged_rows_show_every_ship_as_seen_where_it_truly_is` | truth rows from `visible`, and from `since` (each failed) |
| 6 | `test_perception_rules.py::test_the_observed_respawn_wait_is_in_seconds` | `* rules.tick_length` dropped |
| 7 | `test_mode_rules.py::test_team_one_sees_a_moved_front_negated`, `::test_a_tick_with_a_moved_front_and_its_mirror_agree` | the front not negated for view 1 |
| 8 | `test_mode_rules.py::test_a_dead_ship_inside_a_zone_does_not_count_toward_its_capture` | `& alive[..., None]` dropped in `phase.py` |
| 9 | `test_mode_rules.py::test_every_zones_observed_movement_is_zero_on_the_tick_a_capture_completes` | movement not zeroed on a completion |
| 10 | `test_mode_rules.py::test_spawn_protection_is_by_each_ships_own_team` | `repeat_interleave(S, 1)` to `repeat(1, S, 1)`, at S = 3 |
| 11 | `test_mode_rules.py::test_spawns_count_every_enemy_on_a_respawn_and_same_group_rivals_on_a_reset` | `counts = rival` (respawn case) and `rival \| True` (reset case) |
| 12 | `test_mode_rules.py::test_a_match_with_no_living_ship_respawns_at_the_uniform_fallback_points` | `any_living` forced true |
| 13 | `test_reward_paths.py::test_the_objective_progress_level_carries_objective_progress` | the level built from `objective` |
| 14 | `test_reward_paths.py::test_a_restore_after_boundary_damage_heals_the_boundary_row_by_its_share` | deficit without the boundary row; the boundary row's attribution removed; the ship rows' total including the boundary (each failed) |
| 15 | `test_reward_paths.py::test_a_hit_and_a_restore_in_one_tick_share_the_ticks_ledger_total`, `::test_a_kill_is_shared_by_the_ledger_including_the_killing_ticks_damage` | `ships_total = old_total`; the kill's `ledger_total=old_total` |
| 16 | `test_rollout_records.py::test_the_stored_reward_is_the_learners_seat_teams` | `reward[environments, 0]`, and `reward[environments, role]` |
| 17 | `test_rollout_records.py::test_seats_are_redrawn_at_episode_ends_and_only_there` | seat never redrawn, and redrawn every tick |
| 18 | `test_rollout_records.py::test_the_time_since_seen_rises_from_tick_to_tick_in_the_stored_views` | the `api.perceived` copy-back removed |
| 19 | `test_rollout_records.py::test_each_stored_done_is_its_own_environments` | `events.done[:rows]`, and the environments reversed |
| 20 | `test_pool_arithmetic.py::test_a_share_that_gives_a_fraction_of_an_environment_is_refused` | tolerance `1e-6` to `0.5` |
| 21 | `test_pool_arithmetic.py::test_the_batch_multiple_counts_the_fogged_share` | the fogged denominator dropped from the lcm |
| 22 | `test_pool_arithmetic.py::test_a_time_just_off_a_whole_tick_is_refused` | tolerance `1e-2`, and the check removed |
| 23 | `test_yemong_inputs.py::test_a_ship_that_spawned_this_tick_enters_with_the_null_pending_action` | `where(spawned, null, forecast)` to `forecast` |
| 24 | `test_yemong_inputs.py::test_presence_counts_only_ships_with_a_valid_belief` | presence from every row |
| 25 | `test_yemong_inputs.py::test_the_time_since_seen_input_is_in_seconds` | `* tick_length` dropped |
| 26 | `test_yemong_inputs.py::test_the_nearest_image_crosses_the_period_boundary`, `::test_the_forecast_position_of_a_hidden_enemy_is_the_image_nearest_its_belief` | `_nearest` removed in `step`; `round` to `floor` |
| 27 | `test_yemong_inputs.py::test_unroll_from_a_carry_with_history_whose_first_step_is_a_reset_equals_a_fresh_start` | unroll's `keep = ~reset[0]` all true |
| 28 | `test_yemong_inputs.py::test_unroll_equals_the_step_loop_around_a_reset_mid_segment`, and an argument | removing the decay cut fails it; see below |

- **Item 28, equivalent mutation:** `_later(starts)` and `starts` differ only at step 0.
  If `starts[0]` is false they agree; if it is true, `unroll` has already cleared the
  trunk state and the convolution buffer for that view (`keep = ~reset[0]`), so the cut
  multiplies or masks zeros. Both mutations pass every test, as the argument predicts.
  The test still pins the mid-segment resets (removing the cut fails it).
- **Item 11** needs a 300 m arena in its reset case: in a normal arena the same-group
  rivals are the nearest anyway, so the reset rule cannot be seen.
- **Item 17** is statistical: it uses the rollout's own seat draws (false-failure
  chance about 3e-5, in its docstring).
- **Spies:** items 11 and 12 wrap `spawns._farthest_from_enemies`, and items 16 and 17
  wrap `rewards.reward_step` and the resolve stage, with `monkeypatch`, to record what
  the real code computed; nothing is replaced.
- **Item 7** imports `test_seat_symmetry`'s helpers (`_melee`, `_random_tick`,
  `_assert_mirrored`) to reach a moved front in the mirror check.

### 4. Docs, docstrings and contracts.md

- **`STYLE_GUIDE.md`:** deleted. Nothing outside `docs/rewrite/` referred to it;
  `docs/contributing/code-style.md` replaces it.
- **Docstrings against the code:**
  - `cli.py`: names `bnb train` (which exists now), and says it is an entry point and
    which others there are (the play shell, `bench`); the seed is set there, and only
    there in library terms (`test_layering.py`).
  - `Privileged.ships`: every row as if seen (`visible` 1, `time_since_seen` 0, the true
    hidden columns), the dead included, unlike a fog-off observation, where a dead enemy
    is not visible. Pinned by sweep item 5.
  - `obs.py`: only ship rows carry `team_relative`.
  - `config.py` and `test_contracts.py`: `MAX_SHIPS` is a design cap, ship indices are
    int32 with -1 for none.
  - `model.py`: the memory is `(..., M + Z, Dm)`.
  - `api.py`: names the functions that take the ship card (`reset`, `move`, `resolve`,
    `step`), and the runtime's functions (section 2).
- **Defaults and counts in prose:** every place the audit lists now names the field
  (`boundary_damage_base`, `group_centre_fraction`, `spawn_margin`,
  `spawn_zone_fraction`, `MAP_SCALE_RADIUS`) or the constant (`NUM_JOINT`,
  `SHIP_HIDDEN_ZERO`, `len(REWARD_LEVELS)`, `len(MEANS)`, `state_code.width`,
  `trajectory_width`, `sum(SIZES)`), and `REDRAW` in `scripted.py`. `spawning.py`'s
  "-0.8 R_a" comment says the side instead. `inputs.ship_inputs` added a literal 9 for the
  respawn features; it is `1 + 2 * RESPAWN_HARMONICS` (the same number). Yemong's
  "2C" density cells are "2 cells" (`C` is spawn candidates now).
- **Missing docstrings:** 82 written (on `section/cleanup-docstrings`, merged),
  `tests/docs/test_docstrings.py` holds the count at zero outside `bnb.play` and
  `bnb.models.yemong.kernels` (68 there, owned by open branches: request below).
- **`contracts.md`:** the Game API list is complete (with the runtime's functions); the
  global columns no longer say the mode's come later; `values` is `(R, S, levels)`;
  `GameConfig.num_teams` is deleted in the code (`TEAMS` is one constant in
  `bnb.contracts.config`) and the doc; the ship-index convention says int32 -1; the
  skeleton notes keep the three settled points (controller signature with `reset`,
  perception and `ticks_since_seen`, the seat) and drop the stale ones (`last_seen`, the
  seat-layout xfail, `presets.json`, the ledger, uint8 indices). "Required argument of
  every game function" was already gone. A "cleanup pass" entry records the renames.
- **Contract dtypes (rule 8):** `tests/contract/test_game_dtypes.py` checks the dtype and
  shape of every field of the `Events`, `Obs` and `Privileged` that one game tick
  produces, in all three modes, at batch 1 and 6.
- **Same-day reviews:** the staleness check already compares against the page's own last
  commit; what blocked a second review on one day was a `reviewed:` value that did not
  change. The writing guide now says to write the date and time (`2026-11-03 14:20`).

### 5. Layering and duplication

- **`test_imports.py`** resolves a relative import against the importing module's
  package before checking the layer order. It found nothing upward; checked by adding
  `from ...runtime import lineup` to `game/physics/step.py`: it fails.
- **`PRESETS`:** not moved (request below). The play shell imports it from
  `bnb.game.api`, and `bnb/play/` is off limits; `bnb.game` cannot re-export from
  `bnb.runtime` (an upward import), and a copy would be a second definition. The runtime
  now reads it through `bnb.game.api`.
- **`test_game_section.py`** no longer imports `race_arm`: it holds its own copies of the
  three name maps it reads (`FLIGHT_SHARED`, `CARD`, `RULES`).
- **One definition each:**
  1. a ship's team: `bnb.game.state.ship_team(ships_per_team, device)`, used by
     `observe.py` (twice), `phase.py`, `spawning.py` and (through the API) the rollout;
  2. the enemy flip: `bnb.game.state.enemy` (moved from `sight.py`), used by `sight.py`,
     `spawns.py` (twice) and `phase.py`'s score (its enemy's deaths);
  3. flight-path velocity: `numerics.flight_path_velocity`, the same expression, in
     `bullets.launch` and `observe.py`;
  4. disc sampling: `spawning.uniform_disc`, used by the map (`new_map`) and every spawn
     rule;
  5. magic numbers:
     - 0.85: `FIELD_DISC_FRACTION` (`bnb.game.rules`), the field disc of both maps
       (`new_map`, `frontline.mirror_fields`). `ModeRules.group_centre_fraction` stays its
       own field: a spawn rule of team deathmatch that equals it by choice, tunable alone;
     - the zone count: `ZONE_SLOTS` in `bnb.contracts.obs`; `frontline.ZONES` is defined
       from it (Frontline fills every slot) and `FRONTLINE_ZONES` is gone.
       `inputs.ZONE_ROLES` (5) stays: the roles a zone can have, a different count;
     - 2600 and 40 m: kept apart. `MAP_SCALE_RADIUS` and `Rules.field_band_width` are game
       rules; Yemong's `ARENA_SCALE`, `FIELD_BAND_SCALE` and `GRADIENT_SCALE` are input
       scales, constants of the model's `VERSION` that must not follow a rule change.
       A comment in `inputs.py` says so;
  6. mode names: `TEAM_DEATHMATCH`, `FRONTLINE`, `SANDBOX` and `MODES` in
     `bnb.contracts.config`; `game/modes/rules.py`, `GameConfig`'s default and the CLI's
     import them;
  7. two-hot: `codes.two_hot_at` (public now); `critic.two_hot` computes its bins and calls
     it;
  8. action sizes: `bench/cell.py` and the four tests use `NUM_POWER`, `NUM_STICK`,
     `NUM_SHOOT`; `test_learner.py` and `test_yemong_contract.py` use `to_joint`;
  9. the batch multiple: `bench.sizes.environments_for` asks `environment_multiple` of
     the skeleton lineup (8) or the mixed pools (64); the drawn cells keep 360, 1024 and
     1576 environments;
  10. `INFINITY`: only `spawns.py` defines it now (`combat.py`'s went in section 6);
  11. the core radius: `sight.core_radius(fields)`, used by `visibility` and by
      `api.core_radius`.
- **Same-day reviews:** see section 4.

## Decisions and deviations

1. **The off-limits files are not formatted** (`bnb/models/yemong/kernels/`,
   `bnb/runtime/store.py`, `bnb/play/`): the brief keeps this diff out of them; the
   formatter's exclude list names them until their branches merge.
2. **Reward levels are written `levels`, not `L`** (1.2): `L` is a segment's steps in every
   learner shape.
3. **Kept names:** `obs` as a variable and parameter (the `Trajectory.obs` field and
   `Controller.step`'s parameter, hundreds of uses; see the requests); `bench/results.jsonl`'s
   keys; `Realtime(compile=...)` (the play shell calls it).
4. **`PRESETS` stays in `bnb.game.modes.rules`** (5): the play shell imports it from the
   game API.
5. **The runtime-state test** counts `.ships` and `.mode` only on a state expression, since
   `Obs.ships`, `Privileged.ships` and `GameConfig.mode` share the names.
6. **Global random state** (6): code that draws from torch's default generator by design
   (model initialisation, the rollout's seat draws, `Random`, Yemong's offsets, Gumbel
   sampling) cannot take a local generator without a behaviour change. Those tests run under
   the `global_generator` fixture (the default generator seeded at 0 inside `fork_rng`,
   restored after), set per module with `pytestmark`; the guard fixture fails any other
   test that leaves the global CPU generator changed. Every direct draw in a test uses a
   local `torch.Generator` (`test_perception.py`'s two were the last).
7. **Contracts in configs:** renaming config fields (`num_envs`, `compile`,
   `episode_s`, `respawn_s`, `num_teams`) and the carry's keys (`ctrl`, `t_idx`) means a
   run directory written before this pass cannot be resumed (its `config.json` and carry
   use the old names). Policy files are unaffected (their build records already used
   `tick_length`).

## Bugs found

None. No sweep test found a wrong behaviour, so there is no `xfail`.

## Measurements

- **The rollout change (section 2), on its own:** `/tmp/bnb-scratch-cleanup/rollout_bench.py`
  (`Rollout.run` alone: the mixed preset's three games, one pool of the chaser against
  random at fog share 0.5, 1024 environments, 5v5, 128-tick segments; twelve segments to
  compile, capture and settle, then the best of five runs of four segments; benchmark lock
  held; RTX 4060 Laptop, 8 GB):
  - before (`f007def3`, the formatted tree): 2 692 138 environment-ticks/s (load 3.0);
  - after (`perceived` copied through the API, working tree on `63ca1c94`): 2 699 598
    (load 5.4, provisional).

  +0.3%, within noise: the copy is the same single `copy_` of `ticks_since_seen`.

## Requests and open questions

1. **For the play branch (`bnb/play/`):**
   - import `PRESETS` from wherever the coordinator settles it, then the cleanup's move of
     `PRESETS` into `bnb.runtime` can land (spec 5; play-notes request 5 asks the same);
   - `Realtime(config, compile=...)` becomes `compiled=` (spec 1.3) once `MatchLoop`
     passes the new name;
   - `ruff format` its files, drop its entries from `[tool.ruff.format] exclude` and from
     `tests/docs/test_docstrings.py`'s `EXEMPT` (65 objects lack docstrings);
   - `snapshot.py` reads state fields (`state.mode.score`, `state.match.fogged`, ...):
     the "state is opaque outside `bnb.game`" rule does not hold there yet.
2. **For `fix/intermittent` (`kernels/`, `store.py`):** format the files and drop their
   exclude entries; the kernels lack 3 docstrings (`EXEMPT`).
3. **`obs` as a name:** the code style forbids it as a variable, but it is also the
   `Trajectory.obs` field and every controller's `step(carry, obs, reset)` parameter. A
   rename to `observation` is a contract change across every controller and model; worth
   its own small pass, or an exception in the code style.
4. **The skeleton MLP** (`bnb.models.mlp`, registered with a compatibility entry) is still
   there; runs-notes expected the cleanup to delete it, but no spec item names it.
5. **Spec letters:** if `L` for reward levels is wanted after all, the segment steps need
   another letter first (proposal: keep `L` steps, `R` views, `levels`).
6. **The bench cells do not run on this machine since the kernels' autotuning:** before
   this pass, the 114-ship cell ran out of memory inside Triton's autotune benchmarking,
   and the 40- and 26-ship cells hit `bench.run`'s 900 s timeout (cold autotune caches for
   those shapes). Only the realtime cell finished (p50 0.19 ms, p99 0.21 ms). The rollout
   comparison above stands in for the train cells; the timeout, or warm caches, is for the
   coordinator.

## Size
