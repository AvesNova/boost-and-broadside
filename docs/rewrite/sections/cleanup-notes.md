# Cleanup pass: notes

Branch `section/cleanup`, from v2 at `f51d89a8`. Spec: `cleanup-spec.md`; brief:
`cleanup-brief.md`.

## Progress

The order of the brief. Each group is one or more commits on `section/cleanup`, pushed.

- [x] 1.0 the formatter
- [x] 1.1 contract renames
- [x] 1.2-1.3 names
- [ ] 2 dead code and the opaque state
- [ ] 3 the sweep's tests
- [ ] 4 docs
- [ ] 5 layering and duplication
- [ ] 6 test hygiene
- [ ] 8 additions

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

## Decisions and deviations

## Bugs found

## Measurements

## Requests and open questions

## Size
