# Cleanup pass: notes

Branch `section/cleanup`, from v2 at `f51d89a8`. Spec: `cleanup-spec.md`; brief:
`cleanup-brief.md`.

## Progress

The order of the brief. Each group is one or more commits on `section/cleanup`, pushed.

- [ ] 1.0 the formatter
- [ ] 1.1 contract renames
- [ ] 1.2-1.3 names
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

## Decisions and deviations

## Bugs found

## Measurements

## Requests and open questions

## Size
