# Section 11 (runs and checkpoints): notes

Branch `section/runs`. The spec is `runs-spec.md`; the decisions are in `runs.md`.

## Progress

- **For the coordinator: `save_policy` and `load_policy` are pushed, tested and
  merge-ready** (commit 2a4c36e7 on `section/runs`). The BC hand-off can call:
  - `bnb.runs.checkpoint.save_policy(directory, model, *, return_scale=None, steps=None,
    game=None, types=None)`: `return_scale` is the learner's `ReturnScale`, `steps` a
    dict such as `{"environment_steps": n}`;
  - `bnb.runs.checkpoint.load_policy(directory, device) -> model`;
  - `bnb.runs.checkpoint.read_policy(directory) -> PolicyRecord`, with `return_scale`
    `(low, high)` and `steps`, for the warm start.
  - Every model is now built from its build record: `Yemong(yemong_build(config, rules,
    types))`, `Relational(RelationalBuild(config, ships_per_team, num_ships))`.
- **Next:** training checkpoints and the exact resume; the run directory and the CLI; the
  probe, provenance and uploads; docs.
