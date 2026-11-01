# Sections 4 and 7 (the scripted agent and BC): notes

Branch `section/scripted`. The spec is `scripted-bc-spec.md`.

## Progress (working state)

- **Done:**
  - part A code: `bnb.game.handling` (characteristics), `bnb.controllers.scripted`
    (`Scripted`, `distribution`, `marginals`) with `scripted_parameters`, `scripted_combat`,
    `scripted_frontline`; `bnb.runtime.evaluate` (fresh games to their end; `win_rate` uses
    it). Tests: `tests/sections/test_scripted.py`, `test_handling.py` (CPU, passing).
  - part B code: `bnb.learn.bc` (BehaviourCloning, labels, calibrate, StopRule),
    `bnb.runtime.cloning` (LINEUP, clone, warm_start, teacher_distributions), the
    `Learner._policy` hook in ppo.py, `Rollout(extra_reads=...)`, CLI options
    (`--objective bc`, `--teacher-temperature`, `--teacher-floor`, `--bc-budget-steps`,
    `--calibrate-entropy`). Tests: `tests/sections/test_bc.py` (CPU, passing; the policy
    file round trip skips until section 11 merges).
  - search stage 1 (3 rounds x 32 candidates x 128 games, 203 s):
    `/tmp/bnb-scratch-scripted/search_stage1.jsonl`.
- **In flight:** search stage 2 (8 rounds x 32 x 256 games; `search_stage2.jsonl`), then
  queued behind the bench lock: the toy BC curve (`toy_bc.py`, old parameters, curve shape
  only) and the speed measurement (`speed.py`).
- **Next:** set the chosen parameters as defaults; acceptance runs (2048 vs random, 1024 vs
  chaser, per mode); the toy BC test with its budget; BC throughput (`throughput.py bc`,
  `throughput.py ppo`); natural entropy (`bnb --preset mixed --calibrate-entropy 1.0`);
  the docs (explanation pages, a BC guide, roadmap, stale pages); the full suite.
