# Fixes 10 progress (queue item 10, branch `fixes-10`)

Author decisions of 2026-11-28 ~13:30: option C on the team-wide levels only; the objective
levels optional and off by default; the hit labels of bullets that cannot report a miss
masked. GPU jobs are `bnb-10-*` units under the GPU lock, one at a time; logs and the core
worktree in `data/fixes10/` (not committed); scripts in `fixes-10/` (from item 6's).

## 2026-11-28

- 13:42 core's smoke queued (BC 3 updates, RL 3 updates, compile check; `data/fixes10/wt-core`
  at 2f4951dc), waiting for the CPU replay unit (`bnb-replays-p3`) to finish first.
- ~14:05 code in, targeted tests pass (19): levels in `LearnConfig.level_names` (contract
  `level_names`), the model's critic sized by `ModelConfig.value_levels` (train.py sets it
  from the learner), option C on the team-wide levels (`returns.fallen_team_values`), the
  store unlabels unresolved rows (`store.unresolved_rows`, in `end_segment`).
- 13:48 core's queue restarted (the first try had no venv in the scratch worktree; failed at
  import, nothing ran). The CPU replay unit had finished: load 1.6, performance profile. ETA
  for core's three jobs 14:15 (worst 14:25); then the quick tier and the slow tier on the CPU
  (alone), then the branch's three jobs. The trainer also prints the enemy-hit share of the
  labelled observed rows, all of them and away from the ends (from core RL on).
