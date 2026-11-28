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
- 14:20-15:06 branch smoke, compile and the slowdown check (one bnb-10 unit at a time; a
  browser opened at ~14:23 kept the load at 3-8, so no run was fully quiet). The branch's
  first BC/RL runs (116.9 / 115.9 s per BC update, 119.9 / 116.4 s RL) were slower than core's
  earlier runs. Rerun back to back (14:42-15:00): core BC 108.7 / 110.9 s per update (8,612 /
  8,756 steps/s), branch BC 106.4 / 106.5 s (8,946 / 8,967 steps/s): the slowdown was noise
  (the machine), the branch is ~3% faster. `end_segment` alone (training sizes): core
  0.18-0.35 ms per segment, branch 0.27-0.43 ms (+0.1 ms). Store 2,601.5 -> 2,409.4 MiB, BC
  peak 1.16 -> 1.06 GiB, RL peak 1.28 -> 1.15 GiB. Hit share on the branch: equal on all
  labelled rows and away from the ends in every update (core: +2-9% relative).
- Compile (cold, budget 175 s): branch 189.6 s at load 7.0 (FAIL), rerun 164.8 s at load 2.9
  (pass); core 170.3 s at load 6.8, rerun 174.7 s at load 7.0. Operations 11,509 (core 11,511);
  minibatch_loss 1,645 either way. 6 graphs, no graph break, no recompile.
