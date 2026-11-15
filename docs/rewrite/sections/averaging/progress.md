# Post-hoc averaging: progress

Branch `feat/averaging-tool` (from `feat/imugi-boundary`). Milestone notes, newest last.

## 2026-11-15: the saving path

- **Checked:** with `update_policies` on, `bnb.runs.session.train` calls `save_update_policies` after every update (`every_update or snapshots.due(...)`), and it writes the live weights (`policies/step_<n>/live/`, `trainer.model`, the same state dict the averages fold) as well as both averages. So every update's live weights are there. BC and RL share the path, and a chain passes `--update-policies` to both stages.
- **Size:** a real-scale Yemong policy file (`real_scale_all_modes`, default build) is 2.85M float32 parameters, 11.4 MB of weights; 11 MB on disk per file, measured on a saved run. Each update writes up to three (live, exponential, uniform): about 34 MB.
- **Disk:** the default 2048 environments x 128 ticks x 4 shards is 1,048,576 steps per update, so 100M steps is 96 updates: about 3.3 GB per stage with all three files, 6.6 GB for 100M BC + 100M RL (1.1 GB per stage for the live files alone). Small; nothing to change.
- **Fixed (a gap, not a bug in what was saved):** the run's exponential average starts from the run's starting weights, which were never saved, so it could not be rebuilt exactly from the per-update files. A run with `update_policies` now also writes `policies/step_<start>/` before its first update (`updates` 0 in its record). A resume's start is its checkpoint's update, already written, so nothing is added there.
- **Replaced:** `bnb runs ema` (`bnb.runs.ema`) by `bnb average` (`bnb.runs.averaging`), which does what it did and more.
