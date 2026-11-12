# Steady speed: progress

Branch `perf/steady-speed` (from `fix/learner-recompile` 4ed4b950). The question: is the
release tree's steady training speed, before and after the compile fixes of 2026-11-12, what
the 5090 dry run should see, so that no slowdown turns up mid-run?

- **AFTER:** `fix/learner-recompile` plus this branch's 2048 mix (cbf0584e). It has the eager
  `Learner.head_sums`, equal micro-batches, `RUN_SLACK` 1.05 (BC's lanes are one stacked
  run), and the champion's model as the defaults.
- **BEFORE:** `rel/candidate` 9f8c20fc (the merge of `feat/launch`). It has the same model,
  with the champion's settings given explicitly, and none of the fixes.

## Method

- `steady.py` runs the trainer's own `Trainer.update`: collect, analytics, learn, and the
  cadence's diagnostics and heads' sums every tenth update and on the first. The compile
  bench is not used for steady numbers: its train case runs the heads' sums on every
  update and never runs the cadence. `timed=1` wraps each compiled function with the
  bench's timer (device syncs) for per-unit compile times. `timed=0` is a run's own path.
- Config: `configs.sh` resolves the config with AFTER's code, and the same JSON feeds both
  trees (BEFORE's own mix, 50/30/20, would not split 1024; the BEFORE config differs from
  BEFORE's own resolution only in the pool shares).
  - The settings: preset `real_scale_all_modes`, the lineup Yongma x2 + Jujak x3, the
    champion's model, `host_store=false`, the default 1024 lane environments.
  - gpubox4 (RTX 5060 Ti 16 GB) runs 1024 environments, half the launch's 2048, with the
    same divisibility: 6144 views divide 256, 128, 32 and 16 minibatches.
  - The laptop runs 256 environments.
  - Micro-batch rows: BC 2560 (the launch's; at these sizes BC's minibatch is one
    micro-batch), RL 320.
- Steady state: the updates after the first two. Plain updates are averaged, and the SPS is
  weighted one cadence update in ten, as a run has it (`summarise.py`).
- Records: `records/records.jsonl`, one line per machine x tree x stage x mode.

## 2048 environments (the author's change, 2026-11-12)

`real_scale_all_modes` is now 0.5 / 0.3125 / 0.1875: 1024 / 640 / 384 of 2048, and each
mode's pools are whole (512 / 320 / 192). `launch.toml` is set to 2048. New quick test:
`test_the_all_modes_mix_splits_2048_environments_into_whole_pools`, and the compile-split
test checks 12,288 views at 256, 128, 32 and 16 minibatches. Micro-batch sizes
(`micro_batch_sizes`, the learner's rule):

| Config | 2048 envs: loss sizes | cadence adds | 2040 envs: loss sizes | cadence adds |
|---|---|---|---|---|
| BC launch (2560 rows, 256 minibatches) | 48 | 24 | 47, 48 | 24 |
| RL launch (640 rows, 32 minibatches) | 64 | none | 63, 64 | none |
| BC minibatch2x (128 minibatches) | 96 | 48 | 95, 96 | 48 |
| RL minibatch2x (16 minibatches) | 64 | none | 51 | 63, 64 |

At 2048 every stage's loss compiles one size. BC's cadence still adds a second size: the
gradient decomposition's halves of the first minibatch, 24 views. The cadence's own
`_terms` is its own compiled function anyway, but the label codes (`Learner._codes`) and
the shot tick compile again at that size.

## Log

- 22:28 gpubox4 round 1: AFTER BC and AFTER RL cold, side by side (5 compile threads each,
  `timed=1`, one update).
- 22:30 laptop: AFTER BC at 256 environments, cold to steady (13 updates, `timed=0`),
  `systemd-run --user` with MemoryMax=8G, power profile `performance`.
