# Overnight calibration: progress

Branch `calib/overnight` (from `perf/steady-speed`). Hard stop 08:30 KST, 2026-11-13.

**Goal (the author's, before the 5090 dry run).**
1. One minibatch count for BC and RL (today BC 256, RL 32: inherited, never A/B tested).
2. BC's plateau stop (`bnb.learn.bc.StopConfig`: 5% over 10M since 5M, smoothing 5M,
   hold 10M) recalibrated to where the BC policy's Elo against the scripted agent plateaus,
   at the chosen minibatch count.

## Setup

- `calib.toml`: the arms; `calib.py`: each arm's `bnb train` line from
  `../launch/launch.toml` and `calib.toml` (committed files only); `box.sh`: one run on a
  box in tmux; `plateau.py`: the plateau rule replayed offline on a run's logged BC KL
  (the training code's own `Plateau`) and the grid fit to the Elo plateaus.
- Boxes: gpubox4 and gpubox7, RTX 5060 Ti 16 GB on one host (each container 5.76 cores),
  everything under `/workspace/calib/` (repo, venv, runs, logs, per-box caches).
- Scale: 1024 environments (launch mix shares: 512 / 320 / 192), 524,288 steps per
  update, 6144 views per update.
- gamma / lambda: 0.9999 / 0.99 for every value level in both stages (`calib.toml`'s
  `set`: the discount and trace-decay tuples), not the code's defaults.
- BC arms: minibatches 256, 128, 64, 32; everything else equal (learning rate 3e-4,
  4 epochs); micro-batch 240 rows (24 views) for all four, so all share one compiled shape;
  40M budget, plateau stop off; policy files every 5M (live, both averages).

## Log

- 23:47 Started. Worktree `/home/vizia/avesnova/bnb-calib`; box repo and venv set up on
  `/workspace/calib` (gpubox4, niced, during the speed agent's run).
- 00:25 Change of allocation (coordinator): gpubox4 only (gpubox7 goes to ship experiments).
  BC arms in sequence, priority 256, 64, 32, 128; budget 30M each (a budget-stopped BC run
  resumes with a larger one if time allows); RL forks only if the BC arms finish early.
- 00:30 gpubox4 free (the speed agent's runs done). Merged `perf/steady-speed` 7f7fbeb1
  (the heads' sums sample) as 0f3ec4b4. The venv's Python is on the volume
  (`UV_PYTHON_INSTALL_DIR=/workspace/calib/python`; the default install is container-local).
  The box cache `cache-box4` was seeded with the speed agent's AFTER BC and RL caches.
- 00:32 **bc-mb256 launched** (cold for its rollout without lanes; the learner's 24-view
  shape may hit the seeded cache).
