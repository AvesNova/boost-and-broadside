# gpubox2 log: the one-head A/B (interruptible RTX 3090, 2026-11-12)

The champion's pipeline repeated with `yemong.heads=1`, on an interruptible (half-price)
vast.ai instance. Entries: KST time, what, why. Newest last. Scripts in this directory.

## Reproduction

See the report section at the end. Records of the champion: `champion/*.json` (copied from
`results/gpu-box/runs/` on `perf/gpu-box`).

## Log

- **00:30 Champion reconstructed from its records** (`champion/{bc,rl}-{run,config}.json`,
  from `results/gpu-box/runs/` on `perf/gpu-box`):
  - BC `2026-11-09-bc-deadreckon-split-hiddenphysics`: 4 segments, 0-30.4M on fadd6065
    (budget 30M), 30.4-43.0M on 6736d1d4 (budget 60M; stopped by STOP at the 09:40 KST
    pivot), 43.0-60.8M on e84ee16a, 60.8-120.6M on 09db261c (budget 120M). Settings: the
    segment-0 command (round4.sh's COMMON + DR + `physics_hidden_decay=0.5`), 2560-row
    micro-batches, store on the GPU.
  - RL `2026-11-09-rl-deadreckon-split-hiddenphysics`: from the BC step_120586240 uniform
    average, 0-60.8M on 6af08e09 (58 updates, 320-row micro-batches after 640 ran out of
    memory), 60.8-121.6M on 82e12ff9 (`--updates 116`), 121.6-180.4M on f45ada61 (172).
  - src/: 09db261c = e84ee16a = 6af08e09 = 82e12ff9 = f45ada61 (no diff). fadd6065 and
    6736d1d4 differ only by the physics-hidden-fade switch (off; `delta.fade` runs only
    with it on) and tournament code (`--reuse`). So the arm runs **every stage on 6af08e09's
    src** (this branch: 6af08e09 + scripts in this directory only).
  - Checks (`check_config.sh`, `--print-config` against the recorded config.json): BC 3
    differences, all expected: `yemong.heads` 2 -> 1, `stop.budget_steps` 30M -> 120M (the
    champion's resumes raised it), `yemong.physics_hidden_fade` absent -> False. RL's
    command with the champion's BC file resolves to the champion's RL config exactly (0
    differences); the arm's RL is checked again at the hand-off (`heads` the only one).
  - `heads=1` passes validation: head width 128 >= 2 x 20 rotary pairs.
- **00:37 Box set up** (gpubox2, RTX 3090, driver 595.84; torch 2.15.0+cu132 from the lock,
  `uv sync --frozen` 22 s). `/workspace` is **not a volume** (`workspace_is_volume` false):
  a stop/start keeps it, a recycle or destroy does not; the laptop pulls every 30 min.
  Code: a depth-1 clone of this branch rsynced (anchored excludes), so the runs record a
  clean commit. Supervisor programs `heartbeat` and `onehead` (`supervisor.conf`), both
  autostart. Checkpoints every 4 BC updates (~6 min) and every 3 RL updates (~6 min): an
  interruption loses at most ~6 min of training plus the restart's compile.
- **00:38 Restarted BC once by hand** (dirty tree: an edit not yet committed). Found:
  `supervisorctl stop` lets `bnb train` finish its graceful stop (SIGTERM -> stop after the
  update), so the trainer outlives the stop for minutes; killed it. Fresh start 00:40 on 9f11922.
