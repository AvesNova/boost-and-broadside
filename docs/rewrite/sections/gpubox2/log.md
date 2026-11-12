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
- **00:40 BC fresh start** (`2026-11-12-bc-onehead`, commit 9f11922). Compiling to at least
  00:53 KST (GPU 0%, 9.2 GB held; the heartbeat's 00:52 line shows the first GPU work, 110 W).
- **~00:55-01:10 Interruption 1.** The direct port (80.188.223.200:30148) and the proxy
  (ssh9.vast.ai:22629) both refuse connections. The last contact was 00:53 KST. The laptop's
  pull at 01:11 found the box unreachable. Lost: the compile, about 15 min, and no update
  (there was no checkpoint yet). `reach.sh` on the laptop now logs both ports every minute.
- **02:03 Back on its own (interruption 1 measured).** The box's last heartbeat was 01:09:26 KST
  and its first after the restart 02:02:53, so it was down **53.5 min**. Same address and SSH port
  (80.188.223.200:30148; the proxy came back too). The same kernel boot_id means a container
  stop/start, not a host reboot. Supervisor restarted `heartbeat` and `onehead`. The driver
  found no checkpoint, moved the partial run to runs/failed/, and started BC fresh at 02:02:53.
  **Lost:** 29 min of compile (00:40-01:09; no update had finished: the first update's compile
  plus graph capture was still running) and 53.5 min of downtime.
- **02:05 Interruption 2.** The box went down again about 2.5 min after it came back. The
  laptop's reach log shows it up at 02:03:25 and 02:04:26, and down at 02:05:26.
- **02:10 Back again (interruption 2 measured).** The last heartbeat was 02:04:53 and the
  restart 02:09:58, so about **5 min** down after about 2 min up. Same address, same boot_id.
  The driver started BC fresh again: still no checkpoint.
- **02:37 Interruption 3**, after about 27 min up. The laptop's reach log shows up at 02:36:45
  and down at 02:37:45.
- **02:39 Back (interruption 3 measured).** The last heartbeat was 02:37:01 and the restart
  02:39:08: about **2 min** down after 29 min up. That attempt also ended inside the first
  update (27 min of compile, no update). Fresh start 4 at 02:39.
  - **The first update compiles on one core** (`bnb` at 100% of one CPU; the host is 86% idle).
    This EPYC 7402P compiles more slowly than the first box (the champion's cold first update
    took 1,436 s there; here more than 29 min).
  - The inductor cache in /tmp survives a stop, so each attempt reuses the kernels the last one
    compiled. On the first box a warm first update took 250-454 s.
  - The real risk on this box: interruptions every 2-30 min can keep a cold first update from
    ever finishing.
- **02:47 Interruption 4**, after about 8 min up. The first update had still not finished.
- **02:51 Back (interruption 4 measured).** The last heartbeat was 02:47:09 and the restart
  02:51:13: about **4 min** down. The compile cache has grown to 6.2 GB. Fresh start 5.
- **02:55 Interruption 5**, after about 4 min up.
- **04:56 Back (interruption 5 measured).** The last heartbeat was 02:54:13 and the restart 04:55:53: **2 h 02 min** down. Fresh start 6. Up so far: about 77 min of 4 h 16 min since launch, and still no finished update.
- **04:58 Interruption 6**, after about 2 min up.
- **05:04 Back (interruption 6 measured).** The last heartbeat was 04:56:53 and the restart 05:03:43: about **7 min** down. Fresh start 7.
- **05:14 Interruption 7**, after about 10 min up.
- **05:14-06:44 Interruptions 7-11** (`downtime.py` on the heartbeat): 7.1, 21.1, 7.9, 7.5
  and 46.3 min down, with up windows of 10, 1, <1, <1 and <1 min. Several restarts lived less
  than a minute.
- **07:03 BC's first update finished** (fresh start 12, from 06:44:34): rollout 101 s,
  learn 1,001 s (18.4 min, with the cache warmed by the earlier attempts; the cold compile
  never finished inside one up-window). Memory probe: 16.95 GB allocated, 17.95 GB reserved
  of a 23.06 GB budget (the champion: 18.2 / 19.3 GB). KL 2.964 at update 1.
- Tally at 07:04: 11 interruptions in 6.4 h since launch; up 1.67 h (26%), down 4.72 h; one
  BC update done.
- **07:09 Steady BC:** 108 s per update (rollout 10.9 s, learn 97 s), about 9,700 SPS; the
  champion's was 92.6 s and 11,330 SPS. KL by update (arm / champion): 2.964/2.965,
  1.459/1.452, 0.721/0.717, 0.696/0.699, 0.661/0.661. Checkpoint step_4194304 is 344 MB.
  - **Speed confound: this host caps the GPU at 275 W.** nvidia-smi gives power.limit 275 W
    (max 375), active throttle reason 0x4 (SW power cap), 87-100% utilisation at 258-274 W,
    SM clock 1530-1815 MHz. The first box drew up to 411 of 420 W. The container cannot raise
    the limit.
- **07:10 Deliberate resume test:** kill -9 of the driver and the trainer mid-update 5.
  - Supervisor restarted the driver within a second. It marked the stale record stopped and
    resumed from step_4194304.
  - Interruptions 12 (07:11-07:14) and 13 (07:14-07:35) then hit the restarted process. Each
    later restart resumed from the same checkpoint on its own.
  - Update 5 finished at 07:41. The warm restart's first update took 102 s rollout + 227 s
    learn (5.5 min, against 108 s steady). KL 0.6605 (champion 0.6609).
  - run.json carries one segment per attempt (start 4194304; the killed ones have no end).
- Tally at 07:42: 13 interruptions in 7.0 h; up 1.92 h (27%), down 5.11 h.
- **07:35-08:02 Up 27 min, then interruption 14** (the laptop's reach log). The BC reached
  about 12.6M steps (step_12582912, seen by the coordinator). gpubox2 never came back. The
  author destroyed it about 09:15. Its final records (the run directory past 7.3M, its
  checkpoints, the heartbeat after 07:42) were lost with the box. The laptop keeps the run
  directory to 7.3M (checkpoint step_4194304), the heartbeat to 07:42
  (`gpubox2-heartbeat.log`) and the reach log 01:44-09:10 (`gpubox2-reach.log`).

## Verdict on gpubox2 (interruptible, $0.138/h including storage)

- **14 interruptions in 8.5 h** (00:40-09:10 KST): 13 measured in the heartbeat, plus the
  final one in the reach log.
  - Down spells: 2.1, 3.0, 4.1, 5.1, 6.8, 7.1, 7.5, 7.9, 20.4, 21.1, 46.3, 53.5 and 121.7 min,
    and the last open-ended (over 70 min, then destroyed).
  - Up windows: under 1 min (four times), 1, 1, 2, 3, 8, 10, 27, 27, 27 and 29 min.
- **Up about 2.3 h of 8.5 h (27%)**: the heartbeat to 07:42 gives 1.92 h up of 7.03 h; the
  reach log gives 110 of 442 minutes up from 01:44 to 09:10.
- **Each stop was a container stop/start**: the same kernel boot_id, the same address and SSH
  port. The proxy went down and up with it. The box always came back on its own until the
  last time, and supervisor restarted the heartbeat and the driver within a second of each
  boot.
- **Lost work:**
  - The cold first BC update compiles on one core for more than 29 min on this CPU, longer
    than most up windows. It needed 12 starts and 6.4 h of wall time to finish once (07:03).
  - After a checkpoint, a restart costs about 5.5 min (warm compile and graph capture) plus
    up to 4 updates (7 min) since the last checkpoint.
  - Several restarts lived under a minute and did nothing.
- **Throughput:**
  - Uninterrupted, the BC ran at 108 s per update: 9,700 SPS, about 35M steps per hour.
  - Effective: about 12.6M steps in 8.5 h of wall time, **about 1.5M steps per hour, 4% of
    uninterrupted**.
- **Cost:** 8.5 h x $0.138 = $1.17 for about 12.6M steps, **about $0.093 per million
  steps**. Uninterrupted the same box gives $0.138 / 35M = $0.0039 per million: 24x cheaper.
  - Even if stopped hours were not billed (storage is), the cost per step of up-time is still
    raised by the restarts: about 11 of 27 up-minutes per hour went to compiles that never
    finished.
- **Also on this host:** the 3090 was capped at 275 W (SW power cap; max 375 W), about 15%
  slower than the first gpu box.
- **Not worth the half price for this job.** A pipeline whose first step needs a 30-min
  single-core compile cannot make progress when the box is outbid every 2-30 min. The
  supervisor resume design worked; the market did not. Interruptible could suit work with no
  long cold start (a tournament with a warm cache, short jobs), or at a bid that is not
  outbid.
