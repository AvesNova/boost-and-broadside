# Launch: progress

Branch `feat/launch` (from `integrate/next-runs` abcecd53). Worktree
`/home/vizia/avesnova/bnb-launch`. Times KST, 2026-11-12. Runbook: `runbook.md`.

## 14:20-14:45 Merges and the training-side switches

- **Tried / worked.** Merged `feat/position-codes` (c82e34b5) and `feat/bc-rl-chain`
  (7322d5cb, the chain is not on `integrate/next-runs` yet): both clean. Quick tier of the
  touched files passes. Fixed the merge's doc leftovers (the learner page's rotary read-out
  label, `COUNTS` in full, missing docstrings in the merged benches and the geometry
  target).
- `TrainConfig.lineup` (slot type names, both teams, every game of the batch; checked at the
  config) and `bnb ratings --lineup`, `--seed`. The mix `real_scale_all_modes` (0.5 / 0.3 /
  0.2): it splits only at multiples of 20 environments, so the final config runs 2040.
- `bnb queue` on several GPUs: `run --slot K [--gpu G]`, `add --priority`, `add --warm`.
- **Bug fixed:** a chain's BC cap (`--bc-set stop.budget_steps`) was lost on a resume in BC
  (the resume passed the combined budget as BC's). Tested.
- **Geometry head weight:** kept at 0.007 in the control; x3 (0.021) is a Stage A arm.
- **Didn't:** pure self-play has no switch (`pools` cannot be `--set`): the arm is marked
  `needs` and refused by the launch.

## 14:45-15:00 The launch scripts (re-prioritised: the single-5090 dry run first)

- `bootstrap.sh`, `env.sh`, `runner.sh`, `heartbeat.sh`, `cache.sh`, `launch.py`,
  `replays.py`, `reward_check.py`, `pull.sh`, `launch.toml`, `arms.toml` (the mainline at
  300M, Stage A, the ablations; a proposal).
- **Bug found by the laptop test** (the uncompiled rollout under CUDA graph capture): the
  covariance pack and unpack made their index on the host and copied it in the capture.
  Fixed (made on the device).

## 15:00-15:40 Laptop end-to-end (tiny: 40 environments, uncompiled, budget 400k)

- **Worked:** `check` refuses the placeholder; `probe` stepped BC 1280 (fits) and RL
  640 -> 160 (out of memory at 640 on the 8 GB GPU), writing `micro_batch.json`; `enqueue`
  queued two chains (control, geometry x3) and skipped them when run again; the runner
  (`runner.sh 0` as a user unit) ran job 1. **Killed the unit with SIGKILL** mid-BC (the
  reboot case): restarted, the runner marked job 1 interrupted and queued it again as a
  resume of the chain, which went on from its step-0 checkpoint and reproduced the first
  segment's KL to 4 digits (2.8790, 2.8979, 2.9146 / 2.9147), handed over to RL at the BC
  cap (163,840) and finished. The tournament round's three ratings jobs (priority 100) went
  before job 2. `status`, `pull.sh` (local mode: no checkpoints pulled), `replays.py` (a
  whole elimination match, saved as a replay, `matches.jsonl`) and `reward_check.py` (its
  flags caught the untrained policy's boundary deaths) work.
- **Bugs found and fixed:** the probe parsed the peaks line wrong and stepped down on any
  failure (now only on out-of-memory); replays could not load a mixed batch's policy in
  elimination (the play shell's game check: now `any_mode`).
- **Noted:** before the warm-up ends (10M) a BC snapshot has no uniform average, so early
  tournament fields hold only live policies (the production steps start at 25M).
- **Hub path (coordinator, 15:30):** `hub.sh` and bootstrap's `hub:NAME` and `hub-sync`;
  downloads tested against the public repo; uploads need the author's write token on the box
  (not testable from the laptop: no token here).

## Next

- The laptop end-to-end: probe, two tiny chains on one slot, a kill and a resume, a
  tournament round, the leaderboard, replays, the reward check, the status, the pull.
