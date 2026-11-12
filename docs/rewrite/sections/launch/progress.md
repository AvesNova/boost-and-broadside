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

## Next

- The laptop end-to-end: probe, two tiny chains on one slot, a kill and a resume, a
  tournament round, the leaderboard, replays, the reward check, the status, the pull.
