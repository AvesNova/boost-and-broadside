# GPU box log (rented RTX 3090, 2026-11-08)

What I did and why on the rented box. Goals from the author: (1) make the runs fast on
the big GPU (target 30,000 SPS), first with changes that leave training unchanged, then
others if the GPU is still not full; (2) finish the arms research (round 2); (3) a 200M BC
run, then a 200M RL run from it; commit the run (git LFS or otherwise). "Solve unexpected
errors yourself and note everything." Entries: time (box clock, UTC), what, why. Newest last.

## Setup

- **10:05** The box is an **RTX 3090** (24 GB, sm 8.6, 420 W cap), not a 4090 as briefed:
  everything below is tuned and measured on the 3090. Driver 580 (CUDA 13.0); torch
  2.15.0+cu132 in the repo's `.venv` works (`cuda.is_available()` true). 10:05.
- **10:05** `/workspace` is **not a volume**: nothing here survives a recycle or destroy;
  runs must leave the box (git LFS push, or another sync) to be kept.
- **10:05** No W&B, Hugging Face or GitHub credentials on the box; git-lfs 3.4.1 installed.
  The round-2 run directories (C-fixed's 30M policy, arm 2d's 8.4M checkpoint, every
  tournament) live on the laptop only: not in git, not on any branch. Consequence for the
  arms: the tournament references must be retrained here (see the arms section).
- **10:06** Repo at v2 60c80a41 (= origin/v2 after a fetch). Remaining arms per
  `round2-log.md` 11:31: arm 2d (to 30M), 3d, 4d, each with its tournament.

## Speed (goal 1)

Every number on the 3090, arm 2d's command (`--preset mixed --objective bc --set
yemong.order=temporal_first --set yemong.next_state=delta --set yemong.delta_baseline=persistence
--set yemong.spread_inputs=scalar`), steady state (update 2-3; update 1 compiles). One update is
1,048,576 environment steps (2048 environments x 128 ticks x 4 shards). SPS = steps / (rollout +
learn). Scratch scripts: `bench`-style in-process profiler (`prof.py`, kept in the box's scratch).

- **10:28 Baseline (defaults):** rollout 9.0 s, learn 127.5-128.3 s, 136.5 s per update =
  **7,700 SPS** (laptop 3070: ~232 s learn). BC KL at update 3 0.7155 (laptop arm 2: 0.712):
  same behaviour. GPU **14% busy, 130 of 420 W** during the learn phase, the process at ~145%
  CPU: launch- and host-bound, not GPU-bound. Memory probe: 4.6 GB allocated of 23.6.
- **10:45 Profile (1 epoch, torch profiler on 4 minibatches):** the learn phase is `prepare`
  (29 s: shot-label re-simulation and the next-state counts, no grad, over 512 micro-batches
  of 16 views) + 4.3 s teacher carries + the minibatch loop (~100 ms per minibatch, 1024 per
  update). In the loop: ~10 ms per minibatch is fp32 sgemm outside autocast (the shot
  trajectory query, `_next_state_logits`); ~12 ms device per optimizer step is `_step`'s
  non-finite guard (a clone and a `where` per parameter and state tensor, ~1,400 kernels).
  (py-spy cannot attach here: no SYS_PTRACE.)
- **10:51 `--micro-batch-rows 2560` (training-neutral, the docs' "memory only"):** a minibatch
  is 32 views (320 rows), so any value >= 320 gives one micro-batch per minibatch; `prepare`
  uses the same split, so 2560 (256 views) cuts its 512 chunks to 32. `prepare` 29.0 -> 2.5 s.
  BC KL at update 2 equal to the baseline's to the printed digits (1.6064 both, 1 epoch).
  Memory probe 6.9 GB allocated / 12.6 reserved.
- **10:58 `_step` on the fused Adam (code, training-identical):** the fused CUDA Adam takes
  `found_inf` and then leaves parameters and moments untouched and takes its step count back;
  `_step` now uses that instead of the clone/where copy-back (kept for non-fused optimizers).
  Test `tests/contract/test_fused_skip.py` (GPU): finite steps equal the plain fused step bit
  for bit over 5 steps; a skipped step (first or later) changes nothing. Branch `perf/gpu-box`.
