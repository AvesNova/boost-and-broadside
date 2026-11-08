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
- **11:10 Full config, `--micro-batch-rows 2560` (old `_step`, host store):** learn 94.0 s,
  103 s per update = **10,200 SPS**. KL by update 1.4169 / 0.7160 / 0.7000 (baseline 1.4169 /
  0.7155): equal within run-to-run noise.
- **11:20 + `--set host_store=false`** (the store on the GPU: 24 GB fits it; training-neutral):
  learn 80.3 s (this run already had the new `_step`, see 11:35) = **11,800 SPS**. GPU 100% busy,
  411 of 420 W, 20.1 GB of 24 used (probe: 17.7 allocated, 22.8 reserved of the 23.06 budget:
  little headroom), CPU one core of 12, RAM 7.2 GB. The GPU is now the bound: more CPU cores
  would not help (the user asked); dispatch was the bound only at 160-row micro-batches.
- **11:35 `_step` isolated:** the old `_step` from a clean v2 worktree (`/workspace/bnb-v2-clean`),
  same settings: learn 83.4 s; new: 80.3 s (-3.1 s, ~4%). KL 0.6987 vs 0.6998 at update 4:
  run-to-run noise (two old-code runs differing only in `host_store` differ by 0.0013).
- **11:38 TF32 (the user asked):** not on. `--precision bfloat16` (the default) is bf16 autocast
  over the encoder, trunk and heads and `"highest"` (true fp32) for every other fp32 matmul;
  `tf32` is either/or with bf16. The user's rule: no TF32 for anything that accumulates many
  small numbers. TF32 rounds a matmul's inputs (10-bit mantissa) and still sums in fp32, but a
  global switch would also catch the game, the scripted teacher and the belief's covariance
  propagation, so **no global TF32**: only targeted changes to products of activations and
  weights. Profile by shape: the fp32 GEMMs are the shot trajectory query (737,280 rows =
  32 views x 128 ticks x 10 ships x 18 bullet ages; 128x128 and 128x154 with their backward),
  ~10.5 ms of ~77 ms per minibatch.
- **11:40 The shot trajectory query under the model's matmul autocast** (bf16 like every other
  head; precision change, small): learn 80.3 -> **71.1 s, 80 s per update = 13,100 SPS**. The
  head's own losses unchanged at 4M (trajectory 16.370 vs 16.369, outcome 0.230 vs 0.228); KL
  within noise. CPU tests touching the head and the learner pass (90). Fixed a pre-existing
  ruff-format failure in `tests/sections/test_yemong_error_head.py` (from ae729475).
- **11:42 Committed** on `perf/gpu-box` (23303d81). Push refused: the box's SSH key
  (`~/.ssh/id_ed25519_github`, given to the author as a deploy key) is not accepted yet.
- **11:44 Scope from the author:** this compute is rented; explore only what needs this box
  (sizes that only fit in 24 GB), not code speedups that could be done on the 3070.
  Remaining box-only idea: larger environment counts / minibatches. Below 30k SPS the gap is
  GPU work per sample (4 epochs x 1024 minibatch steps), not anything else.

## Arms (goal 2)

- **11:45 Setup.** C-fixed is retrained here as the reference (the author: OK). Every arm runs
  with the same settings, so the comparisons stay like for like: the laptop's algorithm
  (2048 environments, 4 shards, 4 epochs, 256 minibatches, 30M budget) plus the neutral speed
  settings (`--micro-batch-rows 2560 --set host_store=false`) and the code on `perf/gpu-box`
  (bf16 shot head), with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation; VRAM
  is near the budget). On failure the runner retries a run once with the host store.
  Lines (`runs/arms.sh`; all `--preset mixed --objective bc --bc-budget-steps 30000000 --set
  yemong.order=temporal_first`): C-fixed nothing more; 2d `--set yemong.next_state=delta --set
  yemong.delta_baseline=persistence --set yemong.spread_inputs=scalar --set
  learner.next_state=0.0033`; 3d = 2d + `--set yemong.delta_split=true --set
  learner.correction=0.0033`; 4d = 3d + `--set yemong.delta_baseline=extrapolation`.
  **Note:** the code's default `learner.next_state` is still the absolute head's 0.0041; the
  laptop's queue lines are not in git, but the round-2 log gives 0.0033 for the delta heads
  (04:13, 06:20), so it is passed explicitly. Checked with `--print-config` for all four.
  Tournaments (`bnb ratings <arm> --no-league --with <newest uniform averages> --presets
  frontline deathmatch`, ladder on, scripted = 1000): 2d vs C-fixed; 3d vs 2d, C-fixed; 4d vs
  3d, C-fixed. Output `runs/ratings/<arm>/`.
- **11:46 W&B fails from here:** with the author's key, the client logs in to
  `api.forge.coreweave.com` (wandb 0.31's default cloud) and gets an HTML 403 ("your client
  does not have permission"); `api.wandb.ai` too, and also `ifconfig.me`, while GitHub, PyPI,
  Hugging Face and example.com answer. So it is a network block on this path (possibly my
  command sandbox), not the key. Runs go without `--wandb`; `metrics.jsonl` holds every row
  and can be synced later. Hugging Face token verified (user julianzia, fine-grained).
- **11:52 Arms runner relaunched** (detached: `setsid nohup runs/arms.sh`; log `runs/arms.log`).
  C-fixed started.
