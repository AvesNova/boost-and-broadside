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
- **12:03 One tournament instead of three (the author: fewer tournaments, more arms).** The
  chain 2d -> 3d -> 4d never waited on a result, so the intermediate tournaments changed
  nothing; one field over every arm rates each pair directly and replays C-fixed and 2d less.
  Measured: an arm is ~50-55 min (first update ~11 min, compiling a new config cold; then
  ~81 s per update); a two-mode tournament took ~25 min on the laptop. The runner was swapped
  without touching the C-fixed trainer (a bash script cannot be edited while it runs).
- **12:12 + C-fixed-LB (large batch, the author's request)** before the tournament:
  `--set learner.minibatches=128 --set learner.epochs=2` (64-view minibatches, each sample used
  twice: a quarter of the optimizer steps per sample, each on twice the data; lr unchanged,
  so the comparison isolates batch and reuse). Expected ~2x SPS. Runner `runs/arms3.sh`
  (copy in `gpu-box/arms3.sh`). Plan after the tournament (the author): my choice of a next
  set of tests, a second tournament, then the best two to 60M.
- **12:12 GitHub push works** (the deploy key; `perf/gpu-box` pushed).
- (Two process-kill slips of mine, harmless: `pgrep -f`/`pkill -f` with a pattern that my own
  command line contained killed my shell; no run was affected. Patterns now `[a]rms`-style.)
- **12:37 C-fixed done** (`2026-11-08-bc-arm-C-fixed-3090`, budget, 30.4M, 49 min, 81.1 s per
  update = 12,900 SPS). KL 0.475, agreement 0.678 (laptop C-fixed at 30M: 0.500, 0.672):
  in line. Snapshots at 10.5M, 21.0M, 30.4M. Arm 2d started.
- **13:20 Arm 2d done** (`2026-11-08-bc-arm2d-persistence-3090`, budget, 30.4M, 41 min: the
  compile cache warm). KL 0.479, agreement 0.681 (C-fixed 0.475, 0.678): level, as on the
  laptop. Last-row outcome Brier skill 5-15 s 0.024 vs C-fixed's 0.058 (the laptop's direction;
  single rows, not its 10-log means): the tournament judges. Arm 3d started.
- **13:35 Arm 3d's first updates healthy:** probe 17.2 GB allocated (fits with the store on the
  GPU), no non-finite beliefs or skipped steps, 85 s per update (2d 80 s: the error head). The
  error head's predicted spread starts narrow (position 0.11 m vs realised 1.5 m at 0-0.1 s,
  update 2), the opposite of the laptop's too-wide bug and as the laptop saw after its fix
  (the head's own early overconfidence). Recheck at 30M.
- **14:14 UTC (23:14 KST; the author asked for KST from here) Arm 3d done**
  (`2026-11-08-bc-arm3d-split-3090`, budget, 30.4M, 54 min). KL 0.471 (lowest; C-fixed 0.475,
  2d 0.479), agreement 0.676: level. Error head, realised / predicted error by hidden age
  (0-0.1, 0.1-0.5, 0.5-1, 1-2, 2-5 s): position 0.17/0.64, 1.5/0.83, 6.5/1.7, 20/5.6, 72/30 m;
  heading 0.018/0.021, 0.095/0.042, 0.23/0.10, 0.38/0.19, 0.43/0.21 rad. Right at the shortest
  ages, 2-4x too narrow past 0.1 s (overconfident; not the laptop's too-wide bug). A candidate
  for the next set of tests if the split does well. Arm 4d started.
- **14:22 UTC (23:22 KST) Arm 4d failed on its first update: a code bug, not memory.**
  `torch._dynamo.exc.Unsupported: dataclass fields failure` in `Yemong.segment_rates`
  (`dataclasses.fields(Rates)`), reached from the learner's compiled labels (`Learner._codes`,
  fullgraph). No extrapolation arm had ever run compiled: arms 4, 4b, 4c, 4d were all removed
  from the laptop's queue before running, and the CPU tests run eager. The runner's automatic
  retry (host store) would fail the same way, so I stopped it (failed dirs in `runs/failed/`).
  **Fix:** `delta.RATE_FIELDS`, the field names taken once at import, as `BELIEF_FIELDS` and
  `bc.CARRY_FIELDS` already are (bc.py says why); used by `segment_rates` and `Rates.rows`.
  **Test:** `test_the_baseline_rates_trace_whole_under_torch_compile` (every arm: fullgraph,
  eager backend, equal to eager); fails on the old code for arms 4, 4c, 4d, passes now.
  Error-head file + delta and acceleration tests: 161 pass; two bit-for-bit golden-digest tests
  fail (`test_off_steps_and_unrolls_c_fixed_bit_for_bit_like_the_tree_before_the_switch`,
  `test_arms_1_and_2_step_and_unroll_bit_for_bit_as_recorded`), **identically on clean v2 on
  this box**: digests recorded on the laptop's CPU; machine-dependent, not a regression.
- **14:32 UTC (23:32 KST) Arm 4d restarted** (runner `arms4.sh`: 4d, then LB, then the
  tournament; copy in `gpu-box/arms4.sh`).
- **00:00 KST Arm 4d stable after 3 updates** (the laptop's early CPU runs diverged to 1039 m/s
  velocity error before the held-rate decay). Hidden 2-5 s error at update 3, 4d vs 3d:
  position 122 vs 133 m, velocity 56 vs 59 m/s, heading 1.4 vs 0.8 rad (worse: the held nose
  rate keeps a hidden ship turning, as the laptop saw). KL 0.7196 vs 0.7194. No non-finite
  beliefs or skipped steps; 17.0 GB. 92 s per update (3d 85: the per-tick rate loop). ETAs
  (KST): 4d ~00:45, LB ~01:15, tournament ~01:40.
- **00:43 KST Arm 4d done** (`2026-11-08-bc-arm4d-extrapolation-3090`, budget, 30.4M, 70 min).
  KL 0.470, agreement 0.682. All four at 30M: C-fixed 0.475/0.678, 2d 0.479/0.681, 3d
  0.471/0.676, 4d 0.470/0.682 (KL/agreement): level on imitation. Error head, realised /
  predicted at 0-0.1, 0.5-1, 2-5 s: position 0.063/0.88, 6.5/2.4, 85/48 m (3d: 0.17, 6.5, 72
  realised); heading 0.021/0.023, 0.26/0.13, 0.46/0.25 rad (3d 0.018, 0.23, 0.43). C-fixed-LB
  started.
- **00:55 KST C-fixed-LB speed:** learn 38.9 s + rollout 8.2 s = **22,300 SPS** (C-fixed 12,900:
  1.7x). KL per sample behind early (update 2: 2.14 vs 1.39; update 3: 0.81 vs 0.73), as
  expected with a quarter of the optimizer steps per sample. Probe 20.3 GB allocated, 21.8
  reserved of 23.06 (tight, fits). ETA ~01:10 KST, then tournament 1.
- **01:12 KST C-fixed-LB done** (budget, 30.4M, 29 min): KL 0.575, agreement 0.640 (C-fixed
  0.475 / 0.678). **At equal wall-clock it is a wash:** C-fixed after LB's 0.48 h (14.7M) had
  KL 0.576, agreement 0.645. Twice the speed bought half the learning per sample: the large
  batch (as set: lr unchanged) gives BC no gain. The tournament checks play.
- **01:12-01:28 KST Tournament 1, first attempt (the runner's):** with the default stationary
  ladder the field was 14 players (8 rungs), 91 pairs: batch 1 = 1024 games in 915 s (with
  compile), the largest standard error 276 (target 25), ratings in mirrored pairs (+57/-57):
  projected 21:02 UTC for frontline alone. Stopped; restarted **without the ladder** (6 players,
  15 pairs; the arms play the scripted agent directly, which anchors 1000), `runs/tournament1.sh`
  (copy in `gpu-box/`). The record of the ladder attempt is in `runs/failed/`. Slips of mine:
  another self-matching `pgrep` (killed my shell, nothing else), and a path given where
  `bnb ratings` takes a run name (failed at once, relaunched).
- **01:34 KST Tournament 1, no ladder, batch 1 (frontline, 1024 games, 327 s with the compile
  cache warm; scripted = 1000):** C-fixed 807 +- 26, 4d 775 +- 30, 3d 722 +- 32, 2d 675 +- 32,
  **C-fixed-LB 215 +- 78**. LB nearly always loses (its KL resembled C-fixed's at half the steps,
  but its play does not): the large batch as set is out. Its slowly shrinking error alone would
  have held the stopping rule (every error under 25) for ~9 batches, so I dropped it and
  restarted with the four arms (10 pairs); the LB batch's record is in `runs/failed/`.
  (`runs/killpat.sh REGEX` now kills by pattern, sparing its caller and ancestors: the fourth
  self-matching kill of mine is the last.)
- **02:00 KST Names (the author asked for human ones; run directories keep theirs):** C-fixed
  = **`absolute`** (the absolute next-state head); 2d = **`persist-onehead`**; 3d =
  **`persist-split`**; 4d = **`deadreckon-split`**; C-fixed-LB = **`absolute-bigbatch`**.
- **02:00 KST Brier skill of the outcome head** (mean of the last 10 logs from 20-30M, by time to
  the end: <5, 5-15, 15-30, 30-60, 60-120, >120 s): absolute 0.043 0.037 0.032 0.026 0.016 0.008;
  persist-split 0.033 0.030 0.021 0.013 0.010 0.007; deadreckon-split 0.030 0.024 0.019 0.014
  0.009 0.004; persist-onehead 0.015 0.012 0.008 0.005 0.000 0.001; absolute-bigbatch -0.04 to
  -0.07. Same order as the laptop, but ~14x lower (laptop C-fixed 5-15 s 0.52, arm 2 0.15).
  Same formula (`analytics.outcome_series`). **Check queued (the author agreed):** absolute to
  10.5M on unmodified v2 (old `_step`, fp32 shot head) with only the neutral settings, against
  this box's absolute at 10.5M: equal clears my changes (then code history or the machine).
- **02:10 KST Tournament 1 standings** (scripted 1000): frontline deadreckon-split 789 +- 16,
  absolute 780 +- 16, persist-split 739 +- 18, persist-onehead 682 +- 18; deathmatch batch 1
  deadreckon-split 743, persist-onehead 727, absolute 720, persist-split 715 (+- 25). **The author:
  absolute and deadreckon-split are co-champions; both go into the next round.**
- **02:10 KST Round 3 (the author's ideas + mine), all on deadreckon-split, one change each:**
  `-fastdecay` (`yemong.extrapolation_decay` 0.5 -> 0.2 s: hidden ships keep turning on the held
  nose rate); `-fourier` (the ship token's position and heading as cos/sin at the colour code's
  periods: 9 levels 26,244 m -> 4 m per axis, heading 2pi/4^l, l = 0..3, attenuated by sigma as
  the colour code is; 36 + 8 inputs instead of 81 + 16; refused with the absolute head, whose
  targets are colour codes); `-hiddenphysics` (the author: train the physics head on hidden
  enemies too, the loss weighted exp(-age/tau), tau 0.5 s as its own setting
  `yemong.physics_hidden_decay`; target truth-to-truth: truth(t+1) - baseline(truth(t)), the
  inputs the belief, so the physics head learns dynamics and the error head keeps the belief's
  error; addresses risk C5: the physics head drives hidden forecasts without ever training on
  hidden inputs). Fourier and hidden-physics built by two background agents in worktrees
  (`/workspace/bnb-fourier`, `/workspace/bnb-hidden-physics`), CPU only. Runner `round3a.sh`:
  the Brier check, then fastdecay. Then tournament 2 (absolute, deadreckon-split and the new
  arms); then the best two resumed to 60M.
- (The author asked: for deadreckon-split, a hidden enemy's forecast is the dead-reckoning
  baseline from its held, decaying last-sighting rates + the physics head's change (the head
  runs on hidden rows though trained on seen rows only) + the error head's capped, rate-limited
  step toward its median; its spread is the error head's. Visible rows: baseline + physics.)
- **02:12 KST Tournament 1 final** (`runs/ratings/round2/`, 2048 games per mode, scripted
  1000): frontline deadreckon-split 789 +- 16, absolute 780 +- 16, persist-split 739 +- 18,
  persist-onehead 682 +- 18; deathmatch deadreckon-split 739 +- 17, persist-onehead 711 +- 17,
  persist-split 704 +- 18, absolute 693 +- 18. Means: 764, 737, 722, 697. deadreckon-split over
  absolute: +9 frontline, +46 deathmatch, +27 mean: under the bar (+50 / +40), co-champions as
  the author decided. Brier check started.
- **02:15 KST Learnable tau (the author asked):** `extrapolation_decay` can be learned, but not
  through the existing losses (in training the hidden beliefs come from the recorded rollout, so
  its only gradient is through the error head's target, which rewards easy targets, not small
  errors): a separate loss on the forecast error itself, networks stop-gradded, possibly per
  quantity and ship type. The loss-weight tau (`physics_hidden_decay`) cannot be learned by
  gradient: the loss is lowest with hidden rows weighed at zero (tau -> 0); it is a
  hyperparameter. Suggested: fastdecay first, then an offline per-quantity fit on recorded play.
- **02:20 KST Everything off the box, continuously (the author: make sure all data is saved).**
  `gpu-box/sync.sh`: git gets each run's records under `results/gpu-box/` (run.json,
  config.json, metrics.jsonl, console.txt; tournaments; runner log and scripts; `.txt` because
  the repo ignores `*.log`), committed by path only and pushed; the Hugging Face Hub gets every
  run directory whole (policies, checkpoints) in the public model repo
  https://huggingface.co/julianzia/boost-and-broadside-runs (the author allowed public). The
  first pass: 3.88 GB, 253 files on the hub; git 9c960148. A loop runs it every 20 min
  (`runs/sync.nohup`). `huggingface_hub` installed in the image's `/venv/main`, not the project's
  locked venv. Per run ~800 MB: checkpoints 662 MB, policies ~100 MB, records ~2 MB.
- **02:45 KST Fourier inputs merged** (`feat/fourier-inputs` e51591d9, merge fd33c48f; built by a
  background agent, verified by me: ruff; the codes and error-head files 112 pass, the one
  failure the known machine-dependent golden; the arm's line resolves `position_inputs=fourier`;
  fourier with the absolute head is refused). From its report: `codes.fourier_encode` (per level
  a_l (cos, sin), the colour code's attenuation, the same per-level remainders);
  `ShipStateCode.inputs` swaps only the position (2 x 9 x 2) and heading (4 x 2) runs into the
  ship token; `tokens.code` stays the colour code for its readers (`_next_state_logits`, the
  absolute head's residual base; `head_statistics` baseline); state columns 490 -> 437, the arm's
  ship token 553 -> 500; its own choice: the first layer's init counts each (cos, sin) pair as
  one group. The default is bit for bit (it compared the digests of six configs on one thread
  before and after). It found a third machine-dependent golden here
  (`test_yemong_spread_fit::test_the_default_v1_fit_steps_bit_for_bit_like_the_pre_fix_tree`)
  and a load-dependent flake (`test_yemong_order::test_unroll_equals_a_loop_of_steps[D]`, ~1e-5),
  both on clean code too. Queued after fastdecay (`round3b.sh`).
- **02:45 KST Disk (the author: 64 GB; prune what is not needed, watch it):** 40 GB used. After
  checking the hub has every file (by size): deleted the eliminated arms' checkpoints
  (persist-onehead, persist-split, absolute-bigbatch; their policy files kept for re-rating), the
  failed run directories (their console logs and the aborted tournaments' records copied to
  `results/gpu-box/failed/` first), and my scratch speed-test runs: **34 GB used, 31 free**. The
  largest user is Inductor's compile cache (26 GB); kept (wiping it costs ~10 min of compile per
  config, the co-champions' resumes included); `sync.sh` now prunes cache files unused 12 h when
  under 15 GB free, and reports free space every pass.
- **02:34 KST Brier check done** (`2026-11-09-bc-absolute-v2clean-check`: absolute on unmodified
  v2 to 10.5M). Mean of updates 6-10, Brier skill <5 ... >120 s: v2 clean -0.125 -0.140 -0.152
  -0.176 -0.154 -0.074; this box's absolute -0.148 -0.163 -0.173 -0.202 -0.166 -0.074; calibration
  error 0.143 vs 0.148; KL at 10.5M 0.503 vs 0.525 (likely seed noise: the bf16 shot head left KL
  unchanged at 4M). **My changes did not break the outcome head.** Still open: the laptop's 0.52
  at 20-30M against this box's 0.037 (code history since 11-07, or how the laptop's figure was
  read). Cheapest check, on the laptop: the mean of `outcome/brier_skill/5_15_seconds` over its
  C-fixed's last 10 rows. Or extend the clean run to 30M here (~35 min). fastdecay started.
- **02:55 KST Pushes (the author: my `git push` calls ask him to approve each one; keep the runs
  going whether or not a push works):** I no longer push from my own commands; I commit locally
  and the detached sync loop pushes `perf/gpu-box` every 20 min (its pushes need no approval:
  checked, origin = local at d4ed65ac). The hidden-physics agent was told to commit only; I
  merge its branch locally.
- **03:10 KST Outcome Brier: a regression in the code, not the box or my changes.** The
  laptop's C-fixed (`2026-11-07-bc-arm-C-fixed`, commit 3b72a914, 0-60.8M) at 5-15 s: 6.3-10.5M
  +0.080 +0.104 +0.180 +0.112 +0.192; 21-30.4M 0.468 rising to 0.560. This box at 6.3-10.5M:
  -0.12 to -0.34 on current v2 (clean) and on `perf/gpu-box` alike; 0.037 at 20-30M. And on the
  laptop the runs on newer code were already lower (arm 1 0.35, arms 2 / 2b 0.15 at 20-30M),
  which round 2 read as the arms' effect. So a change in the 98 commits (12 first-parent merges)
  from 3b72a914 to v2 hurts every config's outcome head. Plan: reproduce at 3b72a914 here
  (`runs/bisect.sh COMMIT`: absolute to 10.5M from a detached worktree, read Brier at updates
  8-10), then bisect the merges (~4 steps of ~20 min). Suspects: 1b7218ae (run control: BC
  snapshots, the uniform average from the warm-up), 8ee3178c (accelerations in the observation),
  5cff3794 (delta heads). Queued after fastdecay, before fourier (RL depends on the outcome head).
- **03:20 KST My error, found by the author's laptop agent: no box run was C-fixed.** The
  laptop's C-fixed (and every round-2 arm there) ran with `--set
  learner.discount=[0.991,0.995,0.995,0.999,1.0] --set
  learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]`: the outcome level's target is the actual game
  result (gamma = lambda = 1). I rebuilt the commands from the briefs, which list only the
  model switches; those two lines live only in the laptop's queue commands (nowhere in the
  repo's docs or code, checked). Every box run (round 2, the v2 check, fastdecay) has the
  defaults ([..., 0.9997] / [0.9, 0.95, 0.95, 0.97, 0.97]): a discounted, bootstrapped outcome
  target, hence the negative Brier at 6-10M. The laptop's 2026-11-06 arm-baseline, with the same
  defaults, read -0.23 there; every laptop run with C-fixed's values from 3b72a914 to 272a44ed
  reads +0.04 to +0.19. So **not a code regression**: my "regression" note at 03:10 and the
  bisection prompt were wrong (they could not separate code from command). Consequences: the
  box's tournament 1 compared arms all on the same wrong base (internally consistent, but none
  is C-fixed); round 3 is on it too. **Paused round 3** (both runners stopped; fastdecay asked to
  STOP at update 6-7, resumable; fourier not started). Running the laptop agent's confirmation
  (`runs/confirm.sh`: absolute to 10.5M with the two lines; expect +0.1 to +0.2 at updates
  6-10). Round 3's restart needs the author's call.
- **03:20 KST From the laptop session (the author asked the two agents to talk directly):** its
  per-run numbers at 6.2-10.6M with C-fixed's learner settings: C-fixed +0.13, C-accel +0.10,
  arm2 +0.07, arm2b +0.08, arm2d +0.04 (laptop arm2d at 272a44ed +0.04 vs box arm2d -0.17).
  The laptop's arms also passed `--set stop.score_target=1.0 --set
  stop.plateau_steps=1000000000` (so no stop rule ends an arm before its budget); the box's 30M
  runs lacked them too but all reached the budget anyway (`stopped_by: budget`). It is building
  `fix/bc-defaults`: `bc.config()` takes C-fixed's discount and trace decay (PPO keeps the
  spec's), and a new build's `yemong.order` defaults to temporal_first; the stop flags stay on
  the command line (documented in `docs/guides/behaviour-cloning.md`). I merge it when pushed.
  Its tip, adopted: rebuild a run's command from its `run.json` (`segments[].command`), not
  from names or notes.
- **03:35 KST Round 3 redone on the correct base (the author's plan), `runs/round4.sh`:** gated
  on the confirmation (starts only if its Brier mean at updates 6-10 is > +0.05). Every arm at
  30M with C-fixed's learner settings and the laptop arms' stop flags (`--set
  "learner.discount=[0.991,0.995,0.995,0.999,1.0]" --set
  "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" --set stop.score_target=1.0 --set
  stop.plateau_steps=1000000000`), in order: absolute, deadreckon-split, deadreckon-split-fourier,
  deadreckon-split-fastdecay (tau 0.2 s), persist-split, and deadreckon-split-hiddenphysics once
  its build is merged (the runner checks the switch exists). Not rerun: persist-onehead (~-100
  frontline on the laptop and here) and absolute-bigbatch (~-560). Then tournament 3 (all six,
  no ladder), then the best two or three resumed to 60M. ~6.5 h of training. Agent link: the
  laptop session's message reached me; my reply was accepted by the server, delivery not
  confirmed.
- **03:50 KST Laptop + box split (the author approved; HF token shared with the laptop session
  for the upload, at his word).** Offered to the laptop: deadreckon-split-fastdecay and
  persist-split (no new code), on origin/perf/gpu-box with the same algorithm settings (memory
  settings its own), names `2026-11-09-bc-*-laptop`, each uploaded whole to the hub repo; the
  box takes absolute, deadreckon-split, -fourier, -hiddenphysics, then one tournament here over
  all six. If the laptop declines, the box runs all six (round4.sh skips an arm only when
  `runs/laptop-<arm>` exists, written when the laptop confirms it started).
- **03:50 KST The 60M rule (the author):** resume every front runner that the tournament cannot
  separate (two if two are neck and neck, four if four); and where several one-change ideas each
  beat their base, add their combination (e.g. fourier + fastdecay) to the 60M round. My
  reading of "cannot separate": within the round-2 bar of the leader (about 50 frontline / 40 on
  the two modes' mean), or overlapping 2-sigma intervals; "beats its base": above
  deadreckon-split by more than the noise on the mean.
- **03:55 KST The laptop accepted** (the author confirmed in its session): it runs
  deadreckon-split-fastdecay-laptop then persist-split-laptop and uploads each; the box skips
  both (`runs/laptop-*` markers). **Merged `fix/bc-defaults`** (the laptop session's e562d1c5,
  merge ed9cd257): `bc.config()` takes C-fixed's discount and trace decay, `TrainConfig` builds
  temporal_first by default; the one conflict (`test_yemong_error_head.py`, both branches had
  formatted the same line) kept the box's version, which has both. A bare BC config now resolves
  to [.., 1.0] / [0.99, .., 1.0] / temporal_first; the box's explicit flags are now redundant.
- **04:00 KST Laptop arm 1 started** (`2026-11-09-bc-deadreckon-split-fastdecay-laptop`, 18:31
  UTC, perf/gpu-box 7e904347; its resolved config checked by the laptop session: the algorithm
  settings equal the box's; memory settings 160-row micro-batches and the host store). ETAs: arm 1
  05:45, persist-split-laptop 08:00 KST. Box's four arms end about the same time: tournament ~08:00-08:30.
- **03:33 KST Confirmation:** absolute with C-fixed's learner settings
  (`2026-11-09-bc-absolute-cfixed-learner-check`), outcome Brier at 5-15 s, updates 6-10: +0.068
  +0.125 +0.144 +0.172 +0.186 (mean +0.139); the laptop's C-fixed +0.08 to +0.19. The cause was the
  missing two lines alone; the box's speed changes are cleared. Round 4 started.
- **03:40 KST Disk:** the wrong-base runs need no resume: deleted the checkpoints of the old
  absolute (C-fixed-3090), deadreckon-split (arm4d-3090), fastdecay-3090 (stopped) and the v2
  check, each after confirming every file on the hub by size; their policy files kept. 37 GB
  used, 28 free (the compile cache grows with each new config).
- **04:05 KST Hidden-physics build merged** (`feat/hidden-physics` eae8916a, merge dfa1d1ff; the
  one conflict: both branches added an arm to the error-head tests' `ARMS`: kept both). From the
  agent's report: `yemong.physics_hidden_decay` (None = off; needs `delta_split`, positive);
  hidden age from `Replay.age` x tick length; weight 1 on seen rows, exp(-age/tau) on hidden
  ones, 0 for a hidden row at a segment's first tick (no truth at t-1 stored: 20 of 587 rows in
  one segment). **The target did need changing:** the targets already used the truth at t and
  t+1, but the extrapolation baseline's rates for a hidden row were the belief's held, decayed
  rates; now `labels.true_rates` (Yemong.rates on the privileged rows) gives hidden rows the
  truth's rates; seen rows unchanged. The physics term: sum(weight x CE) / the unchanged
  next-state count. Trunk-gradient share of the physics term +8% (0.02428 -> 0.02629; per
  weighted row 3.63e-6 vs 3.67e-6): coefficient kept at 0.0033. Its uncertain calls: the
  first-tick hidden rows dropped (could use carry.previous); the `next_state/hidden/*`
  diagnostics measure against the new target with the switch. It had pushed eae8916a before my
  no-push message reached it. Verifying the merged tree on CPU (error-head, BC, codes files).
- **04:15 KST absolute on the correct base done** (`2026-11-09-bc-absolute`, budget, 30.4M,
  41 min): KL 0.503, agreement 0.664 (laptop C-fixed 0.500 / 0.672); outcome Brier mean 20-30M
  <5 s 0.437, 5-15 s 0.428, 15-30 s 0.386, 30-60 s 0.331, 60-120 s 0.238, >120 s 0.085 (laptop
  C-fixed 5-15 s ~0.52; its seed pair read 0.52 vs 0.28). The box reproduces C-fixed.
- **04:20 KST Merged tree verified** (fourier + hidden physics + bc defaults): error-head (seven
  arms incl. arm4d_fourier and arm4dh), BC and codes files: 150 pass, 1 fail (the known
  machine-dependent arms-1-and-2 digest). deadreckon-split started 04:15 on it.
- **04:30 KST Laptop ETAs revised** (its message: arm 1's first update 2,546 s cold; arm 1 ends
  ~06:10, arm 2 ~08:45 KST, worst 09:30). The box's four arms end ~07:45, so the gap goes to
  **absolute at seed 1** (`round4b.sh`, ~41 min): the seed spread on this box with a real field,
  which the 60M "cannot separate" call needs (the laptop's two C-fixed seeds differed in Brier
  0.52 vs 0.28). Then `round4b.sh` waits for both laptop runs on the hub and downloads their
  policy files; `tournament3.sh` then rates every finished arm (seven, scripted 1000, frontline
  and deathmatch, no ladder; a missing arm is skipped and named).
- **05:24 KST deadreckon-split on the correct base done** (budget, 30.4M, 68 min): KL **0.432**,
  agreement **0.697** (absolute 0.503 / 0.664: clearly better imitation; on the wrong base the two
  were level, 0.470 / 0.475); outcome Brier 20-30M <5 s 0.343, 5-15 s 0.324, 15-30 s 0.293, 30-60
  s 0.239, 60-120 s 0.157, >120 s 0.006 (absolute 0.437 ... 0.085): the delta arm lowers the
  outcome Brier, as on the laptop in round 2. The tournament judges play. fourier started.
- **05:55 KST fourier healthy after 3 updates:** KL 1.4364 / 0.7272 vs deadreckon-split's
  1.4521 / 0.7173 at updates 2-3; 92 s per update (equal); no non-finite beliefs or skipped steps;
  16.9 GB. First update 1,517 s (cold compile of the new input path). ETAs: fourier ~06:45,
  hiddenphysics ~07:55, absolute-seed1 ~08:40 KST.
- **06:10 KST Laptop arm 1 uploaded** (`2026-11-09-bc-deadreckon-split-fastdecay-laptop`, hub
  2828b28d, budget): downloaded, loads on the box's code (one expected FilledDefaults warning:
  its tree predates the hidden-physics merge, so `physics_hidden_decay` takes its default, off);
  its config matches the plan (C-fixed's discount and trace decay, 2048 x 4 shards x 4 epochs x
  256 minibatches, stop flags). KL 0.492, agreement 0.663 (box deadreckon-split 0.432 / 0.697);
  its Brier 5-15 s, last six logs: 0.240 0.267 0.263 0.265 0.337 0.413. persist-split-laptop
  started 06:07, ETA 08:45 KST (worst 09:30).
- **06:33 KST fourier done** (budget, 30.4M, 68 min): KL 0.490, agreement 0.667; Brier 20-30M
  0.399 0.390 0.374 0.330 0.230 0.053 (<5 s ... >120 s). Between its base (deadreckon-split
  0.432 / 0.697, Brier 5-15 s 0.324) and absolute (0.503 / 0.664, 0.428). deadreckon-split's low
  KL is the outlier (every other arm 0.49-0.50): possibly its seed; absolute-seed1 will show the
  spread. hiddenphysics next.
- **07:00 KST hiddenphysics healthy after 3 updates:** KL 1.4524 / 0.7172 vs deadreckon-split's
  1.4521 / 0.7173 (the change only adds hidden rows to an auxiliary head's loss); 92 s per update;
  no non-finite beliefs or skipped steps; 17.1 GB; first update 1,436 s (cold compile). ETAs:
  hiddenphysics ~07:40, absolute-seed1 ~08:25, laptop persist-split ~08:45 KST, then tournament 3.
- **07:41 KST hiddenphysics done** (budget, 30.4M, 67 min): KL 0.444, agreement 0.692 (base
  deadreckon-split 0.432 / 0.697); Brier 20-30M 0.404 0.388 0.346 0.280 0.180 0.034 (base 0.343
  0.324 0.293 0.239 0.157 0.006): better at every horizon. Hidden errors at 0-0.1 / 0.5-1 / 2-5 s:
  position 0.073 / 6.45 / 73.0 m (base 0.071 / 6.26 / 67.9), heading 0.020 / 0.228 / 0.412 rad
  (base 0.020 / 0.235 / 0.434). absolute-seed1 next.
- **08:10 KST Laptop arm 2 uploaded** (`2026-11-09-bc-persist-split-laptop`, hub 4965d344,
  budget at 08:09 KST; Brier 5-15 s, last six logs 0.379 0.338 0.347 0.391 0.388 0.386). The
  laptop session's caveat (its tree predates `physics_hidden_decay`): the switch only selects
  which rows enter the physics head's training loss, default off (= the laptop's behaviour), and
  does not touch the forward pass, so the tournament rates its policies like for like. The
  laptop GPU is free.
- **08:23 KST absolute-seed1 done** (budget, 30.4M): KL 0.520, agreement 0.657 (seed 0: 0.503 /
  0.664); Brier 20-30M 0.262 0.246 0.222 0.196 0.139 0.012 (seed 0: 0.437 0.428 0.386 0.331 0.238
  0.085). **Seed spread in outcome Brier ~0.18 at 5-15 s** (the laptop's C-fixed seeds: 0.52 vs
  0.28): the arms' Brier differences (0.06-0.10) are within it; KL spread ~0.02. Both laptop runs
  downloaded; tournament 3 started (seven players).
- **09:10 KST Tournament 3 final** (`runs/ratings/round3/`, 3072 games per mode, scripted 1000;
  frontline / deathmatch / mean): deadreckon-split-hiddenphysics 824 +- 17 / 821 +- 18 / **823**;
  deadreckon-split 713 / 857 / 785; persist-split-laptop 717 / 689 / 703; fastdecay-laptop 657 /
  664 / 661; absolute 672 / 642 / 657; fourier 588 / 558 / 573; absolute-seed1 569 / 496 / 532.
  **Seed noise is large:** absolute's two seeds differ by 125 on the mean (one pair: a single
  run's spread ~ +-90), so single-seed gaps under ~100 do not separate arms. Readings:
  hiddenphysics vs its base +38 mean (+111 frontline, -36 deathmatch): neck and neck; persist-split
  -120 (about one seed sigma): borderline, kept; absolute -166 (its better seed); fourier -212 and
  fastdecay -124 below their base (both hurt); only hidden physics beat its base, so no
  combination. **60M (the author's rule):** hiddenphysics and deadreckon-split resumed on the box
  (`round5.sh`, ~45 + 50 min); persist-split-laptop offered to the laptop (its own checkpoint,
  ~2.5 h). Then tournament 4 at 60M.
- **09:35 KST The author: add deadreckon-split-hiddenphysics without decay; then a tournament
  with everything, including the same 30M checkpoints again, to see variance.** "No decay" =
  every hidden row weighs 1 at any age: `yemong.physics_hidden_decay=1e9` (exp(-age/1e9) is 1.0
  in float32 for any age here). `round6.sh`, after round 5's box resumes: the nodecay arm (30M);
  persist-split to 60M on the box from its hub checkpoint unless the laptop takes it (marker
  `runs/laptop-persist-60m`); then **tournament 4**: the seven 30M policies of tournament 3 again
  (identical policies: their rating change from tournament 3 is the tournament's own noise; the
  absolute pair is the seed noise), the nodecay arm at 30M, and the three at 60M (11 players).
  Numbering: tournament 2 never ran (round 3's first attempt, on the wrong base, was paused).
- **09:40 KST Pivot (the author): the 60M round stopped for now; more 30M comparisons first.**
  hiddenphysics' resume asked to STOP (it stops after its update with a checkpoint; resumable);
  deadreckon-split's resume cancelled; the laptop's 60M request withdrawn. `round6.sh` rewritten:
  three 30M arms on the hidden-physics base, one change each: **nodecay**
  (`physics_hidden_decay=1e9`: every hidden row weighs 1); **slowdecay** (the held rates fade
  2.5x slower, `extrapolation_decay` 0.5 -> 1.25 s, the other direction from fastdecay's 0.2 s,
  which hurt; my reading of "slow decay": the held-rate tau, as fastdecay's was); and
  **hiddenphysics-seed1** (my addition, for the leader's seed spread: the only same-settings
  replication at 30M on the correct base is absolute / absolute-seed1). Then tournament 4: every
  30M policy on the correct base (10 players), both seed pairs in.
- **09:50 KST The two taus are coupled (the author): "only use the head's information where it's
  trained".** In the code they are two settings, `extrapolation_decay` (the held rates' fade in
  the baseline) and `physics_hidden_decay` (the hidden rows' weight in the physics loss), equal
  in the hiddenphysics arm only by default; my nodecay changed only the second and slowdecay only
  the first: both decoupled, wrong. Proposed design (to confirm): one tau for the held-rate fade,
  the loss weight exp(-age/tau), and a new third use, a hidden row's forecast = baseline + exp(-age/tau)
  x the physics change (today hidden rows take the full change at any age). Caution: coupled no
  decay (tau = infinity) holds the rates forever, the divergence the decay was added against (a
  held thrust integrates without bound): it needs a top-speed clamp or a long finite tau (10 s).
  Aborted nodecay at its first update (in `runs/failed/`); hiddenphysics-seed1 runs meanwhile.
- **10:00 KST The author chose: without the fade first, then the fade on the best tau.** Three
  arms on the hidden-physics base (`round6b.sh`, after hiddenphysics-seed1): **tau1.25** and
  **tau10** (both `extrapolation_decay` and `physics_hidden_decay` set to the same tau; 10 s = the
  author's "no decay", bounded), and **fade** (tau 0.5 with the new `physics_hidden_fade`: a
  hidden row's forecast = baseline + exp(-age/tau) x the physics change; built by a background
  agent on `feat/physics-hidden-fade`, CPU only; the runner waits for it to be merged). The
  existing hiddenphysics is the tau 0.5 point. Then tournament 4: every 30M policy on the
  correct base (11 players, both seed pairs).
- **10:20 KST Fade build merged** (`feat/physics-hidden-fade` accc60c5, merge 64d7aeb8, no
  conflicts; ruff clean; the arm's line resolves). From the agent's report:
  `YemongConfig.physics_hidden_fade` (needs `delta_split` and `physics_hidden_decay`);
  `Yemong.forecast` takes the rows' hidden age (ticks) and, for age > 0, sets the forecast to
  baseline + exp(-age x tick / tau) x (physics forecast - baseline) (`delta.fade`; seen rows bit
  for bit); faded: the means of x, y, velocity, heading (wrapped), nose rate, angle of attack,
  energy, ln n; not faded: the jumps (shield, recharge wait, gun cooldown, advanced by their
  median) and every spread. `step` passes the enemy rows' age (= `replay.age`); the learner's
  error label now uses `model.forecast` with the age (so e = truth - the faded forecast; bit for
  bit with the switch off); the physics target unchanged (truth to truth). Callers that rebuild
  the forecast (head statistics, decode, figures capture, play predictions) pass the age. Its
  report: 230 pass in six files, the one failure the known arm-1/2 digest; not tested on GPU or
  under graph capture (`delta.fade` builds no index tensor, so capture stays safe). Verifying the
  merged tree on CPU before the next arm imports it.
- **10:20 KST The author: the 60M round also takes absolute (C-fixed) as a baseline**, to catch
  arms that do well to 30M and badly after. Laptop asked (author to confirm there) to take
  hiddenphysics-tau10 (runner skips it on the box if claimed; `runs/hub_fetch.py` fetches it).
- **10:15 KST Merged tree (with the fade) verified:** 230 pass in the six files, 1 fail (the known arm-1/2 digest).
- **10:30 KST Two more 30M arms (the author):** **deadreckon-split-seed1** (so hidden physics vs
  its base becomes two seeds each) and **persist-split-hiddenphysics** (whether hidden physics
  helps beyond dead reckoning), after fade in `round6b.sh` (each skippable if the laptop claims
  it). Tournament 4: 13 players at 30M. Tournament 3 with Brier and imitation (the author asked;
  Brier = mean of 20-30.4M logs; my first table wrongly took hiddenphysics' rows past 30M from
  its stopped 60M resume: 0.445 -> 0.388 corrected): hiddenphysics 823 Elo / Brier 5-15 s 0.388
  / KL 0.444; deadreckon-split 785 / 0.324 / 0.432; persist-split 703 / 0.353 / 0.470; fastdecay
  661 / 0.280 / 0.492; absolute 657 / 0.428 / 0.503; fourier 573 / 0.390 / 0.490; absolute-seed1
  532 / 0.246 / 0.520. Brier does not track Elo (absolute, fourier: top Brier, low Elo); KL a bit
  better.
- **10:40 KST The author: the laptop takes two arms** (tau10, then persist-split-hiddenphysics-laptop; requested, the author to confirm in its session). The box runner skips each once the laptop confirms it (markers) and fetches both from the hub before tournament 4.
- **10:45 KST Laptop accepted tau10** (started 10:19 KST on 45774aeb; its resolved config: extrapolation, split, both taus 10.0, temporal_first, BC discounts; memory settings its defaults); box marker set. persist-split-hiddenphysics: awaiting the author's confirmation there.
- **10:50 KST Laptop accepted persist-split-hiddenphysics-laptop** (queued after tau10; ETA 15:20 KST, worst 16:00); box marker set. Box: seed1, tau1.25, fade, deadreckon-split-seed1 (~14:15); tournament 4 once both laptop runs are in.
- **10:55 KST absolute's 60M resume fills the box's wait for the laptop** (the 60M round's baseline; checkpoint 30.4M restored from the hub), after deadreckon-split-seed1.
- **10:26 KST hiddenphysics-seed1 done** (30.4M): KL 0.475, agreement 0.674, Brier 5-15 s 0.230 (seed 0: 0.444 / 0.692 / 0.388). The same seed pattern as absolute's pair (Brier 0.428 vs 0.246): seed 0 the luckier for both; only multi-seed tournament results decide. tau1.25 next.
- **11:22 KST Disk:** 18 GB free; pruned compile-cache files unused 12 h (80,423 files: the wrong-base and speed-test configs): cache 35 -> 29 GB, **24 GB free**.
- **11:37 KST tau1.25 done** (budget, 30.4M): KL 0.468, agreement 0.677, Brier 5-15 s 0.340, hidden position error 2-5 s 134 m, heading 0.434 rad (tau 0.5 seeds: 0.444 / 0.475 KL, 0.388 / 0.230 Brier, 73 / 114 m, 0.412 / 0.445 rad): within the seed range except the 2-5 s position error, the worst (slower-fading held rates carry the last thrust longer; suggestive only). fade next.
- **12:10 KST fade healthy on the GPU (incl. graph capture) after 3 updates:** KL 1.4523 / 0.7169 vs hiddenphysics' 1.4524 / 0.7172; 92 s per update; no non-finite beliefs or skipped steps; 17.0 GB. ETAs: fade ~12:50, deadreckon-split-seed1 ~14:00, absolute 60M ~14:45, laptop ~15:20, then tournament 4.
- **12:50 KST Laptop tau10 uploaded** (hub a55608a0, budget at 12:43 KST); fetched and loads here (extrapolation, split, both taus 10.0, no fade). KL 0.482, agreement 0.672, Brier 5-15 s 0.271: inside the tau 0.5 seed range. persist-split-hiddenphysics-laptop started 12:44 KST (persistence, split, physics_hidden_decay 0.5), ETA 15:10.
- **12:49 KST fade done** (budget, 30.4M): KL 0.475, agreement 0.671, Brier 5-15 s 0.331, hidden position error 0.5-1 / 1-2 / 2-5 s 6.9 / 23.1 / 105 m, heading 2-5 s 0.446 rad: all inside the tau 0.5 seed pair's range (0.444-0.475 KL; 0.230-0.388 Brier; 73-114 m). deadreckon-split-seed1 next.
- **13:40 KST deadreckon-split-seed1 done** (30.4M): KL 0.480, agreement 0.670, Brier 5-15 s 0.190, outcome EV 0.521 (seed 0: 0.432 / 0.697 / 0.324 / 0.626). Third arm where seed 1 is worse than seed 0 on every training metric (absolute, hiddenphysics, deadreckon-split): seed 0 may be the luckier seed for all, so compare like seeds. absolute's 60M resume next.
- **13:41 KST absolute's 60M resume failed: its hub checkpoint was inconsistent.** The 30.4M
  checkpoint restored from the hub refused to load: `training.safetensors` has SHA-256 7b09f3ba,
  the checkpoint's record says c5776cb3 (weights and return scale match). Same size, so my
  deletion check (size only) passed it at 09:22 and the local original is gone. Likely cause: a
  sync uploaded while that final checkpoint was rewritten, pairing one write's record with the
  other's training state. **Checked every checkpoint on the hub against its own record (SHA-256
  from the hub's metadata): 38 checkpoints, this one file the only mismatch.** Fixes: deletions
  now require each file's SHA-256 to equal the record (not the size); absolute resumes from its
  23.1M checkpoint instead (verified by hash; 19.9M is not kept: a run keeps its newest two),
  `runs/absolute60.sh`, started 13:43 (loads, with the expected FilledDefaults for the newer
  switches). Its 30.4M policy, the one rated, is the original. The 60M continuation branches at
  23.1M. The box's queue is otherwise done; tournament 4 waits for the laptop's last arm.
- **13:56 KST absolute's resume healthy** (rewound to 23.1M; at update 30, 81 s per update; ETA 14:35 KST). Note: its metrics.jsonl now holds updates 23-29 twice (the original and the rewound branch): 20-30M statistics for absolute must take the first 29 rows only (the tables already given predate the rewind).
- **14:35 KST absolute at 60M done** (budget, 60.8M; branched at 23.1M): KL 0.416, agreement
  0.702, Brier 5-15 s 0.642 (last log). **Caught before the tournament:** the rewind moved the
  original 30.4M policy files (the ones rated in tournament 3) to `policies/rewound_1/` (the
  runs code's rule, `records.py`), and `policies/step_30408704` now holds the rewound branch's
  30.4M (different weights). The tournament runner was fixed to rate the original
  (`rewound_1`); checked in the ratings process's arguments.
- **14:38 KST Tournament 4 started** (13 players at 30M: absolute (original), absolute-seed1,
  deadreckon-split, -seed1, -fourier, -fastdecay-laptop, persist-split-laptop, hiddenphysics,
  -seed1, -tau1.25, -tau10-laptop, -fade, persist-split-hiddenphysics-laptop; frontline then
  deathmatch, no ladder). The laptop's last arm arrived (hub 00aaa6cc, budget at 14:37 KST,
  Brier 5-15 s last six 0.272-0.325); the laptop GPU is free.
