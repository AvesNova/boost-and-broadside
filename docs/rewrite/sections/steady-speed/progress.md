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
- gpubox4 shares its host with gpubox7 (from 23:50, another agent's frontline runs, at most
  4 compile threads). A sibling container may load the CPU during these steady runs. The
  AFTER RL plain updates 5-11 stayed within 0.3 s (89.7-90.4 s), so no jitter shows there.
- **Two learn phases do not fit one 16 GB GPU at 1024 environments.** BC at 2560 rows ran
  out of memory at 10.1 GB beside RL's 5 GB, BC at 640 rows at 7.5 GB beside RL's 8 GB, and
  BEFORE's build at 4.3 GB beside AFTER RL's 11.1 GB. From 23:28 on, one process at a time.

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
- 22:45 AFTER BC (2560 rows) out of memory in the learner's first `drift` call, an eager
  unroll of `micro_views` = 256 views against 24-view micro-batches. **Bug, fixed (d8a0ee55):**
  the drift unrolls the update's first micro-batch (`Learner.first_micro_batch`); quick test
  `test_the_drift_never_unrolls_more_views_than_a_micro_batch`. BEFORE has it too.
- 23:03 BC at 640 rows (the same 24-view micro-batches, drift on 64 views), equal lane
  weights: out of memory again in its first learn, beside RL.
- 23:22 **AFTER RL cold** (old lanes, tree cbf0584e, `timed=1`): 3,216 s to the end of the
  first update. Contended with the BC runs until 23:05. Compiles: `_act` 1,091 s (31,729
  ops), `_advance` 34, `_resolve` 68, `_perceive` 18, shot `tick` 14, `Learner._codes` 786,
  `_loss` 455, `_terms` (the cadence) 438. One micro-batch shape (32 views), as printed.
  First update's eager heads' sums: 67.8 s.
- 23:52 **AFTER RL warm, steady** (`timed=0`, 12 updates): warm start 449 s to the end of
  update 1.

  | Updates | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 (cadence) | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | seconds | 94.2 | 92.5 | 90.8 | 90.4 | 90.0 | 90.0 | 89.7 | 90.0 | **163.8** | 89.9 | 94.8 |

  - Plain updates: rollout 16.25 s (flat), learn 75.0 s mean (77.9 down to 73.5: the KL
    early stop trims epochs). That is **5,747 SPS**, and 524,288 steps an update.
  - Weighted one cadence update in ten: **5,323 SPS**.
  - The cadence update: +73.8 s, of which the eager heads' sums are **67.4 s (192 calls,
    0.35 s each)** and the cadence's own gradient decomposition 4.7 s.
  - Peak memory: 10.19 GB allocated, 10.69 GB reserved.
- **Finding: the eager heads' sums cost about 8% of RL's steady time** (+7.4 s an update
  averaged over the cadence). The coordinator chose a sample. **Fixed (b234a644):** on a
  cadence update the heads' sums read the first epoch's first micro-batches until they hold
  `HEAD_SUM_VIEWS` = 256 views: 8 calls at 32 views, so about 2.8 s. The series are ratios
  of totals, so a sample keeps their meaning. Quick test
  `test_the_heads_sums_read_a_sample_of_head_sum_views_views`; documented in
  `head_statistics` and learner.md. Expected cadence update about 91 + 4.7 + 2.8 = 98.5 s,
  so **about 5,700 SPS weighted (-0.8% against plain)**. Not yet confirmed on the GPU
  (needs the final tree's rollout compile; command below).
- 23:52 BEFORE RL on a copy of AFTER RL's caches (`timed=0`, 12 updates), alone.
- 00:29 **BEFORE RL done.**
  - Update 1: 1,210 s. Of that, **822 s is the compile of `head_sums`** (the 40k-op graph;
    the copied caches covered everything else: rollout 143 s, against AFTER's warm 138 s).
  - Plain updates 2-12: 89.9-91.2 s, rollout 16.25 s, learn 74.5 s: **5,776 SPS**.
  - Cadence update 10: 104.7 s, with the compiled heads' sums **7.78 s (192 calls, 0.04 s
    each)** and the cadence 4.7 s. **5,688 SPS** weighted.
  - Peak memory: the same as AFTER's (10.19 GB allocated, 10.69 GB reserved).
  - The `.gpu` file has no exit line: `run.sh` was copied over while it ran.

## Results (gpubox4, RTX 5060 Ti 16 GB, sm_120; RL, 1024 environments, 320 rows, old lanes)

| Tree | Cold to the first update | Warm start | Plain SPS | SPS, one cadence in ten | Rollout / learn (plain) | Cadence update | Peak reserved |
|---|---|---|---|---|---|---|---|
| BEFORE 9f8c20fc | not measured cold; +822 s for `head_sums` over AFTER's units | 1,210 s (with the `head_sums` compile) | 5,776 | 5,688 | 16.25 / 74.5 s | 104.7 s (heads 7.8) | 10.69 GB |
| AFTER cbf0584e | 3,216 s (contended until 23:05) | 449 s | 5,747 | **5,323 (-6.4%)** | 16.25 / 75.0 s | 163.8 s (heads 67.4) | 10.69 GB |
| AFTER + heads' sample (b234a644) | the same as AFTER | the same | about 5,750 | **about 5,700 (projected)** | the same | about 99 s (heads about 2.8) | the same |

- The plain updates are the same within 0.5% (BEFORE 90.9 s, AFTER 91.2 s mean of updates
  2-12). Only the eager heads' sums separate the trees. The other fixes cannot show at
  1024 environments: the micro-batches are one size in both trees, and RUN_SLACK leaves
  RL's old lanes at two runs in both.
- Per call, the compiled heads' sums take 0.041 s and the eager ones 0.351 s. The 256-view
  sample (8 eager calls, about 2.8 s) costs less than BEFORE's compiled whole epoch (7.8 s)
  and has no compile.
- BC is not measured: two learn phases do not fit one 16 GB GPU at 1024 environments, and
  the BC runs lost the race for memory twice (log). Command in `queue.sh` (`bc`).

## Rating precision of RL's lanes (analytic, calibrated on the champion BC run)

Calibration is the champion BC run (onehead-bc, 2048 environments, 1024 lanes over 2
modes). Its exponential average against the scripted agent had 205 lanes per mode and
played 19.8 (frontline) and 18.4 (deathmatch) games an update. Over the 10-update window
the run's own standard error was a median 44 Elo: k = SE x sqrt(games in the window), about
620. The ideal Bradley-Terry value at even odds is k = 347. Elimination's rate (0.15 games
per lane per update) is an estimate from its 180 s cap. RL's new lanes give each of 7
players a 21st of the lanes per mode. One window is 10 updates, about 10.5M steps at 2048
environments.

| Lane environments | Lanes per (mode, player) | Games per window, frontline | SE per mode, frontline (ideal) | SE over 3 modes pooled |
|---|---|---|---|---|
| 1024 | 48.8 | 47 | 90 (50) | about 52 (29) |
| 512 | 24.4 | 24 | 127 (71) | about 73 (41) |
| 256 | 12.2 | 12 | 180 (101) | about 104 (58) |
| 0 | none | none | no rating | no rating |

**The target of +-25 Elo per 10M steps looks wrong for this layout.** Per mode and per
player it needs about 615 frontline games a window: about 13,300 lane environments (4,200
at ideal odds). Pooled over the three modes, 1024 lanes reach about 29-52. To rate the
product, the exponential average, to +-25 would take either a longer window for the
averages or more of the lanes for the averages. The second breaks the equal rows that make
the one stacked run.

## Projection for the 5090 (not measured: assumed factors)

Assumptions (none measured on a 5090):
- RL about 22,000 SPS at 2048 environments and 640 rows. That is about 4x the 5060 Ti: the
  memory bandwidth is 4.0x and the SMs 4.7x.
- BC about 24,000 SPS: the champion's 3090 ran 11,350 SPS, and a 5090 is about 2.1x a 3090.
- Cold compile per stage about 55 min for AFTER, single-threaded tracing as here. BEFORE
  as shipped (2040 environments) adds 14 min for `head_sums` and about 20 min for the second
  micro-batch shape (`_codes`, `_loss` again).

| Job | BEFORE as shipped | AFTER (eager heads' sums, every micro-batch) | AFTER + sample (final) |
|---|---|---|---|
| 100M BC | 69 min steady + about 1.5 h compile = about 2.6 h | about 2.0 h (+6% steady) | **about 1.9 h** |
| 500M BC -> RL (100M + 400M) | 6.2 h steady + about 3.0 h compile = about 9.2 h | about 7.7 h (+24 min) | **about 7.2 h** |
| 8-arm suite (150M chains, 8 GPUs at once) | about 3.4 h wall | about 3.0 h | **about 2.8 h wall** |

The steady differences come from the heads' sums alone (BEFORE's compiled whole epoch 1.5%,
AFTER's eager whole epoch 7-8%, the sample about 0.6% of steady time). The compile
differences come from BEFORE's `head_sums` graph and from its second micro-batch shape at
2040 environments; the 2048 environments remove the second for both trees. Not in these
numbers:
- the final tree's rollout: the new lanes (7 network players in one run, no network-vs-
  network rows) and the pool's always-on scripted pass;
- the drift fix's saving in BC.
Both are in `queue.sh`.

## The author's decisions applied on this branch (2026-11-12, evening)

| Commit | Change |
|---|---|
| cbf0584e | 2048 environments; the all-modes mix 0.5 / 0.3125 / 0.1875 |
| d8a0ee55 | drift on the first micro-batch (bug) |
| f53fad04, 2a80ba32 | BC lanes equal weight, then **no BC lanes in the launch**; RL lanes: every network player (live, both averages, each slot) against the full-strength scripted agent, equal weight, one stacked run |
| b38315fa, ef75b5a9, f591f31d | (feat/pool-handoff, merged) the RL pool's only scripted entry is the full-strength agent (no rungs, no random); the BC -> RL hand-off is always the exponential average |
| 04e0e2bd | a warm start from a parent with no lane ratings starts at the scripted agent's 1000 |
| b234a644 | the heads' sums read a 256-view sample |

**Which setup each number used:** every gpubox4 number above is the OLD lane setup: RL lanes
with league slots, the ladder chain, live 0.5 against the averages; the pool without the
scripted entry. The AFTER RL rows are tree cbf0584e (eager heads' sums over every
micro-batch, the drift on `micro_views`); BEFORE is 9f8c20fc, given the same config JSON.
None of the final tree's rollout (new lanes, the pool's always-on scripted pass) has been
compiled or measured yet.

## 2026-11-13 dry-run readiness on gpubox7 (stopped early: the host stopped answering)

Tree: `calib/overnight` 0f3ec4b4 (its code is `perf/steady-speed` 7f7fbeb1 plus the
calibration kit), cloned to `/workspace/drycheck/repo` on gpubox7 (RTX 5060 Ti 16 GB, shares
its host's 31 GB RAM and CPU with gpubox4). Configs: `configs.sh`'s launch lines at 2048
environments, BC `lanes.environments=0`, BC 256 / RL 32 minibatches (the defaults), RL's
1024 lane environments. 8 GB emulation: `memory_fraction=0.4908` (8/16.3), which caps the
allocator at 7.60 GiB.

- **Fixed:** `steady.py`'s `memory_fraction` passed `torch.device("cuda")`, which
  `set_per_process_memory_fraction` refuses (it needs an index): ac77ba2c.
- **BC at 2048 environments does not fit 8 GB with `host_store=false`.** The build runs out
  of memory while allocating the segment store (7.48 GiB allocated, store not complete),
  before any compile. The laptop needs `host_store=true`.
- **With `host_store=true` the store pins about 11 GB of host RAM** (BC alone: the process
  held 13.4 GB resident, 11.1 GB of it shared/pinned; the host then had 0 GB free and 5 GB
  available). Pinned memory cannot swap.
- Micro-batch note, BC at 1280 rows: "learner micro-batches of 48 views (12288 views in
  256 minibatches); the cadence adds 24". So 1280 and 640 rows are the same run for BC (the
  minibatch, 48 views, is one micro-batch); two shapes, as expected.
- The seeded compile caches did not hit: the calibration's `cache-box4` holds BC at 1024
  environments and 240 rows (24 views, cadence 12), not 48. BC compiled cold.
- About 12 minutes into that cold compile (GPU at 6.2 GB of the 7.6 GiB cap, no update
  yet), **gpubox7 and gpubox4 both stopped answering ssh** (banner timeouts from 05:40 KST).
  The likely cause is host RAM: the 11 GB pinned store plus four compile workers on a host
  that also runs the calibration's BC. The run was not finished; the coordinator is
  recovering the boxes. No BC or RL update completed under the cap, so **the 8 GB verdict,
  the laptop profile, BC's half-minibatch compile count (check 2), the final RL rollout's
  timings (check 3) and the lanes / memory pairs (check 4) are not measured.**

What is known for the laptop without a measurement:
- `host_store=true` is required at 2048 environments, and it needs about 11 GB of free host
  RAM for BC (RL's store is likely larger). On an 8 GB laptop with 16 GB of RAM this may
  not fit; fewer environments may be the laptop's real limit.
- BC micro rows: by `micro_batch_sizes`, any rows from 240 to 479 give 24 views; 24 divides
  the 48-view minibatch and equals the cadence's halves, so BC compiles **one** shape there
  (computed, not run). Rows of 480 and more give 48 + 24. Speed cost not measured.
- RL micro rows: 640 and 320 give one shape (64 or 32 views; the 192-view halves divide);
  1280 gives 128 + 96. RL at 320 rows reserved 10.69 GB at 1024 environments with the store
  on the device, so at 8 GB only 320 or less with `host_store=true` is plausible.
- Never run this emulation again beside another job on a shared host: the pinned store is
  host RAM the other job cannot get back.
