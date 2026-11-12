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
