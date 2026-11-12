# Compile split: progress

Branch `perf/compile-split` (from `origin/feat/launch` 4b09b70b), box gpubox6 (vast.ai, RTX
5060 8 GB, sm_120; Ryzen 9 3900X, 11.5-core quota; torch 2.15.0+cu132, Triton 3.9.0).
Question: does cutting the compile into smaller units cut the cold compile enough to pay
for a small steady-speed cost, across a suite of ~100M-step arms on 5090s?

## What is compiled today (read 18:05-18:20)

- **Rollout:** four stages compiled separately already (`Rollout._act`, `_advance`,
  `_resolve`, `_perceive`; `torch.compile(fullgraph=True, dynamic=False)`), then a block of
  ticks captured in one CUDA graph. `_act` holds every controller's step: the learner's
  Yemong step, the lanes' and league slots' stacked (vmap) Yemong forwards, the scripted
  teacher. So "game tick | model forward" is already two units (physics stages | act).
- **Learner:** `Learner._loss` (the model's unroll, every loss term, the diagnostics) is one
  compiled graph (forward and AOT backward); beside it `_head_sums`, `_terms` again for the
  cadence diagnostics, `shots.tick` (shot re-simulation), `_codes`. Compiled with
  `comprehensive_padding=False`, default mode: **no max-autotune** (no GEMM autotuning).
- **Autotune:** only the custom Triton kernels (`yemong/kernels`: scan, temporal mix,
  gated MLP, attention, heads) autotune over 3-4 configurations per shape key.
- Trunk: 2 blocks (`YemongConfig.blocks`), so a regional compile can save at most one
  block's copy of the trunk; most of a forward's ops are the input and belief codes and the
  heads (compile-overhaul notes).

## Variants (config switches, default = today)

- A: today.
- B `compile_regions=true`: each trunk block of the learner's model one
  `torch.compiler.nested_compile_region` (act stage and loss); the opponents' and lanes'
  vmapped copies stay inlined.
- C `compile_split=true`: the learner's loss compiled as two units, `unroll` and the loss
  over its outputs; the cadence terms reuse the same unroll unit.
- BC: both.

Same math: eager the switches change nothing bitwise (quick test); compiled, the slow test
`test_the_switches_compile_to_the_same_arithmetic` and the first update's series in the
records.

## Method

- Config: `launch.toml` (preset `real_scale_all_modes`, the shared --set lines), resolved
  by `configs.sh`. **Lineup stand-in** `vizia vizia Spitfire-IX x3`: feat/launch has no
  Yongma/Jujak (they are on feat/bullet-damage); same 5 slots, same compiled shapes.
- **200 environments** (2040 does not fit 8 GB; 1020 held 5 GB during its first compile,
  so no two could run at once), micro-batch 160 rows (the default), identical everywhere.
- Cold: empty inductor + Triton caches per variant x stage, `steps=1`, the four variants
  of a stage **side by side** (4 processes, 3 compile threads each: one 11.5-core box;
  Dynamo, AOT and lowering are single-threaded). Like for like across variants; the
  absolute times carry the contention.
- Warm: same caches, new process, alone, 10 compile threads, `steps=4` (the first update
  warm, three steady): warm start, steady SPS, peak memory.
- Records: `records/records.jsonl`, one line per variant x stage x mode.

## Log

- 18:12 A bc cold at 1020 environments started alone; aborted 18:17 (5 GB on the device
  during the first compile: no room for side-by-side runs). Its build phase alone took
  207 s (single-core).
- 18:18 round 1: A, B, C, BC bc cold side by side at 200 environments.
- 18:28 round 1's rollout compiles (4-way): `_act` A 617.5 s at 27,291 ops; C 629.9 s
  (same graph); B and BC 628-631 s at 27,047 ops. **B removes 244 of 27k act ops and is not
  faster.** `_advance` 26-28 s (3,485 ops), `_resolve` 63-75 s (3,254), `_perceive` 14-16 s
  (854), the shot tick 11 s (763); the first collect 742-759 s in all.
- 18:30 C and BC ran out of memory in the first learn (four processes on 7.5 GB usable);
  relaunched C; BC's relaunch ran out at build (three others held 6.3 GB). From here at
  most three GPU processes (`plan.sh` chains the rest).
- **The build phase does not grow with environments.** The aborted 1020-environment run's
  207 s build was the cold measurement of the ship card's characteristics (two files written
  09:14 and 09:15 UTC into `~/.cache/boost-and-broadside/characteristics`, the `bnb.cache`
  disk cache, which the compile caches' reset does not touch); every later run found them
  there and built in 34 s. So a fresh box pays about 170 s once per card (and the rule set it
  reads), then never again; `cache.sh` could carry that directory with the compile caches.
  The cold numbers below exclude it.
- 18:35 next (the coordinator's list): the act-stage ablations, trace-only op counts first
  (`ablate.sh eager`: lanes off; PPO one league slot; frontline only), then a real compile of
  the largest cut to check time against ops.
