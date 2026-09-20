# 50v50 realtime deadline experiment

Date: 2026-09-20 KST  
Branch: `perf/realtime-experiment-handoff`  
Intake revision: `b12bb5c8bd59a981dee613cf0dd523f5f6bacd5e`

## Decision

The final evidence changes the candidate disposition from the initial intake
report below:

1. **Keep the opt-in 30 Hz simulation/decision candidate**: CUDA-graph tick,
   compiled pure perception (including projectiles), and compiled belief compose/advance.
   The three-pair no-render A/B completed 299/300 candidate frames within
   33.3 ms (p50/p95/p99/max **24.337/29.758/30.546/36.089 ms**); paired action
   traces were exact. This supports a near-30 Hz sim+decisions path, not a
   zero-miss guarantee.
2. **Keep the packed renderer boundary, but do not claim sustained rendered
   30 FPS.** In the same two-policy A/B with 900×900 team fog and projectile
   perception, candidate p50 was 31.799 ms but p95 was 58.654 ms and
   **127/300** frames missed 33.3 ms. A renderer-only smoke did validate
   snapshot extraction, actual NVIDIA OpenGL rendering, and display flip, but
   it was only 20 samples and excluded policy/environment work.
3. **Reject/defer the remaining candidates.** The two-policy vmap screen is
   policy-call-only and inconsistent across pairs; field transport failed
   strict parity; no collision/precision change has a correctness-backed
   end-to-end win. Whole-tick compilation and earlier LOS alternatives remain
   rejected as described in the history below.

The original intake findings are retained below as experiment history; the
final candidate evidence and its narrower claims are recorded after that table.

Initial intake decision:

1. **Keep the lean interactive environment step** (`7082876`). Play/watch now
   bypass reward, episode-statistic, and perception-diagnostic bookkeeping while
   retaining the authoritative physics, terminal/reset behavior, two team
   observations, and one-tick policy action buffer. Across four alternating
   pairs it reduced complete-frame p50 from **164.745 ms to 153.687 ms**.
2. **Retain but do not activate the ModernGL renderer prototype** (`815ac6d`).
   Snapshot extraction plus NVIDIA rendering and flip measured **7.757 ms p50,
   10.794 ms p99** in the renderer-only test, versus **90.518/116.373 ms** for
   the old renderer without flip. In a complete two-policy frame it reduced p50
   from **165.968 ms to 83.478 ms**. It still lacks health/shield UI, trails,
   capture-progress styling, and robust large-disk drawing, so it is not a
   drop-in gameplay presentation yet.
3. **Defer compiled pure perception.** It passed 40-step parity and consistently
   reduced the environment phase in the legacy-renderer A/B, but its complete
   GPU-rendered combination was not repeatable: pooled p50 was only
   77.391→75.704 ms, the candidate had a 535.193 ms maximum, and it missed all
   150 decision deadlines. Cold perception compilation cost 4.53–12.83 seconds.

The final no-render candidate is a demonstrated near-30 Hz sim+decisions path
under this bounded test (1/300 misses); it is not an unconditional deadline
guarantee. The rendered complete path does not sustain 30 FPS. Renderer-only
measurements are useful boundary timings, not evidence that the full loop
meets either 30 or 60 FPS.

## Measurement contract

The machine remained on its existing `quiet` platform profile and `powersave`
CPU governor/energy policy. No power, sleep, governor, GPU-limit, or keep-awake
setting was changed.

- Hardware: RTX 4070 Laptop GPU (8,188 MiB), Intel i7-13620H; NVIDIA driver
  595.84. PRIME offload selected the RTX for measured OpenGL runs.
- Software: Python 3.13.11, PyTorch 2.13.0+cu130, Pygame 2.6.1, ModernGL 5.12.0
  and glcontext 3.0.0 in the isolated `/tmp/bnb-deadline-moderngl` install.
- Game: one 50v50 Frontline environment, 100 ships, 72 fields, 10 projectile
  slots per ship, 900×900, team-0 fog, zone occlusion, projectile perception,
  and one decision per physics tick.
- Policies: two distinct randomly initialized policies, labeled as such. The
  available trained checkpoint uses `recursive_belief_v8`, not the runtime
  `frontline_shields_v9` observation schema, and was not loaded.
- Final complete-frame results use three or four alternating-order pairs,
  `default` policy compilation, 10 warmup frames, CUDA completion timing, and
  50 measured frames per arm. Startup/compilation is reported separately.
- Tail numbers describe these bounded runs, not long-run tail guarantees.
  Phase values are diagnostic windows from the same arm but are not added to
  reconstruct frame medians.

## Final candidate comparison

| Candidate | Exact scope | Correctness/parity | Complete latency, p50 / p95 / p99 / max | Rendering | Environment throughput | End-to-end training | Startup / memory | Limitations | Decision |
|---|---|---|---|---|---|---|---|---|---|
| Final no-render candidate | B=1 50v50 Frontline; two distinct random policies compiled `default`; compiled perception and belief; fixed-storage CUDA-graph authoritative tick; projectile perception and both team views | Exact paired action traces in all three A/B pairs; exact 200-tick variable-action graph parity including reset and CUDA RNG; perception parity at 2e-6; belief parity at atol 2e-4/rtol 2e-6 with exact discrete state and retained ownership | **24.337 / 29.758 / 30.546 / 36.089 ms**, n=300, 1 miss; pair medians 24.663/24.601/24.048 ms. Reference 60.560/64.819/65.996/73.361 ms, 200 misses, with one transient 20.827 ms pair | Excluded | Not a training path; environment/observation diagnostic mean 6.701 ms | Not measured; benefit unproven | Cold combined first frame observed up to 28.379 s; isolated graph capture 20.95 ms; peak allocated 290.69 MiB in both final arms | Random policies; bounded tail sample; opt-in fixed-shape interactive execution | **Keep** |
| Final packed-rendered candidate | Same candidate plus one-transfer packed snapshot, ModernGL team fog, 900x900 framebuffer and display flip | Same exact paired actions; 14 packed snapshot/render contract tests; actual RTX OpenGL execution | **31.799 / 58.654 / 75.747 / 110.366 ms**, n=300, 127 misses. Reference 71.153/130.060/173.140/183.907 ms, 300 misses | Renderer-only smoke: snapshot 1.196 ms p50; snapshot+render+flip 4.186 ms p50, n=20 | Not a training path | Not measured | Renderer startup 0.620 s in smoke; final peak CUDA allocation 290.69 MiB | Presentation parity incomplete; GL/flip tails prevent sustained 30 FPS | **Keep boundary; decouple/interpolate presentation** |
| Two-policy vmap | Two distinct weights and team views, policy calls only | CPU output/sampling checks passed | Not a complete-frame measurement | Excluded | Excluded | Excluded | Separate first compile 62.889 s; vmap 35.085 s | Pair medians were inconsistent: separate 11.569/11.631 ms, vmap 12.382/11.116 ms | **Reject unchanged** |

Final evidence: [no-render summary](performance-experiments/final-realtime-no-render-summary-50v50.json),
[compressed raw no-render samples/actions](performance-experiments/final-realtime-no-render-ab-50v50.json.gz),
[packed-rendered summary](performance-experiments/final-realtime-packed-rendered-summary-50v50.json),
[compressed raw rendered samples/actions](performance-experiments/final-realtime-packed-rendered-ab-50v50.json.gz),
[CUDA-graph parity](performance-experiments/cuda-graph-tick-parity-50v50.json),
[compiled-belief parity](performance-experiments/compiled-belief-parity-50v50.json), and
[packed-renderer smoke](performance-experiments/packed-renderer-screen-50v50.json).

## Candidate comparison (intake history)

| Candidate | Exact scope | Correctness/parity | Complete rendered frame, p50 / p95 / p99 / max | Rendering result | Environment throughput | End-to-end training | Startup / memory | Limitations | Decision |
|---|---|---|---|---|---|---|---|---|---|
| Current reference | CUDA, two random policies, standard wrapper, old 900² offscreen renderer, no flip | Reference remained runnable; final suite passed | Intake check: 167.291 / not recorded / 206.902 / 209.982 ms, n=30, 30 misses. Four-pair reference: 164.745 / 185.801 / 210.118 / 277.748 ms, n=200, 200 misses | 76.13–81.83 ms diagnostic range in the four-pair run | Not re-run; archived environment-only data below | Not measured | First cold compiled-policy frame+setup 45.019 s; later arms 0.256–0.332 s. Peak CUDA allocation accumulated from 262.19 to 313.16 MiB in-process | Random policies; no display flip | Baseline |
| Shared-observation assembly | Reuse common observation channels for both team views | 40-step CUDA parity passed at atol/rtol 2e-6 | One-shot 165.997 ms versus 167.291 ms; not a promotion signal | No demonstrated render change | Isolated observation p50 17.627 ms versus 17.314 ms; slower in two of three pairs | Not measured | Same 238.63 MiB allocated / 316 MiB reserved in isolated arms | No repeatable complete or isolated gain | **Reject**; production edit reverted; reproducible copy remains in harness |
| Lean interactive step | Standard authoritative tick, reset and observations; omit reward/training metrics and diagnostic accumulation only in play/watch | 40-step 50v50 CUDA state/event/two-view/ownership parity; terminal/reset CPU coverage | **153.687 / 178.708 / 192.459 / 200.145 ms**, n=200, 200 misses, versus 164.745 / 185.801 / 210.118 / 277.748 | Unchanged old offscreen renderer; no flip | Not a training path. Environment phase improved in every pair: reference 52.16–56.52 ms, lean 40.03–47.62 ms | Not applicable; reward/stat work is intentionally omitted | No new persistent tensors. In-process peak allocation is order-dependent, 276.75–305.88 MiB for lean | Cannot be used for PPO or reward-bearing evaluation | **Keep**, committed in `7082876` |
| Whole mutable tick compilation | `torch.compile(TensorEnv.tick)` at B=1, 50v50 | **Failed first tick**: lost/divergent power, cooldown, shooting, bullet cursor/ring and projectile tensors | No valid timing | N/A | N/A | N/A | First parity pair took 52.453 s | Mutable writes are not replayed faithfully; any apparent speed is invalid | **Reject** |
| Pure compiled perception | Lean step, old offscreen renderer, production buffered eager versus compiled pure/unbuffered two-view perception | 40 fixed-action CUDA steps passed, including projectile visibility and retained ownership at 2e-6; action traces matched in all A/B pairs | **158.415 / 178.253 / 186.669 / 189.526 ms**, n=150, 150 misses, versus 167.295 / 200.426 / 217.782 / 255.053 | Old renderer, no flip; 90.45–106.90 ms reference and 94.14–101.02 ms candidate diagnostic ranges | Environment phase improved in all pairs: 39.76–51.66 ms to 33.42–34.03 ms | Not measured | Cold perception compile 4.529 s in final run and 12.826 s with a colder cache in screening; full-arm peak allocations accumulated 262.19–291.31 MiB | Modest complete-frame benefit, compile specialization/startup, and final GPU combination not repeatable | **Defer** |
| ModernGL renderer | Immutable one-transfer snapshot; batched ships/projectiles/map; quarter-resolution exact-LOS fog; toroidal interpolation; actual RTX GL | 10 snapshot/geometry/visibility/ring-reuse/ownership tests; real GL shaders and full team-fog frame executed | With lean step and two policies: **83.478 / 96.201 / 105.843 / 113.759 ms**, n=150, 150 misses, versus flip-inclusive legacy 165.968 / 181.851 / 199.371 / 223.884 | Renderer-only snapshot+GPU+flip: **7.757 / 10.337 / 10.794 / 11.291 ms**, n=300, 0 misses at 16.7 or 33.3 ms. Old draw without flip: 90.518 / 113.780 / 116.373 / 165.332 | No effect on headless environment throughput | No effect; not measured | Shader/renderer startup 0.715–0.809 s. Process max RSS 1.268–1.269 GB versus 1.210–1.211 GB reference; CUDA env allocation was identical | Missing visual/UI features; point primitives clip very large disks; ModernGL is not yet a project dependency | **Retain prototype, defer default integration** |
| GPU renderer + compiled perception | GPU snapshot/render/flip on both arms; buffered eager versus compiled pure perception; lean step; two policies | Perception parity passed separately; pair action traces exact | Candidate **75.704 / 90.476 / 102.740 / 535.193 ms**, 150/150 misses; reference 77.391 / 92.600 / 95.512 / 101.651 ms, 102/150 misses | 8.92–9.82 ms candidate render diagnostic range | Candidate env phase 30.43–34.28 ms; reference was unstable at 13.49, 44.55, 46.44 ms | Not measured | Cached first compiled-builder calls 5.9–6.3 ms after a 1.076 s first call in this process; peak allocations accumulated to 305.87 MiB | Pair 0 strongly favored reference (25.811 vs 80.998 ms); candidate won pairs 1–2. Large thermal/order variation and candidate tail spike | **Reject as final combination; not repeatable** |
| Archived compiled-real and Warp LOS | Isolated projectile/ship LOS kernels, then complete two-policy frames | 240,012 sampled boolean comparisons per alternative | Archived complete-frame runs showed no repeatable improvement | No render effect | Archived environment-only Warp path was neutral/slower depending on batch | Not measured | See archived audit | Fast isolated kernels do not establish a game improvement | **Reject repeating unchanged substitution** |

Raw artifacts: [intake reference](performance-experiments/current-reference-baseline-50v50.json),
[one-shot rejected observation screen](performance-experiments/current-candidate-screen-50v50.json),
[lean screen](performance-experiments/lean-interactive-screen-50v50.json),
[lean parity](performance-experiments/lean-interactive-parity-50v50.json),
[lean final A/B](performance-experiments/lean-interactive-final-ab-50v50.json),
[shared-observation parity](performance-experiments/shared-observation-parity-50v50.json),
[shared-observation isolated A/B](performance-experiments/shared-observation-isolated-ab-50v50.json),
[whole-tick rejection](performance-experiments/whole-tick-compile-50v50.json),
[compiled-perception parity](performance-experiments/compiled-perception-parity-50v50.json),
[compiled-perception screen](performance-experiments/compiled-perception-screen-50v50.json),
[compiled-perception final A/B](performance-experiments/compiled-perception-final-ab-50v50.json),
[renderer-only summary and raw links](performance-experiments/gpu-renderer-ab-summary-50v50.json),
[GPU complete-frame A/B](performance-experiments/gpu-complete-final-ab-50v50.json), and
[final combination A/B](performance-experiments/final-combination-ab-50v50.json).

## Correctness result

The full repository suite ran:

```text
1580 passed, 12 skipped, 1 documentation-policy failure in 672.92s
```

The sole failure found legacy `--mode` command text in this report and two
pre-existing archived audit documents; no runtime or numerical assertion
failed. Earlier focused integrated verification passed 318 tests with four
CUDA-sandbox skips, and the authorized CUDA parity runs below passed separately.

It covered environment physics, projectile lifecycle and fields, collisions,
shields and hull destruction, Frontline capture/transitions, boundary damage,
rewards, reset/termination/truncation, interactive buffered actions and belief
state, and GPU snapshot contracts. The four skipped cases require CUDA inside
pytest's sandbox. Authorized CUDA checks separately established:

- 40-step lean interactive parity for every `TensorState` field, exact discrete
  events, both team views, projectiles/visibility, retained ownership, and
  buffered actions;
- 40-step compiled pure-perception parity for both views and retained ownership;
- exact action-trace equality for every paired complete-frame run reported as
  a candidate comparison;
- actual RTX OpenGL context and a complete team-fog frame.

Floating comparisons used `atol=rtol=2e-6`; discrete values were exact. Fixed
input actions are stored in the parity artifacts, while every complete-frame
artifact stores the resulting policy action trace. The paired complete-frame
traces were compared exactly, so comparison does not rely on seed alone.

## Reproduction commands

All intensive commands were serialized with
`artifacts/deadline-experiment/benchmark.lock` and an external timeout. The
principal commands were:

```bash
# Isolated dependency used by the prototype; no project dependency was changed.
UV_CACHE_DIR=/tmp/bnb-deadline-uv-cache uv pip install --python .venv/bin/python \
  --target /tmp/bnb-deadline-moderngl moderngl

# Exact-revision intake baseline in the detached reference worktree.
timeout 600s flock -x artifacts/deadline-experiment/benchmark.lock \
  /usr/bin/time -v .venv/bin/python /tmp/bnb-reference.YVmdWu/benchmarks/realtime_latency.py \
  --device cuda --scenario 50v50 --execution sequential --steps 30 --warmup 10 \
  --compile default --window-size 900 --sides 2 --seed 99 \
  --out docs/engineering/performance-experiments/current-reference-baseline-50v50.json

# Lean correctness and final complete-frame A/B.
timeout 300s flock -x artifacts/deadline-experiment/benchmark.lock \
  .venv/bin/python benchmarks/deadline_experiment.py --mode lean-parity \
  --device cuda --steps 40 --seed 271828 \
  --out docs/engineering/performance-experiments/lean-interactive-parity-50v50.json
timeout 1200s flock -x artifacts/deadline-experiment/benchmark.lock \
  .venv/bin/python benchmarks/deadline_experiment.py --mode lean-rendered \
  --device cuda --warmup 10 --samples 50 --pairs 4 --policy-compile default \
  --seed 424242 \
  --out docs/engineering/performance-experiments/lean-interactive-final-ab-50v50.json

# Renderer-only alternating arms; N=0,1,2, reversing arm order on pair 1.
timeout 180s flock -x artifacts/deadline-experiment/benchmark.lock \
  env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  .venv/bin/python benchmarks/gpu_renderer_benchmark.py --arm reference \
  --device cuda --warmup 20 --samples 100 \
  --out docs/engineering/performance-experiments/gpu-renderer-ab-pairN-reference.json
timeout 180s flock -x artifacts/deadline-experiment/benchmark.lock \
  env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  .venv/bin/python benchmarks/gpu_renderer_benchmark.py --arm gpu --device cuda \
  --moderngl-path /tmp/bnb-deadline-moderngl --warmup 20 --samples 100 \
  --out docs/engineering/performance-experiments/gpu-renderer-ab-pairN-gpu.json

# Complete legacy-versus-GPU renderer A/B, both using the lean step.
timeout 1200s flock -x artifacts/deadline-experiment/benchmark.lock \
  env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  .venv/bin/python benchmarks/deadline_experiment.py --mode gpu-rendered \
  --device cuda --warmup 10 --samples 50 --pairs 3 --policy-compile default \
  --moderngl-path /tmp/bnb-deadline-moderngl \
  --out docs/engineering/performance-experiments/gpu-complete-final-ab-50v50.json

# Whole-tick rejection and pure-perception promotion.
timeout 900s flock -x artifacts/deadline-experiment/benchmark.lock \
  .venv/bin/python benchmarks/performance_audit.py --mode tick-compile \
  --scenario 50v50 --batch 1 --steps 10 --pairs 1 --device cuda \
  --out docs/engineering/performance-experiments/whole-tick-compile-50v50.json
timeout 900s flock -x artifacts/deadline-experiment/benchmark.lock \
  .venv/bin/python benchmarks/deadline_experiment.py --mode compiled-parity \
  --device cuda --steps 40 --seed 161803 \
  --out docs/engineering/performance-experiments/compiled-perception-parity-50v50.json
timeout 1200s flock -x artifacts/deadline-experiment/benchmark.lock \
  .venv/bin/python benchmarks/deadline_experiment.py --mode compiled-perception \
  --device cuda --warmup 10 --samples 50 --pairs 3 --policy-compile default \
  --seed 57721 \
  --out docs/engineering/performance-experiments/compiled-perception-final-ab-50v50.json

# Final combined A/B.
timeout 1200s flock -x artifacts/deadline-experiment/benchmark.lock \
  env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  .venv/bin/python benchmarks/deadline_experiment.py \
  --mode gpu-compiled-perception --device cuda --warmup 10 --samples 50 \
  --pairs 3 --policy-compile default --seed 141421 \
  --moderngl-path /tmp/bnb-deadline-moderngl \
  --out docs/engineering/performance-experiments/final-combination-ab-50v50.json
```

## Training and throughput boundary

No deadline candidate changes the reward-bearing training path. The lean step
deliberately omits rewards/statistics and the GPU renderer is presentation-only;
therefore an environment-only B=1/32/128/large sweep or PPO A/B would not test
their claimed benefit. Compiled perception was not integrated after its final
combination failed the repeatability gate. End-to-end training benefit is
**unproven**, and no new training-throughput claim is made.

For context only, the archived environment-only CUDA runs (no policies,
rollout storage, evaluator, or PPO) measured 66.77, 69.21, 68.13, and 73.00 ms
per batch step at B=1, 32, 128, and 960, respectively. The interrupted PPO run
recorded a 578.93-second warmup and one 215.94-second update with two PPO epochs;
it has no completed aggregate. Neither set of numbers is rendered-frame or
complete training throughput.

## Handoff

- The runnable reference remains in the repository; no archived audit file was
  modified.
- ModernGL remains an isolated prototype dependency, not a project dependency.
- Failed production observation edits were reverted; failed candidates live
  only in benchmark code and raw evidence.
- The final implementation and evidence are checkpointed on the experiment
  branch; publishing status is recorded in the Git history/handoff message.
- No benchmark process remains. Durable continuation state is in the ignored
  `artifacts/deadline-experiment/WORKLOG.md` and `state.json`.
- Exact-session resume was available, but reset-time telemetry and a safe
  persistent process identity were not, so no unattended restart supervisor was
  installed or claimed.
- Model-usage telemetry was unavailable; usage is unknown.
