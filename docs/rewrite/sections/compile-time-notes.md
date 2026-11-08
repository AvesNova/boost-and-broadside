# Compile time: where it goes (2026-11-08)

Branch `perf/compile-time`. Backlog item "Compile time". The aim is a shorter first update
with steady-state speed unchanged.

## Measurements

Arm 2d's config (`2026-11-08-bc-arm2d-persistence`: BC, mixed preset, 2048 environments +
1024 lane environments), 2 updates, RTX 4070 Laptop, torch 2.15.0, inductor cache warm for
this exact config (arm 2d ran today). Probe 1 ran with `TORCH_TRACE` and py-spy, probe 2
without them.

| | probe 1 (traced) | probe 2 (clean) | arm 2d's real run |
|---|---|---|---|
| process start to update 1 | ~58 s | ~47 s | |
| update 1: rollout / learner (incl. cadence) | 123 / 396 s | 101 / 319 s | 327 / 497 s |
| update 1 total | 519 s | 420 s | 823 s |
| update 2 total | 351 s | 283 s | 242 s |
| Dynamo compile, all frames | 189 s | 143 s | |
| of which bytecode tracing | 157 s | 122 s | |
| of which backend (all cache hits) | 23 s | 16 s | |

The warm cache saved 415 s of inductor work in probe 1 (`remote_cache_time_saved_s` summed
over the frames), so a fully cold first update costs roughly 400 s more than probe 2. The
real runs (550-1110 s) sit between warm and cold: each arm changes the graphs, so part of
the cache misses.

Update 2 ran slower in both probes than in the real run (learner 265 s against 221 s).
Steady-state comparisons therefore need an A/B in the same session over several updates,
not a comparison against old runs.

## Where the time goes (warm cache)

1. **Dynamo tracing, about 120 s.** Every inductor and autograd graph was a cache hit, but
   Dynamo retraces the Python in every process. The rollout's `_act` alone is 18,780 FX
   nodes (78 s in probe 1). It holds 4 Yemong forwards (the learner and three lane
   policies, 16 attention calls) and many small ops from `codes.py` (`circular_encode`,
   `invert_mean`, `scalar_encode`, ...). Then come `Yemong.step` (`_evaluate`, 11 s), the
   learner's `_loss` (25 s), the cadence's `_terms` (15 s), and smaller ones.
2. **Three avoidable recompiles, about 35 s.**
   - `_loss` 7/1: guard `heads == True` (ppo.py:509). The first epoch's measured minibatch
     passes `heads=True`, which retraces the whole loss.
   - `_codes` 6/1 and `_terms` 8/1: the cadence's gradient decomposition splits the first
     minibatch (16 views) into halves of 8 (`views.tensor_split(2)`), a second shape.
3. **Scripted characteristics, about 35-40 s per process.** `Trainer.__init__ ->
   characteristics -> measure_characteristics` flies every manoeuvre eagerly
   (`manoeuvres.fly`, `shown.trajectory`). It is deterministic in the rules and the ship
   card, and it also runs in tournaments (`league/matches.py`) and play.
4. **Store allocation, about 5 s.** Model build about 3 s.

On a cold cache (after every reboot) add about 400 s of inductor and Triton compilation.
The inductor cache lives in `/tmp/torchinductor_vizia`, which the reboot wipes. It had grown
to 48 GB since the 2026-11-07 boot. The Triton cache (`~/.triton`, 1.3 GB) survives
reboots.

CUDA-graph capture and Triton autotuning are negligible here (under 10 s together).

## Options, by saving per run and risk to steady speed

| option | saves | steady-speed risk |
|---|---|---|
| A. durable inductor cache (`TORCHINDUCTOR_CACHE_DIR` outside `/tmp`, with a size cap) | ~400 s on the first run after a reboot | none |
| B. cache the scripted characteristics on disk (key: rules + ship card + code version) | ~35 s every process | none |
| C. remove the three recompiles (heads sums outside the traced loss, or one shape; cadence halves at the training micro-batch size) | ~35 s warm, ~100 s cold | none if done as stated; measure |
| D. Dynamo precompile (`torch.compiler` aot_compile / `load_compiled_function`), keyed by config + source hash | ~120 s on any repeat build (resume, tournaments, reruns); 0 for a new arm | none expected (same graphs); experimental API |
| E. `nested_compile_region` on the Yemong forward (trace once, reuse) | part of `_act`'s tracing; the lanes run different batch sizes, so less than 3/4 | a region boundary can block fusion; measure |
| F. cheaper tracing of the code encoders (fewer, larger ops in `codes.py`) | unknown share of 120 s | the kernels change; measure |

## Cold compile (the author's target, 2026-11-08 evening)

The aim is the cold compile: the wait after a new parameter, model change or bug fix, when
the graphs change and inductor compiles them again. A warm cache does not help there.

Arm 2d's config, fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`, 1 update, traced:
**first update 2,523 s** (rollout 1,062, learner 1,462 of which cadence 500), against
about 240 s steady. Compile 1,730 s, of which Dynamo tracing 162 s; the rest is inductor.
Peak memory 29 GB (unit cap 30 GB).

| graph | cold s | note |
|---|---|---|
| rollout `_act` | 617 | 4 Yemong forwards (learner, 3 lane policies) + scripted teacher; 906 kernels |
| `_advance` / `_resolve` / `_perceive` | 59 / 82 / 27 | physics |
| learner `step` (`Rollout._evaluate`, the bootstrap) | 148 | a 5th copy of the Yemong forward, called once per update |
| shots `tick`, `_codes` | 17, 27 | |
| `_loss` (forward + backward) | 235 | |
| `_loss` again (`heads == True` guard) | **140** | recompile |
| `_terms` (cadence) | 168 | |
| `_codes`, `_terms` again (cadence halves at 8 views) | **28, 180** | recompile |

Where inductor's time goes (cold, all graphs): `Scheduler.codegen` 638 s (main thread,
writing kernel source: `LoopBody.__call__` per node, sympy indexing), waiting on the Triton
workers 414 s, `Scheduler.__init__` (fusion) 269 s, runtime autotune benchmarks 214 s,
backward compiles 203 s, lowering 138 s. About 5,900 Triton compiles for about 1,850
kernels (3 configs each). Most of this is single-threaded Python, so neither a GPU nor
more cores help much; the lever is fewer ops reaching inductor.

Setup also grows cold: 221 s against 47 s warm, mostly the Yemong Triton kernels'
`triton.autotune`, which compiles each config serially in the main thread on first use.

The graphs are dominated by the code functions: in one Yemong forward, `codes.py` accounts
for 1,067 of about 1,590 annotated FX lines (the trunk's kernels are small). They are
called once per feature (12 delta components, 5 scalars, ...), each call 30-60 ops, with
per-component constants (HL-Gauss matrices, centres, smoothing) rebuilt in every graph.

## Changes on this branch

| commit | change | cold saving (estimate) | steady cost |
|---|---|---|---|
| 7c62a408 | `Rollout._evaluate` (the bootstrap, once per update) eager | ~148 s (its graph) | 0.1 s per update (102 ms per call, measured) |
| 7c62a408 | the heads diagnostic by `torch.cond` on a host bool: one `_loss` graph, not two | ~140 s | none (no device sync; checked with `set_sync_debug_mode("error")`) |
| 7c62a408 | the cadence's gradient decomposition over the first two minibatches, one half each | ~208 s (`_codes` and `_terms` at 8 views) | none; **its series now cover twice the views** (a diagnostic definition change, for the author) |
| 3b4b4b83 | `StateCode.encode`'s five scalars in one `codes.scalar_encode_many` call | encode 160 s to 93 s cold alone; it is in every Yemong graph | none within noise (2.03 vs 2.09 ms GPU per call) |

All bit for bit where goldens exist: the arm goldens, the C-fixed and v1 spread-fit
digests pass unchanged; a new test checks `scalar_encode_many` against the per-feature
encode in float32 and float64.

### Tried and not kept

- **A batched delta decode** (12 `symlog_decode` calls to 3, one per padded range; the
  medians from the same batches; constants as exact buffers). Ops 856 to 313, cold
  compile alone 62 s to 26 s, bit for bit. But GPU time per call rose from 0.94 ms to
  1.2-1.4 ms (the 8-component batch's kernel is about 45% slower per component than the
  separate kernels; precomputing centres and smoothing did not help). Estimated +0.75% on
  an update, so not kept. Each component's decode as its own fused kernel is what
  inductor does best here.
- **Batching the scalars with per-feature tensor constants** (low and spacing as buffers):
  inductor split each fused persistent reduction into several kernels (+25% GPU time).
  Keeping the coordinates in per-feature Python-float arithmetic and batching only the
  constant-free core kept the fusion.

Lesson: fewer ops cut compile time roughly in proportion, but batching changes inductor's
fusion, so every batched rewrite needs its kernel time measured
(`torch.profiler`, the compiled function warm, against the old tree).
