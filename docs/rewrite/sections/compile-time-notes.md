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
