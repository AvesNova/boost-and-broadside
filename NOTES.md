# Swift Yemong: notes

A runtime-fast Yemong built from `docs/yemong-spec.md` and the equivalence fixtures alone
(brief: `BRIEF.md`). Package: `src/bnb/models/swift/`.

## Design

- **Plain modules, fixture names.** Every network piece is an `nn.Module` whose attribute
  names are the fixture weight names, so the fixture adapter
  (`tests/reference/yemong/yemong_arm.py`) loads goldens with `load_state_dict`. The model
  composes the same modules, so the fixtures test the production code path.
- **Time-major everywhere.** The trunk and temporal sublayers take `(T, B, Q, D)` and
  `starts[t]` ("an episode starts at t"). The fixtures' `done[t]` ("ended at t") maps to
  `starts[t] = done[t - 1]`. `step` is the same code with `T = 1` and no starts.
- **Branch-free codes.** The codes, belief and critic functions have no data-dependent
  control flow, so `torch.compile(fullgraph=True)` fuses them and `step` captures in a CUDA
  graph.
- **The linear scan** (`scan.py`) is a Triton kernel on CUDA float32 (one program per
  block of channels walks time; the backward is the reverse recurrence), registered as a
  `torch.library.custom_op` with autograd so it compiles under `fullgraph=True`. The CPU /
  float64 path is a plain loop (the fixtures).
- **Kernels as `triton_op`s.** Every Triton kernel (`attention.py`, `gated.py`, `head.py`,
  `temporal.py`, `scan.py`) is a `torch.library.triton_op` with autograd, so it compiles
  under `fullgraph=True` and Inductor launches it from its own wrapper code (no Python
  custom-op dispatch per call). Each has a switch in `network.py` (`ATTENTION`,
  `FUSED_MLP`, `FUSED_TEMPORAL`, `FUSED_HEAD`) and the plain-torch path stays for the CPU,
  float64 and the fixtures.
- **Precision.** Under bf16 autocast the matmuls run in bf16; RMSNorm upcasts to float32
  (with float32's epsilon, so autocast does not silently switch to bf16's epsilon); the
  residual stream, the scan state, the codes, the belief and the critic expectations stay
  float32 (`upcast` on every head output).
- **Replay.** `step` returns `Replay(valid, age, means, spreads, action, previous_action)`,
  the belief after observing. The spec lists means, spreads, action and previous action;
  `valid` (local presence's source) and `age` (time since seen) depend on history too, so
  they are in the record. `unroll` detaches the replay (stored data, no gradient).

## Spec piece to code

| spec | code |
|---|---|
| 3.1 belief channels, 3.2 layout, 3.4 codec | `state_code.py` (`ShipStateCodec`, `ShipCard`) |
| 3.3 circular, position, scalar, velocity, grouped | `codes.py` |
| 4 inputs (ship, field, global tokens, presence, rotary geometry) | `model.py` `Yemong.compose`, `targets.local_presence`, `network.rotary_tables` |
| 5.1 carry | `model.py` `Carry` |
| 5.2 step (reset, observe, compose, trunk, heads, advance, act) | `model.py` `Yemong.step`, `belief.py` |
| 5.3 unroll | `model.py` `Yemong.unroll` |
| 6.1 encoder | `network.Encoder` |
| 6.2 rotary | `network.rotary_tables`, `network.apply_rotary` |
| 6.3 spatial sublayer | `network.SpatialBlock`, `network.GatedMLP` |
| 6.4 temporal sublayer | `network.TemporalBlock`, `network.RGLRU`, `network.DepthwiseConvolution`, `scan.py`, `temporal.py` (CUDA) |
| 6.5 trunk | `network.Trunk` |
| 7 heads and initialisation | `network.MLPHead`, `network.ShotTrajectoryHead`, `Yemong.initialise`, `head.py` (CUDA, 16-bit) |
| 8 critic, return scale | `targets.py` (`two_hot`, `expectation`, `critic_values`, `outcome_targets`, `ReturnScale`) |
| 9 hex density, shot codes | `targets.py` (`hex_cells`, `hex_counts`, `poisson_terms`, `shot_*`) |

Decisions the spec left open (all small):
- Decoded positions are wrapped to `[-P/2, P/2)` (the code is periodic with
  `P = 26 244 m`; the arena is centred on the origin, so this is the identity on the
  arena and keeps rotary and presence correct for negative coordinates).
- `time_since_seen` on the ship token is the belief's (`age * tick`), as the fixtures'
  `belief_observe` returns it.
- `ShipCard` holds the game constants the model reads (tick, per-type shield and energy
  maxima, `alpha_s`, ticks between shots, bullet life, muzzle speed, hit radius). The
  defaults are stand-ins; two ship types make the ship token 553 wide as the spec says.
- The cooldown code's centres are `0, h, 2h, 3h` with `h` the tick.
- `step` samples with Gumbel-max (`torch.rand`), so CUDA-graph replays draw fresh noise.

## Tests

- `tests/reference/yemong`: 43 fixtures (float64 to 1e-9, float32 network tests to 2e-4).
- `tests/test_swift_model.py`: `unroll == loop(step)` in float64 on the CPU (1e-9; resets
  at the first step and mid-segment, and one view-only reset) and float32 on CUDA eager
  and compiled (1e-4); shapes, dtypes, finiteness; bf16 autocast forward + backward on the
  CPU and CUDA; the belief rules (sharp code when visible, spawn voids, fog off zero
  spread, hidden-zero columns unread); permutation within a team; sizes 1v1 to 10v10; the
  state-code round trip; recomputing the MLPs or every sublayer gives the same gradients.
- `tests/test_swift_kernels.py`: every Triton kernel against plain torch (the fused
  temporal mix against the float64 block, forward and every gradient, with resets; the
  fused head against float64, at least as close as the module path under bf16).

## Benchmark

`bench/speed.py` (see its docstring for the exact protocol). Results: `bench/results.jsonl`
(one line per cell, with the commit). Cells: training (`unroll` forward + backward, L = 128,
V = 16 and 64) and rollouts (`step`, V = 2048), float32 (strict, no TF32) and bf16
autocast, eager / compiled / compiled + CUDA graphs. Rollout "compiled" and "cudagraphs"
use `runtime.CapturedStep` (static carry updated in place inside the compiled function;
cudagraphs replays it as one graph). A training cell that runs out of memory retries with
the trunk's sublayers recomputed in the backward and is marked `recompute`.

The GPU is shared with other agents' jobs; a cell that still runs out of memory is
recorded as such. `bench/profile.py` prints a `torch.profiler` kernel table for one cell.

Work per unit (FlopCounterMode): training 152 MFLOP per view-tick forward + backward
(1.25 TFLOP per V = 64 segment); a decision 47 MFLOP per view (97 GFLOP at V = 2048). With
D = 128 every GEMM has K = 128, so most of the model is memory-bandwidth bound (the laptop
4070 has ~256 GB/s): fusion and 16-bit storage matter as much as tensor cores.

### Baseline (commit 5efbb2ed, plain modules, SDPA, Triton scan)

| cell | eager | compiled | compiled + graphs |
|---|---|---|---|
| training V=16 fp32 | 125.5 ms (16.3k vt/s) | 83.8 ms (24.4k) | 82.9 ms (24.7k) |
| training V=16 bf16 | 93.2 ms (22.0k) | 43.9 ms (46.6k) | 43.1 ms (47.5k) |
| training V=64 fp32 | OOM | OOM | OOM |
| training V=64 bf16 | OOM | 196.2 ms (41.8k) | 195.6 ms (41.9k) |
| rollout V=2048 fp32 | 53.6 ms (38.2k vs/s) | 29.8 ms (68.7k) | 29.5 ms (69.5k) |
| rollout V=2048 bf16 | 44.5 ms (46.0k) | 18.0 ms (113.7k) | 17.2 ms (118.9k) |

vt/s = view-ticks per second; vs/s = view-steps per second. Peak memory: V=16 fp32 eager
2.9 GB, V=64 bf16 compiled 4.6 GB.

## Speed changes and what each bought

Profiles first (`bench/profile.py`, bf16 compiled):
- **rollout, baseline** (18.3 ms/step): GEMMs 3.2 ms; flash attention 2.0 ms (its 128-row
  tiles on 11 queries x 21 keys) plus 1.4 ms of rotary / cat / transpose around it; the
  gated MLP's `gelu(gate) * up` 2.1 ms (memory bound); codec encode + decode ~2.5 ms;
  carry write-back 0.7 ms; RMSNorms 0.8 ms.
- **training V=16, baseline**: flash attention forward + backward 20% of GPU time, GEMMs
  27%, gated-MLP elementwise forward + backward 12%, RMSNorm backward 5%.

Changes (spatial sublayer micro-benchmark, compiled, B = 8192 views fwd + bwd / 2048 fwd):

| change | train fp32 | train bf16 | rollout fp32 | rollout bf16 |
|---|---|---|---|---|
| SDPA (baseline) | 42.6 ms | 25.8 ms | 3.46 ms | 1.87 ms |
| Triton small attention (`attention.py`) | 35.5 | 18.3 | 2.76 | 1.21 |
| + fused gated MLP (`gated.py`, bf16 only) | 35.4 | 16.6 | 2.76 | 0.96 |
| broadcast multiply-reduce attention (rejected) | 41.2 | 27.3 | 3.28 | 2.02 |

- **Triton small attention:** one program per (view, head), 16 x 32 tiles, keys read by
  stride from the token and map projections (no cat, no transpose); backward recomputes
  the softmax in registers. 1.2-1.5x on the sublayer.
- **Fused gated MLP:** both GEMMs in one kernel, `gelu(g) * u` in registers; the backward
  recomputes g and u, so the 8D-wide pre-activations are neither written nor saved
  (activation memory down by 2 KB per token per MLP).
- **Rotary inside the attention kernel:** q and k are loaded as even and odd lanes,
  rotated in float32 registers; the backward rotates the gradients back. Removes a
  rotated copy of q, k and the map keys per sublayer (and, under autocast, their float32
  promotion and cast back). Spatial sublayer micro-benchmark (B = 8192 / 2048, compiled):
  train bf16 16.74 -> 15.79 ms, train fp32 35.7 -> 35.2, rollout bf16 0.959 -> 0.850,
  rollout fp32 3.00 -> 2.66.
- **Exact fine-level circular codes:** `u = (m mod p_l)/p_l * c` instead of
  `(m mod P)/p_l * c` (identical math). At the 4 m level u reached 3^9, so one float32 ulp
  was ~1e-3 of the fraction; compiled and eager disagreed by ~0.1 in next-state logits
  where the code is near zero (`log(code + 1e-3)`). Now exact (the level periods are
  integers in metres).

## Phase 2 (speed)

Profile of the training cell (V = 16, compiled, `bench/profile.py`, which now also prints
GPU time per kernel group), before phase 2, ms of GPU time per segment:

| group | bf16 (32.8 total) | fp32 (74.1 total) |
|---|---|---|
| GEMMs (cuBLAS) | 11.0 | 35.5 |
| fused pointwise (Inductor) | 5.4 | 6.3 |
| RMSNorm backward | 4.2 | 6.9 |
| gated MLP kernel | 3.5 | - (gelu * up: 10.4) |
| attention kernel | 2.7 | 5.8 |
| RMSNorm forward | 2.6 | 5.1 |
| scan kernel | 1.0 | 1.4 |

By part (compiled fwd + bwd, bf16 / fp32 ms): compose 2.2 / 2.3 (forward only), encoder
2.2 / 4.3, trunk 22.8 / 53.4 (a spatial sublayer 3.7 / 8.7, a temporal 4.5 / 9.3), heads
8.1 / 15.5 (the shot trajectory head alone 3.5 / 7.0: 81 920 rows).

Readings:
- The GEMMs run at the roof: bf16 ~29 TFLOP/s (the GeForce fp32-accumulate rate), strict
  fp32 ~9 TFLOP/s. What is left is memory traffic between them, and CPU launch overhead:
  the compiled V = 16 bf16 cell spent ~2-3 ms more wall time than GPU time (CUDA graphs
  close the gap).
- A strict-fp32 fused gated MLP (`tl.dot` with IEEE inputs) is *slower* than cuBLAS SGEMM
  plus the elementwise pass (spatial micro-benchmark 40.2 vs 35.8 ms): rejected; fp32
  keeps cuBLAS for every GEMM.
- `max-autotune` does nothing here: Inductor refuses GEMM autotuning on 36 SMs.

Changes:
- **Fused temporal mix** (`temporal.py`): the RG-LRU gates, the scan and `h * gelu(branch)`
  in one kernel per direction. Time is walked in chunks of 16 steps, each an associative
  scan in registers (the one-step loop was latency bound: 128 dependent loads); only h
  (float32) is stored between the two GEMMs. Temporal sublayer alone: bf16 4.48 -> 3.98
  ms, fp32 9.34 -> 8.56 ms. As a `custom_op` it *lost* 1.5 ms of wall time in the
  compiled bf16 cell (CPU dispatch) while winning 1.3 ms with CUDA graphs; hence:
- **`triton_op` for every kernel**: Inductor launches them itself.
- **Fused heads** (`head.py`): `W2 gelu(RMSNorm(W1 x + b1)) + b2` as one kernel per
  direction under 16-bit autocast (the hidden row, 128 or 256 wide, lives in registers;
  the backward writes only dh and the activation for the cuBLAS weight GEMMs). Covers the
  seven MLP heads, the shot trajectory head's tail and the encoder's three kind
  projections with the shared layer (whose input gradient GEMM is skipped).
- **MLP-only recompute** (`Trunk.recompute = "feedforward"`): the bench tries it before
  full sublayer recompute when a cell runs out of memory.

### Benchmark, every cell (ms; eager / compiled / compiled + CUDA graphs)

| cell | baseline 5efbb2ed | phase 1 end cb9398a9 | phase 2 a39c1463 |
|---|---|---|---|
| train V=16 fp32 | 125.5 / 83.8 / 82.9 | 111.2 / 74.4 / 74.0 | 120.0 / 75.8 / 75.0 |
| train V=16 bf16 | 93.2 / 43.9 / 43.1 | 71.2 / 34.0 / 33.3 | 71.2 / 34.0 / 34.0 (A/B: 33.0 / 32.7) |
| train V=64 fp32 | OOM / OOM / OOM | 573 / 368 / 367 (recompute all) | 521 / 333 (recompute MLPs) / 362 (all) |
| train V=64 bf16 | OOM / 196.2 / 195.6 | 355 / 154.8 / 154.3 | 313 / 142.5 / 142.4 |
| rollout V=2048 fp32 | 53.6 / 29.8 / 29.5 | 48.2 / 26.5 / 26.3 | 63.0 / 27.8 / 26.8 |
| rollout V=2048 bf16 | 44.5 / 18.0 / 17.2 | 35.4 / 12.8 / 12.6 | 54.5 / 13.8 / 13.9 |

Peak memory, phase 2: V=16 bf16 compiled 0.82 GB (was 0.91), V=64 bf16 3.1 GB (was 3.5),
V=64 fp32 compiled with MLP recompute 5.1 GB.

**Caveat on the phase 2 column.** The GPU lock was held by other agents' jobs for ~90
minutes; the run went through while other processes still had GPU memory, and the
unchanged-code control in the A/B below drifted by 1-4 ms between runs. In-process A/B
(same process, flags toggled, training V=16 bf16, `/tmp/bnb-scratch-yemong-blind/p2/ab2.py`):

| setting | compiled | CUDA graphs | peak |
|---|---|---|---|
| neither kernel | 35.2 ms | 37.8 (outlier) | 0.94 GB |
| fused temporal | 33.5 | 32.9 | 0.93 |
| fused temporal + fused heads | 33.0 | 32.7 | 0.80 |

An earlier same-process A/B of the temporal kernel alone (CUDA graphs, before
`triton_op`): 33.4 -> 32.15 ms. The eager rollout and eager V=16 fp32 cells got slower
(35 -> 54 ms, 111 -> 120 ms); unexplained, and not yet A/B'd: suspects are `triton_op`
dispatch in eager mode and the fused head's low parallelism on small row counts (the
outcome and density heads have 2048 rows = 32 programs). The compiled rollout cells are
1 ms slower too. A rollout A/B (`p2/ab_rollout.py 2048 bfloat16 compiled,cudagraphs`,
log `p2/ab_rollout.log`) was queued on the lock when this phase stopped.


## What to try next

- **First: the rollout A/B** (above). If the fused head loses on the rollout, try
  BLOCK_M = 32 (or 16) when rows are few, or keep it for training only; check the eager
  regression (`triton_op` vs `custom_op` dispatch in eager).
- fp32 V = 64: it needs ~8.5 GB without recompute. Storing less in strict fp32 means
  recomputing something; "feedforward" recompute is the cheapest ladder step.
- The shot trajectory head's front (latent + age projection, norm, gelu) into `head.py`'s
  kernel as a prologue (81 920 rows: the largest head).
- RMSNorm into the following GEMM's prologue (attention and feedforward norms): K = 128
  fits a row in registers; saves a bf16 write + read of every normed activation.
- Residual stream: every sublayer reads and writes the float32 residual 2-3 times
  (norm, add, norm backward). A kernel that does the residual add and the next norm
  together would halve that traffic.
- CPU overhead in the compiled (no graphs) cells at small V: fewer, larger kernels.

## Progress

- Done: every fixture; the whole model; tests; every kernel (scan, attention, gated MLP,
  temporal mix, heads) as a `triton_op`; the benchmark; phase 2 profiling and changes.
- In flight: the rollout A/B of the phase 2 kernels (queued on the shared lock; its log is
  `/tmp/bnb-scratch-yemong-blind/p2/ab_rollout.log`). Rerun `bench.speed` on a quiet GPU.
- Next: "What to try next".
