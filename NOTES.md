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
| 6.4 temporal sublayer | `network.TemporalBlock`, `network.RGLRU`, `network.DepthwiseConvolution`, `scan.py` |
| 6.5 trunk | `network.Trunk` |
| 7 heads and initialisation | `network.MLPHead`, `network.ShotTrajectoryHead`, `Yemong.initialise` |
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
  state-code round trip.

## Benchmark

`bench/speed.py` (see its docstring for the exact protocol). Results: `bench/results.jsonl`.

(table below)

## Speed changes and what each bought

(below)

## Progress

- Done: every fixture; the whole model; tests; Triton scan; the benchmark.
- In flight: baseline numbers, profiling.
- Next: see "What to try next".
