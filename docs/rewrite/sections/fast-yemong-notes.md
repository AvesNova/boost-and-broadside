# Fast Yemong: notes

Brief: `fast-yemong-brief.md`. Audit: `../audits/2026-11-02-fast-yemong.md`. Kernel source:
`race/yemong-blind`, `src/bnb/models/swift/` (read-only).

## Piece map

Both models are time-major over a segment: v2's `unroll` runs tokens (L, R, Q, D), map
memory (L, R, M, Dm) and the temporal carry (R, Q, D); the blind model's (T, B, Q, D). The
kernels flatten every leading axis but time (and, for attention, time too: one view-step
is one program). The table names the v2 piece each kernel replaces and what differs.

| Kernel (blind file) | v2 piece it replaces | Layout and parameters |
|---|---|---|
| scan (`scan.py`) | `temporal.parallel_scan` (the chunked scan in `RGLRU.sequence`) | Same layout: decay and contribution (L, ..., D), initial (..., D), flattened to (L, C). v2 cuts the recurrence at a start by zeroing that step's decay, which the kernel takes as given. No parameters. |
| attention with rotary (`attention.py`) | `SpatialBlock.forward` from the rotations to the output projection's input: `apply_rotary` on queries, keys and map keys, the two `cat`s, the transposes and SDPA | v2's `query_key_value` (3D, D) is (q, k, v) by rows then (heads, head width): q, k and v are strided views of its output, which the kernel reads by stride (no copy); `map_key_value` (2D, Dm) likewise. Rotary: both rotate interleaved (even, odd) pairs of a head's first 2R dims (R = 20 of 32 lanes at head width 64); v2's tables are (..., tokens, 1, R), the kernel's (views, tokens, R): a reshape. SDPA's default scale is the kernel's W^-1/2. |
| gated MLP (`gated.py`) | `GatedMLP.forward`'s `gate_up` GEMM and `gelu(gate) * up` (spatial and temporal sublayers) | v2's `gate_up.weight` (2H, D), gate rows first, is the kernel's (gate, up) pair as two row blocks of one matrix: the port takes the one matrix, so its weight gradient is one GEMM into it. `down` stays cuBLAS. 16-bit only (strict fp32 lost to cuBLAS in the blind model). |
| temporal mix (`temporal.py`) | `TemporalBlock.sequence` from `RGLRU._terms` to `recurrent * gelu(gate_branch(normed))`: the gates, the start mask, the scan and the output gate | v2's `recurrence.gates` (2D, D), recurrence rows first, is the kernel's pre-activation layout (p_r, then p_i). v2's `gate_branch` is its own (D, D) Linear: its output is the kernel's branch (unit stride, contiguous). keep = 1 - (a start after the first step). `log_decay` (D,) is shared. The convolution and the three projections stay plain. Segments only (L > 1); `step` keeps the plain path. |
| heads (`head.py`) | `HeadGroup.forward` per head (`last(gelu(norm(first x)))`), `ShotTrajectoryHead.query`'s tail (`last(gelu(middle_norm(middle h)))`), and the encoder's `KindProjection` + `shared` (`shared(gelu(first_norm(first x)))`, before `shared_norm`) | v2's `HeadGroup.first` is one matrix for the group: head i is its row block i (a view). The encoder's queries are `shared(cat(ship, global))`: row-wise, so `cat(head(ship), head(global))`. 16-bit only. |

What the blind kernels compute differently from v2, and how the port keeps v2's function:
- **Norm epsilon under bf16 (heads, encoder): no difference.** v2's `norm` is
  `RMSNorm(eps=None)`, and under autocast a head's norm reads the bf16 output of its first
  projection. Measured: torch upcasts and uses float32's epsilon there too (an input of
  0.001 gives 0.945 for both dtypes, where bf16's epsilon would give 0.011), which is the
  blind head kernel's constant.
- **Autotuning (gated MLP).** The blind kernel autotunes over four tile configurations,
  which differ in BLOCK_K and so in accumulation order: the configuration a process picks
  by timing decides the bits. The port fixes one configuration (determinism).
- **Separate gate-branch projection (temporal mix).** The blind model computes the gate
  branch inside a fused projection and passes a strided view; v2's `gate_branch` is its own
  Linear, so the port passes its contiguous output. Same function.
- **The encoder's shared projection, kernel path only.** v2 projects the ships and the
  global token with one `shared` product over their concatenation; the head kernel runs it
  per kind (row-wise, the same function). The plain path is unchanged.

## Progress

- Step 1 (piece map): done.
- Code for all five kernels, their switches (`kernels.Switches`), the tests
  (`tests/sections/test_yemong_kernels.py`), the bf16 default (`TrainConfig.precision`
  and the CLI's `--precision`), the precision signals (`nonfinite_gradients` in the PPO
  update, `nonfinite_beliefs` in the trainer's row, both and the drift in the printed
  line) and the docs (`yemong.md` Kernels section, `learner.md`, `training.md`): written,
  not yet committed. GPU tests pending the lock.
- Commits: one per kernel, built with `/tmp/bnb-scratch-fast-yemong/stages.py k` (writes
  the intermediate `kernels/__init__.py`, `temporal.py` and test file for stage k from the
  final versions saved in `/tmp/bnb-scratch-fast-yemong/final/`).
- Measurements: `bench.yemong_speed` (`--plain`, `--kernels name...`) into
  `/tmp/bnb-scratch-fast-yemong/speed.jsonl`; training throughput and memory with
  `/tmp/bnb-scratch-fast-yemong/cell.py` (bench.cell's train cell with precision, switches
  and micro-batch rows).
