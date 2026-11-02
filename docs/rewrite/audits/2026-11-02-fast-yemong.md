# A faster Yemong: what the blind race found (2026-11-02)

## The two models

- **v2's Yemong** (`src/bnb/models/yemong/`, about 2,100 lines): plain PyTorch. It is
  compiled with `torch.compile`, and bf16 is a model option (`bfloat16_matmuls`).
- **The blind model** (branch `race/yemong-blind`, `src/bnb/models/swift/`, about 2,850
  lines): built from the spec and the fixtures alone, with speed as the goal. It passes all
  43 Yemong fixtures (float64 to 1e-9, float32 to 2e-4), so it computes the same function.
  It is plain PyTorch plus five Triton kernels (about 950 lines):

  | Kernel | Replaces | Buys (its own micro-benchmark) |
  |---|---|---|
  | `attention.py` | SDPA plus the rotary rotation, the concatenation and the transposes around it | 1.2–1.5× on the spatial sublayer; flash attention's 128-row tiles waste most of their work on 11 queries × 21 keys |
  | `gated.py` | the gated MLP's two GEMMs and `gelu(g) · u` (bf16 only) | about 10% on the sublayer; the wide pre-activations are never stored |
  | `temporal.py` | the temporal mix's gates, scan and output gate (training segments only; it loses at one step) | 4.5 → 4.0 ms on the sublayer |
  | `head.py` | the heads' linear–norm–gelu–linear (bf16 only) | 0.3–0.5 ms at V = 16 |
  | `scan.py` | the linear recurrence | the scan without Python loops |

  Both models are time-major over a segment. The structural difference outside the
  kernels is the recurrence: v2 runs a log-depth parallel scan in PyTorch (about seven
  passes over the state for 128 steps, with the intermediates kept for the backward pass);
  the blind model walks time once in `scan.py`.

## The speed-up

Both models ran the same benchmark back to back on a quiet machine (load average 0.8–2.1,
RTX 4070 Laptop). Training is `unroll` forward and backward over 128 steps; the rollout is
one `step` for 2048 views. Each cell is the best compiled mode, in ms.

| Cell | v2 | Blind, scan kernel only* | Blind | Speed-up |
|---|---:|---:|---:|---:|
| Training V=16, fp32 | 86.3 | ~84 | 73.7 | 1.17× |
| Training V=16, bf16 | 51.7 | ~44 | 32.7 | 1.58× |
| Training V=64, bf16 | 226 (6.0 GB) | ~196 | 141 (3.1 GB) | 1.60× |
| Training V=64, fp32 | out of memory | out of memory | 329 (recomputes its MLPs) | runs |
| Rollout, bf16 | 16.3 | ~17 | 12.6 | 1.29× |
| Rollout, fp32 | 28.8 | ~30 | 26.6 | 1.08× |

\* The blind agent's own first measurements, with SDPA attention and only the scan kernel,
taken on a possibly shared machine: approximate.

**Where the gain comes from (bf16 training):** about 15% with only the scan kernel (very
likely mostly the scan, not yet separated), and about 25% more from the other kernels.
Memory halves.

**What it could mean for training (estimated, not measured):** the update is about 93% of
training time. Its V = 16 micro-batch takes 108 ms in fp32 (the model about 86 ms, the rest
about 22 ms), and bf16 alone cuts the update by 31%, to about 75 ms (the model 52 ms).
With the blind model in bf16, the micro-batch would take about 33 + 23 = 56 ms:
- against bf16 today, the update about 1.3× faster and training about 1.3×;
- against today's fp32 default, the update about 1.9× faster and training about 1.8×.

The freed memory allows larger micro-batches, and may let the 26-ship cell fit.

## The maintenance burden

- **About 950 lines of Triton** (forward and backward for each kernel), harder to read and
  review than PyTorch. Each kernel has a switch, a plain-PyTorch twin (used on the CPU, in
  float64 and by the fixtures) and a test that the two agree.
- **Two paths per piece.** Every change to a kernelled piece must be made twice, in the
  plain path and the kernel, and kept equal by the test.
- **Toolchain risk.** It uses `torch.library.triton_op`, which is recent, so a PyTorch or
  Triton upgrade can break a kernel. The switches turn it off, at the plain path's speed.
- **CUDA only, and tuned for this GPU.** Tile sizes were chosen for an RTX 4070 Laptop;
  another GPU runs correctly but may want different tiles.

## Changing the model

- **Prototype on the plain path.** With the switches off, any change (width, heads, a new
  token, another activation) runs at the plain path's speed, which with the scan kernel
  is still a little faster than v2's Yemong today. Write or adapt a kernel only for a change you keep.
- **What a kernel handles without rewriting:** the number of ships, fields and zones (the
  attention pads its query and key counts to a power of two of at least 16), the width,
  the number of heads, the batch, the sequence length and the number of blocks.
- **What needs a kernel rewritten:**
  - another attention form: windowed or sparse attention, a new positional encoding, or a
    change to the rotary layout;
  - another MLP activation or gating;
  - a new temporal mix (other gates, another recurrence);
  - a head that is not linear–norm–gelu–linear.

  Each rewrite costs roughly a day's agent work with its equivalence test.
- **Large token counts:** the attention keeps one view's whole query and key set in one
  tile. That is fine up to a few dozen keys; at hundreds it needs a tiled kernel, or the
  SDPA path.

## Options

1. **The scan only:** v2's Yemong with the sequential scan kernel (about 110 lines). Up to
   about 15% on bf16 training, and less memory.
2. **Layout plus the two biggest kernels** (attention with rotary, fused gated MLP): most
   of the gain, about 400 lines of Triton.
3. **All of it:** the full 1.6×, and all five kernels to maintain.

Each option keeps v2's interfaces, passes the 43 fixtures and the learner tests, and is
measured on real training throughput and learning curves before it merges.
