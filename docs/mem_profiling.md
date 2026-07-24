# Memory-footprint reduction: measured savings & drawbacks

Goal: fit **more backbone blocks** and/or **larger batch (`_MAX_TOKENS`)** without a bigger
(24 GB) GPU. Three independent changes contribute, each targeting a different pool:

| Pool | What | Grows with | Knob |
|---|---|---|---|
| **Update activations** (transient peak) | autograd graph in `evaluate_actions` backward | **depth**, micro-batch, T, D | SDPA fused kernel + `ModelConfig.grad_checkpoint` |
| **Rollout buffer** (persistent, resident) | pre-allocated obs + per-component arrays | batch (`num_envs`), T, K | bf16 / uint8 buffer storage |

Run `scripts/bench_mem.py` (one config per process) to measure any point on the frontier for
your production `d_model` / `microbatch_tokens`.

## 1. SDPA fused-kernel fix (activation, free)

`TransformerBlock._attn` previously built the additive attention mask in `x.dtype`. Under bf16
autocast the `qkv` Linear emits bf16 while `nn.RMSNorm` keeps `x` fp32, so the mask came out fp32
on bf16 q/k/v. That dtype mismatch **disqualifies the flash / mem-efficient SDPA kernels** and
silently falls back to the math kernel, which materializes the full `(B, H, N, N)` score matrix —
slower and a much larger activation peak. Building the mask in `q.dtype` restores the fused path.

- **Saving:** removes the `O(B·H·N²)` score-matrix materialization from the attention activation
  peak. No precision change (the mask values are unchanged; only their dtype).
- **Cost:** none. This is a latent bug fix, not a trade-off — do it regardless of the other two.
- Guarded by `tests/models/test_encoder.py::test_attn_mask_dtype_matches_query_under_autocast`.

## 2. Reduced-precision buffer (batch axis)

The rollout buffer is the batch-axis hog — every array scales with `num_envs`. Its read-once leaf
channels are downcast under two hard rules that keep this a pure memory change:

- **bf16, never fp16.** bf16 keeps fp32's full exponent range, so a reward/value spike cannot
  silently overflow to `inf` the way fp16 (max 65504) can. The ~0.4% mantissa rounding on
  read-once channels is negligible. Applies to the per-component reward/value/advantage/return
  arrays and `expert_probs`.
- **Accumulators stay fp32.** Anything that sums or runs an EMA over stored data upcasts first:
  `compute_gae` seeds `lastgaelam` fp32 and stores only the downcast advantage; `AdvantageScaler`
  and `ReturnScaler` upcast before their reductions; `_precompute_lambda_aggregates` upcasts
  `returns` before the fp32-lambda einsum. bf16's ~0.4% resolution would otherwise let a small
  increment vanish under a large running value (the swamping failure), so no running statistic is
  ever bf16.

Per-channel observation storage (`_obs_storage_dtype`):

| channel | dtype | why |
|---|---|---|
| `pos` | **fp32** | Fourier position encoder needs sub-pixel accuracy; bf16's 8-bit mantissa resolves ~1 part in 256 (~4 px at a 1024 map, linearly worse as maps grow). fp16 would reach 2048 px but breaks the no-fp16 rule and still caps out on large maps. |
| other floats (`vel`, `att`, `ang_vel`, `health`, `power`, `cooldown`, `radius`) | bf16 | bounded, precision-tolerant; feature transforms upcast to fp32 on read |
| `team_id`, `previous_action` | uint8 | small non-negative indices (0–2, 0–6); read path upcasts via `.long()`/`.float()` |
| `alive` | bool | unchanged |

- **Saving:** roughly halves the buffer's float footprint (positions excepted). Linear in
  `num_envs`, so the absolute win scales with batch.
- **Cost:** none to SPS or activation; ~0.4% relative rounding on the bf16 leaf channels, with all
  accumulation still in fp32.
- Targets the **batch** axis. Guarded by `tests/train/test_buffer.py::TestStoragePrecision`.

## 3. Gradient checkpointing (`grad_checkpoint=True`, depth axis)

Recompute each Yemong block's activations in the backward pass (`torch.utils.checkpoint`,
`use_reentrant=False`) instead of storing them, so activation memory stops scaling with depth.

- **Saving:** activation becomes essentially flat with depth instead of growing ~1 GB/block. Depth
  stops being memory-bound (it becomes SPS-bound instead).
- **Cost:** **~20–30% on the update phase** (rollout is no-grad and untouched, so end-to-end cost =
  that × the update-phase share of a step). Numerically **exact** — outputs and gradients match the
  non-checkpointed path (`tests/models/test_encoder.py::TestGradCheckpoint`). Composes with
  `torch.compile` in all modes; none of the compile partitioners rematerialize this on their own.
- Targets the **depth** axis.

## Runtime flag (no code)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trims reserved memory (~3% here; larger in
  long multi-shape runs). Free — recommended always-on.

## Measuring the frontier

> Prior indicative numbers (8 GB RTX 4070, 36-env micro-batch, D=128) put activation at ~2.9 GB
> (2 blocks) / ~5.0 GB (4 blocks, OOM past that on 8 GB), flattening to ~1.3 GB with
> checkpointing, and the full-batch buffer around ~0.9 GB. **Those were measured on the earlier
> fp16 buffer and before the SDPA fix**, so both the buffer figure (fp16→bf16, positions now fp32)
> and the activation baseline (math-kernel fallback now removed) have shifted. Re-measure on the
> target card before relying on absolutes:

```bash
uv run --no-sync python scripts/bench_mem.py --blocks <N> \
    --num-envs <full> --minibatches 32 --microbatch-tokens 37500 \
    --compile max-autotune --grad-checkpoint
```

`buffer_mb` is summed analytically (exact, linear in `num_envs`); `activation_mb` is
`update_peak - persistent` and is independent of `num_envs`, so it is representative even at a
reduced batch. Note `_MICROBATCH_TOKENS` derives from `_MAX_TOKENS`: to grow batch without growing
activation, hold `_MICROBATCH_TOKENS` fixed while raising `num_envs`.

## Recommendation

Adopt all three — they stack (activation vs buffer are different pools; the SDPA fix and
checkpointing both cut activation but compose). Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `ModelConfig.grad_checkpoint=True`, and keep
the bf16/uint8 buffer. Budget ~20–30% update-phase SPS for the depth checkpointing buys you.

## Measured frontier: RTX 4070 Laptop 8GB (Jul 2026)

One-factor-at-a-time sweep (`scripts/bench_mem.py`, extended with `--num-ships`, `--epochs`, and
`reduce-overhead` compile support) across `_MAX_TOKENS`, `_NUM_SHIPS`, `_NUM_MINIBATCHES`,
`_MICROBATCH_TOKENS`, `grad_checkpoint`, and `--compile`. Full grid would be ~4800 runs; since the
knobs are independent, each axis varies alone against a shared baseline (`blocks=2`, `_MAX_TOKENS=
5_000_000`, `_NUM_SHIPS=8`, `_NUM_MINIBATCHES=32`, `grad_checkpoint=True`, `compile=none`,
`num_envs` computed from the real `rl.py` formula, 1 PPO epoch measured — production runs 4, so
scale `update_ms` ×4 for real per-update cost). 21 configs, ~35 min total.

**⚠ The microbatch divisor currently in `runs/rl.py` (uncommitted: `_MAX_TOKENS=5_000_000 //
_NUM_MINIBATCHES // 1`, i.e. `microbatch_tokens=156250`) OOMs on this card even with
`grad_checkpoint=True`** — confirmed directly (`microbatch_d1` row below). Divisor **5**
(`microbatch_tokens=31250`, the previously-committed value) is the largest that fits and is used
as the baseline below.

### 1. `_MAX_TOKENS` → `num_envs` (ships=8, mb=32, microbatch=31250, gc=True, compile=none)

| tokens | envs | persistent MB | activation MB | reserved peak MB | rollout ms | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|---|---|---|
| 1M | 960 | 346 | 1076 | 1986 | 6856 | 4658 | 85,374 |
| 2M | 1952 | 647 | 1128 | 2808 | 9081 | 9492 | 107,623 |
| 3M | 2912 | 937 | 1141 | 3866 | 11610 | 14206 | 115,507 |
| 4M | 3904 | 1236 | 1158 | 5148 | 14263 | 18977 | 120,268 |
| 5M | 4864 | 1525 | 1171 | 6328 | 16308 | 23540 | 124,993 |

`persistent_mb` (buffer) scales linearly with `num_envs`, exactly as expected. `activation_mb`
stays flat (~1.1 GB) — confirms it's governed by `microbatch_tokens`, not batch size.
Ship-tokens/sec rises with diminishing returns as rollout's fixed per-step overhead amortizes over
more envs — bigger `_MAX_TOKENS` is free throughput headroom until VRAM runs out (~6.3 GB at 5M
here, of 7.62 GB usable).

### 2. `_NUM_SHIPS` (tokens=5M, mb=32, microbatch=31250, gc=True, compile=none)

| ships | envs | persistent MB | activation MB | reserved peak MB | ship-tok/s |
|---|---|---|---|---|---|
| 2 | 19520 | 1536 | 1564 | 6746 | 101,958 |
| 4 | 9760 | 1530 | 1155 | 6322 | 115,866 |
| 8 | 4864 | 1525 | 1171 | 6328 | 124,993 |
| 16 | 2432 | 1527 | 1199 | 6592 | 125,937 |

`ScaleConfig.num_envs` is inversely proportional to `num_ships` by design (docstring: "keeps
total ships-per-update constant across scales") — confirmed: `persistent_mb` and ship-tok/s are
both nearly invariant across this axis (buffer ±1%, throughput 102k–126k). `_NUM_SHIPS` mostly
just trades env count for ships/env at fixed total throughput; the only real VRAM wrinkle is at
the extreme (ships=2, 19520 envs), where activation rises ~34% — worth a wider check before
running that scale for real.

### 3. `_NUM_MINIBATCHES` (tokens=5M, ships=8, microbatch=31250, gc=True, compile=none)

| minibatches | activation MB | reserved peak MB | ship-tok/s |
|---|---|---|---|
| 4 | 1622 | 6944 | 124,765 |
| 8 | 1365 | 6622 | 124,751 |
| 16 | 1236 | 6460 | 124,641 |
| 32 | 1171 | 6328 | 124,993 |
| 64 | 953 | 6304 | 125,664 |

Clean, useful result: more minibatches → monotonically lower activation VRAM (1622→953 MB, −41%
from 4→64) with **no throughput cost** (ship-tok/s flat within noise, 124.6k–125.7k). Raising
`_NUM_MINIBATCHES` is close to a free way to buy VRAM headroom — total compute per update doesn't
change, just how it's chunked.

### 4. `_MICROBATCH_TOKENS` divisor (tokens=5M, ships=8, mb=32, gc=True, compile=none)

Per-minibatch token count at this baseline is fixed at 156,250 (`_MAX_TOKENS // _NUM_MINIBATCHES`);
the divisor sets how finely that gets split for the backward pass.

| divisor | microbatch_tokens | activation MB | reserved peak MB | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|---|
| 1 | 156250 | — | — | **OOM** | — |
| 2 | 78125 | 2767 | 6632 | 26484 | 116,127 |
| 3 | 52083 | 1883 | 6284 | 26250 | 116,757 |
| 4 | 39062 | 1423 | 6314 | 24828 | 121,060 |
| 5 | 31250 | 1171 | 6328 | 23540 | 124,993 |
| 6 | 26041 | 989 | 6344 | 23181 | 126,110 |

Smaller microbatches are both smaller **and faster** here (activation 2767→989 MB and update time
26484→23181 ms from divisor 2→6) — on this VRAM-constrained 8GB card, finer chunking avoids
allocator/fragmentation pressure near the ceiling. This ordering likely reverses on a GPU with
slack VRAM (per-chunk kernel-launch overhead would then dominate instead). Divisor 1 (the
uncommitted `rl.py` value) is the exact OOM boundary — confirms the divisor must be ≥2 on this card.

### 5. `grad_checkpoint` (tokens=5M, ships=8, mb=32, microbatch=31250, compile=none)

| grad_checkpoint | activation MB | reserved peak MB | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|
| False | 2569 | 6818 | 19522 | 139,391 |
| True | 1171 | 6328 | 23540 | 124,993 |

Matches the documented tradeoff almost exactly: checkpointing cuts activation by **54%** for
**+20.6%** update time (doc estimated ~20–30%). `grad_checkpoint=False` is faster when it fits —
use it whenever VRAM allows; flip it on only when the batch/microbatch you actually want doesn't
fit otherwise.

### 6. `--compile` (tokens=5M, ships=8, mb=32, microbatch=31250, gc=True)

| compile | rollout ms | update ms (1ep) | reserved peak MB | ship-tok/s |
|---|---|---|---|---|
| none | 16308 | 23540 | 6328 | 124,993 |
| reduce-overhead | 15935 | 22169 | 6688 | 130,713 |
| default | 15882 | 22164 | 6688 | 130,916 |
| max-autotune | 15920 | 22185 | 6688 | 130,711 |

All three compiled modes land within noise of each other (~6% faster than eager, both rollout and
update) — no measurable extra win from `default`/`max-autotune` over `reduce-overhead` in this
config, and compile adds ~360 MB reserved (workspace). Caveat: this used only 1 warmup iteration
before measuring, which may undersell `max-autotune` — its autotuning cache can need more warm
calls to fully engage. `reduce-overhead` gets the same speedup for the cheapest compile-time cost,
so it's the reasonable default here (matches `main.py`'s existing default).

**Bottom line for this card:** `grad_checkpoint=True` + microbatch divisor ≥2 is required just to
fit `_MAX_TOKENS=5_000_000` at all; divisor 5–6 and `_NUM_MINIBATCHES=32–64` give the best VRAM/
speed balance found. The uncommitted divisor-1 change in `runs/rl.py` will OOM as-is on this GPU.
