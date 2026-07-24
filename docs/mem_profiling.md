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
