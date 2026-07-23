# Memory-footprint reduction: measured savings & drawbacks

Goal: fit **more backbone blocks** and/or **larger batch (`_MAX_TOKENS`)** without a bigger
(24 GB) GPU. Two knobs were added and profiled; this documents what each buys.

All numbers below were measured on an 8 GB RTX 4070 Laptop with `scripts/bench_mem.py` at the
production micro-batch (`microbatch_tokens=37500` → 36 envs/micro-batch), `d_model=128`,
`num_steps=128`. VRAM peak splits into two pools that the two knobs target independently:

| Pool | What | Grows with | Knob |
|---|---|---|---|
| **Rollout buffer** (persistent, resident) | pre-allocated obs + per-component arrays | batch (`num_envs`), T, K | `buffer` fp16/int8 storage |
| **Update activations** (transient peak) | autograd graph in `evaluate_actions` backward | **depth**, micro-batch, T, D | `ModelConfig.grad_checkpoint` |

## Measured results (36-env micro-batch, D=128)

| config | compile | blocks | buffer MB | activation MB | update-peak MB | update ms |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 2 | 11.4 | 2911 | 2984 | 739 |
| baseline | none | 4 | 11.4 | 5005 | 5106 | 1155 |
| baseline | none | 6 | — | **OOM** | — | — |
| buffer-dtypes | none | 2 | **5.8** | 2906 | 2972 | 722 |
| grad_checkpoint | none | 2 | 11.4 | **1291** | 1363 | 919 |
| grad_checkpoint | none | 4 | 11.4 | **1321** | 1423 | 1448 |
| grad_checkpoint | none | 6 | 11.4 | **1354** | 1485 | 2019 |
| grad_checkpoint | none | 8 | 11.4 | **1388** | 1548 | 2596 |
| baseline | max-autotune | 2 | 5.8 | 2906 | 2972 | 632 |
| combined | max-autotune | 2 | 5.8 | **1282** | 1349 | 770 |

### Buffer dtype shrink (fp16 floats + int8 obs indices)
- **Saving:** halves the buffer (11.4 → 5.8 MB here). Linear in `num_envs`, so at the full
  config (`num_envs≈5856`) it is **~1.85 GB → ~0.95 GB, ≈0.9 GB saved**.
- **Cost:** none — activation and SPS unchanged. Advantages carry ~3e-4 relative rounding
  (fp16), negligible for PPO; GAE and scaler reductions still run in fp32.
- Targets the **batch** axis.

### Gradient checkpointing (`grad_checkpoint=True`)
- **Saving:** activation drops **56% at 2 blocks, 74% at 4 blocks, and is essentially flat with
  depth** (1291 → 1388 MB from 2 → 8 blocks; baseline grows ~1.05 GB/block and OOMs past 4 on
  8 GB). Depth stops being memory-bound.
- **Cost:** **+19–24% on the update phase** compiled (+24–30% eager). Rollout is untouched
  (no-grad), so end-to-end cost = that × the update-phase share of a step. Numerically exact —
  outputs and gradients match the non-checkpointed path (see `tests/models/test_encoder.py::TestGradCheckpoint`).
- Composes with `torch.compile` in **all three modes** (none/default/max-autotune, incl. CUDA
  graphs). Notably, none of the compile partitioners rematerialize this on their own — the
  manual checkpoint is required.
- Targets the **depth** axis.

### Runtime flags (no code)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: trimmed reserved 1522 → 1472 MB here
  (~3%); larger in long multi-shape runs. Free — recommended always-on.

## Applying to the 24 GB card

- **More blocks** is gated by activation. With `grad_checkpoint` activation is ~1.3 GB flat, so
  depth becomes limited by **SPS (compute), not VRAM**. Add blocks freely; watch throughput.
- **Bigger batch** is gated by the buffer. `buffer` fp16 halves it. Note `_MICROBATCH_TOKENS` is
  derived from `_MAX_TOKENS`, so raising `_MAX_TOKENS` also raises the micro-batch (→ more
  activation). **Hold `_MICROBATCH_TOKENS` fixed** to grow batch without growing activation.
- The relative savings above transfer across GPUs; absolute frontier depends on your production
  `d_model` and `microbatch_tokens`. Confirm on the rental with one run at the real batch:

  ```bash
  uv run --no-sync python scripts/bench_mem.py --blocks <N> \
      --num-envs <full> --minibatches 32 --microbatch-tokens 37500 \
      --compile max-autotune --grad-checkpoint
  ```

## Recommendation

Adopt **both** (they stack: buffer=persistent, checkpoint=transient). Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `ModelConfig.grad_checkpoint=True`, and keep
the fp16/int8 buffer. Budget ~20% update-phase SPS for the depth you gain.
