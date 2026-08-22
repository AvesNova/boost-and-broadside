# Memory optimization and measured tradeoffs

Training has to fit a fixed logical batch onto whatever card is available. `bnb train --vram`
makes that decision; this page is where its knobs come from, what each one costs, and what
has actually been measured on real hardware.

Three pools compete for VRAM, and each has its own knob:

| Pool | What | Grows with | Knob |
|---|---|---|---|
| **Update activations** (transient peak) | autograd graph in `evaluate_actions` backward | **depth**, micro-batch, T, D | SDPA fused kernel + `ModelConfig.grad_checkpoint` |
| **Rollout buffer** (persistent, resident) | pre-allocated obs + per-component arrays | batch (`num_envs`), T, K | bf16 / uint8 buffer storage |
| **Host arena** (pageable CPU RAM) | logical batch beyond the resident shard | logical tokens | `rollouts_per_update` |

The SDPA fix and the reduced-precision buffer are unconditional and always on. Gradient
checkpointing and microbatch size are what `--vram` reaches for when a launch does not
otherwise fit; both are numerically equivalent to leaving them alone. Nothing `--vram` can
do changes the experiment.

## Resolving a launch for a card (`--vram`)

The knobs below are not edited into a profile. `bnb train --vram` selects them, and the
three categories differ in what they promise, so they are never mixed:

| Tier | Knobs | Guarantee |
|---|---|---|
| 1: same mathematical objective | gradient checkpointing, microbatch tokens | Equivalent objective and update within floating-point tolerance; not bit-identical |
| 2: same nominal logical batch | rollout shard width and shard count at a fixed total | Same nominal tokens and optimizer-step count; different env-stream count, temporal correlation, and minibatch composition |
| 3: experiment change | total token budget, minibatch count, fleet size | Changes the optimization or the task |

`--vram` may move tier 1 and tier 2 only. A profile's logical batch is fixed, so a width is
valid only when it divides that batch exactly and stays minibatch-aligned; anything else is
rejected rather than silently rounded. For the `rl` profile the valid widths are 7776 (1
shard), 2592 (3), 864 (9), 288 (27), 96 (81), and 32 (243) -- there is no two-shard split,
so a preset that cannot afford 7776 proposes 2592 rather than inventing a width between.

### Policies

| Value | Behaviour |
|---|---|
| `auto` (default) | Use a cached measurement of *this* machine if one matches; otherwise keep the profile's derived sizing |
| `probe` | Measure this machine unless it is already cached, then store and use the result |
| `reprobe` | Measure again and replace the stored entry |
| `off` | Ignore cache and presets entirely |
| `8`/`16`/`24`/`32` | Apply that memory-tier preset row |

A numeric row is applied on whatever device was asked for, including `--device cpu`, which
is how a launch for a card that is not in this machine gets printed; the record notes that
the device is not an accelerator, where `auto` would have reported nothing to size instead.

Explicit `--num-envs`/`--microbatch-tokens` outrank all of it, and `--print-config` reports
every resolved value with its source (`profile`, `derived`, `vram-cache`, `vram-preset`, or
`cli`), which is also stored in every training checkpoint.

### Presets are starting points, measurements are not

Only the 8 GB row is measured. Probing that card directly in August 2026, running one
complete `rl` update in eager mode under Torch 2.13 / CUDA 13.0, accepted it on the first
candidate:

| candidate | allocated peak | reserved peak | of total | outcome |
|---|---:|---:|---:|---|
| 3904 envs, 25,000 microbatch tokens, no checkpointing | 6.00 GB | 7.88 GB | 8.19 GB | fit |

Reserved peak is 96% of the card, so this row has essentially no allocator headroom on
8 GB. It fits; a slightly larger shard or microbatch would not. The ladder below it
therefore reaches for gradient checkpointing before it narrows the shard.

That row was measured at eight entity tokens per environment, when `rl` was field-free and
resolved to 3904 envs. The profile now carries four fields and resolves to 2592, which is
31,104 resident entity tokens against the measured 31,232 -- within 0.4%, and the microbatch
is capped at 25,000 tokens either way. The number is therefore expected to carry, but it has
not been re-probed at the current width, and it is a measurement rather than a derivation.

The 16, 24, and 32 GB rows are linear extrapolations of the persistent-buffer and
rollout-peak figures in the production comparison below, and have never been run. Applying
*any* row, including the measured one, is reported as `provisional`, because a measurement
belongs to the card it was taken on. Only a probe of the current machine is reported as
`measured`.

### Probing

Each candidate runs one complete PPO update at production width in its own interpreter, so
an out-of-memory failure never leaves a fragmented allocator behind for the next attempt.
The ladder starts at the largest preset row the card's advertised memory could hold and
descends; below the smallest row it enables gradient checkpointing first, because that is
numerically exact and only costs time, and narrows the rollout shard only after that. The
first candidate that survives a full update wins.

The result lands in `.vram.json` beside the working tree, gitignored and recomputable rather
than an artifact. Each entry carries the question it answered, written out: GPU name, UUID,
MIG status, total memory, compute capability and SM count; the Torch/CUDA/cuDNN/Python
versions; the autocast dtype and compile mode; the network architecture; the arena the
tokens come from; and the token geometry. Change any of those and the entry stops matching,
so `auto` falls back to the profile's own sizing instead of reusing a measurement of a
different question.

The list is deliberately short of a whole profile. A learning rate or a reward weight cannot
move a byte, so a measurement of this card survives editing them, and `rl` and `bc` share
one entry because they differ in objective rather than in architecture. What does invalidate
an entry is anything that changes the shape of the work: token width, network size, the
logical batch. The stored hash is only the dictionary key; the identity beside it is what
says whether an entry still applies, and it is readable.

A cache file that cannot be read is an error naming `--vram reprobe` or `--vram off`, never
a silent resize. A reprobe reaches the file only after it has measured the card, so it
replaces the damaged entry instead of raising the same error and throwing the measurement
away.

A launch record claims only the tiers the launch actually moved, measured between the
profile's own derived sizing and what it runs at. The 8 GB row restates the shipped launch
exactly, so it claims nothing: tier 2 warns about a changed env-stream count and minibatch
composition, and a knob reset to the value it already held changes neither. A width or
microbatch chosen on the command line does claim its tier, since the cost is the same
whoever picked it. `proposed`, `applied`, and the per-value source map record who did.

`--compile` changes the reserved workspace, which is why compile mode is part of the cache
identity: a measurement taken under one mode does not answer for another. Probe with the
flags you intend to train with. `bnb train --profile rl --vram probe` uses the run's own
compile mode, so one command line probes and then trains against its own measurement.
Switching `--compile` afterwards is a cache miss, and `auto` says so instead of reusing the
wrong number.

## The knobs and what they cost

### SDPA fused-kernel fix (activation, free)

`TransformerBlock._attn` previously built the additive attention mask in `x.dtype`. Under bf16
autocast the `qkv` Linear emits bf16 while `nn.RMSNorm` keeps `x` fp32, so the mask came out fp32
on bf16 q/k/v. That dtype mismatch **disqualifies the flash / mem-efficient SDPA kernels** and
silently falls back to the math kernel, which materializes the full `(B, H, N, N)` score matrix:
slower, and a much larger activation peak. Building the mask in `q.dtype` restores the fused path.

- **Saving:** removes the `O(B·H·N²)` score-matrix materialization from the attention activation
  peak. No precision change (the mask values are unchanged; only their dtype).
- **Cost:** none. This is a latent bug fix rather than a trade-off, and applies regardless of the
  other two.
- Guarded by `tests/models/test_encoder.py::test_attn_mask_dtype_matches_query_under_autocast`.

### Reduced-precision buffer (batch axis)

The rollout buffer is the batch-axis hog, since every array scales with `num_envs`. Its read-once
leaf channels are downcast under two hard rules that keep this a pure memory change:

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

### Gradient checkpointing (`grad_checkpoint=True`, depth axis)

Recompute each Yemong block's activations in the backward pass (`torch.utils.checkpoint`,
`use_reentrant=False`) instead of storing them, so activation memory stops scaling with depth.

- **Saving:** activation becomes essentially flat with depth instead of growing ~1 GB/block. Depth
  stops being memory-bound and becomes SPS-bound instead.
- **Cost:** **~20–30% on the update phase** (rollout is no-grad and untouched, so end-to-end cost is
  that times the update-phase share of a step). Numerically **exact**: outputs and gradients match
  the non-checkpointed path (`tests/models/test_encoder.py::TestGradCheckpoint`). Composes with
  `torch.compile` in all modes; none of the compile partitioners rematerialize this on their own.
- Targets the **depth** axis.

### Allocator flag (no code)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trims reserved memory (~3% here; larger in
long multi-shape runs). Free, and recommended always-on.

## Host-backed logical rollout shards

`TrainConfig.rollouts_per_update` separates logical experience per PPO update from the
GPU-resident rollout width:

- Collection runs one fixed-width shard at a time without changing policy weights.
- Bootstrap values, GAE, and next-state labels are computed on GPU before the shard moves to
  pageable CPU RAM.
- Return percentiles use a deterministic, evenly spaced entity sample bounded by
  `return_quantile_samples`; advantage RMS remains an exact whole-logical-batch reduction.
- Lambda aggregates reload only team IDs, alive/actor masks, advantages, and returns, one shard
  at a time.
- PPO gathers one shard minibatch into pinned RAM, copies it on a dedicated CUDA stream, and
  slices microbatch views on GPU. The following shard minibatch is prefetched during compute.

The bulk host arena is pageable on purpose. Only the bounded minibatch staging tensors are
page-locked, which prevents a large logical batch from permanently removing the same amount of
RAM from the host pager.

The production profile uses a 12,000,000-token logical batch, a 6,000,000-token shard, and two
rollouts per update. The actual logical count is 11,993,088 after rounding `num_envs` down for
minibatch divisibility. Microbatch tokens derive from the shard width, so increasing the logical
batch does not raise activation memory.

### Measured production comparison

RTX 4070 Laptop 8GB, two backbone blocks, 8 ships, 128 rollout steps, 32 minibatches,
`microbatch_tokens=37500`, gradient checkpointing enabled, eager mode, one warmup and one
measured update in a fresh process. One PPO epoch was timed so the transfer/preprocessing cost
is not hidden by repeated compute.

| storage | logical tokens | shard tokens | shards | persistent MB | rollout peak MB | update peak MB | host MB | rollout ms | update ms (1ep) | ship-tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU resident | 5,996,544 | 5,996,544 | 1 | 1825.5 | 4142.9 | 3211.0 | 0.0 | 23179.8 | 32372.9 | 107,943 |
| host backed | 11,993,088 | 5,996,544 | 2 | 1825.2 | 3726.0 | 3211.5 | 2211.6 | 51375.7 | 67116.2 | 101,214 |

Doubling experience therefore adds effectively **zero persistent VRAM** (−0.2 MB measured);
the update allocated peak is unchanged and the rollout allocated peak is 417 MB lower. CPU
storage grows linearly, as intended: about 2.16 GiB for the 12M logical batch.

The one-epoch comparison is deliberately harsh and loses **6.2% ship-token throughput**.
Production uses four PPO epochs; extrapolating the separately measured rollout/update phases
gives 305.3 s for two GPU-resident 6M updates versus 319.8 s for one host-backed 12M update, a
**4.5% throughput cost** at equal experience. This is an inference from the measured phase times,
not a separately timed four-epoch run.

The single-shard 6M baseline emitted a recoverable allocator warning while requesting a
temporary 504 MiB block; the two-shard 12M run completed without one. Allocated peaks are the
reliable comparison here, because reserved-memory peaks depend on allocator history and
fragmentation.

Use shard width, not logical batch size, to tune rollout throughput. A matched 5M experiment
split into two 2.5M shards lost much more throughput because small rollout batches became
launch-bound; transfer and host preprocessing accounted for only about 2.5 seconds of that
run. Keep the rollout shard near the efficient 5–6M frontier on this GPU and raise
`rollouts_per_update` to grow total experience.

## Appendix: measured frontier, RTX 4070 Laptop 8GB

Measured in July 2026, before `--vram` existed, so the knobs appear here under the profile
constant names they had at the time. The relationships they establish are what the preset rows
and the probe ladder are built on.

One-factor-at-a-time sweep (`scripts/bench_mem.py`, extended with `--num-ships`, `--epochs`, and
`reduce-overhead` compile support) across `_MAX_TOKENS`, `_NUM_SHIPS`, `_NUM_MINIBATCHES`,
`_MICROBATCH_TOKENS`, `grad_checkpoint`, and `--compile`. A full grid would be ~4800 runs; since the
knobs are independent, each axis varies alone against a shared baseline (`blocks=2`, `_MAX_TOKENS=
5_000_000`, `_NUM_SHIPS=8`, `_NUM_MINIBATCHES=32`, `grad_checkpoint=True`, `compile=none`,
`num_envs` computed from the profile's real formula, 1 PPO epoch measured, so scale `update_ms` ×4
for the production per-update cost). 21 configs, ~35 min total.

**⚠ A tested temporary microbatch divisor of 1 (`_MAX_TOKENS=5_000_000 //
_NUM_MINIBATCHES // 1`, i.e. `microbatch_tokens=156250`) OOMed on this card even with
`grad_checkpoint=True`**, confirmed directly (`microbatch_d1` row below). Divisor **5**
(`microbatch_tokens=31250`, the previously-committed value) is the largest that fits and is used
as the baseline below.

### A1. `_MAX_TOKENS` → `num_envs` (ships=8, mb=32, microbatch=31250, gc=True, compile=none)

| tokens | envs | persistent MB | activation MB | reserved peak MB | rollout ms | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|---|---|---|
| 1M | 960 | 346 | 1076 | 1986 | 6856 | 4658 | 85,374 |
| 2M | 1952 | 647 | 1128 | 2808 | 9081 | 9492 | 107,623 |
| 3M | 2912 | 937 | 1141 | 3866 | 11610 | 14206 | 115,507 |
| 4M | 3904 | 1236 | 1158 | 5148 | 14263 | 18977 | 120,268 |
| 5M | 4864 | 1525 | 1171 | 6328 | 16308 | 23540 | 124,993 |

`persistent_mb` (buffer) scales linearly with `num_envs`, exactly as expected. `activation_mb`
stays flat (~1.1 GB), confirming it is governed by `microbatch_tokens` rather than batch size.
Ship-tokens/sec rises with diminishing returns as rollout's fixed per-step overhead amortizes over
more envs, so a bigger logical batch is free throughput headroom until VRAM runs out (~6.3 GB at 5M
here, of 7.62 GB usable).

### A2. `_NUM_SHIPS` (tokens=5M, mb=32, microbatch=31250, gc=True, compile=none)

| ships | envs | persistent MB | activation MB | reserved peak MB | ship-tok/s |
|---|---|---|---|---|---|
| 2 | 19520 | 1536 | 1564 | 6746 | 101,958 |
| 4 | 9760 | 1530 | 1155 | 6322 | 115,866 |
| 8 | 4864 | 1525 | 1171 | 6328 | 124,993 |
| 16 | 2432 | 1527 | 1199 | 6592 | 125,937 |

`ScaleConfig.num_envs` is inversely proportional to `num_ships` by design (docstring: "keeps
total ships-per-update constant across scales"), and that is confirmed here: `persistent_mb` and
ship-tok/s are both nearly invariant across this axis (buffer ±1%, throughput 102k–126k).
`_NUM_SHIPS` mostly just trades env count for ships/env at fixed total throughput. The only real
VRAM wrinkle is at the extreme (ships=2, 19520 envs), where activation rises ~34%, which is worth
a wider check before running that scale for real.

### A3. `_NUM_MINIBATCHES` (tokens=5M, ships=8, microbatch=31250, gc=True, compile=none)

| minibatches | activation MB | reserved peak MB | ship-tok/s |
|---|---|---|---|
| 4 | 1622 | 6944 | 124,765 |
| 8 | 1365 | 6622 | 124,751 |
| 16 | 1236 | 6460 | 124,641 |
| 32 | 1171 | 6328 | 124,993 |
| 64 | 953 | 6304 | 125,664 |

Clean, useful result: more minibatches gives monotonically lower activation VRAM (1622→953 MB,
−41% from 4→64) with **no throughput cost** (ship-tok/s flat within noise, 124.6k–125.7k). Raising
`_NUM_MINIBATCHES` is close to a free way to buy VRAM headroom, since total compute per update does
not change, only how it is chunked.

### A4. `_MICROBATCH_TOKENS` divisor (tokens=5M, ships=8, mb=32, gc=True, compile=none)

Per-minibatch token count at this baseline is fixed at 156,250 (`_MAX_TOKENS // _NUM_MINIBATCHES`);
the divisor sets how finely that gets split for the backward pass.

| divisor | microbatch_tokens | activation MB | reserved peak MB | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|---|
| 1 | 156250 | n/a | n/a | **OOM** | n/a |
| 2 | 78125 | 2767 | 6632 | 26484 | 116,127 |
| 3 | 52083 | 1883 | 6284 | 26250 | 116,757 |
| 4 | 39062 | 1423 | 6314 | 24828 | 121,060 |
| 5 | 31250 | 1171 | 6328 | 23540 | 124,993 |
| 6 | 26041 | 989 | 6344 | 23181 | 126,110 |

Smaller microbatches are both smaller **and faster** here (activation 2767→989 MB and update time
26484→23181 ms from divisor 2→6), because on this VRAM-constrained 8GB card finer chunking avoids
allocator/fragmentation pressure near the ceiling. This ordering may reverse on a GPU with
slack VRAM, where per-chunk kernel-launch overhead would dominate instead. Divisor 1 is the
measured OOM boundary for this sweep, confirming that the divisor had to be ≥2 on this card.

### A5. `grad_checkpoint` (tokens=5M, ships=8, mb=32, microbatch=31250, compile=none)

| grad_checkpoint | activation MB | reserved peak MB | update ms (1ep) | ship-tok/s |
|---|---|---|---|---|
| False | 2569 | 6818 | 19522 | 139,391 |
| True | 1171 | 6328 | 23540 | 124,993 |

Matches the documented tradeoff almost exactly: checkpointing cuts activation by **54%** for
**+20.6%** update time, against an estimate of ~20–30%. `grad_checkpoint=False` is faster when it
fits, so use it whenever VRAM allows and flip it on only when the batch and microbatch you actually
want will not fit otherwise.

### A6. `--compile` (tokens=5M, ships=8, mb=32, microbatch=31250, gc=True)

| compile | rollout ms | update ms (1ep) | reserved peak MB | ship-tok/s |
|---|---|---|---|---|
| none | 16308 | 23540 | 6328 | 124,993 |
| reduce-overhead | 15935 | 22169 | 6688 | 130,713 |
| default | 15882 | 22164 | 6688 | 130,916 |
| max-autotune | 15920 | 22185 | 6688 | 130,711 |

All three compiled modes land within noise of each other, about 6% faster than eager on both
rollout and update, with no measurable extra win from `default`/`max-autotune` over
`reduce-overhead` in this config, and compile adds ~360 MB reserved (workspace). One caveat: this
used only 1 warmup iteration before measuring, which may undersell `max-autotune`, whose autotuning
cache can need more warm calls to fully engage. `reduce-overhead` gets the same speedup for the
cheapest compile-time cost, so it is the reasonable default here, and it matches `bnb train`'s
default.

**Bottom line for this card before host-backed batches:** `grad_checkpoint=True` plus a microbatch
divisor ≥2 was required to fit a 5,000,000-token batch; divisor 5–6 and `_NUM_MINIBATCHES=32–64`
gave the best VRAM/speed balance. The host-backed design above supersedes the requirement that the
entire logical batch fit at once.

### Measuring the frontier yourself

Run `scripts/bench_mem.py` (one config per process) to measure any point on the frontier for your
production `d_model` / `microbatch_tokens`:

```bash
uv run python scripts/bench_mem.py --blocks <N> \
    --num-envs <full> --minibatches 32 --microbatch-tokens 37500 \
    --compile max-autotune --grad-checkpoint
```

`buffer_mb` is summed analytically (exact, linear in `num_envs`); `activation_mb` is
`update_peak - persistent` and is independent of `num_envs`, so it is representative even at a
reduced batch. Microbatch tokens derive from the shard width, so to grow the batch without growing
activation, hold the microbatch fixed while raising `num_envs`.

Earlier indicative numbers on the same card (36-env micro-batch, D=128) put activation at ~2.9 GB
(2 blocks) and ~5.0 GB (4 blocks, OOM past that on 8 GB), flattening to ~1.3 GB with
checkpointing, and the full-batch buffer around ~0.9 GB. Those were taken on the earlier fp16
buffer and before the SDPA fix, so both the buffer figure (fp16 to bf16, positions now fp32) and
the activation baseline (math-kernel fallback now removed) have shifted. Re-measure on the target
card before relying on absolutes.
