# Micro-batch check (2026-11-15)

Question: does `LearnerConfig.micro_batch_rows` (or `host_store`) change the update? rl-v20
(480 rows, device store, 5090) and its seed twin rl-v20-seed1 (96 rows, host store, 4070)
started from the same BC checkpoint. The twin finished 70–150 Elo weaker and logged a lower
"drift" (0.16–0.22 against 0.33–0.41).

## Verdict

The update does not depend on the micro-batch size, within rounding. The drift did: it was a
max over the first micro-batch only, so its sample grew with `micro_batch_rows`. That is now
fixed. The twin's Elo gap still needs another explanation: seed noise, the GPU, or the commit.

## What is computed per micro-batch (code read, `src/bnb/learn/ppo.py`, `losses.py`)

- Every loss term is a masked sum over the micro-batch's tokens divided by the **minibatch's**
  token count (`gradient`: `prepared["counts"][chosen].sum(0)`). The policy, entropy, critic,
  next-state, correction, enemy-action, density, shot and geometry terms all work this way,
  so the micro-batches' gradients add up to the minibatch's.
- There is no per-micro-batch advantage normalisation. The advantage is divided by the
  return scale, which updates once per update in `prepare`.
- The reported statistics (clip fraction, approximate KL, the divergence histogram, entropy
  buckets) and the diagnostics' sums use the same minibatch denominators and are summed over
  micro-batches.
- Gradient clipping and the optimizer step run once per minibatch, after accumulation.
  `.grad` accumulates in float32. bf16 is autocast on the matmuls only.
- The shot labels and their random draws are indexed by view, not by micro-batch.
- `host_store` is a data path only: pinned copies on a GPU, nothing on the CPU. The GPU suite
  already checks its equality (`test_learner_streaming`:
  `test_the_update_through_pinned_memory_equals_the_device_path`).
- **The one per-micro-batch quantity was `policy/ratio_drift/{max,mean}`.** It unrolled
  `first_micro_batch` views. At rl-v20's split (12,288 views in 32 minibatches of 384,
  10 ships per view) that is 48 views at 480 rows and 8 views at 96 rows. A max over 6x
  fewer tokens is smaller. The mean is unbiased and did not differ.

## Experiment (CPU, `micro_check.py`, `bf16_floor.py` here)

Setup: a small Yemong (`yemong_cases.model`) and one stored rollout of 16 environments
(24 views). Two minibatches of 12 views, 2 epochs, no warmup, no early stop. Micro-batches of
12, 6 and 2 views, measured against 12:

| | float32, 6 views | float32, 2 views | bf16 matmuls, 6 views | bf16 matmuls, 2 views |
|---|---|---|---|---|
| first minibatch gradient, 1 − cos | 2.8e-13 | 2.9e-13 | 4.6e-6 | 4.7e-6 |
| gradient \|Δg\|/\|g\| | 6.2e-7 | 6.3e-7 | 3.0e-3 | 3.1e-3 |
| loss terms, max \|Δ\| | 3.8e-6 | 3.8e-6 | 1.9e-6 | 3.8e-6 |
| parameters after 2 updates, \|Δp\|/\|step\| | 8.1e-6 | 1.0e-5 | 2.7e-2 | 3.9e-2 |

For scale, bf16 against float32 at the same split gives 1 − cos 9.2e-5 and
|Δg|/|g| 1.4e-2. That is about 4x the difference any split makes under bf16. The bf16
difference between splits is batch-shape-dependent rounding: it is smaller than bf16's own
error and has no direction. Every logged series (losses, value losses, KL, clip fraction,
diagnostics) matched to about 4e-5 relative or better, except the drift. With the old code,
after one update's steps on this rollout, `ratio_drift/max` read 0.582 at 12 views and 0.524 at
2 views. Its mean was 0.147 and 0.146.

## The runs' logs (`box-5090-backup/runs/*/metrics.jsonl`, median over updates)

| | ratio_drift/max | ratio_drift/mean | kl_first | clip_fraction |
|---|---|---|---|---|
| rl-v20 (48-view drift sample) | 0.237 | 0.00705 | 3.5e-5 | 0.074 |
| rl-v20-seed1 (8-view sample) | 0.161 | 0.00696 | 3.8e-5 | 0.071 |

The mean, the KL before the first step and the clip fraction agree. Only the max differs, as
its sample size predicts. The console's `drift` column is the max (`cli.py`). The drift is
|exp(log π_unroll − log π_rollout) − 1| over living own ships, before any step. It measures how
far the learner's `unroll` is from the rollout's `step` on the same weights, which is
bf16/step-unroll rounding. It does not measure how far the policy moved.

## Fix

`Learner.drift` now reads a fixed sample: the first minibatch's first `DRIFT_VIEWS` = 48
views, or the whole minibatch when it is smaller. It unrolls them in micro-batches, so memory
still follows `micro_batch_rows`. 48 is rl-v20's sample, so its series compare with later runs.
`first_micro_batch` became `drift_views`. The test
`test_the_drift_reads_the_same_sample_at_every_micro_batch_size` fails on the old code
(max differed by 0.36 at 1-view micro-batches) and passes on the new.
