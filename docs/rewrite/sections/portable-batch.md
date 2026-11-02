# Portable batches: decisions (2026-11-02)

Status: decided, not yet specified. It runs after the fast-Yemong kernels merge, beside
update profiling phase 2 (both touch the learner), before the cleanup pass.

## Why

The publication run (about 500M steps) will be on a rented GPU with far more memory than
the 8 GB development laptop. Moving between them must not mean retuning (author,
2026-11-02).

## Decided

- **Hyperparameters live on the logical batch:** environments × segment length × shards
  per update, and the learner's epochs and minibatches. A config means the same training
  on any GPU.
- **Memory knobs change only memory and speed, never the maths:** `micro_batch_rows`
  (gradient accumulation), shards (segments collected one after another),
  `host_store` (segments in CPU memory). They are already overridable on resume.
- **The split is chosen automatically:** from the memory probe, the run picks the largest
  micro-batch and the fewest shards that fit the device (an explicit setting still wins),
  and records the choice in `run.json`.
- **A test pins the equivalence:** two splits of one logical batch give the same update,
  bitwise where the order of sums allows, otherwise within float rounding of the
  reductions.
- **The kernels autotune per GPU** (told to the fast-Yemong agent): nothing tuned to the
  laptop's RTX 4070.

## To settle in the spec

- Whether more shards (consecutive segments of the same environments) count as the same
  logical batch as more environments. The data differs: shards reuse fewer environments
  over more time. Measure whether learning differs before treating them as
  interchangeable; until then, environments × length is fixed per config and only the
  memory knobs move.
