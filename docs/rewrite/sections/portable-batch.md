# Portable batches: decisions (2026-11-02)

Status: decided, not yet specified. It runs after the fast-Yemong kernels merge, beside
update profiling phase 2 (both touch the learner), before the cleanup pass.

## Why

The publication run (about 500M steps) will be on a rented GPU with far more memory than
the 8 GB development laptop. Moving between them must not mean retuning (author,
2026-11-02).

## Decided

- **Hyperparameters live on the logical batch:** environments, segment length and shards
  per update, and the learner's epochs and minibatches. A config means the same training
  on any GPU.
- **Shards are a hyperparameter, not a memory knob (author, 2026-11-02):** they are the
  same environments continuing over consecutive segments, so they change the batch's
  diversity (fewer independent matches, correlated samples), the advantages (each segment
  boundary is a bootstrap cut) and the per-update statistics (the return scale's mix).
  More shards is a different batch from more environments.
- **Memory knobs change only memory and speed, never the maths:** `micro_batch_rows`
  (gradient accumulation) and `host_store` (segments in CPU memory). They are already
  overridable on resume.
- **The split is chosen automatically:** from the memory probe, the run picks the largest
  micro-batch that fits and whether the segments live on the host (an explicit setting
  still wins), and records the choice in `run.json`. If a config cannot fit even then, it
  refuses with the probe's numbers instead of changing the batch.
- **A test pins the equivalence:** two splits of one logical batch give the same update,
  bitwise where the order of sums allows, otherwise within float rounding of the
  reductions.
- **The kernels autotune per GPU** (told to the fast-Yemong agent): nothing tuned to the
  laptop's RTX 4070.
