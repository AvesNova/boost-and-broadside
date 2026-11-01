---
covers: []
---

# Runs and checkpoints (in progress)

<!-- generated: section-status -->

Today a training run keeps nothing on disk: `bnb` prints one line per update.

This section will cover the run directory and its record, checkpoints that carry the
settings they were made with (so models of different sizes load side by side), resuming
and forking a run, a warm start from a policy file, `bnb runs`, and a memory check before
the first update.
