---
covers: []
---

# The learner (in progress)

<!-- generated: section-status -->

Until it merges, training uses the skeleton's plain PPO
(see [The training loop](../training-loop.md)).

It will cover:

- PPO over Yemong with v1's math: per-level returns, each level with its own discount and
  lambda, the categorical critic, the return scale and the auxiliary losses;
- the labels, including shot labels at every age of each flight, re-simulated with the
  game's own launch, flight and hit code;
- streaming the rollout through pinned CPU memory;
- several games in one update.

Its equivalence fixtures are already in `tests/reference/learner`.
