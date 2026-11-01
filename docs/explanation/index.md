---
covers: []
---

# Explanation

How the system works, and why it is built that way.

- [Architecture](architecture.md): the layers and the dependency rule, the split between
  pure sections and the stateful runtime, compiled stages and CUDA graphs.
- [Contracts](contracts.md): the types between sections, with the observation's columns.
- The game: [physics](game/physics.md), [perception and fog](game/perception.md),
  [modes](game/modes.md), [rewards](game/rewards.md) and
  [mixed batches](game/mixed-batches.md).
- [The model (Yemong)](yemong.md): its belief, tokens, network, heads and carry.
- [The training loop](training-loop.md): the rollout, the lineup and the skeleton learner.
- [Roadmap](roadmap.md): the sections of the rewrite and their status.

Sections in progress have a short page each under *In progress*, saying what they will
cover.
