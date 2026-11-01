---
covers: []
---

# Reference

One page per package, generated at build time from the docstrings and signatures in the
code, so it always describes the code in the tree. Packages are listed in the order of the
dependency rule (see [Architecture](../explanation/architecture.md)): each imports only from
its own layer and the layers above it in this table. `bnb.controllers` and `bnb.models` share
a layer.

| Page | Package |
|---|---|
| [bnb.contracts](contracts.md) | types shared between sections, no logic |
| [bnb.game](game.md) | physics, perception, modes, the batch layout |
| [bnb.controllers](controllers.md) | scripted controllers, model adapters |
| [bnb.models](models.md) | the skeleton MLP, policy heads, Yemong |
| [bnb.learn](learn.md) | rewards, the skeleton PPO |
| [bnb.runtime](runtime.md) | lineup, rollout, training loop, realtime |
| [bnb.cli](cli.md) | the `bnb` command |
| [bench](bench.md) | the benchmark harness, outside the package |
