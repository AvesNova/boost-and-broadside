---
covers: []
---

# Boost and Broadside

Boost and Broadside is a game in which two teams of ships fly, fight and hold zones, and
the stack that trains agents to play it. This is version 2: a rewrite in PyTorch, built
for batched training on one GPU. The game is pure functions of tensors: a batch of
environments steps together in compiled stages, and the same functions run one match on
a CPU for interactive play. The previous code is at the git tag `v1-torch-final`.

The rewrite is built in sections. Their status, from the [roadmap](explanation/roadmap.md):

<!-- generated: roadmap-summary -->

## Where to start

| If you want to | Read |
|---|---|
| install the project and run something | [Getting started](getting-started/index.md) |
| run training, benchmark, or work with the fixtures | [Guides](guides/index.md) |
| understand how the system works and why | [Explanation](explanation/index.md) |
| look up a function, class or constant | [Reference](reference/index.md) |
| look up a term (carry, view, pytree, seat, ...) | [Glossary](explanation/glossary.md) |
| change these docs | [Writing docs](contributing/writing-docs.md) |

The packages and their layers are tabled in [Architecture](explanation/architecture.md#layers-and-the-dependency-rule).
