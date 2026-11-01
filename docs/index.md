---
covers: [src/bnb/]
---

# Boost and Broadside

Boost and Broadside is a game in which two teams of ships fly, fight and hold zones, and
the stack that trains agents to play it. This is version 2: a rewrite in PyTorch, built
for batched training on one GPU. The game is pure functions of tensors: a batch of
environments steps together in compiled stages, and the same functions run one match on
a CPU for interactive play.

The rewrite is built in sections. The game, its modes and rewards, mixed batches and the
model (Yemong) are merged. The learner, the scripted agent, behaviour cloning, the league,
diagnostics and the play shell are specified or planned; the [roadmap](explanation/roadmap.md)
lists each with its status. The previous code is at the git tag `v1-torch-final`.

## Where to start

| If you want to | Read |
|---|---|
| install the project and run something | [Getting started](getting-started/index.md) |
| run training, benchmark, or work with the fixtures | [Guides](guides/index.md) |
| understand how the system works and why | [Explanation](explanation/index.md) |
| look up a function, class or constant | [Reference](reference/index.md) |
| change these docs | [Writing docs](contributing/writing-docs.md) |

## The packages

| Package | What it holds |
|---|---|
| `bnb.contracts` | the types shared between sections, with no logic |
| `bnb.game` | physics, perception and modes, as pure functions |
| `bnb.controllers`, `bnb.models` | scripted controllers; models and their heads |
| `bnb.learn` | rewards and learners |
| `bnb.runtime` | the only stateful code: rollout, compiled stages, CUDA graphs, training loop |
| `bench` | the benchmark harness, outside the package |
| `tests/reference` | equivalence fixtures: goldens from float64 references, run through adapters |
