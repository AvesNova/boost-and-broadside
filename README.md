# Boost and Broadside (v2, in progress)

A game in which two teams of ships fly, fight and hold zones, and the stack that trains
agents to play it. Version 2 is a from-scratch PyTorch rewrite, built for batched training
on one GPU: the game is pure functions of tensors, stepped in compiled stages and CUDA
graphs.

- The previous code is at tag `v1-torch-final`.

## Status

Merged: the contracts and walking skeleton (section 0), the game (1), modes (3), rewards
(3b), mixed batches (3c) and the model, Yemong (5). Training still runs the skeleton's MLP
and plain PPO; the learner (6), the ship card (1b), the scripted agent (4), behaviour
cloning (7), the league (8), diagnostics (9) and the play shell (10) are specified or
planned. The [roadmap](docs/explanation/roadmap.md) is kept current from
`docs/roadmap.yml`.

## Quickstart

```
uv sync --extra docs                          # the environment, with the docs tools
uv run --no-sync pytest -m "not gpu"          # the tests that run on the CPU
uv run --no-sync bnb --preset mixed           # train the skeleton model (needs CUDA)
uv run --no-sync python -m bench.run          # the skeleton benchmark, about five minutes
uv run --no-sync mkdocs serve                 # these docs, at http://127.0.0.1:8000
```

Python 3.14 and torch 2.15.0 (CUDA 13.2 wheels) come from the lock file. Training and
benchmarks need an NVIDIA GPU; the tests run on the CPU apart from those marked `gpu`.

## Documentation

The documentation is a MkDocs site built from [`docs/`](docs/index.md):

- [Getting started](docs/getting-started/index.md): installation, tests, a first training
  run, a benchmark.
- [Guides](docs/guides/index.md): training, benchmarking, the reference fixtures, adding a
  mode.
- [Explanation](docs/explanation/index.md): the architecture, the contracts, the game, the
  model, the training loop and the [roadmap](docs/explanation/roadmap.md).
- Reference: generated from the code by `mkdocs build`.
- [Writing docs](docs/contributing/writing-docs.md): how the docs are written and checked.

The design records of the rewrite are in [`docs/rewrite/`](docs/rewrite/scaffold.md),
outside the site.
