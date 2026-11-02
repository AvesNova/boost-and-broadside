# Boost and Broadside (v2, in progress)

A game in which two teams of ships fly, fight and hold zones, and the stack that trains
agents to play it. Version 2 is a from-scratch PyTorch rewrite, built for batched training
on one GPU: the game is pure functions of tensors, stepped in compiled stages and CUDA
graphs.

- The previous code is at tag `v1-torch-final`.
- Which sections are built, and which are in progress: the
  [roadmap](docs/explanation/roadmap.md), rendered from `docs/roadmap.yml`.

## Quickstart

<!-- not-runnable: changes the environment; the training run and the benchmark need CUDA -->
```bash
uv sync                                       # the environment, with the docs tools
uv run --no-sync pytest -m "not gpu"          # the tests that run on the CPU
uv run --no-sync bnb train --preset mixed           # train the skeleton model (needs CUDA)
uv run --no-sync python -m bench.run          # the skeleton benchmark (needs CUDA)
uv run --no-sync python -m bnb.play           # watch or fly a match (needs OpenGL)
uv run --no-sync mkdocs serve                 # these docs, at http://127.0.0.1:8000
```

The Python and torch versions and the GPU requirement are on the
[installation page](docs/getting-started/installation.md).

## Documentation

The documentation is a MkDocs site built from [`docs/`](docs/index.md):

- [Getting started](docs/getting-started/index.md): installation, tests, a first training
  run, a benchmark.
- [Guides](docs/guides/index.md): training, benchmarking, the reference fixtures, adding a
  mode, playing and spectating.
- [Explanation](docs/explanation/index.md): the architecture, the contracts, the game, the
  model, the training loop, the roadmap and a [glossary](docs/explanation/glossary.md).
- Reference: generated from the code by `mkdocs build`.
- [Writing docs](docs/contributing/writing-docs.md): how the docs are written and checked.

The design records of the rewrite are in [`docs/rewrite/`](docs/rewrite/scaffold.md),
outside the site.
