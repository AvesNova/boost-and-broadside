---
covers: [pyproject.toml, .python-version]
---

# Installation

## Requirements

- **uv** manages the environment. The lock file, `uv.lock`, pins every dependency.
- **Python 3.14.** `pyproject.toml` requires `>=3.14,<3.15`, and `.python-version` names
  3.14; uv installs it if it is missing.
- **torch 2.15.0**, from the PyTorch CUDA 13.2 wheel index (`pytorch-cu132` in
  `pyproject.toml`).
- **An NVIDIA GPU** with a driver that supports CUDA 13.2, for training and benchmarks.
  The rollout captures CUDA graphs and the `bnb` command defaults to `--device cuda`. The
  tests run on the CPU, apart from those marked `gpu`.

## Set up

From the repository root:

<!-- not-runnable: changes the environment -->
```bash
uv sync                  # the project and its development tools
uv sync --extra docs     # the same, plus MkDocs for building these docs
```

`uv sync` makes the environment match the lock file exactly, so a plain `uv sync` removes
the docs packages again. Use `--extra docs` whenever you work on the docs.

## Run commands through uv

Run everything as `uv run --no-sync ...`. `--no-sync` uses the environment as it is,
so running a command never changes it.

<!-- runnable -->
```bash
uv run --no-sync python -c "import bnb, torch; print(torch.__version__)"
```

Next: [running the tests](running-tests.md).
