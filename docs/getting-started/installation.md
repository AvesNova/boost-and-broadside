---
covers: [pyproject.toml, .python-version]
---

# Installation

## Requirements

- **uv** manages the environment. The lock file, `uv.lock`, pins every dependency.
- **The versions** below come from `pyproject.toml`. uv installs the Python version if it
  is missing.
- **An NVIDIA GPU** with a driver that supports the CUDA version of the torch wheels
  (their index is named after it), for training and benchmarks. The rollout captures CUDA
  graphs and the `bnb` command defaults to `--device cuda`. The tests run on the CPU, apart
  from those marked `gpu`.

<!-- generated: versions -->

## Set up

From the repository root:

<!-- not-runnable: changes the environment -->
```bash
uv sync
```

`uv sync` makes the environment match the lock file exactly. It installs the project and
the default dependency groups, which include the tools that build these docs, so the strict
docs build runs in every test run.

## Run commands through uv

Run everything as `uv run --no-sync ...`. `--no-sync` uses the environment as it is,
so running a command never changes it.

<!-- runnable -->
```bash
uv run --no-sync python -c "import bnb, torch; print(torch.__version__)"
```

Next: [running the tests](running-tests.md).
