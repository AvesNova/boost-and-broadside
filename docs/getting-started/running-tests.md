---
covers: [pyproject.toml, tests/conftest.py]
reviewed: 2026-11-01
---

# Running the tests

<!-- not-runnable: runs the whole suite -->
```bash
uv run --no-sync pytest                    # everything; the gpu tests need CUDA
uv run --no-sync pytest -m "not gpu"       # everything that runs on the CPU
uv run --no-sync pytest -m "not slow"      # the fast tests
uv run --no-sync ruff check .              # the linter, configured in pyproject.toml
```

## Markers

Two markers, declared in `pyproject.toml`:

- `gpu`: the test needs a CUDA device. These tests also skip when none is available.
- `slow`: the test takes more than a few seconds.

## Layout

| Directory | What it tests |
|---|---|
| `tests/contract` | every implementation of every contract: shapes, dtypes, the import direction, ONNX export |
| `tests/reference` | sections against goldens from float64 references, through adapters (see [Reference fixtures](../guides/reference-fixtures.md)) |
| `tests/sections` | each section's own unit and property tests |
| `tests/loop` | the training loop end to end: the lineup, toy learning |
| `tests/docs` | these docs: the strict build, code references, examples, staleness |

One test file runs in a few seconds:

<!-- runnable -->
```bash
uv run --no-sync pytest tests/contract/test_imports.py -q
```

The docs build test needs the `docs` dependency group, which `uv sync` installs by
default; without it, that test skips and says so.

Next: [a first training run](first-training-run.md).
