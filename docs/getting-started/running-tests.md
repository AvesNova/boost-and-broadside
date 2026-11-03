---
covers: [pyproject.toml, tests/conftest.py]
reviewed: 2026-11-03 09:45
---

# Running the tests

<!-- not-runnable: runs the whole suite -->
```bash
uv run --no-sync pytest -m "not slow"      # the quick tier: what to run while working
uv run --no-sync pytest                    # the full tier: at the end of a task and at a merge
uv run --no-sync pytest -m "not gpu"       # everything that runs without a GPU
uv run --no-sync ruff check .              # the linter, configured in pyproject.toml
```

The suite also runs `ruff format --check` and `ruff check`.

## Markers

Two markers, declared in `pyproject.toml`:

- `gpu`: the test can touch a GPU: a CUDA device, or an OpenGL 3.3 context (EGL) for the
  play shell's renderer. Without CUDA, every CUDA test skips (`tests/conftest.py`); the
  play shell's tests skip when they cannot open a context.
- `slow`: the test takes more than 10 s on a quiet machine. Where a smaller case of it is
  fast, that case stays in the quick tier.

## Shared state

The GPU and torch's default generator are shared by every test in the process, and the
GPU by every process on the machine. `tests/conftest.py` keeps them apart:

- a `gpu` test holds the benchmark lock (`bench.lock.bench_lock`) while it runs, so another
  process on the GPU cannot make it run out of memory;
- a test without `gpu` that initialises CUDA fails, so `-m "not gpu"` never does;
- a test that leaves the default CPU generator changed fails. Draw from a local
  `torch.Generator`; where the code under test draws from the default generator (a
  model's initialisation, the rollout), use the `global_generator` fixture, which seeds it
  and restores it afterwards;
- `fresh_compiler` empties dynamo's caches for a test whose compiles must not depend on
  the tests before it.

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
