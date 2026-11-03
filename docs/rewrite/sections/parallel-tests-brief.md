# Brief: parallel CPU tests, and the quick tier on GitHub Actions

The suite runs one test at a time: the quick tier (`-m "not slow"`) takes about 3.3
minutes, and with CUDA hidden the full suite about 17. Two changes (author, 2026-11-03):
1. **Run the CPU tests in parallel** with pytest-xdist, on any machine.
2. **A GitHub Actions job on every push** runs the quick tier on CPU. The slow tier stays
   by hand, run when relevant.

## 1. Parallel tests

- **Add `pytest-xdist`** to the dev dependencies. Make parallel the documented default way
  to run the CPU tests, for example `uv run --no-sync pytest -n auto -m "not slow"`.
- **The GPU tests stay serial.** They share one GPU behind the benchmark lock
  (`bench.lock`), so workers would only queue on it. Choose one approach and document it:
  - run `-m "not gpu"` in parallel and `-m gpu` serially (a small `make`-style command or
    a script in `tools/` that runs both), or
  - a mechanism that keeps every `gpu` test on one worker (xdist's `--dist loadgroup`
    with an `xdist_group` mark added to `gpu` tests by a conftest hook).
- **Make the tests safe in parallel:**
  - temporary files only in `tmp_path`;
  - no shared paths (`runs/`, `replays/`, caches) written by two tests;
  - the guards in `tests/conftest.py` (CUDA initialisation, the global generator) still
    work per worker;
  - module and session fixtures behave per worker;
  - compile caches (Inductor and Triton) written concurrently do not corrupt one another.
    If they can, give each worker its own cache directory.
- **Check for order dependence:** run the parallel quick tier at least 5 times, with
  `-p randomly` or a shuffled order if available, and fix any test that fails in one
  order and passes in another (a flaky test). Report what you fixed.
- **Measure** the quick tier and the CUDA-hidden full suite, serial against parallel, on
  this machine, with the load average.

## 2. GitHub Actions

- **The repo:** `github.com/AvesNova/boost-and-broadside`. The working branch is `v2`.
  There is no `gh` CLI here.
- **The workflow:** `.github/workflows/quick-tests.yml`, on `push` to every branch and on
  `workflow_dispatch`. A newer push to a branch cancels its older run
  (`concurrency` with `cancel-in-progress`).
  - Ubuntu, Python 3.14, uv (the `astral-sh/setup-uv` action), with uv's cache.
  - **CPU torch:** `pyproject.toml` pins torch to the CUDA index (`pytorch-cu132`). The
    runner must install the CPU wheel of the same version instead, without changing
    developers' installs. Find a clean way: a uv index override or environment variable in
    the workflow, or an optional extra. Do not commit a second lock file unless there is
    no other way, and explain it if so.
  - Run `ruff check`, `ruff format --check`, then the quick tier in parallel with CUDA
    absent: `-m "not slow"`. `gpu` tests skip cleanly.
  - The tools extra and the docs build: include them if the quick tier's docs tests need
    them, as they do locally.
- **Verify it runs:** push the branch, then read the run's result through the public API,
  for example
  `curl -s https://api.github.com/repos/AvesNova/boost-and-broadside/actions/runs?branch=ci/parallel-tests`.
  If the repo is private and the API needs a token, say so in your report; don't try to
  find credentials. Fix the workflow until the run is green, and report its time.
- **Never** add secrets, deploy steps, or anything that publishes.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-parallel-tests`, branch `ci/parallel-tests`
  (from v2). Work only here. Scratch in `/tmp/bnb-scratch-parallel-tests`.
- **Do not read** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** the diagnostics agent is working on the GPU. Hold the benchmark lock for any
  GPU run (the wrapper `/tmp/bnb-scratch-fast-yemong/locked.py <command...>`); most of
  your work is CPU.
- **May change:** `pyproject.toml` (dev dependencies, pytest config), `tests/conftest.py`
  and test files (only for parallel safety), `.github/`, `tools/`, and the contributing
  docs (`docs/contributing/`: how to run the tests, the tiers, parallel runs, CI).
  `src/` only if a test exposes a real shared-state bug; then write it up.

## Done

- The parallel quick tier passes 5 times in a row; the CUDA-hidden full suite passes in
  parallel once; the GPU tests pass serially under the lock once.
- The Actions run on your branch is green, or you report exactly why it cannot be checked.
- Commit to `ci/parallel-tests` and push. Author Julian Kizanis <julianzia@gmail.com>;
  end every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Your final message, in half a page: the times before and after, the CI run's time and
  link, and the flaky tests you fixed.
