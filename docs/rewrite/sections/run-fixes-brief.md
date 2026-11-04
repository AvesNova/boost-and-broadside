# Brief: fixes from the first BC run

The first full BC run (2026-11-04, 100M steps, W&B streaming) turned up five problems. Fix
each, with a test, and update the docs that describe the behaviour.

1. **BC runs write no training checkpoints.** `run.json` shows `checkpoint_every: None`
   and no checkpoints after 7.5 hours, so a crash would have lost the run. A BC run should
   write training checkpoints on the usual cadence (`bnb.runs.records.retained`) and
   resume from them with `--resume`, with the same integrity rule as PPO (exact restore;
   continuation within rounding, `tests/runs/test_resume.py`). That includes the stop
   rule's window and the BC budget. If a resumable BC run turns out to be a large change,
   stop and report what it needs before building it.
2. **The `entropy_collapse` alert fired on every BC update.** The rule is for RL; in BC a
   falling student entropy is the goal. Make the trigger rules know the objective and
   keep RL-only rules quiet in BC (check every one of the 21 rules for the same problem).
3. **`shutdown can only be called once`** is printed after `run ... finished` at the end
   of a `--wandb` run. Find which shutdown runs twice and fix it.
4. **`bnb runs dashboard` prints links that 404.** `wandb-workspaces` builds the view URL
   without the app's path prefix: on the host `forge.coreweave.com` the browser URL is
   `https://forge.coreweave.com/wandb/<entity>/<project>?nw=<id>` (the run's own URL,
   which wandb prints correctly, shows the prefix). Print the working link, derived from
   the configured base URL, not hard-coded to this host.
5. **Dashboards:**
   - **Dead panels:** a test that every panel pattern in `bnb.runtime.dashboards` matches
     at least one series that a short CPU training run logs, PPO and BC; RL-only panels
     are exempt in BC. A pattern that matches nothing in either fails, naming the panel.
   - **The BC views** leave out RL-only panels and sections (ratings, the league, PPO's
     KL and clip fraction).
   - **Save at start:** a `--wandb` run saves its headline and curated views once its
     first record is logged and prints both links; a failure there warns and the run goes
     on. `wandb-workspaces` is in the `tools` extra: without it, warn once.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-run-fixes`, branch `fix/first-run` (from v2).
  Work only there. Scratch in `/tmp/bnb-scratch-run-fixes`. Do not read other directories
  under `/home/vizia/avesnova`.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **CPU only.** A training run holds the GPU for the next ~4 hours: run everything with
  `CUDA_VISIBLE_DEVICES=""`. Do not take the benchmark lock. GPU tests are run later.
- **W&B:** only offline mode (`WANDB_MODE=offline`) in tests and checks; never create runs
  or views on the server. Credentials are in `~/.netrc`; never print them.

## Done

- The quick tier (`-m "not slow"`) and the slow CPU tests you touched pass with CUDA
  hidden; `ruff check` and `ruff format --check` are clean.
- Commit to `fix/first-run` and push. Author Julian Kizanis <julianzia@gmail.com>; end
  every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message, half a page: each fix and its commit, the tests added, and anything left
  for the GPU.
