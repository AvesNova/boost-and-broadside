# Brief: keep every update's policies; tournaments over mixed architectures

Two decided changes (author, 2026-11-06; `docs/rewrite/decisions.md`, "Round four"), both
needed before the next experiments. CPU work only: the GPU is in use.

## 1. Every update's policies are kept (for the next RL run)

"For now, let's save all checkpoints", so EMA decays can be compared afterwards.

- After **every** update, write three small policy files (the format `bnb.runs.checkpoint`
  writes for `policy/`): the live weights, the EMA and the uniform average (when it has
  started), under the run directory, for example `policies/step_<steps>/{live,
  exponential_average, uniform_average}/`. Never pruned. About 10 MB each.
- **Training states** (the large resumable checkpoints): keep only the 2 newest, at the
  existing interval; and one at the run's end that holds both averages.
- **The league's snapshot interval** is still measured from one update (the bug the
  checkpoint interval had; `src/bnb/league/league.py` around line 318, compare
  `bnb.runs.session.CADENCE_UPDATES`): fix it the same way (the slowest of updates 2-4).
- A small offline tool that builds an EMA of any decay from a run's saved live policies
  (`bnb runs ema RUN --decay 0.9 --at STEP`, or as you see fit) and writes it as a policy
  file, so the tournament can rate it. Document the caveat: EMA per update.
- Option to turn the per-update policies off (BC runs don't need them every update; your
  call on the default per objective; say which).
- Docs: `docs/explanation/runs.md`, the guides that mention checkpoints.

## 2. Tournaments over mixed architectures

The fixed-slot engine (`bnb.league.matches.Arena`) stacks one architecture's weights and
refuses a field that mixes builds. The Griffin-only BC arms (`yemong.forecast_feedback`
on and off) differ in build and must be rated in one field.

- Group the network slots by build: each group its own template, stacked weights and
  compiled forward over its rows; a block's two seats may come from different groups.
  Allocation, record, resume and ETA unchanged.
- Measure (CPU is enough for a small test; the real cost comes when it runs on the GPU
  later) and say the expected cost per extra group.
- Tests (CPU): a two-build field plays and rates; one build gives today's results exactly.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-keep-mix`, branch `feat/keep-and-mix` (from v2).
  Work only there. Scratch in `/tmp/bnb-scratch-keep-mix`. Don't read other directories
  under `/home/vizia/avesnova`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""` for everything; never take the bench lock; no
  training runs.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append to `/tmp/bnb-scratch-keep-mix/progress.md`, `## HH:MM`
  heading (from `date`), 1-3 bullets; ETAs as local clock times. Start with a plan note.
- **Tests:** CPU tests for what you add, the quick tier (`pytest -m "not slow and not
  gpu"`), the CPU resume tests if you touch checkpoints, ruff, docs.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/keep-and-mix` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** what each part does, disk per 100M-step run, the tests,
  anything left open.
