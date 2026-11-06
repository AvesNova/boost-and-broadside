# Brief: inference-time ablations of a trained policy (rotary, forecast feedback)

**Asked by the author (2026-11-06):** before the next BC run, measure how much the best
model gets worse when its rotary attention is turned off. The forecast-feedback ablation
(the belief's forecast not fed back) rides along: same machinery, same tournament.

Both are inference-time ablations of a model trained with the feature on: they measure how
much the trained policy *relies* on it, not what a model trained without it would reach.
Say so in every result.

## What to build

- **Per-player ablations in the tournament engine** (`bnb.league.matches`, `Player`,
  `Arena`): a player may carry a set of ablations, applied only on the rows that player
  acts on. The engine stacks one architecture's weights in slots and refuses mixed builds;
  an ablated player is the same build, so it must stay in the same tournament.
  - **`rotary`:** the rotary rotation is the identity for that player's rows: every angle 0
    (position and heading, token and map tables; `bnb.models.yemong.layers.rotary_tables`,
    `Yemong` around `model.py:360`), so cos = 1, sin = 0. Check the fused attention kernel
    path (`kernels.attention`) gets the same tables, or falls back to the plain twin.
  - **`forecast`:** as `yemong.forecast_feedback=false` (merged; read its docs and
    `belief.py`): hidden rows keep their last sighting, enemy action uniform, nothing
    written back, for that player's rows only.
  - Per-row, not per-model: build the tables (or the belief write) from a per-row mask, so
    the compiled graph stays one graph. Ablation off must be bit for bit today's output.
- **CLI:** a way to name an ablated copy of a policy in `bnb ratings` / `bnb head-to-head`
  (for example `--with PATH:rotary`, `PATH:forecast`; your choice, documented).
- **Tests (CPU):** off equals today exactly; `rotary` gives the same output as a model with
  zero angles everywhere on those rows and leaves other rows untouched; `forecast` matches
  the config switch's behaviour on those rows.
- **Docs:** `docs/explanation/league.md` (the tournament) and the Yemong page, briefly.

## The measurement (you run it once the code is in and tested)

Field, frontline and deathmatch, target standard error 25 (the engine's record, resume and
ETA as merged):

- the best model `/home/vizia/avesnova/runs-v2/players-calm-fern/step_100663296-exponential_average`,
  and the same policy with `rotary` off, and with `forecast` off;
- the ladder: `players-calm-fern/step_7340032-live`,
  `2026-11-06-mixed-calm-fern/league/step_18874368/checkpoint`,
  `2026-11-06-mixed-calm-fern/league/step_37748736/checkpoint`,
  `players-calm-fern/step_51380224-live`, `players-calm-fern/step_73400320-live`,
  `players-calm-fern/step_100663296-live`;
- the BC clone `/home/vizia/avesnova/runs-v2/averages/eager-acorn-uniform-66M-rated`;
- scripted, playing only the BC clone and the 7.3M model.

Before launching, write a launch note in progress.md: the field, where the per-batch record
is, the expected and worst-case finish as local clock times. Copy the results to
`/home/vizia/avesnova/runs-v2/ratings/2026-11-06-ablations/` when done. Report each
ablated player's Elo and its difference from the unablated EMA, with the difference's
standard error (from the fit's covariance if you have it, else say how you estimated it).

If cheap, also: on one CPU capture (`bnb figures capture`), each ablation's mean KL from
the unablated policy's action distribution, per mode. Skip if it costs more than an hour.

## The machine

The GPU is free. Hold the bench lock for every GPU command:
`uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-ablations`, branch `feat/inference-ablations`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-ablations`. Don't read other
  directories under `/home/vizia/avesnova`, except the run and players named above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to `/tmp/bnb-scratch-ablations/progress.md`,
  `## HH:MM` heading (from `date`) and 1-3 bullets; ETAs as local clock times (expected and
  latest), never "minutes left". Start with a plan note.
- **Tests:** CPU tests for what you add, the quick tier (`pytest -m "not slow and not gpu"`),
  ruff, docs. No long GPU suite: the tournament is the test.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/inference-ablations` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the mechanism, the tests, the results table (Elo and
  difference per ablation per mode), anything left open.
