# Brief: a Griffin-only memory arm (no forecast fed back)

**Decided by the author (2026-11-06):** a clean Griffin-only BC experiment: the same model
and training with and without the forecast fed back into the belief, compared at matched
steps. Nothing else changes in this experiment.

## Why

Yemong's belief (`src/bnb/models/yemong/belief.py`, `model.py`) keeps, for each hidden ship
the team has seen, the model's own forecast: after each decision the next-state head's
decoded mean and spread (and the enemy-action head's probabilities) are stored and read
back as that ship's row on the next tick (`beliefs.advance`). The figures
(`docs/guides/figures.md`) show this loop is broken for positions: errors grow to kilometres
while the decoded sigma stays near 1 m (the position code's finest levels set the variance),
and the overconfident forecast is fed back as input. No gradient reaches it through the
policy. The author's alternative: no forecast in the input at all, and the Griffin layers
(the temporal mixing) carry memory.

## What to build

- **One switch in `YemongConfig`** (name it clearly, e.g. `forecast_feedback: bool = True`):
  - **on:** today's model, bit for bit;
  - **off:** a hidden ship's row is what the team last saw: its last-seen state (position,
    velocity, shield and the rest, as at its last sighting, not advanced) and its age (time
    since seen), with the enemy action at uniform. The forecast is never written back. The
    next-state, enemy-action and other heads still train as auxiliary losses, unchanged.
  - Every input that derives from the belief follows the switch the same way, in
    particular the rotary positions of hidden ships (their last-seen position when off)
    and anything else that reads `Belief.means`/`spreads`/`action` for hidden rows. List
    every reader you found and what each does when off.
- **Selectable from the CLI** (`--set yemong.forecast_feedback=false` should already work;
  check), recorded in the run's `config.json`.
- **Tests:** on equals today (a recorded step or unroll matches bit for bit); off writes no
  forecast (a hidden row's inputs equal its last sighting at every later tick until it is
  seen again); `unroll` still equals a loop of `step` with the switch off.
- **Docs:** the Yemong page and the belief section say what the switch does.
- Don't add other features (a separate hidden-ship encoder, new heads, a new code): those
  are open questions, not part of this experiment.

## The machine

An RL run owns the GPU for about two more hours. Work on the CPU
(`CUDA_VISIBLE_DEVICES=""`), never take the bench lock, no training runs: the two BC arms
are launched by the coordinator afterwards.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-griffin-only`, branch `feat/griffin-only` (from
  v2). Work only there. Scratch in `/tmp/bnb-scratch-griffin-only`. Don't read other
  directories under `/home/vizia/avesnova`.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-griffin-only/progress.md`, `## HH:MM` heading (from `date`) and 1-3
  bullets. Start with a plan note.
- **Tests:** CPU tests for what you add, plus the quick tier; ruff clean; docs updated.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/griffin-only` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the switch, every belief reader and its off behaviour,
  the tests, anything left open.
