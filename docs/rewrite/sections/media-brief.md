# Brief: periodic media while a run trains (W&B videos and head figures)

The author wants to see a training run, not only its scalars: every so often, a short video
of the current policy playing and the head figures, in W&B next to the run's curves, so a
glance says whether the policy still looks sane. `bnb figures` exists now
(`docs/guides/figures.md`, `src/bnb/figures/`); the play shell renders offscreen
(`bnb.play.app.run_offscreen`, EGL). This round joins them to a run.

## What to build

- **A watcher process,** started by hand next to a training run (for example
  `bnb figures watch RUN`), that never touches the trainer: it polls the run's
  `checkpoints/` (and the league's snapshot directory if useful), and for each new
  checkpoint:
  1. captures a small rollout on the CPU (the existing capture, a few environments);
  2. renders the figure sets that are quick (`shots`, `belief`, `next_state`, `density`,
     `enemy_action`; `critic` only if it stays cheap) into the run's
     `artifacts/figures/<checkpoint>/`;
  3. renders a 20-30 s video of one game: the checkpoint's policy against the scripted
     agent, one per training mode (frontline, deathmatch), from the play shell offscreen,
     encoded to MP4 with ffmpeg (installed at `/usr/bin/ffmpeg`), small enough to upload;
  4. logs the video and the figures to W&B at the checkpoint's environment steps.
- **W&B:** don't write into the trainer's own W&B run from a second process (concurrent
  writers to one run are not supported). Use a companion run in the same project, grouped
  with the training run and named after it (e.g. `<run>-media`), with environment steps as
  the x-axis, and say in its notes which run it watches. The W&B run id of the training
  run is in its `run.json`. Credentials are in `~/.netrc`: never print them; the repo is
  public, never commit them.
- **A video option on `bnb figures`** (e.g. `bnb figures video --policy P`) so the same
  video can be made by hand.
- **Overlays in the video** are a later round (play-mode overlays); plain games now. Keep
  the renderer call where the overlays will plug in.
- **Docs and tests:** the figures guide gains the watcher and the video; CPU tests for the
  polling and the logging (W&B offline mode in tests), the existing docs coupling.

## The machine

A training run owns the GPU for hours. Captures and the model's inference run on the CPU
(`CUDA_VISIBLE_DEVICES=""` for torch). The play shell's offscreen context is OpenGL over
EGL: it uses the GPU's graphics side, not CUDA, and a little memory (the trainer peaks near
5.8 of 8 GB). Measure what it uses and how long one video takes; if it slows the trainer's
updates measurably (`train/time/update_seconds` in the run's `metrics.jsonl`), say so and
fall back to rendering frames another way. Never take the bench lock.

## Test it on the live run

`/home/vizia/avesnova/runs-v2/2026-11-06-mixed-calm-fern` is training now (checkpoints
every ~7.3M steps, about 29 minutes). Once the watcher works, run it on that run with
`--wandb`: that is the real test. Read only from the run directory, except
`artifacts/figures/`.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-media`, branch `feat/media` (from v2). Work only
  there. Scratch in `/tmp/bnb-scratch-media`. Don't read other directories under
  `/home/vizia/avesnova`, except the run above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to `/tmp/bnb-scratch-media/progress.md`,
  `## HH:MM` heading (from `date`) and 1-3 bullets: a part landed (with its commit), a
  measurement (video time, memory, any slowdown of the trainer), something failed, a change
  of plan. Start with a plan note.
- **Tests:** CPU tests for what you add, plus the quick tier; ruff clean; docs updated.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/media` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Leave the watcher running on calm-fern when you finish, and say how to stop it (its PID
  and command).
- **Final message, half a page:** what was built, the measurements, the W&B companion
  run's name, what the first videos show, anything left open.
