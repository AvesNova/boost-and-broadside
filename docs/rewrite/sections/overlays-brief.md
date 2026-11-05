# Brief: prediction overlays in the play shell

The play shell (`src/bnb/play`, `docs/guides/playing.md`) shows the game. The author wants
to see, on the same map and while it plays, what each agent thinks: the real flight of
every bullet, and for a network agent its heads' predictions; for the memory scripted
agent, where it thinks the enemies it cannot see are. The decoder layer
(`src/bnb/models/yemong/decode.py`) already turns every head into world-space predictions
with their uncertainty; the offline figures use it (`docs/guides/figures.md`). The
overlays must use the same functions: no decoding of their own.

## What to build

- **Overlays, each toggled on its own,** in a card or submenu (not always open; follow the
  shell's existing cards and menus):
  - **True bullet paths** (physics): each live bullet's path ahead, from the game's own
    transport (the learner's re-simulation, as the figures' shots use). For every agent.
  - **Predicted shot path** (network agents): the shot head's decoded path for the
    selected ship's next shot, with its sigma.
  - **Next-state ghost** (network agents): each ship's predicted next state, as a ghost
    with its sigma.
  - **Belief ghosts** (network agents): each hidden enemy's believed position with a sigma
    ellipse; with the truth view on, its true position beside it.
  - **Dead-reckoning ghosts** (the memory scripted agent, `bnb.controllers.scripted_memory`):
    each remembered enemy's predicted position; it has no sigma, only a fade weight, so
    draw the weight as the ghost's opacity. Don't invent an ellipse.
  - **Density** (network agents): the hex field as predicted, and with the truth view, true
    and the signed error.
  - **Enemy action** (network agents): the predicted action distribution of the selected
    enemy, as a small readout.
  - **Value and outcome** (network agents): a small gauge or readout per level.
- **Only what exists:** each controller says which overlays it can provide; the card lists
  only those for the sides being played. Scripted against scripted offers true bullet paths
  and the dead-reckoning ghosts, and nothing a network head would provide.
- **Per side:** a network on each side has its own predictions; let the viewer pick whose
  (the selected ship's side by default).
- **The video hook:** `bnb.figures.video.draw_frame` is where overlays plug into the
  watcher's videos; add an option to turn chosen overlays on there.
- **Cost:** overlays off cost nothing; on, the shell should stay smooth (measure the frame
  time with a network agent and all overlays on).

## The machine

A training run owns the GPU for hours and will keep it. Run the agents' inference on the
CPU (`CUDA_VISIBLE_DEVICES=""` for torch; the shell's `agent_device` then picks the CPU).
The shell's OpenGL uses the GPU's graphics side, which is fine (the media watcher renders
there now, +100 MiB, no slowdown of the trainer). Never take the bench lock. Check your
overlays by rendering offscreen frames (`bnb.play.app.run_offscreen`) and looking at them:
you can read PNGs.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-overlays`, branch `feat/overlays` (from v2). Work
  only there. Scratch in `/tmp/bnb-scratch-overlays`. Policies to use (read only):
  `/home/vizia/avesnova/runs-v2/averages/eager-acorn-uniform-66M-rated` and the checkpoints
  of `/home/vizia/avesnova/runs-v2/2026-11-06-mixed-calm-fern`. Don't read other
  directories under `/home/vizia/avesnova`.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-overlays/progress.md`, `## HH:MM` heading (from `date`) and 1-3 bullets:
  a part landed (with its commit), what an overlay shows on a real game, a measurement,
  something failed, a change of plan. Start with a plan note.
- **Tests:** CPU tests for what you add (which overlays each controller offers, the
  decoded geometry the overlays draw), plus the quick tier; ruff clean; docs (the playing
  guide) updated, with the docs tests passing.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/overlays` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** what was built, a few offscreen frames' paths that show
  each overlay, the frame-time measurement, what the overlays reveal on a real game, and
  anything left open.
