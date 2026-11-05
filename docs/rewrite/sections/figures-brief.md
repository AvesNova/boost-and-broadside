# Brief: figures, round one (the scaffolding, then the shot head)

The author wants to see what each head has learned, plot by plot: make a figure, look at
it, say what it shows, then make the next. The first question is the shot head's: **do its
predicted trajectories bend through the fields as real shots do, or only go straight?**
v1's head learned straight lines only. The training numbers cannot tell: v2's
shot-trajectory loss falls to about 15.3 by 8M steps and stays there for 140M more.

This round builds the scaffolding and the shot figures, then goes one head at a time.
Play-mode overlays and periodic W&B media come in a later round; design for them, don't
build them.

## Principles (author, 2026-11-06)

1. **One decoder layer.** One module turns the model's outputs into world-space
   predictions with their uncertainty: the shot path (mean and sigma at each age), the
   next state, the belief of each hidden enemy, the density field, the enemy action
   distribution, the value and outcome. Offline figures use it now; play overlays will use
   the same functions later. No figure decodes for itself.
2. **Measure once, render many times.** A capture step plays a rollout (or builds a
   scenario), runs the model, and stores observations, truth and model outputs as an
   artifact. Renderers read artifacts only: cheap, CPU, repeatable, never running the
   model.
3. **One CLI.** `bnb figures` (sets such as `shots`, `scenarios`, `belief`, ...) and a
   capture command or option. Output to the run's `artifacts/figures/` (or a given
   directory). No free-standing scripts.
4. **Light provenance.** Each PNG carries its source in its own metadata (PNG text
   chunks): the run or policy path, the checkpoint step, the commit, the command line.
   No manifest, no registry beyond a plain list of figure sets. (v1's figure system was
   overly pedantic; keep this small.)
5. **Docs coupling.** One docs page lists every figure set: what it shows, how to read
   it. A docs test fails if a figure set is missing from the page or the page names one
   that does not exist (the repo's existing code-docs checks show the pattern).
6. **Every figure that matters gets a number.** For the shot head: the error on bent
   flights against a straight-line baseline, so curvature learning becomes a metric.

## Part 1: the scaffolding

The decoder layer (start with what the shot figures need, plus whatever else is cheap),
the capture artifact, `bnb figures` with provenance, the docs page and its test. Where
the model's heads and codes live: `src/bnb/models/yemong/` (`shots.py`, `codes.py`,
`state_code.py`, `belief.py`, `density.py`, `heads.py`, `model.py`); the labels and their
statistics: `src/bnb/learn/` (`shots.py`, `labels.py`, `head_statistics.py`); the game:
`src/bnb/game/api.py`. Loading a policy: `bnb.runs.checkpoint.load_policy`.

## Part 2: the shot head

- **The shot figure** (v1 had one; described here, don't look for it): for shots in a real
  rollout, the true flight through the fields (the environment's own launch and transport,
  at zero spread, plus a fan of spread samples) against the head's decoded path at every
  age, with its one-sigma radius; the field map underneath.
- **The scenario lab:** a small set of named, deterministic setups built through the game
  API, each a shot whose true path the physics gives, against the head's prediction:
  - a straight shot through a uniform field (the control);
  - crossing a field boundary from higher to lower refractive index;
  - from lower to higher;
  - a glancing entry that exits again at an angle.
  Add one or two more if the physics has other distinct cases (reflection, if it exists).
  The model needs an observation history to predict from: build each scenario so the
  model sees a plausible state (decide how, and say how in the docs).
- **The number:** displacement error by age, split into bent and straight flights, against
  a straight-line baseline (launch position plus launch velocity times age). Report it in
  the figure and in your notes; propose (don't add) a training metric for it.
- **Check a suspected metric bug on the way:** in every update of the last runs,
  `shot/enemy_hit_rate/predicted` equals `shot/enemy_hit_rate/realised` to three
  decimals, and the same for the ally and clear-miss rates. Find out whether the metric
  reports the label twice. Fix it if so, with a test.

## Part 3: one head at a time

After the shot head, in this order, each a figure set, each looked at and described in
your notes before the next:

1. **Belief of hidden enemies:** the ghost (mean and sigma ellipse) against the true
   position, on a few frames; coverage within one and two sigma against time since seen.
   The training metrics say it is badly overconfident (within one sigma: 48% at under
   0.1 s, 22% at 0.1-0.5 s, 1% at 0.5-1 s; velocity 9%).
2. **Next state:** a ship's real path against the one-step predictions, and the error over
   time with the spans it was hidden shaded.
3. **Density:** the hex field as true, predicted and signed error (the real hexagonal
   cells, not a contour); R^2 against cheap baselines (a constant map, the visible ships
   alone); error against the number of hidden enemies; error by ring. The training
   metric `density/explained_deviance` is NaN; find out why on the way.
4. **Enemy action:** predicted distribution against the action taken, by visibility.
5. **Critic and outcome:** a value trace over an episode with kills and deaths marked; a
   reliability diagram of the outcome head; Brier skill by horizon.

## Checkpoints to look at

- The BC clone, averaged: `/home/vizia/avesnova/runs-v2/averages/eager-acorn-uniform-66M-rated`
  (and its live clone, `/home/vizia/avesnova/runs-v2/2026-11-05-mixed-eager-acorn/policy`).
- The RL run training now, `/home/vizia/avesnova/runs-v2/2026-11-06-mixed-lively-meadow`:
  its checkpoints as they appear. Read only; never write into a run directory except
  under `artifacts/figures/`.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-figures`, branch `feat/figures` (from v2). Work
  only there. Scratch in `/tmp/bnb-scratch-figures`. Don't read other directories under
  `/home/vizia/avesnova`, except the run and checkpoint paths above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`. Add libraries
  if a task needs them (matplotlib is fine), plotting ones in an optional extra.
- **CPU only.** The GPU is running a training run and stays that way for hours: set
  `CUDA_VISIBLE_DEVICES=""` for everything, never take the bench lock, keep captures
  small (a few environments, a few hundred ticks). If something truly needs the GPU, say
  so in your notes and move on.
- **W&B:** none.
- **Look at your figures.** You can read PNGs; check each one shows what it claims before
  you describe it.
- **Progress notes:** append brief milestone notes to `/tmp/bnb-scratch-figures/progress.md`,
  with a `## HH:MM` heading (from `date`) and 1-3 bullets: a part landed (with its
  commit), a figure made and what it shows (the numbers), something failed, a change of
  plan. Start with a plan note.
- **Tests:** CPU tests for what you add, plus the quick tier; ruff clean; docs updated.

## Done

- Commit and push to `feat/figures` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, one page:** what was built; each figure set with what it showed; the
  shot head's answer (does it bend?); the hit-rate metric's verdict; anything left open.
