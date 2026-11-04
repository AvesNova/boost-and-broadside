# Brief: health sweep of the first BC and RL runs

Read the training metrics of two runs and report, for every family of series, what it did
and whether it looks healthy. This is a sweep, not a verdict: cover everything, flag what
looks odd, give the numbers. Another reviewer judges the flags.

## The runs (read only)

- **BC:** `/home/vizia/avesnova/runs-v2/2026-11-04-mixed-wandering-quill/` — behaviour
  cloning of the scripted agent, 100M environment steps, finished. Built before
  "diagnostics round 2", so it lacks many series.
- **RL:** `/home/vizia/avesnova/runs-v2/2026-11-04-mixed-tidy-harbour/` — PPO warm-started
  from the BC clone, 381 updates (~50M steps), **still running**: read whatever rows exist;
  never write, move or lock anything in either directory.
- In each: `metrics.jsonl` (one JSON object per update, every series by name, x-axis
  `counters/environment_steps`), `config.json`, `run.json`. The launch logs are
  `/home/vizia/avesnova/runs-v2/bc-launch.log` and `rl-launch.log`.

## What the series mean

Read, in `/home/vizia/avesnova/boost-and-broadside-v2` (read only):
`docs/rewrite/diagnostics.md` (the audit: every series, its meaning, healthy range),
`docs/rewrite/sections/diagnostics-spec.md`, `diagnostics-notes.md`,
`src/bnb/runtime/dashboards.py` (how the series group into sections),
`src/bnb/runtime/triggers.py` or wherever the 21 alert rules live, and `docs/explanation/`
as needed. Look up a series' definition in the code when the docs don't settle it.

## What to report

For each run, for every family (alerts, ratings and pool scores, episodes and rewards,
policy and entropy, KL and clip fraction, critic and returns, gradients and the cadence
updates' gradient decomposition, representation, every auxiliary head — next-state,
enemy action, density, belief, outcome, shot — inputs, fog, physics, frontline, map scale,
league, BC's own series, throughput and memory):

- start, end, min, max and the trend (rising, falling, flat, oscillating), at a few
  checkpoints along the run;
- whether that matches its healthy range or expected behaviour, citing where the
  expectation comes from;
- any alert that fired: when, how often, and the series behind it.

Specific questions to answer with numbers:
1. RL's entropy rose from about 0.8 to about 1.9 in its first 100 updates. Which action
   factors drive it, is it still rising, and does any strength measure move with it?
2. RL's KL and clip fraction over the run; any update with an outlier.
3. The critic: explained variance and value loss per level, over time.
4. Each auxiliary head against its baseline (the `gain` series: next-state per feature,
   enemy action against repeat and marginal, density, belief). Which features predict well,
   which barely beat the baseline, which are worse than it.
5. The `sigma_miscalibrated` alert in RL: the series behind it and how far off it is.
6. The cadence updates (every 10th update in RL): which series they add, and what the
   gradient norms, cosines and shares between losses show.
7. The live ratings: values, standard errors and games per window, per mode; how often a
   window holds too few games to mean anything.
8. BC: the KL, score and entropy curves, and why the score fell from 0.425 to 0.404 after
   ~76M steps while the KL kept falling (look for anything that co-moves).

Also save a few PNG charts (matplotlib, in the scratch directory) of the most informative
series, for the reports that follow.

## Where you work

- Scratch only: `/tmp/bnb-scratch-run-analysis`. Write scripts and charts there.
- **CPU only**, `CUDA_VISIBLE_DEVICES=""`; the GPU is busy with the RL run. Run Python with
  `uv run --no-sync python` from `/home/vizia/avesnova/boost-and-broadside-v2`.
- **Nothing on W&B:** don't use the W&B API or create anything there; all data is on disk.
- Change no repository files.

## Output

Write the full findings to `/tmp/bnb-scratch-run-analysis/findings.md`: a section per run,
a table per family, then the flags ranked by how much they matter, each with its evidence.
Your final message: the ranked flags with one line of evidence each, the answers to the
eight questions in short, and the paths of the findings and charts.
