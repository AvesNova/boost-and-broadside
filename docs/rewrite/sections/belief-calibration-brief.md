# Brief: the belief's spread is badly overconfident (bug), and baselines for it

**Found 2026-11-07** in BC arm C's last update (`runs-v2/2026-11-06-bc-arm-C/metrics.jsonl`,
the `belief/*` series): the fed-back belief of hidden ships is far from the truth while
claiming metre-level certainty.

| hidden for | position error | truth within 1 sigma | heading error | within 1 sigma |
|---|---|---|---|---|
| 0.1-0.5 s | 8 m | 11% | 0.20 rad | 18% |
| 0.5-1 s | 35 m | 0.5% | 0.52 rad | 6% |
| 1-2 s | 114 m | 0.05% | 0.71 rad | 4% |
| 2-5 s | 545 m | 0% | 0.67 rad | 5% |
| 5-10 s | 1.76 km | 0% | 0.62 rad | 5% |

(`belief/normalised_square_error/*` reaches 1e6-1e29.) Also `next_state/kl/enemy_hidden` is
25 nats per ship (visible 3.2).

## 1. Find and fix the cause

Suspects (measure, don't guess): the decode of a predicted code to (mean, spread)
(`bnb.models.yemong.codes` / `decode.py`, `state_code.py`) under-reporting the spread when
the fine levels are flat or multimodal; the spread not accumulating tick to tick (each tick
re-deriving it from a sharp one-step prediction); the within-one-sigma statistic itself
(`bnb.learn.head_statistics`) being wrong. Fix what is wrong, with a test that a
deliberately flat or bimodal code decodes to a spread covering the truth at the expected
rate. If the cause is a design limit rather than a bug, say so and stop at the measurement.

## 2. Baselines (measurement, CPU)

On CPU rollouts with C's policy (`runs-v2/2026-11-06-bc-arm-C/policy-uniform`, read-only;
a few hundred games of each preset is enough), by time since last seen (the same age
buckets):
- hidden enemies: position and heading error of (a) the belief, (b) persistence (the last
  sighting), (c) dead reckoning (last position + velocity x age; heading held), and for
  (c) also with the game's drag/speed limits if simple;
- visible ships, one tick ahead: the next-state head's nats against the persistence
  baseline (today's) and against a dead-reckoned baseline code.

Write the numbers to `/tmp/bnb-scratch-belief/baselines.md` (a table per quantity).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-belief-calibration`, branch
  `fix/belief-calibration` (from v2). Scratch `/tmp/bnb-scratch-belief`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns
  the GPU. Never take the bench lock.
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-belief/progress.md`, `## HH:MM` from `date +%H:%M`,
  1-3 bullets, ETAs as clock times. Start with a plan; note the cause as soon as known.
- **Tests:** CPU tests for the fix, the quick tier (`pytest -m "not slow and not gpu"`),
  ruff, docs (`docs/explanation/yemong.md` belief section if it changes).

## Done

- Commit and push to `fix/belief-calibration`. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the cause (measured), the fix or why none, the baseline tables, tests,
  commit hash, time.
