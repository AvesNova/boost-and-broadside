# Brief: how much each ship-state quantity changes per tick, and how to code its target

**Goal (author, 2026-11-07):** the next-state head's targets should be coded in a frame
sized to each quantity's real change, centred on its physics baseline (persistence, or
dead reckoning where it applies), not in the input's absolute frame. First measure the
changes; then compare candidate target codes for every quantity, with numbers.

Today's code (`src/bnb/models/yemong/state_code.py`, `codes.py`): position 9 levels x 9
colours (3 x 3 per level, period 26 244 m, finest 4 m); velocity 3 x 81 bins (say exactly
what the three runs are, e.g. projections on three directions, and the bin spacing: knee
50 m/s, spacing 2.5); heading 4 levels x 4 colours (circular); nose rate 41 bins over
+-8 rad/s; angle of attack, shield recharge wait, shield, energy, log refractive index 21
bins each; gun cooldown one bin per tick. The head adds log(code + floor) of the current
state (persistence) to its logits.

## 1. Measure (CPU rollouts, memory-capped)

Players: the scripted agent, and calm-fern's RL agent
`/home/vizia/avesnova/runs-v2/players-calm-fern/step_100663296-exponential_average`
(read-only). Each in self-play, on every preset (`bnb.game.api.PRESETS`: frontline,
deathmatch, deathmatch_wide at least), on enough games for stable 99.9% quantiles; say the
counts.

For every quantity of the 12 means (`MEANS`), for living ships:
- **one tick ahead:** the change from persistence (q_{t+1} - q_t; circular quantities
  wrapped to (-pi, pi]); and from dead reckoning where it applies: position
  (x_t + v_t dt, and x_t + (v_t + v_{t+1}) dt / 2; say which matches the integrator),
  heading (heading_t + nose_rate_t dt), and anything else with an obvious kinematic baseline
  (say what you chose);
- also at **8 and 32 ticks** for position, velocity and heading (the belief's horizons);
- statistics: the fraction exactly zero (or below float32 resolution), mean, and the
  |change| quantiles 50, 90, 99, 99.9, max; per axis and as a norm for vectors;
- split by situation where cheap: in a field or zone or not; thrusting or not; just
  respawned; colliding or hit;
- save each distribution as a histogram (log-spaced bins of |change|, signed where it
  matters) to `/tmp/bnb-scratch-deltas/histograms/<player>/<preset>/<quantity>.json`.

## 2. Compare target codes (analysis, no model change)

For every quantity, evaluate candidate codes of its change on the measured samples, at an
equal bin budget where possible (say the budget):
- uniform categorical over the measured range; symlog (or mu-law) bins; quantile bins
  (equal mass);
- multi-level colours (the hierarchy we use: period = range, c colours per level);
- for 2-D quantities: per axis (x and y), on three directions 120 degrees apart, a joint
  2-D hierarchy (3 x 3 per level), polar (magnitude symlog + angle);
- for circular quantities: circular colours; wrapped bins.

Report per candidate: bins; the coverage of the 99.9% change; the resolution at the median
change (bin width there, and as a fraction of the median); the mean quantization error;
the target entropy in nats (the floor of the cross-entropy a perfect model reaches at that
resolution; equal-mass bins maximise it); and the tail events lost or clipped. Recommend
one code per quantity and say why, and what you would check in training.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-state-deltas`, branch `analysis/state-deltas`
  (from v2). Scratch `/tmp/bnb-scratch-deltas`. Scripts under `bench/` in the branch.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; the GPU is
  training. Never take the bench lock.
- **Memory:** every collection runs in its own capped unit:
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ...`. Stream results to
  disk per segment; never hold all games in RAM (an uncapped collection of 28 GB crashed
  the machine today).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-deltas/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.

## Done

- The report at `docs/rewrite/sections/state-deltas.md`: a table per quantity (both
  players, each preset, persistence and dead reckoning), then the code comparison and the
  recommendations. Commit (with the scripts) and push to `analysis/state-deltas`. Author
  Julian Kizanis <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the headline numbers per quantity, the recommended codes, the commit, the
  time.
