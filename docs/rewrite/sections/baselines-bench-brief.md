# Brief: the baselines bench (persistence, rule, extrapolation; frames; axis correlation)

**Decided by the author (2026-11-08):** before the round-2 delta-head arms (persistence and
dead reckoning) spend GPU time, measure on the CPU, from recorded play, what each baseline
leaves for the head to learn. Offline analysis only: no training, no model changes. The
arms themselves are still under discussion; this bench informs them.

Read first: `docs/rewrite/sections/state-deltas.md` (the earlier analysis: its scripts
`bench/state_deltas_*.py` and its recorded segments), `docs/rewrite/sections/
delta-heads-round2-notes.md`, `bnb.models.yemong.delta` (today's rule baselines and the
symlog HL-Gauss code), `bnb.game.physics.flight` (`flight_tick`, `kick`).

Data: the recorded segments in `/tmp/bnb-scratch-deltas/segments/{scripted,calm-fern}/
<preset>/segment_*.npz` (read only; layout in `bench/state_deltas_collect.py`). Reuse the
earlier scripts' loading, pair masks (living ship at t and t+1, no respawn) and statistics
where they fit. No new games unless something is missing; if so, say why.

## 1. What each baseline leaves (one tick)

For every quantity in the state code (position, velocity, heading, nose rate, angle of
attack, energy, shield, recharge wait, gun cooldown, log refractive index), the residual
truth(t+1) - baseline(t) under three baselines:

| Baseline | Definition |
|---|---|
| P, persistence | nothing changes |
| R, rule | today's `delta.py` baselines (arc dead reckoning, v rotated by omega dt, heading + omega dt, shield recharge, countdowns) |
| E, extrapolation | last tick's rate: x + v dt + a dt^2/2 (a from t-1 to t); v + a dt with a's tangential and normal parts (velocity frame at t-1) applied in the velocity frame at t; heading + omega dt + omega_dot dt^2/2; omega + omega_dot dt; other scalars x + (x_t - x_{t-1}). Shield and countdowns (and energy, if a shot makes it jump): last tick's rate only when last tick had no event, else the rule rate |

Where t-1 is missing (first tick of a life), E falls back to R; count how often.

Per quantity and baseline: median, p90, p99, p99.9 and max of |residual|; the share exactly 0;
the entropy (nats) of the residual coded in the delta code (knee at that baseline's median,
range at its max, 51 bins for position and velocity components, 31 otherwise), which is
roughly what the head must learn. Position and velocity per component in three frames:
world, nose, velocity (below 1 m/s the velocity frame falls back to the nose; count it).

Splits: by player (scripted, calm-fern) and preset; held control vs a control change on t
(power or stick differs from t-1; the segments have both); in a field band vs not; hit vs not.
The held / changed split matters most for E: its residual should be near zero when the
control is held and carry the change otherwise. Say whether that holds.

Check the nose rate under E specifically: the nose is a damped oscillator (damping 0.7), so
omega_dot changes fast; say whether E beats P there or overshoots.

## 2. Axis correlation and the error ellipse

The heads code each axis as an independent categorical, which cannot represent correlation
between the axes. For position and velocity, per frame (world, nose, velocity):
- one tick: the correlation between the two components of each baseline's residual
  (Pearson and a robust one, e.g. Spearman), and the anisotropy of the residual's 2 x 2
  covariance (ratio of the axes, angle of the major axis in that frame);
- the hidden-ship proxy: from a ship's true state at t, dead reckon (constant velocity and
  E) for k ticks, k = 3, 15, 30, 90, 150 (0.1 to 5 s), and measure the same correlation and
  anisotropy of truth(t+k) - forecast. This stands in for a hidden ship's belief error
  before any learned correction.
Say which frame leaves the two axes least correlated at each horizon.

## Output

- Scripts `bench/baselines_*.py` (CPU, streaming over segments, nothing large in memory), a
  test file for the baselines (E on a constant-acceleration and a steady-turn trajectory is
  exact; the event rule), and `docs/rewrite/sections/baselines-bench.md`: the tables, then a
  short plain reading per quantity (which baseline leaves the least, and the cost), with no
  recommendation beyond what the numbers show.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-baselines-bench`, branch `bench/baselines` (from
  v2). Scratch `/tmp/bnb-scratch-baselines`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock. No training runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-baselines/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going.

## Done

- Commit and push to `bench/baselines` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the headline numbers per quantity (best baseline, its median and p99.9,
  entropy against P), the held / changed split, the frame correlation by horizon, anything
  surprising or open, the time.
