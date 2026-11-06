# Brief: why the league plays beaten opponents; simulate, then compare samplers

**Asked by the author (2026-11-06):** "let's do some little sims, do some stats, and see
what actually works for us instead of just guessing." No training runs.

## The finding to explain

In `2026-11-06-mixed-calm-fern` (read only, `/home/vizia/avesnova/runs-v2/`), late in the
run the league's games went about 50% to the ladder's rungs, which the live policy beat
100% of the time (`pool/game_share/rung`, `pool/score/rung` in `metrics.jsonl`), while
past snapshots got about 0-5% (`pool/game_share/checkpoint`) with 4 snapshots in the pool.
The sampling weight is `p (1 - p)` plus a floor of 0.05 spread evenly
(`bnb.league.pool.sampling_weights`; `docs/explanation/league.md`, "Sampling" and "Slots,
the ladder and the drain"). Two candidate causes, neither checked:

1. the floor dominates once every `p (1 - p)` is near 0, and the rungs are most of the
   entries;
2. slot residency, not the weight: a slot holding the ladder serves a whole drain cycle,
   so game shares follow residency.

There may be others (the gauge's predicted `p` for entries with few games, how the ladder
counts as one entry or many, the per-mode mixing). Find out which, from the code and the
run's logs.

## What to do

1. **Reproduce** the logged game shares with the league's own code (`bnb.league.pool`,
   `bnb.league.league`, `bnb.runtime.opponents`) driven by the run's logged ratings and
   pool at each update (from `metrics.jsonl` and the run's `league/` and `run.json`;
   reconstruct what you need, say what you assumed). A simulation that does not reproduce
   the logged shares is not done.
2. **Name the cause** with numbers (for example the share each mechanism accounts for).
3. **Compare samplers** on the same reconstructed inputs, by the game share against
   opponents the live policy beats less than, say, 90% of the time, and by the expected
   learning signal (for example sum of share x p(1-p)): today's; no floor over beaten
   entries; `(1 - p)^2`; a fixed share (for example 30-50%) for the newest snapshots and
   the averages; Nash weighting over the pool's empirical payoff matrix (from the
   recorded games). Also more snapshot slots (say 8). Say each one's cost (games, slots,
   complexity) and its failure modes.
4. **Overlap with the rating games** (the author: "the agents we want here and the agents
   the rating games want probably heavily overlap"): how much of each sampler's training
   games would also serve the in-training rating (information about live's rating).

Write a short report (`docs/rewrite/sections/league-sampling-sim.md`): the cause, the
table of samplers, a recommendation. Commit the simulation as a script under `scripts/` or
a test-quality module if it is reusable (your call). Change no league behaviour.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-league-sim`, branch `feat/league-sampling-sim`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-league-sim`. Don't read other
  directories under `/home/vizia/avesnova`, except the run above.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""` and `nice -n 15` for anything heavy; a BC run
  owns the GPU. Never take the bench lock.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append to `/tmp/bnb-scratch-league-sim/progress.md`, `## HH:MM`
  heading (from `date`), 1-3 bullets; ETAs as local clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/league-sampling-sim`. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the cause with its numbers, the samplers' table, the
  recommendation, what you assumed.
