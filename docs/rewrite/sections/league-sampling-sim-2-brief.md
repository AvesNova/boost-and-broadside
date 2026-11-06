# Brief: league sampling, round 2: turnover, slots, ladder, weighting

Round 1 (`league-sampling-sim.md`, `bench/league_sampling.py`, merged) reproduced the
league of `2026-11-06-mixed-calm-fern` exactly and found three causes: the old gauge's
runaway snapshot ratings (gone at head), serial drains of one game clock each (11 draws in
the run), and the 11-rung ladder weighted as 11 entries. Read the report and the script
first, then extend them.

**The author's principles (2026-11-06):**
- The league's training games serve only the agent. They need not anchor or feed the
  ratings; the rating lanes exist to supply that.
- In RL the ladder need not be in the training pool at all (the clone starts near 900
  Elo, about 0.4 against scripted); it matters only in BC.
- A game cut mid-way is not wasted: its ticks still train the agent. Only its result is
  lost for Elo (or credited by fraction).
- Simulate; don't ask. Cheap variants are just run.

## Compare, on the reconstructed inputs (repaired snapshot ratings as the main case)

- **Turnover:** today's serial drain; all slots draining together; 2 incoming sets; a
  **clean cut** every k updates (k = 1, 2, 4), swapping all or a fraction of the slots
  mid-game (the opponent's carry reset at the cut); a draw per row at reset (the limit).
- **Slots:** 4, 6, 8 (with each turnover rule).
- **Ladder:** 11 entries; one entry (weighted by its best rung); none (the RL case).
- **Weighting:** p(1 - p), (1 - p)^2; with and without the floor.

Not every product: pick a sensible grid (one factor at a time from a good default, plus
the promising pairs), say which you ran and why.

## Measures (per variant, mean and 10th-90th percentile over seeds)

- diversity: distinct opponents per 10 updates, and the entropy of the opponent mix;
- open-game share and learning signal (as round 1);
- games cut short (share of games, share of ticks in cut games);
- opponent carry resets per update;
- memory cost (stacked parameter sets) and per-tick cost (set calls), as round 1 estimated.

## Output

Extend the report with a section "Round 2" (or a new `league-sampling-sim-2.md`): the
grid, the table, a recommendation with its cost. Keep the CPU tests passing; add one for
the clean cut. Change no league behaviour.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-league-sim2`, branch `feat/league-sampling-sim-2`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-league-sim2`. Don't read other
  directories under `/home/vizia/avesnova`, except `runs-v2/2026-11-06-mixed-calm-fern`
  (read only).
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`; GPU runs are going. Never take the
  bench lock.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append to `/tmp/bnb-scratch-league-sim2/progress.md`, `## HH:MM`
  heading taken from `date +%H:%M` (local), 1-3 bullets; ETAs as local clock times. Start
  with a plan note. Target: the report within about an hour.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/league-sampling-sim-2`. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the table's highlights, the recommendation and its
  cost, what you assumed.
