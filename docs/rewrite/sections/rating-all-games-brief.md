# Brief: rate from every game, not only the lanes'

The ratings (`bnb.league.lanes.LaneRatings`, one Bradley-Terry fit per mode with the
scripted agent pinned at 1000) read only the rating lanes' games. In RL the lanes play
about one game per update for each matchup against the scripted agent, so the live and
averaged ratings carry errors of about 150 Elo after 25 updates. Meanwhile half of every
training batch is the live policy against the league's slots (`bnb.league.league.League`,
`after_shard` records them per mode and per entry in `self.record`): snapshots, the
averages, the ladder's rungs, the scripted agent. Each is a real, attributable game. The
author (2026-11-06): "why are we only rating based on the lanes? Shouldn't we be getting
our rating data from all of the games, including the lane games?"

## What to build

- **The league's training games join the fit:** per mode, the live policy's results
  against each league entry it played (`self.record`), as games of the live policy (a
  moving player: kept for the fit's window, as the lanes keep theirs) against that entry
  (a snapshot is stationary; an average moves; a ladder rung and the scripted agent are
  fixed). Self-play games (live against live) carry no information: leave them out.
- **Attribution:** check that a training game is credited to the entry that played it,
  including across a slot's drain and swap (`bnb.runtime.opponents`), and to the live
  policy's weights as the lanes credit them. Say what you found.
- **Names:** the league's entry names and the lanes' player names must meet in one fit
  (the lanes resolve `league_<k>` to the entry the slot holds). Make that one mapping.
- **Report** how much the errors shrink: rerun the fit offline on a real run's games if
  that is possible from what a run stores, else in a CPU test with a synthetic league.
- **The gauge** (`bnb.league.gauge`) has a side path that rates the averages from league
  games; once the fit reads them, remove whatever is now redundant (no legacy paths).
- Docs (`docs/explanation/league.md` and the lanes' docs) and CPU tests.

## The machine

An RL run owns the GPU for hours (`/home/vizia/avesnova/runs-v2/2026-11-06-mixed-calm-fern`,
read only). Work on the CPU (`CUDA_VISIBLE_DEVICES=""`), never take the bench lock, no
training runs.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-rating-games`, branch `feat/rating-all-games`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-rating-games`. Don't read other
  directories under `/home/vizia/avesnova`, except the run above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-rating-games/progress.md`, `## HH:MM` heading (from `date`) and 1-3
  bullets. Start with a plan note.
- **Tests:** CPU tests for what you add, plus the quick tier; ruff clean; docs updated.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/rating-all-games` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** what was built, the attribution check, the error
  reduction, anything left open.
