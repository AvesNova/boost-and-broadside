# Brief: an offline rating tournament on fixed slots

`bnb ratings` (`src/bnb/league/tournament.py`, `matches.play`) rates a field offline: batches
of games split over pairs by A-optimal value, a Bradley-Terry fit after each batch with the
scripted agent pinned at 1000. It cannot rate a real field: every batch builds a new rollout
with one controller per player in the batch, and **every controller evaluates every
environment**, so memory and time grow with players x environments. A 46-player field ran
out of memory on an 8 GB GPU; a 4-network batch of 192 environments takes ~190 s (games are
played to their end, ~9,000 ticks).

## The author's design (2026-11-06)

- **Fixed slots, compiled once.** K network slots plus the scripted agent, and a fixed layout
  of matchup blocks: each block is a pair of seats with a fixed number of environments. The
  shapes never change, so the rollout is built and compiled once.
- **Each batch:** choose which matchups the blocks play (by information, as `allocate` does
  now, coarsened to whole blocks), load the chosen models' weights into the slots (swap in
  place between batches, as the league and the lanes do: `bnb.runtime.opponents`,
  `bnb.runtime.lanes.LaneSeat`), play the games, refit on every game so far, repeat until
  the standard errors are under the target or the budget is spent.
- **Only the relevant models act on their slices:** a slot's network evaluates only the
  environments where it plays. Reuse the training rollout's machinery for this (the lanes'
  per-player rows and functional call are the model to follow) rather than a new engine.
- **Allowed pairs:** the field may restrict who plays whom; the first use is "the scripted
  agent plays only the earliest players" (it anchors the scale through them).
- `bnb ratings` and `bnb head-to-head` both use the new engine; the old per-batch rollout
  goes (no legacy path).

## The first real use (the coordinator launches it once you hand back, unless you are
## done early enough to run it yourself; say which)

The RL run `2026-11-06-mixed-calm-fern` (read only). Players (policy files):

- the BC clone `/home/vizia/avesnova/runs-v2/averages/eager-acorn-uniform-66M-rated`;
- live models: `/home/vizia/avesnova/runs-v2/players-calm-fern/step_7340032-live`,
  `.../2026-11-06-mixed-calm-fern/league/step_18874368/checkpoint`,
  `.../league/step_37748736/checkpoint`, and from `players-calm-fern/`:
  `step_51380224-live`, `step_58720256-live`, `step_73400320-live`, `step_80740352-live`,
  `step_88080384-live`, `step_100663296-live`;
- the final averages: `players-calm-fern/step_100663296-exponential_average` and
  `step_100663296-uniform_average`;
- the scripted agent, allowed to play only the BC clone and the 7.3M model.

Frontline and deathmatch; target standard error 25; report the leaderboard.

## The machine

The GPU is free. Hold the bench lock for every GPU command:
`uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`. Measure the new
engine's memory and time per batch and say them.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-tournament`, branch `feat/tournament-slots` (from
  v2). Work only there. Scratch in `/tmp/bnb-scratch-tournament`. Don't read other
  directories under `/home/vizia/avesnova`, except the run and players above.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append brief milestone notes to `/tmp/bnb-scratch-tournament/progress.md`,
  `## HH:MM` heading (from `date`) and 1-3 bullets. Start with a plan note.
- **Tests:** CPU tests for what you add (the allocation to blocks, the slot swaps, the
  allowed pairs, a small field's ratings recovered), the quick tier, ruff, docs
  (`docs/explanation/league.md`). No long GPU suite: the real tournament is the test.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/tournament-slots` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the engine, the measurements, the leaderboard if you ran
  it, anything left open.
