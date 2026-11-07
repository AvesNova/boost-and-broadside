# Brief: the offline arena is ~15x slower with mixed builds (bug fix)

**Bug (found 2026-11-07):** `bnb ratings` with a field of 4-5 Yemong builds (BC arms A, R,
C, D: `rotary` off, `order=temporal_first`, `rotary_readout`, `forecast_feedback` off) took
**2182 s** for its first frontline batch (1024 games); a single-build field of 15 players
takes ~145 s per batch (277 s for the first). The process sat at 99.9% CPU, GPU ~98%.
`runs-v2/ratings/2026-11-07-bc-arms/frontline.batches.jsonl` holds that one batch; the
command is in `~/.local/share/bnb/queue4.sh` (the "arms tournament" step).

**Known cause, partly:** `bnb.league.matches.MatchSeat` runs every group's network over
every slot, slot by slot (`_network`: a Python loop of `functional_call` per slot), and
keeps each slot's own group's result. Groups x slots forwards; the module docstring
estimates "one more network pass per group", but 4-5 groups explain ~5x, not 15x. Find
the rest (graph capture not used with several groups? the per-slot loop's launch overhead?
the temporal_first or read-out paths slow in this engine?). Measure, don't guess.

## Goal

A mixed-build field costs about what a single-build field does per game (each row runs
only its own player's network, once). One way: fix each slot's group for the whole
tournament (allocate blocks to group pairs from the matchup schedule) so each group runs
only on its own slots; another: batch the slots of one group into one call (stacked
weights via `torch.func.vmap` or a grouped forward). Your call; keep the graph captured
once and results identical in distribution (same games, same tallies, per-batch record and
resume unchanged).

## Tests

- A mixed field gives the same results as each pair played alone (or as today, seeded).
- A timing check in your notes: single build vs 4 builds, per batch, before and after
  (short runs, `--budget` small).
- The quick tier (`pytest -m "not slow and not gpu"`), the GPU tests you touch, ruff, docs
  (`docs/explanation/league.md` or wherever the engine is described).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-arena-fix`, branch `fix/arena-mixed-builds`
  (from v2). Scratch `/tmp/bnb-scratch-arena-fix`. Read policies under
  `/home/vizia/avesnova/runs-v2/` (read-only): the four arms'
  `2026-11-06-bc-arm-{A,R,C,D}/policy-uniform`.
- **GPU:** you may use it; nothing else is running. Every GPU command goes through the bench
  lock: `uv run --no-sync python ~/.local/share/bnb/locked.py <cmd>`. Keep each GPU job
  short (under 15 minutes).
- **CPU:** `nice -n 15`, at most 8 threads.
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-arena-fix/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times ("ETA 11:40 (worst 12:10)"). Start with a
  plan note; note the measured cause as soon as you have it.

## Done

- Commit and push to `fix/arena-mixed-builds`. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the cause (measured), the fix, before/after seconds per batch, tests, the
  commit hash, the time.
