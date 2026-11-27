# Work queue (from 2026-11-27 evening)

Work the author wants done that needs no input from them, run by the coordinator one item after
another (author 2026-11-27 ~21:45: "it'll be done when it's done", within reason). The author
reviews what happened afterwards. Each item has a budget; an agent that goes past it, or into
a rabbit hole (days of searching, a 2% gain for a large change, thousands of lines of new
kernel code), is pulled back and the item is reported as it stands.

Rules: timing, profiling and speed work run one at a time on a quiet machine (the GPU lock
alone is not enough; CPU load moves timings). Read-only work (reviews, docs) and game
statistics (rates, Elo) may run alongside one non-timing job. Every item ends with its results
on its branch, a progress note, tests once at the end, and a line in the Log below. Merges into
core when sensible (author 2026-11-27).

## Running

- R1 Persona study round 2 (`personas-2`): 1v1 / 4v4 / 16v16, fields and fog, longer clocks,
  crowd-dependent parameters, tuning, proposals. Game statistics (not timing-sensitive).
- R2 Two-level league (`league-pool`): pool, PFSP loader, duplicates, drain, scripted personas
  in one call, precompiled routing, resume. Includes timings: nothing else timing-sensitive
  runs beside it.

## Queue (in order)

1. Merge R1 and R2 into core (after the coordinator's check); league personas from R1 into
   R2's pool config. Then reset the compile check's baselines to the merged tree (author
   2026-11-27 ~22:00: cold is about 120 s now, so the budget can come down a lot): op counts
   per graph as measured, the cold-time budget to about 150 s (25% over the measured cold
   compile); a new graph needs its own baseline.
2. Hit-range tool port to the new bullet state (lean/tool, lean/sandbox). Small; budget 1 h.
3. Docs cleanup before the reviews: stale and duplicate docs (e.g. docs/rewrite/questions.md
   beside docs/questions.md), closed one-off briefs, stale backlog rows, notes that contradict
   the code; old docs scripts that no longer run (about 2,450 ruff errors under docs/):
   delete or fix. Budget 2-3 h.
4. Second training hot-path review (measurements, report; first priority after the cleanup):
   - every head's rows: next-state, hit and pairwise losses on all visible enemies, hidden
     enemies masked (author 2026-11-27 ~22:00); report what the code does today; the fix goes
     in item 5;
   - activation checkpointing (per block) and selective recompute of cheap element-wise
     results, with a micro-batch size vs memory vs speed curve;
   - compiler-mode grid per graph (default, max-autotune, max-autotune-no-cudagraphs,
     reduce-overhead): cold and warm compile, steady speed;
   - the CUDA-graph rewrite plan (league tallies out of the graph, outputs into the store,
     random draws as inputs);
   - the remaining host copy per micro-batch; anything new the profile shows.
   Budget 3-4 h.
5. Implement the review's measured wins that keep results within rounding (no recipe
   change), including the CUDA-graph rewrite if the grid shows it pays. Budget by item.
6. The scripted teacher only where it is used once the teacher term is off (its league
   lane), and a compact teacher record (own rows, factored flight x fire) while it is on.
   Budget 2 h.
7. Independent correctness review of the merged tree, including the final review of the
   scripted agent (E5): silent mistakes (labels, signs, masks, off-by-one ticks), no speed
   work; ranked report, no changes. May run beside a non-timing item. Budget 3 h.
8. Fix the review's findings that are bugs (fix and report); anything that is a design
   question goes to the author.
9. Eval counters the persona study lacked: bullet hits and collision hits counted
   separately, a tail-chase share of hits. Small.

## Waiting on the author (not in the queue)

BC persona spread and adopting the re-centred yardstick, persona box and clocks (both need
R1's results); Muon (tabled); Liger-style kernels (tabled); the BC dry run itself (proposed
first).

Decided 2026-11-27 ~22:00: hidden enemies masked in the per-enemy losses (visible only, to
start); residual stream stays float32; compile baselines reset after item 1.

## Log

- 2026-11-27 ~21:45 queue written; R1 and R2 running.
- ~22:00 author: hidden enemies masked, float32 residual, baseline reset after item 1.
- ~00:20 R2 two-level league done (`league-pool` 10f9ae2b, unmerged). Agent's numbers: share-tracking total variation over 100 updates 0.010 (core's capacity rule 0.246); RL tick 28.65 ms at 1 snapshot (core 26.37), 33.06 at 12 (32.51); BC 25.51 (24.19, per-row scripted parameters); smoke run RL 10 updates + resume, 8,100 steps/s, no recompile; fixed a core bug in train.set_random_states (GPU resume). Open for the author: K/R, personas sharing one member's share (scripted members took 88% early from random weights), real personas from R1. Item 2 (hit-range port, `hit-range-2`, CPU only) started beside R1.
