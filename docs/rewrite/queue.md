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
  crowd-dependent parameters; (redirected 00:55) a Bradley-Terry skill surface over the parameters from all games, its robust centre as the proposed yardstick and persona centre, a loose adaptive ladder as validation and as a test of the ladder. Game statistics (not timing-sensitive).
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
- Item 2 done: hit-range tool ported (`hit-range-2` d93bf0ae, unmerged; to merge with R1/R2 in item 1). Agent's check at 200 m astern with the new defaults: DPS 186.7, 287/300 hits, implied TTK 0.536 s, actual ~0.58 s, 2-sigma 2.39 m straight / 12.7 m after the egg (287 lines within 12 m = the 287 hits), half-width 2.375 m; no-store verified. Porting bug (old egg_entry arguments) caught in review and fixed.
- 00:35 author: stop R1's CMA search (gen 1 already good enough; over-optimising risks a brittle agent tuned to the yardstick and eval conditions; no re-centred yardstick from it). R1 redirected to the study's goals, roughly and cheaply: which parameters matter for behaviour and for skill per size (Sobol), couplings and trade-offs, working ranges, iso-Elo diverse personas (league personas, BC good region). Budget ~1.5 h GPU.
- 00:55 author: R1 revised — re-centring the yardstick IS wanted (correction: the 00:35 line's 'no re-centred yardstick' was the coordinator's overreach); fit a skill surface from all games, robust centre (plateau, not peak), uncertainty-directed games, loose ladder over the top candidates + anchors with a small round-robin to test the ladder. Budget ~2 h GPU.
- ~03:10 R1 done (`personas-2` fc7de8f7; results-2.md). Agent's numbers: robust centre (centroid within 50 Elo of the fitted peak; Bradley-Terry quadratic on 627k games, 82.7% of explainable deviance) rates 1278 / 1482 / 2335 at 1v1 / 4v4 / 16v16 vs the old yardstick (1000), within 38 / 11 / 28 of the best tested; 7 rows' best region at the explored edge; ladder test vs round robin: Kendall tau 1.00, all within 0.8 combined SE, no cycles; today's 60 s clock ends 40/72/68% of games by elimination. Proposals for the author: robust centre as yardstick, proposal-2.json persona box, BC good region, league1/2/3/5/6, clocks. Item 1 started: merge league-pool + personas-2 + hit-range-2 (defaults unchanged), compile baselines reset.
- ~03:40 item 1 done: core fast-forwarded to ce27fe64 (league-pool, personas-2, hit-range-2; no defaults changed; two-level league entry left PROPOSAL). Merge agent's checks: quick tier 362 passed, slow learn 6 passed, ruff lean clean; compile baselines reset (act 4,535, advance 3,194, minibatch_loss 861, slot_actions 720, merge_league 28; cold 120.9 s, budget 150 s), check passes. Item 3 (docs cleanup, `docs-cleanup`) started.
