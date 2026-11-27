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

- 8c Heads (`heads-8c`, lean/learn + lean/model).

## Done

R1 persona round 2, R2 two-level league, R3 persona round 3, 8a correctness fixes, 8b scripted delay, items 1 (merge + compile baselines), 2 (hit-range
port), 3 (docs cleanup), 7 (correctness review): see the Log.

## Queue (in order)

8a. Correctness fixes (from item 7; fix and report; all on one branch, tests once at the end):
   - Dead ships (author 2026-11-28 ~04:00, option C): after a ship dies, its value for the
     return is the mean value of its living team-mates (it has no observation of its own;
     a perfect critic on every team-mate predicts the same match outcome), so the outcome
     reaches its earlier ticks at gamma, not (gamma lambda)^k; the death penalty stays. When
     the whole team is dead the game has ended and the outcome arrives directly. Test with a
     perfect critic (the review's check: dying 30 / 100 / 300 ticks before the end).
   - The hit-class prior from today's measured rates (enemy 0.0070, team-mate 0.0015 per
     bullet), and the stale "four hits of about 21" comment; check value_knee's effect
     (review: negligible).
   - Team-shared payments split over living ships only (not dead ones).
   - Separate seeds for the explicit generator and torch's default generators.
   - Not a fix (author ~04:00): EMA / uniform slots taking new weights mid-game is fine (the
     live policy changes mid-game too, and the averages move more slowly).
8b. After R3: the scripted agent's one-tick action delay (its trigger and candidate paths act
   as if the action applied now; 22% of fire decisions move more than the target's half
   width before the shot leaves). Then a quick re-check of the persona centre (Elo of the
   fixed agent vs the unfixed one at 1v1 / 4v4 / 16v16, loose).
8c. Heads (author 2026-11-28 ~04:05), one branch:
   - next-state and hit losses on all visible ships (own and visible enemies); hidden masked.
   - A pairwise relative-geometry head (representation shaping, current state, exact targets
     from what the team sees). Per-ship projections (a shared ship-side projection, a
     type-specific other-side projection: ships, fields, zones later), pairs combined after
     the projections (u_i + v_j or u_i x v_j, GELU, linear per target); gating only in the
     per-ship stage. Targets (categorical bins, nats; 32 bins, 16 where coarse enough):
     ship-ship range, range rate, line-of-sight rate, energy difference (symmetric /
     antisymmetric: upper triangle only, antisymmetric negated), antenna train angle and gun
     solution error (both orders; aspect angle and heading crossing angle follow from ATA in
     both orders, so not predicted). Angles on a circle of bins (HL-Gauss smoothing wraps; no
     seam); gun error with bins concentrated near zero. Ship-field: signed distance to each
     field's edge (negative inside), symlog bins. Map edge: signed distance per ship on the
     per-ship heads. Pairs with a hidden or dead ship masked.
   - Cost control: a fixed number K of valid pairs sampled per environment-tick (fixed shapes:
     static compile and CUDA graphs unaffected; the draw is an input), sampled before the
     pair layers so cost scales with K; unbiased (mean over sampled valid pairs). Measure the
     logits and step cost at K = 4 / 8 / all.
   - Tests: targets from the game's own functions; circular bins wrap; symmetry; sampling
     unbiased; masks.
4. Second training hot-path review (measurements, report), on the tree with 8a-8c:
   - every head's rows match the decisions above;
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
9. Eval counters the persona study lacked: bullet hits and collision hits counted
   separately, a tail-chase share of hits. Small.

## Waiting on the author (not in the queue)

- After R3: the five persona decisions (yardstick = robust centre with crowd rows on?, persona
  sampler / box, BC spread, league personas, clocks: 60 s ends only 40/72/68% of games by
  elimination at 1v1 / 4v4 / 16v16).
- The value loss averaged over 2 zero-target placeholder levels (effective 0.03 vs the
  configured 0.05): average over the live levels only, or leave it.
- The two-level league's K and R; whether the personas share one member's PFSP share.
- Optional: a literature check on team critics / a super-token critic (MAPPO's death masking).
- Muon (tabled); Liger-style kernels (tabled); the BC dry run itself (proposed first).

Decided 2026-11-27 ~22:00: hidden enemies masked in the per-enemy losses (visible only, to
start); residual stream stays float32; compile baselines reset after item 1.
Decided 2026-11-28 ~04:05: dead ships option C; EMA/uniform mid-game reloads fine; the
scripted delay is a bug; next-state and hit on all visible ships; the pairwise head as in 8c.

## Log

- 2026-11-27 ~21:45 queue written; R1 and R2 running.
- ~22:00 author: hidden enemies masked, float32 residual, baseline reset after item 1.
- ~00:20 R2 two-level league done (`league-pool` 10f9ae2b, unmerged). Agent's numbers: share-tracking total variation over 100 updates 0.010 (core's capacity rule 0.246); RL tick 28.65 ms at 1 snapshot (core 26.37), 33.06 at 12 (32.51); BC 25.51 (24.19, per-row scripted parameters); smoke run RL 10 updates + resume, 8,100 steps/s, no recompile; fixed a core bug in train.set_random_states (GPU resume). Open for the author: K/R, personas sharing one member's share (scripted members took 88% early from random weights), real personas from R1. Item 2 (hit-range port, `hit-range-2`, CPU only) started beside R1.
- Item 2 done: hit-range tool ported (`hit-range-2` d93bf0ae, unmerged; to merge with R1/R2 in item 1). Agent's check at 200 m astern with the new defaults: DPS 186.7, 287/300 hits, implied TTK 0.536 s, actual ~0.58 s, 2-sigma 2.39 m straight / 12.7 m after the egg (287 lines within 12 m = the 287 hits), half-width 2.375 m; no-store verified. Porting bug (old egg_entry arguments) caught in review and fixed.
- 00:35 author: stop R1's CMA search (gen 1 already good enough; over-optimising risks a brittle agent tuned to the yardstick and eval conditions; no re-centred yardstick from it). R1 redirected to the study's goals, roughly and cheaply: which parameters matter for behaviour and for skill per size (Sobol), couplings and trade-offs, working ranges, iso-Elo diverse personas (league personas, BC good region). Budget ~1.5 h GPU.
- 00:55 author: R1 revised — re-centring the yardstick IS wanted (correction: the 00:35 line's 'no re-centred yardstick' was the coordinator's overreach); fit a skill surface from all games, robust centre (plateau, not peak), uncertainty-directed games, loose ladder over the top candidates + anchors with a small round-robin to test the ladder. Budget ~2 h GPU.
- ~02:20 R1 done (`personas-2` fc7de8f7; results-2.md). Agent's numbers: robust centre (centroid within 50 Elo of the fitted peak; Bradley-Terry quadratic on 627k games, 82.7% of explainable deviance) rates 1278 / 1482 / 2335 at 1v1 / 4v4 / 16v16 vs the old yardstick (1000), within 38 / 11 / 28 of the best tested; 7 rows' best region at the explored edge; ladder test vs round robin: Kendall tau 1.00, all within 0.8 combined SE, no cycles; today's 60 s clock ends 40/72/68% of games by elimination. Proposals for the author: robust centre as yardstick, proposal-2.json persona box, BC good region, league1/2/3/5/6, clocks. Item 1 started: merge league-pool + personas-2 + hit-range-2 (defaults unchanged), compile baselines reset.
- ~02:45 item 1 done: core fast-forwarded to ce27fe64 (league-pool, personas-2, hit-range-2; no defaults changed; two-level league entry left PROPOSAL). Merge agent's checks: quick tier 362 passed, slow learn 6 passed, ruff lean clean; compile baselines reset (act 4,535, advance 3,194, minibatch_loss 861, slot_actions 720, merge_league 28; cold 120.9 s, budget 150 s), check passes. Item 3 (docs cleanup, `docs-cleanup`) started.
- ~03:00 author: extend the persona study to capture the whole summit (box edges were arbitrary) and draw personas from a Gaussian with covariance proportional to the inverse curvature of the Elo surface. R3 started (`personas-3`, game statistics beside the docs cleanup; budget ~1.5 h GPU). The five persona decisions wait for R3.
- ~03:00 item 3 done: core fast-forwarded to 5b54fccb (docs cleanup: duplicate questions file folded into docs/questions.md; 65 closed briefs/notes and ~80 dead scripts deleted, all still on the v2 branch or a named commit; docs ruff 2,424 -> 0 with a per-file E501 ignore for docs/**/*.py in pyproject.toml; package notes checked against the code; sections/INDEX.md). For the author: value_knee (lean/learn/config.py:166) still derives from a 21-point hit while hit damage is 8. Item 7 (correctness review, read-only, no GPU) started beside R3; item 4 waits for R3 (timing work runs alone).
- ~04:00 item 7 done (`correctness-review` 2319245f, report docs/rewrite/sections/correctness-review-2026-11-28.md). Findings: (1) a dead ship's outcome target decays by (gamma lambda)^k after death (horizon 33 ticks instead of 3,333): dying early on a losing team avoids the -10 — design call for the author; (2) the enemy-head decision (visible enemies trained, hidden masked) is not built: heads train on own ships only, no pairwise head; (3) EMA/uniform league slots reload weights mid-game; (4) the scripted trigger and paths ignore the one-tick action delay (22% of fire decisions move more than the target's half width); (5) hit-class prior 8x off (0.0547 vs 0.0070 measured), stale "hits of about 21" comment; (6-8) low: dead ships' shares of team payments, value loss averaged over 2 zero-target levels (effective 0.03 vs 0.05), the same seed in two generators. Item 8 (fixes) waits: (4) after R3 (it changes the scripted agent R3 measures); (1) and (7) to the author.
- ~04:05 author decisions recorded and queue reordered: 8a correctness fixes (dead ships option C, hit prior, team shares, seeds), 8b scripted delay after R3, 8c heads (visible ships, pairwise relative-geometry head with sampled pairs, field and map-edge signed distances), then 4 (training review on that tree), 5, 6, 9.
- ~04:08 item 8a started (`fixes-8a`, lean/learn only, beside R3).
- ~04:12 author: the two-level league is the default and the only league (the old capacity rule is already gone from the code; core-decisions entry marked decided). Clock times in the 04:xx lines corrected (were estimates).
- ~04:30 R3 done (`personas-3` 5018e6da, merged into core; results-3.md, proposal-3.json). Agent's numbers: round 2's centre was not the top (69% of 96 Sobol points over the enlarged top region beat it); steep rows bounded, plateau rows (collision, threat, power rows, fire tolerance upward, temperature downward) flat to the hard end; robust-centre-3 vs round 2's centre -8 / +102 / +194 (surface predicted +453 / +334 / +533: over-extrapolates at the edge); with the old yardstick at 1000 it rates 1140 / 1587 / 2407. Gaussian sampler as briefed dropped 104 / 79 / 103 Elo for targets 25 / 50 / 100; the cause is temperature (a cliff on the warm side); the same personas with temperature reflected colder dropped 23. League personas league3-1..5 within about 70 of the centre, wider team-style spread than round 2. The five persona decisions go to the author. 8b started (`scripted-delay`).
- ~04:45 8a done (`fixes-8a` 409b7be1, merged into core). Agent's numbers: perfect-critic target at the ship's last living tick, dying 30 / 100 / 300 ticks before the end: 3.974 / 0.462 / 0.001 before, 9.910 / 9.704 / 9.139 after (ideal 10 gamma^k), survivor unchanged; hit prior re-measured over 1.6M bullets (scripted 4v4, 4,200 ticks): miss 0.9865 / enemy 0.0110 / team-mate 0.0025 (the review's 0.0070 / 0.0015 came from mostly approach ticks); damage value_knee 0.0286 -> 0.0109; team payments over living ships; generator seed + 1,000,003. Quick tier lean/learn, model, eval 169 passed; slow learn 6 passed; ruff clean. For the author: on the damage and kill_death levels a team-mate's value also holds that team-mate's own future damage and kills, so option C credits a dying ship's last ticks with most of it; exact only on the outcome level (could restrict C to the outcome level). 8c started (`heads-8c`, lean/learn + lean/model, beside 8b; its timing waits for a quiet GPU).
- ~05:10 8b done (`scripted-delay` b991d706, merged into core). Agent's numbers: fire decisions more than the target's half width off at the shot 20.9% -> 0.4% (yardstick), 15.4% -> 0.2% (centre 3), 1v1 CPU 128 x 1800 ticks; the fixed agent fires 20-28% more and lands 20-29% more hits; Elo fixed vs unfixed (both robust-centre-3) +34 ± 8 / +18 ± 5 / +30 ± 8 at 1v1 / 4v4 / 16v16; scripted act 6.95 -> 7.13 ms (+2.6%), first call 33 -> 40 s; quick tier lean/scripted 31 passed, ruff clean. The centre still holds (gain uniform; not re-tuned; fire_tolerance 4.89 was tuned against the late aim). Item 4 waits for 8c.
