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

- 13 Shrinking zone (`zone-13`, lean/game + learn + scripted).

## Done

R1 persona round 2, R2 two-level league, R3 persona round 3, 8a correctness fixes, 8b scripted delay, 8c heads, items 4 (second training review), 5 (speed wins), 6 (teacher), 9 (eval counters), 10 (morning fixes), 11 (gradient tools), 12 (game rules), 1 (merge + compile baselines), 2 (hit-range
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

10. Fixes (author 2026-11-28 ~13:30), one branch:
   - Option C on the global levels only: the outcome (and the objective levels when they are on).
     Damage and kill_death are each ship's own: a dead ship's value there stays 0 (as before 8a).
   - The objective levels (objective_progress, objective) optional, off by default (unused in
     today's training): no value bins, value loss, returns or store rows for them when off;
     kept in the code for zones later. This also settles the value loss over placeholder levels.
   - Hit-label bias: mask the hit loss on bullets that cannot report a miss, i.e. those fired in
     the last bullet life (12 ticks) before an episode end or before the update's end.
   - Cold-compile budget raised to 175 s (author ~14:00).

11. (author 2026-11-28 ~15:30) One branch:
   - Entropy bonus 0.003 in BC too (the 2026-11-24 decision "on for the whole run").
   - Port v2's gradient decomposition (src/bnb/learn/gradients.py) to lean: per-term gradient
     norm and share of the step, trunk cosines, group agreement, coherent fraction (two
     halves), actor/critic share; value by level, next-state by column, geometry by head/target.
     On a cadence, off by default (speed).
   - Then the diagnostic: a few BC updates (and early RL) with it on, at fresh weights and after
     a few BC updates; report per-term trunk shares and cosines with the policy.

12. Game rules (author 2026-11-28 ~16:30, after the replays showed duels degenerate into nose-to-nose
   fire and mutual collisions), one branch:
   - Nose-cone armour back on for bullets: no damage within 30 deg of the target's nose, a taper
     to full at 45 deg (as before). Visuals (an "absorbed" flash) later.
   - The shared aspect function on the sine of the angle off the nose (|cross(nose, n)|, no
     trig; contacts behind the beam count as beyond the cone), same smootherstep, two angle
     pairs: bullet armour 30-45 deg; collision blame 0-45 deg (full blame dead-on, half by 45).
   - The deflector a circle (radius 24 m, the egg's side and rear) instead of the egg, same
     early-hump turn; scripted gun model, hit-range tool and tests follow.
   - Vision an egg with three settings, ahead / side / behind: defaults 1,500 / 850 / 850 m
     (replaces the ellipse's eccentricity).
   - Then: scripted Elo new rules vs today's (1v1 / 4v4 / 16v16), collision-death share,
     mutual-kill share, tail-chase share; new replays. A persona re-run is proposed separately.

13. Shrinking zone (author 2026-11-28 ~17:30), after item 12 (same game core):
   - The arena radius per game shrinks linearly from spawn, 4,200 m at 35 m/s: 0 at 120 s, and on
     past 0 to minus the ramp (-300 m at ~129 s) so everyone takes full boundary damage. No clock
     end: every game ends by elimination (one team or both). The longest possible game is derived
     (full damage from ~129 s, max shield / net boundary damage rate after recharge and any
     boundary healing) and asserted, not set by hand.
   - Observation laid out for staged zones later: the super token reads the current circle
     (centre, radius), the next circle (centre, radius) and the time to the stage's end (here:
     the same centre, the radius at the end of the stage); per-ship signed distance to the
     current edge. Check the map-centre reference and position codes with a moving centre.
   - Spawns clear of the shrinking band; the scripted agent's boundary term reads the current
     zone; tests; replays.
   - Later (not now): battle-royale stages (hold, then shrink to a random circle inside the
     current one, repeat until gone).
14. Later: a more resilient scripted agent: a sensitivity study that also varies the game's own
   settings (field strengths, ship parameters), so small rule changes don't need hours of
   re-tuning.

15. Calibration under the new rules and the zone (after 13): re-measure the value supports, the
   next-state column ranges, the geometry heads' ranges (pair, field, map edge) and their bin
   balance, the hit-class prior, and check the feature scales. Units stay physical (1 m is 1 m,
   1 m/s is 1 m/s, whatever the map size); only the super token's zone features carry the map's
   size. Ranges are re-measured, never rescaled by the arena.
16. BC dry run (author authorised 2026-11-28 ~18:00, overnight, coordinator's judgement): one
   teacher and one scripted persona, the default ScriptedConfig() (also the league's only
   scripted member and the yardstick); 4v4 training game as the recipe; 96 updates (~100M
   ticks); gradient diagnostics every 16 updates; checkpoints as configured. Loss weights by the
   literature (literature-loss-balance.md): value 0.25 (author confirmed ~18:15; categorical
   critic on a shared trunk, MuZero family), auxiliaries unchanged (small fixed, nats), entropy 0.003 (decided), BC 1.0.
   Watch: no NaN / recompile, bc_kl and the losses falling, gradient shares sensible (no term
   near 99%). Then rate the live and averaged students against the yardstick (4v4, plus a
   small 1v1 / 16v16 check). Success (author): a 0.4 win rate against the teacher within the
   first 100M ticks.
17. RL dry run from the BC checkpoint if BC trains and the student plays (no crash, losses
   sane, the student wins games against the teacher): ~96 updates (~100M ticks), defaults
   (two-level league K 12 / R 4, teacher term with decay and gate), diagnostics every 16;
   rate the result against the yardstick and the BC student.
   Rules overnight: fix bugs found on the way, note every change and choice in the Log and in
   docs/rewrite/sections/dry-run-2026-11-28.md; no major redesign without the author.

18. After 17, until the author is back (no hard deadline; author ~18:30): keep the GPU busy
   with long, low-token experiments chosen by the coordinator from what 16-17 show, one at a
   time, each rated and written up. Candidates, in order: extend RL (`--resume --updates`) while
   it is still improving; a second seed of BC -> RL (run-to-run spread); one-arm ablations
   (geometry heads off; value 0.05 vs 0.25); power modes' steady speed. No redesign; every run
   traced to committed settings.

## Waiting on the author (not in the queue)

- After R3: the five persona decisions (yardstick = robust centre with crowd rows on?, persona
  sampler / box, BC spread, league personas, clocks: 60 s ends only 40/72/68% of games by
  elimination at 1v1 / 4v4 / 16v16).
- Whether the personas share one member's PFSP share (K 12 / R 4 decided ~14:00).
- Optional: a literature check on team critics / a super-token critic (MAPPO's death masking).
- Loss balance (policy, value, BC/teacher, auxiliary heads): literature-loss-balance.md done; the gradient diagnostic (item 11) next, then a proposal; 8c's coefficients and K stay provisional until then.
- Muon (tabled); Liger-style kernels (tabled); the BC dry run itself (proposed first).
- 8c's provisional geometry coefficients (pair 0.1, field 0.05, edge 0.05) and K = 8.
- micro_batch_views (8) and policy_step_chunks (6) are per GPU: re-measure on the 5090.

Decided 2026-11-27 ~22:00: hidden enemies masked in the per-enemy losses (visible only, to
start); residual stream stays float32; compile baselines reset after item 1.
Decided 2026-11-28 ~13:30: option C on the global levels only (outcome; objective levels when on); objective levels optional and off by default; mask hit labels of bullets fired in the last 12 ticks before an episode or update end.
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
- ~05:00 8b done (`scripted-delay` b991d706, merged into core). Agent's numbers: fire decisions more than the target's half width off at the shot 20.9% -> 0.4% (yardstick), 15.4% -> 0.2% (centre 3), 1v1 CPU 128 x 1800 ticks; the fixed agent fires 20-28% more and lands 20-29% more hits; Elo fixed vs unfixed (both robust-centre-3) +34 ± 8 / +18 ± 5 / +30 ± 8 at 1v1 / 4v4 / 16v16; scripted act 6.95 -> 7.13 ms (+2.6%), first call 33 -> 40 s; quick tier lean/scripted 31 passed, ruff clean. The centre still holds (gain uniform; not re-tuned; fire_tolerance 4.89 was tuned against the late aim). Item 4 waits for 8c.
- ~05:20 8c done (`heads-8c` 619fd59c, merged into core; merged tree: quick tier lean/scripted + lean/game 158 passed, ruff clean). Agent's numbers: geometry targets from new game functions (core.py "relative geometry"); next-state and hit on own living + visible enemies; pairwise head (gated per-ship projections, GELU(u_i + v_j), a linear per target), field edge and map edge; ranges from 0.70M visible pairs (99.9% intervals); bullet along the gun solution passes its target within 3 cm. Cost per micro-batch (quiet GPU): off 150.2 ms, K=4 162.1, K=8 165.4 (+10.1%, +0.35 GiB), all 193.1 (+28.6%); most of K=8's cost is fixed (projections, targets, unrolled per-column loss). Cold compile 141.6 s (budget 150 s); loss graph 861 -> 1,627 ops, baseline reset on the branch. Provisional for the author: pair 0.1 / field 0.05 / edge 0.05 coefficients, K=8; pre-8c checkpoints load only with geometry_heads=False. Item 4 started (`review-4`).
- ~06:50 item 4 done (`review-4` 25e2aeec, merged into core; report training-review-2.md). Agent's numbers (laptop 4070, quiet): update 165.6 ms per micro-batch (heads off 150.2), GPU busy 98%, host floor 6.0 ms; BC tick 25.34 ms, RL tick 27.70 ms. Loss rows match the decisions; one bias found: bullets still in flight in the last ~11 ticks before an episode or update end never report, so hits are labelled there and misses are not (est. +2-4% relative hit share) — mask those windows or label them missed: for the author. Smaller micro-batches are faster per view (ms per 64 views: 8 140.9, 16 147.8, 64 165.0; peak 4.6 -> 1.2 GiB at 16). Checkpointing +15-20%: not proposed. max-autotune-no-cudagraphs -1.3% / -1.2% with 300-500 s cold compile; reduce-overhead fails on both graphs as written; CUDA graphs worth at most ~2-3% here (GPU-bound). Live-policy step in 6 seat chunks: BC tick -12.1%, RL -10.8%. Item 5 started (`speed-5`): micro-batch size, seat chunks, critic support on device.
- ~08:34 item 5 done (`speed-5`, merged into core). Agent's numbers (laptop 4070, quiet): micro_batch_views 64 -> 8 (losses equal to 7 digits, gradient within bf16 rounding; BC update alone 124.8 -> 107.4 s, peak 5.13 -> 1.32 GiB); live-policy step in 6 seat chunks via nested_compile_region (static compile only; equal to one call to 4e-8): BC tick 25.34 -> 20.87 ms, RL 27.70 -> 23.22 ms; critic support cached (bitwise). Smoke on the real trainer, 3 updates, no recompile: BC 7,605 -> 8,933 / 8,747 steps/s, RL 7,545 -> 8,678 / 8,786. Quick tier 179 passed, slow learn 6 passed, ruff clean. Cold compile over the 150 s budget already on core before item 5 (160.3 s, after the 8b + 8c merges; not bisected), 164.3 s now; baseline counts reset, budget not raised: for the author. Both new settings are per GPU: re-measure on the 5090. Item 6 started (`teacher-6`).
- ~09:49 item 6 done (`teacher-6` 3f8c379b, merged into core). Agent's numbers: bisect of the cold compile: 8b alone 139.2 s, 8c alone 151.5 s (recorded 141.6 s: ~10 s run-to-run), both 163.9 s (8b +1,630 ops / +9.5 s in act; 8c +766 ops / +13.6 s in the loss). Scripted agent moved to its own graph, run on league environments always and on self-play only while the teacher is used (switching off compiles nothing; covered by the static test); teacher record own rows only, float16 flight 21 + fire 1: store 3,345.5 -> 2,601.5 MiB (teacher on), 2,337.4 MiB (off); teacher off saves 2.8 ms a tick, 1.4 s an update (inside the ~1.5 s spread); BC losses equal core to the printed digits; old vs new record cross-entropy within 1.2e-7. Cold compile 164.7 s, budget 150 s kept: for the author. Quick tier 212 passed, slow learn 6 passed, ruff clean. Item 9 started (`counters-9`).
- ~10:11 item 9 done (`counters-9` 9c3a589e, merged into core; merged tree quick tier lean/learn + scripted + model 179 passed). The old `hits` lumped bullets and collisions (StepOut.hit_damage is their sum); StepOut now also gives bullet_damage and collision_damage; eval style gains bullet_hits, collision_hits, tail_hits (shooter in the target's rear cone, |train angle| > TAIL_CHASE_ANGLE 120°, new core.train_angle) and the styles view a tail_share. Agent's CPU sanity run (yardstick mirror 4v4, 192 games, mean 44.8 s): bullet hits 20.1 per ship-minute, collision hits 1.03, tail share 0.434. Game version hash changed (new eval conditions). The queue is empty; the rest waits on the author.
- ~13:35 author: option C on the global levels only; objective levels optional, off by default; hit-label bias masked; worktrees merged into core pruned (76; two agent worktrees under main/.claude kept: uncommitted files). Item 10 started (`fixes-10`).
- ~14:00 author: compile budget 175 s (item 10 sets it); league K 12 / R 4 kept; fire_tolerance / lead_error re-check not needed now; loss-balance literature check running.
- ~15:00 item 10 done (`fixes-10` 7c2703d2, merged into core). Option C on the outcome only (dead ships' damage / kill_death values 0 again); objective levels off by default (LearnConfig.objective_levels; value head 3 levels; pre-item-10 checkpoints need value_levels=5 + objective_levels=True); hit labels of the last 11 rows before an episode or update end masked (exact rule; core's RL hit share was +2.3 / +8.5 / +5.4% high, the branch's equal near and away from the ends); compile budget 175 s. Back-to-back BC (both at load ~4.5-6, the author's browser open): core 108.7 / 110.9 s per update (8,756 / 8,612 steps/s), branch 106.4 / 106.5 s (8,967 / 8,946); store 2,601.5 -> 2,409.4 MiB; peak 1.16 -> 1.06 GiB; end_segment +0.1 ms. An earlier +8-11% branch slowdown was the busy machine. Cold compile 189.6 s at load 7.0 (fails 175 s; core 170.3 s at load 6.8): to re-measure quiet. Quick tier 222 passed, slow learn 6 passed, ruff clean.
- ~15:06 item 10 follow-up (`fixes-10` 9ccd4e7a merged: notes): cold compile on the branch 164.8 s at load 2.9 (passes 175 s), 189.6 s at load 7.0; core 170.3 / 174.7 s at load 6.8 / 7.0. Little headroom on a loaded machine.
- ~15:30 author: entropy 0.003 in BC too; port v2's gradient tools to lean and run the gradient diagnostic. Item 11 started (`gradients-11`).
- ~16:09 item 11 done (`gradients-11` 51320307, merged into core). Entropy 0.003 in BC; lean.learn.gradients (v2's decomposition: norms, shares, trunk cosines, agreement, coherent fraction, actor share; --gradient-diagnostics-every N, off by default; ~30 s per measured update); fixed geometry.draw_keys treating cuda / cuda:0 as different devices (GPU pair keys came from the default generator). Diagnostic, fresh weights, 6 BC + 3 RL updates, first minibatch, raw gradients: BC trunk shares bc 0.93, value 0.008, next_state 0.010, pair 0.004, field 0.054, edge -0.003, entropy -0.002; RL teacher 0.82, next_state 0.08, value 0.05, pair 0.03, edge 0.03, field 0.01, surrogate -0.02 (40x smaller norm than the teacher, coherent fraction ~0: noise this early). No auxiliary over 50% in any update; in BC pair and field point against bc (cosine negative in 6/6 and 5/6 updates). Quick tier 199 passed, slow learn 6 passed, ruff clean, compile 168.6 s at load 6.6 (budget 175), graphs unchanged.
- ~16:10 author (replays): the round-3 centre and league personas are degenerate in duels: nose to nose, firing, colliding (often both die; inside-egg shots get through). Discussing front armour by hit angle with an "absorbed" visual and collision damage; nothing built yet.
- ~16:30 author: nose-cone armour back; shared aspect function on the sine (armour 30-45, collision blame 0-45); circle deflector 24 m; egg vision 1,500 / 850 / 850 m. Item 12 started (`rules-12`).
- ~17:30 author: shrinking zone as item 13 (linear 35 m/s from spawn, past 0, no clock; stages later; observation laid out for stages); a resilient scripted agent as a later item 14. Item 12's behaviour check trimmed (new rules with the default yardstick at 1v1 / 4v4; old rules with the centre at 1v1 as the baseline).
- ~17:50 item 12 done (`rules-12` 0c07a97b, merged into core). Nose armour on; aspect function on the sine (armour 0 / 0 / 0 / 0 / 0.547 / 1 at 0 / 10 / 22.5 / 30 / 37.5 / 45 deg; blame 1 / 0.951 / 0.712 / 0.577 / 0.511 / 0.5); circle deflector 24 m (hit width 4.75 m from every aspect); sight egg 1,500 / 850 / 850 m. GPU tick 0.738 -> 0.735 ms (noise); compile 11,439 ops, cold 166.4 s (budget 175). Behaviour (mirror games, fields + fog): centre under core rules 1v1: tail-chase share 4.9%, collision deaths 4.0%, mutual 6.6% of fatal collisions, 1.00 kills/min; yardstick under the new rules: 1v1 tail-chase 75.7%, collision deaths 0.7%, 0.43 kills/min, 37.0% end by elimination; 4v4 tail-chase 53.4%, collision deaths 3.1%, 5.68 kills/min. The round-3 centre breaks under the new rules (collision deaths 21.0% at 1v1, 41.5% of fatal collisions mutual). The degeneracy shows in the tail-chase share. Replays data/replays-2026-11-28-rules12/. Quick tier 408 passed + the fixed style test (merged tree: eval + game + scripted 197 passed); slow learn 6; ruff clean. Stale under the new rules: the hit-class prior, feature scales. Item 13 started (`zone-13`).
- ~18:00 author (leaving until ~10:00 tomorrow): after 13, calibrate (15), then BC dry run (16) and RL from its checkpoint (17); one teacher / one persona; loss weights by the literature, watched with the gradient diagnostics; coordinator's judgement for loose ends, every change noted.
- ~18:15 author: value 0.25; BC success = 0.4 win rate vs the teacher within 100M ticks; the scripted agent must see the actual (current) edge; compile 166 s fine; the coordinator may close the author's browser and other programs; if thermals are high, try the power modes (performance / balanced / power-saver) or power limits and measure steady speed in each, at the coordinator's judgement.
- ~18:30 author: no hard deadline; after the RL dry run keep the GPU busy with long low-token experiments by the coordinator's judgement until the author returns (item 18).
