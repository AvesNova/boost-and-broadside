# Sections: index

Results, evidence and progress records, one line per entry: what it is, and what cites it.
Closed one-off briefs and notes were deleted on 2026-11-28; the v2-era ones are on the `v2`
branch (`git show origin/v2:docs/rewrite/sections/<name>`), the lean ones in this branch's
history.

## Lean core (from 2026-11-22)

| Entry | What it is | Cited by |
|---|---|---|
| bullet-deflector/ | the deflector and egg: progress, sketches, egg-experiments.md, workbench numbers | core-decisions.md, lean/game/NOTES.md, lean/tool/hitbench.py, test_deflector.py |
| clock-measurement/ | clock.py and report.json behind the 60 s clock | clock-measurement.md |
| energy-duel/ | energy duel, the 45-degree exchange evidence (exchange45.md) | core-decisions.md "Landed 2026-11-25", lean/game/tuning.py |
| eval-scale/ | evaluation runner at scale and routing | lean/eval/NOTES.md, core-worklog.md |
| first-touch/ | collisions at first touch | core-worklog.md |
| framework-spike/ | PyTorch against JAX benchmark (the framework choice) | core-decisions.md "Framework" |
| game-audit.md | v2 game audit and port map for the lean core (2026-11-22) | departure-audit.md, imugi-behaviours.md (unsure: no current decision cites it) |
| hot-path-audit.md | game hot-path audit (2026-11-27) | hot-path-audit-progress.md; the decisions on it are in core-decisions.md "Static training graphs" |
| hot-path-audit-progress.md | the approved hot-path fixes (branch tick-audit) | core-decisions.md "Static training graphs" |
| imugi/ | 2026-11-22 measurement scripts of the old scripted agent | imugi-behaviours.md |
| lead-turn-yo-yo/ | lead turn and yo-yo sweep (agent since replaced) | lead-turn-yo-yo-review.md |
| league-pool/ | two-level league progress and numbers | core-decisions.md, lean/learn/NOTES.md, test_league.py |
| lookahead-personas/ | persona studies 1 and 2 (results.md, results-2.md, proposal-2.json) | docs/questions.md 17, lean/learn/config.py, lean/scripted/NOTES.md |
| nose-cone/ | nose cone and collisions | core-worklog.md |
| post-stall/ | post-stall regime, literature.md | core-decisions.md, core-backlog.md A13, lean/game/card.py |
| replay/ | replay video script and notes | (unsure: not cited) |
| scoreboard/ | compile check, op counts, feature and target ranges, GPU training scripts | lean/README.md, lean/*/NOTES.md, lean/learn/config.py, scoreboard.md |
| scripted-ghosts/ | scripted ghosts, head-on fire and the head-on stalemate (2026-11-25) | (unsure: not cited) |
| scripted-lookahead/ | look-ahead agent progress, rounds 1 to 3 | core-decisions.md, lean/scripted/NOTES.md, opponent.py, parameters.py |
| scripted-reach/ | scripted reach measurements | scripted-reach.md |
| scripted-research/ | references on scripted dogfight AI (2026-11-26) | (unsure: not cited) |
| scripted-screening/ | Sobol screening of the BFM agent | core-worklog.md, scripted-screening.md, scripted-divergence.md |
| shield-target/ | why most nonzero shield targets are tiny | (unsure: not cited) |
| simplification/ | op-count scripts behind simplification.md | scoreboard.md, simplification.md |
| tick-audit/ | benchmarks of the tick-audit fixes | (unsure: numbers in core-decisions.md) |
| tracking-circle/ | the tracking circle | core-decisions.md |
| training-path-audit.md | training-path audit (2026-11-27) | core-decisions.md "The training-path fixes" |
| training-path-audit-progress.md | the approved training-path fixes (branch train-audit) | core-decisions.md |
| turns/ | turn charts, doghouses, measure_core.py | lean/tool/design.py, lean/game/tuning.py, lean/tool/NOTES.md |
| vision/ | vision range measurement | core-decisions.md |
| vision-ellipse/ | sight ellipse | core-decisions.md |

## v2 section documents (sections 1 to 11)

Specs, section write-ups and notes of v2's code in `src/bnb`, cited by `docs/roadmap.yml`
and the v2 site under `docs/explanation`.

| Entry | What it is | Cited by |
|---|---|---|
| game.md, game-spec.md, game-notes.md | section 1, the game | roadmap.yml, explanation/game/physics.md, src/bnb/game |
| ship-card.md, ship-card-spec.md, ship-card-notes.md | section 1b, the ship card | roadmap.yml, explanation/game/ship-card.md, tests |
| modes.md, modes-spec.md, modes/ | section 3, modes; mode tuning | roadmap.yml, explanation/game/modes.md |
| rewards.md, rewards-spec.md | section 3b, rewards | roadmap.yml, explanation/game/rewards.md |
| mixed-spec.md, mixed-notes.md | section 3c, mixed batches | roadmap.yml, explanation/game/mixed-batches.md |
| scripted.md, scripted-bc-spec.md, scripted-memory-notes.md | section 4, Imugi | roadmap.yml, explanation/imugi.md; scripted.md also by lean/scripted |
| yemong.md, yemong-spec.md, yemong-notes.md, fast-yemong-notes.md | section 5, the model | roadmap.yml, explanation/yemong.md, simplification.md |
| learner.md, learner-spec.md, learner-notes.md | section 6, the learner | roadmap.yml, explanation/learner.md, tests |
| bc.md | section 7, behaviour cloning | departure-audit.md, model-audit.md |
| league.md, league-spec.md | section 8, league and ratings | roadmap.yml, explanation/league.md; league.md also by lean/learn |
| diagnostics-spec.md | section 9, diagnostics | roadmap.yml |
| play.md, play-spec.md, play-m2-spec.md | section 10, the play shell | roadmap.yml |
| runs.md, runs-spec.md | section 11, runs | roadmap.yml |
| cleanup-spec.md | v2 cleanup pass spec | (unsure: not cited) |

## v2 research and arm records

| Entry | What it is | Cited by |
|---|---|---|
| acceleration-inputs-brief.md, arms34-brief.md | round-2 arm briefs, kept for their citations | bench/acceleration_inputs_measure.py, explanation/yemong.md |
| baselines-bench.md, baselines-bench-tables.md, baselines-bench-brief.md | one-tick baselines bench | bench/baselines_*.py, explanation/yemong.md |
| state-deltas.md, state-deltas-tables.md, state-deltas-brief.md | per-tick state deltas and target codes | bench/state_deltas_*.py |
| belief-baselines.md | belief and next-state baselines | (unsure: not cited) |
| round2-plan.md, round2-log.md | round 2 plan and log | explanation/yemong.md |
| league-sampling-sim.md | league sampler simulation | bench/league_sampling*.py, explanation/league.md |
| rl-round-five-notes.md, rollout-lanes-notes.md, dppo-notes.md | RL round five, rollout lanes, DPPO reading | explanation/league.md, explanation/training-loop.md, references.md |
| seat-symmetry-notes.md | the seat-symmetry bug | tests/sections/test_seat_symmetry.py |
| portable-batch.md | portable batches decisions | backlog.md |
| compile-overhaul/, compile-overhaul-notes.md, compile-split/ | v2 compile work | review.md, simplification.md, src/bnb |
| gpu-box/, gpu-box-log.md | rented 3090 log | backlog.md, model-audit.md |
| model-audit/ | blind review of the v2 learner | model-audit.md |
| temperature-sweep.md, temperature-sweep/ | temperature sweep (2026-11-11) | (unsure: not cited) |
| yardstick.md | gate G3: v1's recipe, v2's numbers | departure-audit.md |

## v2 release-push records (2026-11-12 to 2026-11-15)

Progress logs and data of the v2 release push; not on the `v2` branch, so kept. Cited only
where noted; otherwise unsure.

| Entry | What it is | Cited by |
|---|---|---|
| archetypes/, balance-sizes/, doctrines/, gun-study/, tank-armour/ | ship archetype and balance studies | explanation/game/ship-design/archetypes.md (balance-sizes, doctrines, gun-study) |
| bc-v19/ | overnight log 2026-11-14 | departure-audit.md, model-audit.md |
| evaluation-suite/ | `bnb evaluate` design pilot | evaluation-suite.md |
| frontline-lock/, front-reversal/, mode-timing/ | frontline front and game length | briefing.md, src/bnb/game/modes/rules.py |
| imugi-rename/ | the Imugi rename | arms-todo.md |
| launch/ | launch progress | explanation/behaviour-cloning.md, guides/queueing-jobs.md |
| lethality/, maneuvers/, maneuvers-2/ | Imugi lethality and manoeuvres | explanation/imugi.md, src/bnb/game, tests |
| position-codes/, steady-speed/ | position codes, steady speed | src/bnb, bench/rotary_aliasing.py |
| rating-allocator/ | rating allocator and shared pool | explanation/league.md |
| absent-fields/, break-even/, bug-sweep/, bullet-damage/, bullet-life/, cheese-agents/, elimination/, fixed-maps/, game-loop/, map-panel/, micro-batch-check/, overnight-calibration/, release-candidate/, seed-isolation/, spawn-heading/, training-defaults/, zero-sum-rewards/ | progress logs and data of closed v2 work | (unsure: not cited) |
