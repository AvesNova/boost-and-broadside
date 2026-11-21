# Index of v2's notes

Which file discusses which subject, for `docs/` and `docs/rewrite/` on the `v2` branch as of
2026-11-21 (589 entries). It lists subjects only: no findings or conclusions. The notes
record v2's design and conditions, and many of their conclusions do not carry over. Read a
file when working on its subject, as evidence under v2's conditions.

Method: one pass by four agents over every file (topics mostly from headings and a skim),
then a second pass over the 36 logs, notes and reviews of 300 lines or more. The second
pass read headings and sampled passages, not every line, so subjects buried deep in long
logs and table bodies may be missing. Rows marked `(other files)` summarise a folder's
non-markdown files (results, scripts, configurations, images).

| Path | Lines | Kind | Dates | Topics |
|---|---|---|---|---|
| docs/contributing/code-style.md | 231 | guide | 2026-11-06 | Tooling; layers; naming; contracts and types; tensors; randomness; configuration; docstrings and comments; errors; tests; commits |
| docs/contributing/writing-docs.md | 189 | guide | 2026-11-02 to 2026-11-03 | Docs build and preview; kinds of page; code links; generated blocks; fences and examples; voice; doc checks |
| docs/explanation/architecture.md | 151 | explanation | 2026-11-02 | Layers and dependency rule; pure core and stateful edge; compiled stages; CUDA graphs; random draws; shapes and layout; one code path |
| docs/explanation/behaviour-cloning.md | 266 | explanation | 2026-11-02 to 2026-11-14 | Behaviour cloning learner; labels; rollouts; averages; stopping; metrics; hand-off to reinforcement learning; checkpoint retention; measurements |
| docs/explanation/contracts.md | 191 | explanation | 2026-11-03 | Observation; privileged truth; action; events; controllers; pools and trajectories; configuration |
| docs/explanation/game/case-studies.md | 288 | explanation | none | Ship trade-off case studies; Mosquito and Zerstörer cycle; guns on airframes; shield and recharge; climb and store; engine knee |
| docs/explanation/game/case-studies/sensitivity-armour/report.md | 57 | results | none | Sensitivity run; shield and body knobs; game counts per run |
| docs/explanation/game/case-studies/sensitivity-fine/report.md | 49 | results | none | Sensitivity run; steep knobs in small steps; game counts per run |
| docs/explanation/game/case-studies/sensitivity-flight/report.md | 54 | results | none | Sensitivity run; flight and store knobs; game counts per run |
| docs/explanation/game/case-studies/sensitivity-guns/report.md | 50 | results | none | Sensitivity run; gun values; game counts per run |
| docs/explanation/game/case-studies/sensitivity-pairs/report.md | 63 | results | none | Trade-off pairs run; game counts per run; ship knob pairs |
| docs/explanation/game/case-studies/sensitivity.md | 228 | results | none | Ship knob sensitivity; Elo per doubling; recommended step; trade-off pairs; cycles and cyclic share; tuning |
| docs/explanation/game/mixed-batches.md | 138 | explanation | 2026-11-02 to 2026-11-15 | Mixed batches; map-size sampling; field-count sampling; lineup draw; fixed shapes across modes; training mix; cost |
| docs/explanation/game/modes.md | 322 | explanation | 2026-11-12 to 2026-11-15 | Game modes; clocks and presets; mode phase; team deathmatch; elimination; Clover layouts; benched ships; Frontline; mode observation |
| docs/explanation/game/perception.md | 101 | explanation | 2026-11-03 | Visibility; fog of war; time since seen; observation building; the seat |
| docs/explanation/game/physics.md | 257 | explanation | 2026-11-03 | Physics API; the tick; ships and ship card; flight; fields; bullets and combat; spawns and maps |
| docs/explanation/game/real-planes.md | 209 | reference | 2026-11-10 to 2026-11-13 | Real WW2 plane types researched; combat economy proposal |
| docs/explanation/game/real-scale-balance.md | 374 | log | 2026-11-10 to 2026-11-13 | balancing the real-scale ship set; ship-type tournament ratings; rounds of tune, search, rate and explain; equal damage per second rule; rules the rounds follow; round 1 and round 2 data; running a round; researched types removed from the game; pacing of climb; muzzle speed, shield, fire rate, dispersion and recharge as rating factors; move to designing types for balance; Imugi set search |
| docs/explanation/game/real-scale-pacing.md | 141 | results | 2026-11-13 | Real-scale fight pacing; charge and flight model; pacing measures; sweep; choice |
| docs/explanation/game/real-scale-tournament.md | 150 | results | 2026-11-13 | Ship-type tournament; ratings; rating predictors; real-scale variant |
| docs/explanation/game/real-scale.md | 160 | explanation | 2026-11-13 | Real-scale variant; P-51D from standard fighter; map and clock; rules; ships; Imugi real-scale sets; unscaled elements |
| docs/explanation/game/rewards.md | 81 | explanation | 2026-11-03 | Reward levels; reward ledger; one tick of rewards |
| docs/explanation/game/ship-card.md | 265 | explanation | 2026-11-03 to 2026-11-09 | Ship card; ship sets; handles; measuring; solving; estimating; handle independence; reference chart |
| docs/explanation/game/ship-design.md | 378 | explanation | 2026-11-10 to 2026-11-13 | Ship type set design; Elo scale targets; fighting-game design aims; knobs; WW2 envelope; handle independence; ship search; seed cycle |
| docs/explanation/game/ship-design/archetypes.md | 491 | notes | 2026-11-12 to 2026-11-13 | turn fighter and energy fighter archetypes; Second World War aircraft figures; corner speed, sustained turn, ballistic coefficient; mapping of the paradigm to the 2D game; conditions for the paradigm; gun lethality and damage by aspect; proposed starting variants of vizia; Imugi doctrine elements per type; tune loop of flight models, doctrine, balance check; naming Jujak and Yongma; balance across fleet sizes; arena radius and fog; same-type duel joust |
| docs/explanation/game/ship-design/runbook.md | 147 | guide | none | Ship-design search runbook; start check; trade-off case study; stage spec search; exact cards; continuing a search; seven types; Frontline; commit |
| docs/explanation/glossary.md | 96 | reference | 2026-11-03 | Glossary; adapter; belief; carry; controller; decision step and tick; golden; lineup, pool, seat; pytree; rung; stage |
| docs/explanation/imugi.md | 447 | explanation | 2026-11-03 to 2026-11-12 | Imugi scripted agent; observation and outputs; combat layer; doctrines; Frontline layer; logistics, floor and temperature; memory and dead reckoning; parameters; cheese agents; tests |
| docs/explanation/in-progress/diagnostics.md | 12 | other | none | Diagnostics placeholder; in-progress page |
| docs/explanation/in-progress/play-shell.md | 336 | explanation | 2026-11-03 | Play shell; match loop; snapshot and interpolation; render passes; index texture; fog mask; interface; replays; live tuning; themes; keyboard pilot |
| docs/explanation/index.md | 21 | other | none | Explanation section index |
| docs/explanation/league.md | 684 | explanation | 2026-11-02 to 2026-11-15 | League pool; sampling; slots, Imugi and the cut; ratings; live gauge; resuming; metrics; published ratings; decision suite; shared rating pool |
| docs/explanation/learner.md | 447 | explanation | 2026-11-02 to 2026-11-13 | Learner; model protocol; segments; returns and advantage; losses; trust region; shot labels; compilation and precision; toy learner; nearest-neighbour MLP |
| docs/explanation/roadmap.md | 28 | plan | 2026-11-01 | Roadmap; in-progress work |
| docs/explanation/runs.md | 262 | explanation | 2026-11-02 to 2026-11-15 | Run directory; policy files and training checkpoints; build records and compatibility set; resume, fork, warm start; retention; memory probe; offline results and uploads |
| docs/explanation/training-loop.md | 182 | explanation | 2026-11-06 | Training update; controllers; rollout; rating lanes; realtime |
| docs/explanation/yemong.md | 845 | explanation | 2026-11-08 to 2026-11-14 | Yemong model; belief; position offset; tokens and state code; next-state head; error head; spread and position inputs; network; attention output gate |
| docs/getting-started/benchmark.md | 25 | guide | 2026-11-03 | First benchmark |
| docs/getting-started/first-training-run.md | 78 | guide | 2026-11-02 | First training run; command; printed output; run location; Python equivalent |
| docs/getting-started/index.md | 12 | other | none | Getting started section index |
| docs/getting-started/installation.md | 46 | guide | 2026-11-03 | Requirements; setup; running commands through uv |
| docs/getting-started/running-tests.md | 64 | guide | 2026-11-03 | Test markers; shared state; test layout |
| docs/guides/adding-a-mode.md | 115 | guide | 2026-11-01 | Adding a game mode; clock; static rules; state; hooks; observation; presets and training mix; tests; docs |
| docs/guides/behaviour-cloning.md | 181 | guide | 2026-11-09 to 2026-11-11 | Cloning Imugi; fixed-budget BC experiment; resuming a BC run; starting RL from the clone; label entropy; Python usage |
| docs/guides/benchmarking.md | 92 | guide | 2026-11-03 | Benchmark lock; cells; skeleton benchmark; reading results |
| docs/guides/designing-a-ship.md | 103 | guide | 2026-11-03 | Charting the ship card; handles; solving; saving a card; tick-rate check |
| docs/guides/figures.md | 281 | guide | 2026-11-06 | Figures; shots; scenarios; belief; next_state; density; enemy_action; critic; videos; watching a training run |
| docs/guides/index.md | 22 | other | none | Guides section index |
| docs/guides/playing.md | 472 | guide | none | Playing and spectating; screen; views; ship size; camera and grid; flying; trained agents; map making; prediction overlays; replays; themes; keys |
| docs/guides/queueing-jobs.md | 125 | guide | 2026-11-07 to 2026-11-12 | GPU job runner; queueing jobs; order and preemption; multiple GPUs; queue watching; crash recovery |
| docs/guides/reference-fixtures.md | 65 | guide | 2026-11-03 | Reference fixtures; goldens; REFERENCE_ARM; running fixtures |
| docs/guides/resuming-forking-sharing.md | 157 | guide | 2026-11-02 to 2026-11-11 | Starting and stopping runs; listing runs; resume; fork; warm start; memory check; EMA decays; sharing checkpoints |
| docs/guides/training.md | 312 | guide | 2026-11-11 to 2026-11-15 | Training configuration; game choice; pools; mixed batch; model and learner settings; chaining BC and RL; stopping a run; toy learner; seeds |
| docs/index.md | 28 | other | none | Project landing page; where to start |
| docs/rewrite/arm-review.md | 319 | review | 2026-11-13 to 2026-11-15 | candidate experiment arms list; arm fields (paper question, purpose, cost, importance); map size fixed versus variable; variable ship and field counts; seed pairs; PPO minibatches and learning rate; training mix across modes; self-play seats; offensive premium reward shaping; trust region clip versus DPPO; entropy; dense reward decay; auxiliary head removal; belief input versus temporal layers; Griffin versus GRU; input encoding and rotary position; value output form; shot head variants; BC to RL versus scratch; reference runs and run budget |
| docs/rewrite/arms-todo.md | 223 | tracking | 2026-10-30 to 2026-11-15 | Arms decisions; open items for the author; pre-freeze tasks; box setup; cheap checks; candidate arms |
| docs/rewrite/audits/2026-11-01-consistency.md | 185 | review | 2026-11-01 | consistency audit; naming; dead or leftover code; five highest-value fixes |
| docs/rewrite/audits/2026-11-01-v1-gaps.md | 178 | review | 2026-11-01 | v1 against v2 feature gaps; commands; run infrastructure (section 11); tools with no v2 section |
| docs/rewrite/audits/2026-11-02-fast-yemong.md | 97 | review | 2026-11-02 | blind race between two Yemong models; Yemong speed-up; maintenance burden |
| docs/rewrite/backlog.md | 91 | tracking | 2026-11-03 to 2026-11-10 | Deferred work; speed; tournaments; portability; Imugi; later items |
| docs/rewrite/briefing.md | 87 | brief | 2026-11-15 | 5090 pairwise arms preparation; handover; open tickets; interrupted work; working agreements |
| docs/rewrite/contracts.md | 650 | spec | 2026-10-31 to 2026-11-15 | Section-0 contracts; config; game state; action; observation; events; privileged; controllers; pools; trajectory; functions; review log |
| docs/rewrite/decisions.md | 1577 | decisions | 2026-09-10 to 2026-11-07 | rewrite goals and framing; motivating evidence (CPU dispatch, torch.compile, in-loop Elo); JAX paradigm and pure-function seams; agent carry and controller interface; raw observation and privileged labels; N-team generality and game modes; training loop, precision and seeding; live pseudo-Elo league; ship types and ship card; turn levels and stick thirds; flight model redesign (energy store, dive/climb, integrators, rigid-body prototypes); P-51D and Spitfire reference planes; Griffin-only memory and damped rotary geometry; code padding rule; timing contract; feat/resim-shot-labels findings; repository layout options; behaviour cloning rounds and averaged agents; run control and GPU queue; tabled items and post-implementation TODO |
| docs/rewrite/diagnostics.md | 1105 | review | 2026-10-21 to 2026-11-14 | audit of logged trainer metrics by prefix group; generic BC and RL diagnostics checklist; warning and alert trigger rules; watch dashboard layout; histograms; loss terms; policy KL and clip fraction; critic explained variance and return normaliser; outcome calibration ledger; next-state population metrics; belief error by hidden age; enemy-action baselines; hex density deviance; shot-head metrics; episode, fog, physics and frontline analytics; gradient norms, shares and cosines; rating and pool series replacing the evaluator; representation health and input envelope checks |
| docs/rewrite/evaluation-suite.md | 155 | spec | 2026-11-15 | Evaluation suite; players; conditions; score; decision rule; allocation; records; command; design pilot; ladder |
| docs/rewrite/figures/ (other files) |  | other | none | density figures for disc sizes and grid layouts; png images; python scripts |
| docs/rewrite/flight-model.md | 169 | spec | 2026-10-22 to 2026-10-23 | Flight model; units and scale; ship state; controls; forces; integrator; prototype measurements; open questions |
| docs/rewrite/gameplay-brainstorm.md | 108 | notes | 2026-10-25 to 2026-10-28 | Gameplay and visual design brainstorm; match and roles; flight and combat; map, fields and vision; visuals |
| docs/rewrite/ideas.md | 127 | notes | 2026-11-14 | Candidate arms; optimisation; BC hand-off; rewards; critic; model; auxiliary heads; symmetry; league; environment distribution; evaluation |
| docs/rewrite/library-mockups/ (other files) |  | other | none | JAX library mock-ups; python scripts |
| docs/rewrite/library-mockups/README.md | 17 | notes | 2026-10-21 | JAX library choice (tabled item 6); library mock-ups |
| docs/rewrite/physics-checks/ (other files) |  | other | none | physics check scripts; result json; python scripts |
| docs/rewrite/references.md | 171 | reference | 2026-10-21 to 2026-11-14 | Literature references; PPO; GAE; DAPO; DPPO; OpenAI Five; AlphaStar; AlphaZero; ratings |
| docs/rewrite/scaffold.md | 431 | plan | 2026-10-31 to 2026-11-01 | PyTorch rewrite scaffold plan; inputs carried over; design rules for speed and maintainability; repository tree; runtime stages, graphs and memory; section contracts; walking skeleton milestone 0; section order and acceptance; repository mechanics; process per section; author decisions of 2026-10-31 |
| docs/rewrite/sections/absent-fields/progress.md | 44 | log | none | absent fields; game, model and plumbing changes; tests; plan and log |
| docs/rewrite/sections/acceleration-inputs-brief.md | 107 | brief | 2026-11-08 | acceleration inputs; round-2 arm 1; C-fixed configuration; linear and angular acceleration; switch; logged diagnostics; CPU tests |
| docs/rewrite/sections/archetypes/ (other files) |  | other | none | archetypes run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/archetypes/arming2/ (other files) |  | other | none | archetypes arming2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/arming2/sizes.md | 8 | results | none | archetype pairs arming2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm1/ (other files) |  | other | none | archetypes confirm1 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm1/sizes.md | 6 | results | none | archetype pairs confirm1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm10/ (other files) |  | other | none | archetypes confirm10 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm10/sizes.md | 4 | results | none | archetype pairs confirm10 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm11/ (other files) |  | other | none | archetypes confirm11 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm11/sizes.md | 4 | results | none | archetype pairs confirm11 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm12/ (other files) |  | other | none | archetypes confirm12 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm12/sizes.md | 5 | results | none | archetype pairs confirm12 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm2/ (other files) |  | other | none | archetypes confirm2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm2/sizes.md | 7 | results | none | archetype pairs confirm2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm3/ (other files) |  | other | none | archetypes confirm3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm3/sizes.md | 5 | results | none | archetype pairs confirm3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm4/ (other files) |  | other | none | archetypes confirm4 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm4/sizes.md | 6 | results | none | archetype pairs confirm4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm5/ (other files) |  | other | none | archetypes confirm5 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm5/sizes.md | 7 | results | none | archetype pairs confirm5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm6/ (other files) |  | other | none | archetypes confirm6 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm6/sizes.md | 7 | results | none | archetype pairs confirm6 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm7/ (other files) |  | other | none | archetypes confirm7 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm7/sizes.md | 4 | results | none | archetype pairs confirm7 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm8/ (other files) |  | other | none | archetypes confirm8 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm8/sizes.md | 4 | results | none | archetype pairs confirm8 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/confirm9/ (other files) |  | other | none | archetypes confirm9 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/confirm9/sizes.md | 5 | results | none | archetype pairs confirm9 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/d2kit/ (other files) |  | other | none | archetypes d2kit run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/d2kit/sizes.md | 4 | results | none | archetype pairs d2kit results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/kit3/ (other files) |  | other | none | archetypes kit3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/kit3/sizes.md | 4 | results | none | archetype pairs kit3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/mirror/ (other files) |  | other | none | archetypes mirror run artefacts; result json |
| docs/rewrite/sections/archetypes/mirror2/ (other files) |  | other | none | archetypes mirror2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/mirror2/sizes.md | 13 | results | none | archetype pairs mirror2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/mirror3/ (other files) |  | other | none | archetypes mirror3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/mirror3/sizes.md | 12 | results | none | archetype pairs mirror3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/mirror4/ (other files) |  | other | none | archetypes mirror4 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/mirror4/sizes.md | 13 | results | none | archetype pairs mirror4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/progress.md | 1111 | log | none | archetype pairs on tune/archetypes; arming distance card field; pairs.py, sizes.py, score.py, make_map.py tools; 1v1 to 20v20 balance grids; mirror studies and mirror kit; hit-and-run, rate fighter, speed fighter, close fighter, brawler, small ship, scout, interceptor, armed gunner; armoured-nose variants; four-ship set rosters and sets 4 to 8; confirmation runs at 1024 games; maneuvers doctrines and crowd extension; post-merge sense entropy; playtest maps; KST-timed rounds; gpubox7 |
| docs/rewrite/sections/archetypes/roster1/ (other files) |  | other | none | archetypes roster1 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/roster1/sizes.md | 11 | results | none | archetype pairs roster1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/roster2/ (other files) |  | other | none | archetypes roster2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/roster2/sizes.md | 10 | results | none | archetype pairs roster2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/roster3/ (other files) |  | other | none | archetypes roster3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/roster3/sizes.md | 6 | results | none | archetype pairs roster3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round1/ (other files) |  | other | none | archetypes round1 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round1/sizes.md | 22 | results | none | archetype pairs round1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round10/ (other files) |  | other | none | archetypes round10 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round10/sizes.md | 13 | results | none | archetype pairs round10 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round11/ (other files) |  | other | none | archetypes round11 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round11/sizes.md | 9 | results | none | archetype pairs round11 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round12/ (other files) |  | other | none | archetypes round12 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round12/sizes.md | 7 | results | none | archetype pairs round12 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round13/ (other files) |  | other | none | archetypes round13 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round13/sizes.md | 9 | results | none | archetype pairs round13 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round14/ (other files) |  | other | none | archetypes round14 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round14/sizes.md | 8 | results | none | archetype pairs round14 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round15/ (other files) |  | other | none | archetypes round15 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round15/sizes.md | 7 | results | none | archetype pairs round15 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round16/ (other files) |  | other | none | archetypes round16 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round16/sizes.md | 7 | results | none | archetype pairs round16 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round17/ (other files) |  | other | none | archetypes round17 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round17/sizes.md | 5 | results | none | archetype pairs round17 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round18/ (other files) |  | other | none | archetypes round18 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round18/sizes.md | 7 | results | none | archetype pairs round18 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round19/ (other files) |  | other | none | archetypes round19 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round19/sizes.md | 7 | results | none | archetype pairs round19 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round2/ (other files) |  | other | none | archetypes round2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round2/sizes.md | 19 | results | none | archetype pairs round2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round3/ (other files) |  | other | none | archetypes round3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round3/sizes.md | 20 | results | none | archetype pairs round3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round4/ (other files) |  | other | none | archetypes round4 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round4/sizes.md | 20 | results | none | archetype pairs round4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round5/ (other files) |  | other | none | archetypes round5 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round5/sizes.md | 18 | results | none | archetype pairs round5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round6/ (other files) |  | other | none | archetypes round6 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round6/sizes.md | 12 | results | none | archetype pairs round6 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round7/ (other files) |  | other | none | archetypes round7 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round7/sizes.md | 11 | results | none | archetype pairs round7 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round8/ (other files) |  | other | none | archetypes round8 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round8/sizes.md | 11 | results | none | archetype pairs round8 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/round9/ (other files) |  | other | none | archetypes round9 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/round9/sizes.md | 13 | results | none | archetype pairs round9 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/screen1/ (other files) |  | other | none | archetypes screen1 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/screen1/sizes.md | 10 | results | none | archetype pairs screen1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/screen2/ (other files) |  | other | none | archetypes screen2 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/screen2/sizes.md | 10 | results | none | archetype pairs screen2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/screen3/ (other files) |  | other | none | archetypes screen3 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/screen3/sizes.md | 7 | results | none | archetype pairs screen3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/screen4/ (other files) |  | other | none | archetypes screen4 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/screen4/sizes.md | 7 | results | none | archetype pairs screen4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set3ce/ (other files) |  | other | none | archetypes set3ce run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set3ce/sizes.md | 6 | results | none | archetype pairs set3ce results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set4/ (other files) |  | other | none | archetypes set4 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set4/sizes.md | 9 | results | none | archetype pairs set4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set4b/ (other files) |  | other | none | archetypes set4b run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set4b/sizes.md | 8 | results | none | archetype pairs set4b results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set5/ (other files) |  | other | none | archetypes set5 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set5/sizes.md | 10 | results | none | archetype pairs set5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set6/ (other files) |  | other | none | archetypes set6 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set6/sizes.md | 8 | results | none | archetype pairs set6 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set7/ (other files) |  | other | none | archetypes set7 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set7/sizes.md | 4 | results | none | archetype pairs set7 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/set8/ (other files) |  | other | none | archetypes set8 run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/set8/sizes.md | 4 | results | none | archetype pairs set8 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/archetypes/triangle/ (other files) |  | other | none | archetypes triangle run artefacts; result json; run log text |
| docs/rewrite/sections/archetypes/triangle/sizes.md | 5 | results | none | archetype pairs triangle results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/arena-mixed-builds-brief.md | 56 | brief | 2026-11-06 to 2026-11-07 | offline arena; mixed builds; arena speed bug; tests |
| docs/rewrite/sections/arm2-persistence-brief.md | 125 | brief | 2026-11-08 | round-2 arm 2; persistence delta heads; belief advance for hidden ships; ellipse spread inputs; loss weight; diagnostics |
| docs/rewrite/sections/arms34-brief.md | 105 | brief | 2026-11-08 | round-2 arms 3 and 4; error head; delta split; extrapolation baseline; diagnostics |
| docs/rewrite/sections/balance-sizes/ (other files) |  | other | none | balance-sizes run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/balance-sizes/baseline/ (other files) |  | other | none | balance-sizes baseline run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/baseline/sizes.md | 3 | results | none | balance across fleet sizes baseline results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/cards/A/ (other files) |  | other | none | balance-sizes cards A run artefacts; toml config |
| docs/rewrite/sections/balance-sizes/cards/B/ (other files) |  | other | none | balance-sizes cards B run artefacts; toml config |
| docs/rewrite/sections/balance-sizes/confirm/ (other files) |  | other | none | balance-sizes confirm run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/confirm/sizes.md | 4 | results | none | balance across fleet sizes confirm results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/deny2/ (other files) |  | other | none | balance-sizes deny2 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/deny2/sizes.md | 10 | results | none | balance across fleet sizes deny2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/diag20/ (other files) |  | other | none | balance-sizes diag20 run artefacts; result json; shell scripts; run log text |
| docs/rewrite/sections/balance-sizes/diag20/10v10-wide.md | 3 | results | none | balance across fleet sizes diag20 10v10-wide table; single-size diagnostic; candidate score; kill mix; life |
| docs/rewrite/sections/balance-sizes/diag20/arena.md | 3 | results | none | balance across fleet sizes diag20 arena table; single-size diagnostic; candidate score; kill mix; life |
| docs/rewrite/sections/balance-sizes/diag20/long.md | 3 | results | none | balance across fleet sizes diag20 long table; single-size diagnostic; candidate score; kill mix; life |
| docs/rewrite/sections/balance-sizes/diag20/nofog.md | 3 | results | none | balance across fleet sizes diag20 nofog table; single-size diagnostic; candidate score; kill mix; life |
| docs/rewrite/sections/balance-sizes/final/ (other files) |  | other | none | balance-sizes final run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/final/sizes.md | 5 | results | none | balance across fleet sizes final results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/images/ (other files) |  | other | none | trajectory and event images with data; result json; png images |
| docs/rewrite/sections/balance-sizes/mirror1/ (other files) |  | other | none | balance-sizes mirror1 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/mirror1/sizes.md | 11 | results | none | balance across fleet sizes mirror1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/paths1/ (other files) |  | other | none | balance-sizes paths1 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/paths1/sizes.md | 14 | results | none | balance across fleet sizes paths1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/paths2/ (other files) |  | other | none | balance-sizes paths2 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/paths2/sizes.md | 12 | results | none | balance across fleet sizes paths2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/paths3/ (other files) |  | other | none | balance-sizes paths3 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/paths3/sizes.md | 12 | results | none | balance across fleet sizes paths3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/paths4/ (other files) |  | other | none | balance-sizes paths4 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/paths4/sizes.md | 10 | results | none | balance across fleet sizes paths4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/paths5/ (other files) |  | other | none | balance-sizes paths5 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/paths5/sizes.md | 10 | results | none | balance across fleet sizes paths5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/progress.md | 654 | log | none | fleet-size balance tuning branch; Yongma and Jujak ship types; frozen r9 card and r9 scripted agent; candidate cards A, B, C, D and play-map presets; induced drag, shield, engine, bullet life and damage levers; head-on damage fraction and mirror joust; deny doctrine element; sustained turn and cruise characteristics; fight shape tallies; self-play pace and life per ship; arena radius and 20v20 diagnosis; strength check against r9; playtest set |
| docs/rewrite/sections/balance-sizes/round1/ (other files) |  | other | none | balance-sizes round1 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round1/sizes.md | 22 | results | none | balance across fleet sizes round1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round2/ (other files) |  | other | none | balance-sizes round2 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round2/sizes.md | 18 | results | none | balance across fleet sizes round2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round3/ (other files) |  | other | none | balance-sizes round3 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round3/sizes.md | 14 | results | none | balance across fleet sizes round3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round4/ (other files) |  | other | none | balance-sizes round4 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round4/sizes.md | 14 | results | none | balance across fleet sizes round4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round5/ (other files) |  | other | none | balance-sizes round5 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round5/sizes.md | 8 | results | none | balance across fleet sizes round5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round6/ (other files) |  | other | none | balance-sizes round6 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round7/ (other files) |  | other | none | balance-sizes round7 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round7/sizes.md | 12 | results | none | balance across fleet sizes round7 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/round8/ (other files) |  | other | none | balance-sizes round8 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/round8/sizes.md | 14 | results | none | balance across fleet sizes round8 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/balance-sizes/spot20/ (other files) |  | other | none | balance-sizes spot20 run artefacts; result json; run log text |
| docs/rewrite/sections/balance-sizes/spot20/sizes.md | 4 | results | none | balance across fleet sizes spot20 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/baselines-bench-brief.md | 89 | brief | 2026-11-08 | baselines bench; persistence, rule and extrapolation baselines; one-tick residual; axis correlation; error ellipse |
| docs/rewrite/sections/baselines-bench-tables.md | 627 | results | none | full tables for one-tick baselines bench; pair counts per player and preset; persistence, rule-based (R) and extrapolation (E) baselines; pooled entropy by series; held and changed control; win counts between baselines; last-tick change as predictor; axis correlation and error ellipse; mirror-folded correlation; per-situation subsets; Spearman per player and preset; CV and E forecasts at horizons |
| docs/rewrite/sections/baselines-bench.md | 332 | results | 2026-11-07 | one-tick baselines bench; scripts baselines_core, baselines_measure, baselines_report; definitions of persistence, rule-based and extrapolation baselines; E-arc position; entropy under own code and today's code; float32 recording resolution; held and changed control; nose rate decay; axis correlation and error ellipse; mirror folding; error scale at horizons; reading per quantity; old recordings and open points |
| docs/rewrite/sections/bc-averages-brief.md | 66 | brief | 2026-11-05 | behaviour cloning; averaged agents; build task |
| docs/rewrite/sections/bc-stop-brief.md | 82 | brief | 2026-11-02 | behaviour cloning score stop; confidence bound; tests; docs |
| docs/rewrite/sections/bc-v19/ (other files) |  | other | none | bc-v19 run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/bc-v19/overnight-log.md | 452 | log | 2026-11-14 | overnight log of 2026-11-14; bc-v19 NaN crash and game physics 0/0 root cause; softmin fix; bc-v20 run and plateau stop rule; rl-v20 handoff and 300M extension; geometry head loss weight and per-output split; Yemong v8; live Elo by mode against Baekho; run-flag triage (gauge_runaway, rating_drop, reward_hacking, BC KL); deathmatch Elo scale question and kill-share scoring; attention shares and per-head attention; masked-zone head-to-head; E2/E3 mode experiments (rl-dm, rl-kl, rl-mix-kl); kill-lead outcome A/B; bc-v21 with variable map sizes; fused key mask and Qwen gate |
| docs/rewrite/sections/bc.md | 75 | decisions | 2026-10-30 to 2026-11-02 | section 7 behaviour cloning; training review 4c; author decisions; post-implementation decisions |
| docs/rewrite/sections/belief-baselines.md | 572 | results | 2026-11-07 | belief and next-state baselines for BC arm C on CPU self-play; hidden-enemy position and heading error by hidden age; persistence and dead-reckoning baselines; belief sigma coverage; spread-fit decoder variants; next-state head KL per code group; kinematic residual of the flight step; deathmatch, deathmatch_wide and frontline modes; kick-drift-kick integrator midpoint velocity |
| docs/rewrite/sections/belief-calibration-brief.md | 59 | brief | 2026-11-06 to 2026-11-07 | belief spread calibration; overconfidence bug; baselines measurement |
| docs/rewrite/sections/break-even/progress.md | 76 | log | 2026-11-14 | break-even ship count; bnb crossover rewrite; benched-slot cost; open items |
| docs/rewrite/sections/bug-sweep/progress.md | 42 | log | none | bug sweep branch; mixed-batch drift |
| docs/rewrite/sections/bullet-damage/ (other files) |  | other | none | bullet-damage run artefacts; per-game or per-record jsonl |
| docs/rewrite/sections/bullet-damage/progress.md | 87 | log | 2026-11-12 | bullet damage mechanics (game v6); gun-lethality mirrors; recompile notes for the tuning loop |
| docs/rewrite/sections/bullet-life/progress.md | 47 | log | 2026-11-12 | bullet life following gun range; bullet display options; head-on angle |
| docs/rewrite/sections/cheese-agents/progress.md | 30 | log | none | cheese agents; tried items; caveats; next steps |
| docs/rewrite/sections/cleanup-brief.md | 127 | brief | 2026-11-01 | cleanup pass; boundaries; hand back |
| docs/rewrite/sections/cleanup-notes.md | 487 | notes | 2026-11-03 | cleanup pass branch; code formatter and blame-ignore revisions; contract renames; shape letters and reward-level naming; abbreviation sweep; dead code removal; opaque game state and perceived API; sweep tests with mutation checks; docstrings and contracts.md; style guide deletion; layering and import tests; single-definition consolidation (ship team, enemy flip, disc sampling, mode names, two-hot); test hygiene (markers, GPU benchmark lock, CUDA guard, generator guard); game version in policy files; resume drift; test tiers; PRESETS location request |
| docs/rewrite/sections/cleanup-spec.md | 398 | spec | 2026-11-01 to 2026-11-03 | cleanup scope and order; formatter; contract renames; shape letters; abbreviations; dead code; mutation sweep tests |
| docs/rewrite/sections/compile-overhaul-notes.md | 798 | notes | 2026-11-10 to 2026-11-11 | torch.compile op counts and compile-time diagnosis; game VERSION and fingerprint; ratings lanes and stacked vmap forward; act-stage lane grouping; learner loss compile variants; BC teacher label precompute; non-finite step guard and fused Adam; Triton autotune and Dynamo recompile limit; cold versus warm compile caches; Arena tail and started-before-T game counting; real-scale presets baseline; micro-batch rows and VRAM; fleet-size sweep and benching; learner speed profiling (shot trajectory code, grouped cross-entropy, staged head kernel); return units; GPU resume test; shot label validity; RTX 3090 rental box |
| docs/rewrite/sections/compile-overhaul/after-1/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/after-2/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/after-3/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/arena/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/baseline/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/fleet/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/learn/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-overhaul/real-scale/ (other files) |  | other | none | compile overhaul timing measurements; result json |
| docs/rewrite/sections/compile-split/ (other files) |  | other | none | compile-split run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/compile-split/progress.md | 127 | log | 2026-11-12 | compile split; what is compiled; config-switch variants; method; measurement scripts |
| docs/rewrite/sections/delta-heads-brief.md | 107 | brief | 2026-11-07 | next-state delta heads; round-1 challenger; targets; belief advance; switch; loss weight; diagnostics |
| docs/rewrite/sections/delta-heads-round2-notes.md | 69 | notes | 2026-11-07 to 2026-11-08 | delta heads round 2; first arm updates; proposed restart; physics plus capped correction; later work |
| docs/rewrite/sections/diagnostics-brief.md | 97 | brief | 2026-11-03 | diagnostics section 9; tests; boundaries; commits |
| docs/rewrite/sections/diagnostics-notes.md | 278 | notes | 2026-11-03 to 2026-11-04 | diagnostics section 9; test runs; hand-broken checks; cost; round 2; deviations; open questions |
| docs/rewrite/sections/diagnostics-round2-brief.md | 55 | brief | none | diagnostics section 9 round 2; GPU training; commits and hand back |
| docs/rewrite/sections/diagnostics-spec.md | 145 | spec | 2026-10-22 to 2026-11-13 | collection; sinks; trigger rules; dashboards; v2 translations of audit items; acceptance |
| docs/rewrite/sections/docs-brief.md | 144 | brief | none | documentation site; generated content; staleness checks; docs style guide; tests |
| docs/rewrite/sections/docs-notes.md | 355 | notes | none | documentation site branch; MkDocs Material layout and nav tabs; generated content hook; documentation checks (strict build, dotted-name resolution, runnable examples, covers staleness); docstring report; mkdocs version pin; disagreements between docs and code (cli docstring, scaffold.md, contracts.md, game-spec, modes-spec); glossary page; roadmap statuses; writing guide; fix round after blind review |
| docs/rewrite/sections/doctrines/ (other files) |  | other | none | doctrines run artefacts; result json; python scripts |
| docs/rewrite/sections/doctrines/progress.md | 131 | log | 2026-11-12 | doctrines; pieces; round 1 baselines; handoff |
| docs/rewrite/sections/doctrines/round1/ (other files) |  | other | none | doctrines round1 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/doctrines/round1/table-baseline.md | 28 | results | none | doctrines round 1 baseline tables; off and hand variants; lineups; kill and damage mix by approach type |
| docs/rewrite/sections/dppo-notes.md | 47 | notes | 2026-11-06 | DPPO; trust region in LLM reinforcement learning; objective; approximations; joint actions |
| docs/rewrite/sections/elimination/progress.md | 42 | log | 2026-11-12 | elimination mode; Clover map; fixed layouts; mode input; tests, play and docs |
| docs/rewrite/sections/error-head-spread-fix-brief.md | 66 | brief | 2026-11-08 | error head; robust spread; CPU tests |
| docs/rewrite/sections/evaluation-suite/progress.md | 37 | log | 2026-11-14 to 2026-11-15 | bnb evaluate; evaluation suite design pilot; tried items; next steps |
| docs/rewrite/sections/extrapolation-without-inputs-brief.md | 41 | brief | 2026-11-08 | extrapolation baseline; acceleration inputs; CPU tests |
| docs/rewrite/sections/fast-yemong-brief.md | 125 | brief | 2026-11-02 | blind race kernels in v2 Yemong; bf16 default; measurement; docs |
| docs/rewrite/sections/fast-yemong-notes.md | 360 | notes | 2026-11-02 to 2026-11-03 | fast Yemong kernels; piece map of scan, attention with rotary, gated MLP, temporal mix and heads kernels; one commit per kernel; training throughput; memory; autotuning addition; norm epsilon under bf16; deviations and open questions; progress of 2026-11-03; intermittent failures hunt; segment store staging race and copy stream; NaN poison and delayed-copy probes; Inductor autotune cache consistency |
| docs/rewrite/sections/figures-brief.md | 122 | brief | 2026-11-05 to 2026-11-06 | figures round one; scaffolding; shot head; one head at a time; checkpoints |
| docs/rewrite/sections/fixed-maps/progress.md | 40 | log | 2026-11-15 | fixed maps |
| docs/rewrite/sections/front-reversal/ (other files) |  | other | none | front-reversal run artefacts; per-game or per-record jsonl |
| docs/rewrite/sections/front-reversal/results.md | 102 | results | 2026-11-15 | frontline front as a random walk; front reversal; question, definitions and setup |
| docs/rewrite/sections/frontline-lock/ (other files) |  | other | none | frontline-lock run artefacts; per-game or per-record jsonl |
| docs/rewrite/sections/frontline-lock/results.md | 82 | results | 2026-11-13 to 2026-11-15 | frontline lock re-tune; front as a random walk; sweep and choice; fields on; Imugi on both sides |
| docs/rewrite/sections/game-brief.md | 121 | brief | none | section 1 game; boundaries; hand back |
| docs/rewrite/sections/game-loop/ (other files) |  | other | none | game-loop run artefacts; result json; python scripts |
| docs/rewrite/sections/game-loop/final/ (other files) |  | other | none | game-loop final run artefacts; result json |
| docs/rewrite/sections/game-loop/progress.md | 558 | log | 2026-11-12 | tune/game-loop branch; bench/game_loop.py harness and compiled tick; Yongma and Jujak ship cards; doctrine search and ES rounds; card sweeps for balance levers; damage scale and head-on time to kill; league of off, hand, random and hot players; Elo fit; distinctness penalty between doctrines; per-environment fight statistics; pace as kills per minute in 5v5 deathmatch; frontal armour and frozen card; author feedback and regime idea; final state against targets; throughput on laptop GPU |
| docs/rewrite/sections/game-loop/round0-table.md | 24 | results | none | game loop round 0 baseline table; lineups; kills per minute; damage and kill mix by approach type |
| docs/rewrite/sections/game-loop/round1/ (other files) |  | other | none | game-loop round1 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round1/cards.md | 18 | results | none | game loop round1 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round1/self-table.md | 12 | results | none | game loop round1 self-play table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round10/ (other files) |  | other | none | game-loop round10 run artefacts; result json |
| docs/rewrite/sections/game-loop/round10/cards.md | 10 | results | none | game loop round10 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round11/ (other files) |  | other | none | game-loop round11 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round12/ (other files) |  | other | none | game-loop round12 run artefacts; result json |
| docs/rewrite/sections/game-loop/round12/cards.md | 10 | results | none | game loop round12 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round12/verify-frozen.md | 3 | results | none | game loop round12 frozen files verification table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round2/ (other files) |  | other | none | game-loop round2 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round2/cards.md | 18 | results | none | game loop round2 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round2/self-table.md | 12 | results | none | game loop round2 self-play table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round3/ (other files) |  | other | none | game-loop round3 run artefacts; result json |
| docs/rewrite/sections/game-loop/round3/cards.md | 10 | results | none | game loop round3 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round4/ (other files) |  | other | none | game-loop round4 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round4/self-table.md | 12 | results | none | game loop round4 self-play table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round5/ (other files) |  | other | none | game-loop round5 run artefacts; result json |
| docs/rewrite/sections/game-loop/round5/cards.md | 10 | results | none | game loop round5 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round6/ (other files) |  | other | none | game-loop round6 run artefacts; result json |
| docs/rewrite/sections/game-loop/round6/cards.md | 10 | results | none | game loop round6 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round7/ (other files) |  | other | none | game-loop round7 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round7/self-table.md | 12 | results | none | game loop round7 self-play table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round8/ (other files) |  | other | none | game-loop round8 run artefacts; result json |
| docs/rewrite/sections/game-loop/round8/cards-fog.md | 10 | results | none | game loop round8 fog card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round8/cards.md | 10 | results | none | game loop round8 card sweep table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-loop/round9/ (other files) |  | other | none | game-loop round9 run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/game-loop/round9/self-table.md | 12 | results | none | game loop round9 self-play table; variants; kills per minute; damage and kill mix |
| docs/rewrite/sections/game-notes.md | 333 | notes | 2026-11-13 | game (bnb.game) implementation; state types and frozen dataclass pytrees; static Rules versus ship card tensors; explicit randomness inputs; stage cuts between move and resolve; medium as field-axis reduction and its gradient; damage matrix and killer by scatter; killer tie-break; friendly-fire reward stub; sight test bitmasks and per-observer vision range; legacy skeleton name removal; fog of war per environment; two-teams constant; benchmark cells; CUDA-graph capture of scripted controller; hit-test memory and spawn costs |
| docs/rewrite/sections/game-spec.md | 502 | spec | none | game specification; sizes and presets; ship types; the tick; ship flight; constants and functions |
| docs/rewrite/sections/game.md | 123 | decisions | 2026-10-31 | section 1 game; author decisions; constants; agent scope proposal; race review carry-over |
| docs/rewrite/sections/gpu-box-log.md | 929 | log | 2026-11-06 to 2026-11-09 | rented RTX 3090 box; setup, volume and credentials; steps-per-second speed work; micro-batch rows; fused Adam; host store versus GPU store; TF32 and bfloat16 precision; shot trajectory query under autocast; BC arms research (C-fixed, 2d, 3d, 4d, large batch); deadreckon-split, absolute, fourier inputs, hidden physics merges; tournaments and Elo standings; outcome head Brier skill; learnable extrapolation decay; laptop and box work split; disk and LFS/data saving; git push and W&B access; 60M resume rule |
| docs/rewrite/sections/gpu-box/ (other files) |  | other | none | GPU box round, arm, fleet, tournament and hub scripts; python scripts; shell scripts |
| docs/rewrite/sections/griffin-only-brief.md | 65 | brief | 2026-11-06 | Griffin-only memory arm; no forecast fed back; build task |
| docs/rewrite/sections/gun-study/ (other files) |  | other | none | gun-study run artefacts; result json; per-game or per-record jsonl; python scripts |
| docs/rewrite/sections/gun-study/aspect/ (other files) |  | other | none | gun-study aspect run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/gun-study/aspect/table.md | 26 | results | none | gun study aspect table; time to kill by aspect; ship types including vizia |
| docs/rewrite/sections/gun-study/progress.md | 84 | log | 2026-11-12 | gun study; duels; matchups; variants; target; archetype draft |
| docs/rewrite/sections/gun-study/table.md | 43 | results | none | gun study table; time to kill head-on and tail; sustained and instant 180 degree turn; first-merge kills |
| docs/rewrite/sections/gun-study/variants/ (other files) |  | other | none | gun-study variants run artefacts; result json |
| docs/rewrite/sections/gun-study/variants/table.md | 12 | results | none | gun study variants table; damage-scaled P-51D variants; time to kill |
| docs/rewrite/sections/hygiene-brief.md | 67 | brief | 2026-11-03 | model settings in TrainConfig; test hygiene |
| docs/rewrite/sections/hygiene-notes.md | 184 | notes | none | hygiene round; TrainConfig model settings and ship card; GPU resume; CUDA guard; integer dtypes; retention docstring; quick tier |
| docs/rewrite/sections/imugi-rename/progress.md | 28 | log | 2026-11-15 | Imugi rename |
| docs/rewrite/sections/inference-ablations-brief.md | 82 | brief | 2026-11-06 | inference-time ablations; rotary; forecast; measurement run |
| docs/rewrite/sections/intermittent-failure-brief.md | 85 | brief | none | intermittent failure; kernel autotuning; hypotheses; investigation method |
| docs/rewrite/sections/keep-and-mix-brief.md | 60 | brief | 2026-11-06 | keeping every update's policies; tournaments over mixed architectures |
| docs/rewrite/sections/launch/ (other files) |  | other | none | launch, bootstrap and heartbeat scripts; launch config; python scripts; shell scripts; toml config |
| docs/rewrite/sections/launch/progress.md | 87 | log | 2026-11-12 | launch; merges; training-side switches; launch scripts; single-5090 dry run; laptop end-to-end |
| docs/rewrite/sections/launch/runbook.md | 179 | guide | 2026-11-12 | launch runbook; rented box bootstrap; leaderboard; dry run on one 5090; launch.toml lineup; pull.sh; tournament |
| docs/rewrite/sections/league-brief.md | 94 | brief | none | league section 8; tests; measurement; boundaries |
| docs/rewrite/sections/league-notes.md | 224 | notes | 2026-11-03 to 2026-11-07 | league section 8; pool; slots and compiled graph; sampling; live gauge; snapshots and retention; published ratings |
| docs/rewrite/sections/league-sampling-sim-2-brief.md | 65 | brief | 2026-11-06 | league sampling round 2; turnover; slots; ladder; weighting; sampler comparison measures |
| docs/rewrite/sections/league-sampling-sim-brief.md | 67 | brief | 2026-11-06 | league sampling; opponent selection; simulation; sampler comparison |
| docs/rewrite/sections/league-sampling-sim.md | 268 | results | 2026-11-06 | league sampling simulation; reproduction; cause; samplers; overlap with rating games; round 2 turnover, slots, ladder, weighting |
| docs/rewrite/sections/league-spec.md | 178 | spec | 2026-11-01 to 2026-11-14 | league opponents and ratings; pool; slots; sampling; live gauge; snapshots and retention; published ratings; head-to-head; architectures |
| docs/rewrite/sections/league.md | 54 | decisions | 2026-10-30 to 2026-11-01 | section 8 league and ratings; training review 4b and 4e; author decisions |
| docs/rewrite/sections/learner-brief.md | 109 | brief | none | section 6 learner; boundaries; hand back |
| docs/rewrite/sections/learner-notes.md | 429 | notes | 2026-11-01 | learner (bnb.learn) implementation; staging ring and segment store with host streaming; enemy-row compaction of replay and privileged rows; returns and GAE per shard; loss terms and masks; shot re-simulation; tail shots and label validity; no torch.compile or global settings in library code; precision matrix (float32, TF32, bf16); toy relational learner; KL early stop; Adam epsilon; bytes per view-tick; streaming cost and pinned memory; per-update random streams; outcome class of match result; runtime stage list |
| docs/rewrite/sections/learner-spec.md | 194 | spec | 2026-11-01 | PPO and labels; model protocol; data path via pinned CPU RAM; returns; losses; next-state, enemy-action and density labels; shot labels; mixed batches and metrics; precision |
| docs/rewrite/sections/learner.md | 73 | decisions | 2026-11-01 | section 6 learner; author decisions; later ideas; settled points |
| docs/rewrite/sections/lethality/ (other files) |  | other | none | lethality run artefacts; python scripts; toml config |
| docs/rewrite/sections/lethality/diag/ (other files) |  | other | none | lethality diag run artefacts; result json |
| docs/rewrite/sections/lethality/diag/frontline2.md | 3 | results | none | lethality diag frontline2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/ (other files) |  | other | none | lethality geometry run artefacts; result json; shell scripts |
| docs/rewrite/sections/lethality/geometry/A-0.5x.md | 3 | results | none | lethality geometry A-0.5x table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/A-10v10.md | 3 | results | none | lethality geometry A-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/A-2x.md | 3 | results | none | lethality geometry A-2x table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/A-5v5.md | 3 | results | none | lethality geometry A-5v5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/B-0.5x.md | 3 | results | none | lethality geometry B-0.5x table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/B-10v10.md | 3 | results | none | lethality geometry B-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/B-2x.md | 3 | results | none | lethality geometry B-2x table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/C-10v10.md | 3 | results | none | lethality geometry C-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/D-10v10.md | 3 | results | none | lethality geometry D-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/geometry/table.md | 6 | results | none | lethality geometry summary table; variants by size and arena scale |
| docs/rewrite/sections/lethality/round1/ (other files) |  | other | none | lethality round1 run artefacts; result json; run log text |
| docs/rewrite/sections/lethality/round1/sizes.md | 9 | results | none | lethality round1 sizes table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round10/ (other files) |  | other | none | lethality round10 run artefacts; result json |
| docs/rewrite/sections/lethality/round10/frontline.md | 8 | results | none | lethality round10 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round11/ (other files) |  | other | none | lethality round11 run artefacts; result json |
| docs/rewrite/sections/lethality/round11/deathmatch.md | 4 | results | none | lethality round11 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round11/elimination.md | 4 | results | none | lethality round11 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round11/frontline-arena0.5.md | 4 | results | none | lethality round11 frontline-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round11/frontline-arena2.md | 4 | results | none | lethality round11 frontline-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round11/frontline.md | 4 | results | none | lethality round11 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round2/ (other files) |  | other | none | lethality round2 run artefacts; result json |
| docs/rewrite/sections/lethality/round2/elimination.md | 7 | results | none | lethality round2 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round2/frontline.md | 7 | results | none | lethality round2 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round3/ (other files) |  | other | none | lethality round3 run artefacts; result json |
| docs/rewrite/sections/lethality/round3/elimination.md | 13 | results | none | lethality round3 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round4/ (other files) |  | other | none | lethality round4 run artefacts; result json |
| docs/rewrite/sections/lethality/round4/deathmatch.md | 11 | results | none | lethality round4 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round4/elimination.md | 11 | results | none | lethality round4 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round4/frontline.md | 11 | results | none | lethality round4 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round5/ (other files) |  | other | none | lethality round5 run artefacts; result json |
| docs/rewrite/sections/lethality/round5/frontline.md | 7 | results | none | lethality round5 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round6/ (other files) |  | other | none | lethality round6 run artefacts; result json |
| docs/rewrite/sections/lethality/round6/deathmatch.md | 10 | results | none | lethality round6 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round6/elimination.md | 10 | results | none | lethality round6 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round6/frontline.md | 10 | results | none | lethality round6 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/ (other files) |  | other | none | lethality round7-spot run artefacts; result json; shell scripts |
| docs/rewrite/sections/lethality/round7-spot/deathmatch-10v10.md | 5 | results | none | lethality round7-spot deathmatch-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/deathmatch-arena0.5.md | 5 | results | none | lethality round7-spot deathmatch-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/deathmatch-arena2.md | 5 | results | none | lethality round7-spot deathmatch-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/elimination-10v10.md | 5 | results | none | lethality round7-spot elimination-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/elimination-arena0.5.md | 5 | results | none | lethality round7-spot elimination-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/elimination-arena2.md | 5 | results | none | lethality round7-spot elimination-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/frontline-10v10.md | 5 | results | none | lethality round7-spot frontline-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/frontline-arena0.5.md | 5 | results | none | lethality round7-spot frontline-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7-spot/frontline-arena2.md | 5 | results | none | lethality round7-spot frontline-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7/ (other files) |  | other | none | lethality round7 run artefacts; result json |
| docs/rewrite/sections/lethality/round7/deathmatch.md | 9 | results | none | lethality round7 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7/elimination.md | 9 | results | none | lethality round7 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round7/frontline.md | 9 | results | none | lethality round7 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/ (other files) |  | other | none | lethality round8 run artefacts; result json; shell scripts |
| docs/rewrite/sections/lethality/round8/deathmatch-10v10.md | 5 | results | none | lethality round8 deathmatch-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/deathmatch.md | 5 | results | none | lethality round8 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/elimination-10v10.md | 5 | results | none | lethality round8 elimination-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/elimination-arena0.5.md | 5 | results | none | lethality round8 elimination-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/elimination-arena2.md | 5 | results | none | lethality round8 elimination-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/elimination.md | 5 | results | none | lethality round8 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8/frontline.md | 5 | results | none | lethality round8 frontline table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round8b/ (other files) |  | other | none | lethality round8b run artefacts; result json |
| docs/rewrite/sections/lethality/round9/ (other files) |  | other | none | lethality round9 run artefacts; result json; shell scripts |
| docs/rewrite/sections/lethality/round9/deathmatch-10v10.md | 5 | results | none | lethality round9 deathmatch-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/deathmatch-arena0.5.md | 5 | results | none | lethality round9 deathmatch-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/deathmatch-arena2.md | 5 | results | none | lethality round9 deathmatch-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/deathmatch.md | 5 | results | none | lethality round9 deathmatch table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/elimination-10v10.md | 5 | results | none | lethality round9 elimination-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/elimination-arena0.5.md | 5 | results | none | lethality round9 elimination-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/elimination-arena2.md | 5 | results | none | lethality round9 elimination-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/elimination.md | 5 | results | none | lethality round9 elimination table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/frontline-10v10.md | 5 | results | none | lethality round9 frontline-10v10 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/frontline-arena0.5.md | 5 | results | none | lethality round9 frontline-arena0.5 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/lethality/round9/frontline-arena2.md | 5 | results | none | lethality round9 frontline-arena2 table; candidate score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/ (other files) |  | other | none | maneuvers-2 run artefacts; result json; python scripts; shell scripts |
| docs/rewrite/sections/maneuvers-2/confirm/ (other files) |  | other | none | maneuvers-2 confirm run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/confirm/sizes.md | 22 | results | none | maneuvers 2 confirm results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/confirm2/ (other files) |  | other | none | maneuvers-2 confirm2 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/confirm2/sizes.md | 18 | results | none | maneuvers 2 confirm2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/progress.md | 493 | log | none | maneuvers 2 doctrine tuning branch; Imugi doctrine fields (crowd extension, duel extension, hold fire); r9m and r9m2 doctrines and league players; strength rule against r9m; shield-ratio and energy-ratio gates; per-type, mirror-only and team-only gating; choice entropy in bits; mutual head-on joust share; Yongma and Jujak mirrors; card variants (D2, cone, big v small, frozen); energy versus turn balance per fleet size; robustness against league players; 10v10 crowd check; make.py and sizes.py tools |
| docs/rewrite/sections/maneuvers-2/robust/ (other files) |  | other | none | maneuvers-2 robust run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/robust/sizes.md | 10 | results | none | maneuvers 2 robust results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round1/ (other files) |  | other | none | maneuvers-2 round1 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round1/sizes.md | 41 | results | none | maneuvers 2 round1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round2/ (other files) |  | other | none | maneuvers-2 round2 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round2/sizes.md | 29 | results | none | maneuvers 2 round2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round3/ (other files) |  | other | none | maneuvers-2 round3 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round3/sizes.md | 30 | results | none | maneuvers 2 round3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round4/ (other files) |  | other | none | maneuvers-2 round4 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round4/sizes.md | 26 | results | none | maneuvers 2 round4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round5/ (other files) |  | other | none | maneuvers-2 round5 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round5/sizes.md | 26 | results | none | maneuvers 2 round5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/round6/ (other files) |  | other | none | maneuvers-2 round6 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/round6/sizes.md | 26 | results | none | maneuvers 2 round6 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers-2/ten/ (other files) |  | other | none | maneuvers-2 ten run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers-2/ten/sizes.md | 10 | results | none | maneuvers 2 ten results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/ (other files) |  | other | none | maneuvers run artefacts; result json; python scripts; shell scripts |
| docs/rewrite/sections/maneuvers/confirm/ (other files) |  | other | none | maneuvers confirm run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/confirm/sizes.md | 30 | results | none | maneuvers confirm results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/progress.md | 534 | log | none | tune/maneuvers branch; scripted-agent maneuvers as doctrine elements; lead turn; yo-yo; brake; one-circle and two-circle sense after merge; crowd gate and crowd extension; doctrine table columns; joust and side-by-side stalemate; cards frozen, D2 and cone variants; r9 and r9m agents; 1v1, 2v2, 5v5, 10v10 sweeps; 1024-game confirmation; robustness against the league; final state; memory-limited GPU unit launches |
| docs/rewrite/sections/maneuvers/robust/ (other files) |  | other | none | maneuvers robust run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/robust/sizes.md | 8 | results | none | maneuvers robust results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round1/ (other files) |  | other | none | maneuvers round1 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round1/sizes.md | 54 | results | none | maneuvers round1 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round2/ (other files) |  | other | none | maneuvers round2 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round2/sizes.md | 54 | results | none | maneuvers round2 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round3/ (other files) |  | other | none | maneuvers round3 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round3/sizes.md | 46 | results | none | maneuvers round3 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round4/ (other files) |  | other | none | maneuvers round4 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round4/sizes.md | 38 | results | none | maneuvers round4 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round5/ (other files) |  | other | none | maneuvers round5 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round5/sizes.md | 34 | results | none | maneuvers round5 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round6/ (other files) |  | other | none | maneuvers round6 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round6/sizes.md | 38 | results | none | maneuvers round6 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round7/ (other files) |  | other | none | maneuvers round7 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round7/sizes.md | 38 | results | none | maneuvers round7 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/maneuvers/round8/ (other files) |  | other | none | maneuvers round8 run artefacts; result json; run log text |
| docs/rewrite/sections/maneuvers/round8/sizes.md | 26 | results | none | maneuvers round8 results table; candidates by fleet size; energy versus turn score; kill mix; trades; life and kills per minute |
| docs/rewrite/sections/map-panel/progress.md | 25 | log | none | map panel |
| docs/rewrite/sections/media-brief.md | 74 | brief | 2026-11-06 | periodic media during training; W&B videos; head figures; live-run test |
| docs/rewrite/sections/micro-batch-check/ (other files) |  | other | none | micro-batch-check run artefacts; python scripts |
| docs/rewrite/sections/micro-batch-check/results.md | 77 | results | 2026-11-15 | micro-batch check; per-micro-batch quantities in PPO; CPU experiment; bf16 floor |
| docs/rewrite/sections/mixed-brief.md | 86 | brief | none | section 3c mixed batches; boundaries; hand back |
| docs/rewrite/sections/mixed-notes.md | 377 | notes | none | mixed batches section; Pool fields and PoolRules merging; per-environment rules and mode slices; shared versus mode-owned phases; five zone slots and the present column; Obs schema version; Yemong absent_zone token; mixed training preset and TRAINING_MIX; equivalence to single games; seat and offset leak tests; mutation checks; deathmatch_wide preset; game tick and training-cell throughput; speed experiments; bench cell changes; contracts.md requests |
| docs/rewrite/sections/mixed-spec.md | 94 | spec | 2026-11-01 | mixed batches; configuration; game; observation; Yemong; runtime and CLI; acceptance; post-implementation settlements |
| docs/rewrite/sections/mode-timing/ (other files) |  | other | none | mode-timing run artefacts; run log text |
| docs/rewrite/sections/mode-timing/data/ (other files) |  | other | none | mode-timing data run artefacts; per-game or per-record jsonl |
| docs/rewrite/sections/mode-timing/results.md | 195 | results | 2026-11-15 | mode timing; game length of Imugi; real-scale preset values; deathmatch kill rate; tickets for share of games ended |
| docs/rewrite/sections/modes-brief.md | 104 | brief | none | sections 3 and 3b modes and rewards; speed and maintainability goals; boundaries; hand back |
| docs/rewrite/sections/modes-notes.md | 570 | notes | none | sections 3 and 3b notes on modes and rewards; layout, configuration and rules; state and the tick; spawn draws and uniform layout; Frontline and Yemong modes; observation; rewards and ledger; runtime; decisions and deviations; measurements; changes outside bnb.game; tests; requests and open questions; code size; blind review fix round with major, minor and nit items |
| docs/rewrite/sections/modes-spec.md | 258 | spec | 2026-11-01 | modes configuration; game tick additions; team deathmatch; mode-owned spawns; spawn groups; random headings |
| docs/rewrite/sections/modes.md | 132 | decisions | 2026-11-01 | section 3 modes; team deathmatch; Frontline; objective events; mode-owned spawns; observation |
| docs/rewrite/sections/modes/progress.md | 546 | log | 2026-11-12 | mode layer tuning branch; elimination spawns by team side and spawn spread; zone travel times; vast.ai gpubox4 RTX 5060 Ti rental box; semi-random rung temperature; deathmatch kill rate, trades and tickets; release lineup of Yongma and Jujak; Frontline capture dynamics, zone size and capture time; capture lock (hard, soft hold, by distance); per-mode quantities; density sweep over ships per square kilometre; density record for the paper; fog-off runs; mode_dynamics and density_sweep benches; handoff notes |
| docs/rewrite/sections/modes/records/ (other files) |  | other | none | modes records run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/modes/records/zone_travel.md | 64 | results | none | zone travel times table; real-scale frontline map; ship types; distances; coast and boost |
| docs/rewrite/sections/observation-accelerations-brief.md | 89 | brief | 2026-11-08 | policy observation layout records; acceleration columns in observation; observation inputs |
| docs/rewrite/sections/overlays-brief.md | 75 | brief | 2026-11-06 | prediction overlays; play shell; GPU machine usage |
| docs/rewrite/sections/overnight-calibration/ (other files) |  | other | none | overnight-calibration run artefacts; python scripts; shell scripts; toml config |
| docs/rewrite/sections/overnight-calibration/progress.md | 39 | log | 2026-11-13 | overnight calibration; setup; log |
| docs/rewrite/sections/parallel-tests-brief.md | 79 | brief | 2026-11-03 | parallel CPU tests; quick test tier; GitHub Actions |
| docs/rewrite/sections/particles-notes.md | 51 | notes | 2026-11-07 | K-particle position heads; collapse of particle heads; hidden ship positions; survey of methods |
| docs/rewrite/sections/per-block-readout-brief.md | 48 | brief | 2026-11-07 | rotary read-out per block; BC arm D; model block structure; CPU tests |
| docs/rewrite/sections/play-brief.md | 107 | brief | none | section 10 play shell; renderer milestone 1; boundaries and hand-back |
| docs/rewrite/sections/play-m2-brief.md | 93 | brief | none | play shell milestone 2; on-screen interface; tests; commits |
| docs/rewrite/sections/play-m2-fix-brief.md | 46 | brief | 2026-11-03 | play milestone 2 fix round; tests and boundaries |
| docs/rewrite/sections/play-m2-notes.md | 412 | notes | none | section 10 play shell milestone 2; progress; code layout; input; layout and collapsing panels; ImGui theme; live and replayed matches; agents per side; spec items on principles, layout, selection and hit testing, camera, controllers, replays, runtime requests; screenshots; measurements; tests; deviations; fix round; agent step in a CUDA graph |
| docs/rewrite/sections/play-m2-spec.md | 164 | spec | 2026-11-03 | on-screen interface principles; layout; selection and hit testing; camera; controllers per side; trained agents; replays |
| docs/rewrite/sections/play-notes.md | 497 | notes | none | section 10 play shell milestone 1; layout and the loop; snapshot and interpolation; render passes per look element; fog mask; themes; decisions and deviations; frame time and sim rate; per-pass GPU and CPU timings; per-stage CPU profile; index texture; screenshots; milestone 2 foundations; code style draft; fix round; requests and open questions; code size |
| docs/rewrite/sections/play-spec.md | 204 | spec | 2026-11-01 | renderer packages and dependencies; match loop; snapshot; interpolation; OpenGL rendering; views, camera and themes; keyboard pilot |
| docs/rewrite/sections/play.md | 95 | decisions | 2026-11-01 to 2026-11-02 | section 10 play shell; renderer and UI decisions |
| docs/rewrite/sections/portable-batch.md | 33 | decisions | 2026-11-02 | portable batches; batch format |
| docs/rewrite/sections/position-codes/ (other files) |  | other | none | position-codes run artefacts; result json; per-game or per-record jsonl; png images |
| docs/rewrite/sections/position-codes/progress.md | 103 | log | 2026-11-11 to 2026-11-12 | position codes; rotary aliasing sweep; build; geometry head weight |
| docs/rewrite/sections/rating-all-games-brief.md | 56 | brief | 2026-11-06 | rating from every game; rating lanes; GPU machine usage |
| docs/rewrite/sections/rating-allocator/ (other files) |  | other | none | rating-allocator run artefacts; per-game or per-record jsonl; shell scripts |
| docs/rewrite/sections/rating-allocator/progress.md | 127 | log | 2026-11-14 to 2026-11-15 | rating allocator; shared pool; root cause and design; loop check; no prior in the fit |
| docs/rewrite/sections/release-candidate/progress.md | 104 | log | 2026-11-09 to 2026-11-12 | release candidate; merges; lineup; champion model settings as code defaults |
| docs/rewrite/sections/resume-integrity-brief.md | 54 | brief | none | resume test; integrity check; rounding |
| docs/rewrite/sections/rewards-spec.md | 111 | spec | 2026-11-01 | reward levels; ledger with boundary row; one tick of rewards; runtime contracts; acceptance |
| docs/rewrite/sections/rewards.md | 98 | decisions | 2026-11-01 | section 3b rewards; reward levels; prices; attribution; where rewards run; modes relation |
| docs/rewrite/sections/rl-degradation-brief.md | 80 | brief | 2026-11-04 | RL training against the BC clone; suspects list; investigation method |
| docs/rewrite/sections/rl-round-five-brief.md | 91 | brief | 2026-11-06 | league turnover clean cut; ladder removal; memory hand-over test; DPPO; clip-higher |
| docs/rewrite/sections/rl-round-five-notes.md | 115 | notes | 2026-11-06 | league turnover clean cut; memory hand-over test; DPPO and clip-higher; representation tap |
| docs/rewrite/sections/rollout-lanes-brief.md | 99 | brief | 2026-11-04 | rollout batch layout; advantages chained across shards; lookahead ticks; rating lanes; entropy per action factor; dashboards |
| docs/rewrite/sections/rollout-lanes-notes.md | 92 | notes | 2026-11-04 to 2026-11-05 | rollout layout; rating lanes; to-do list; open questions |
| docs/rewrite/sections/round2-log.md | 216 | log | 2026-11-07 to 2026-11-08 | round 2 work log; author-away activity; acceleration arms |
| docs/rewrite/sections/round2-plan.md | 203 | plan | 2026-11-08 | round 2 goals; acceleration inputs; next-state baselines; correction head; experimental arms; components |
| docs/rewrite/sections/run-analysis-brief.md | 74 | brief | 2026-11-04 | health sweep of first BC and RL runs; run series; reporting |
| docs/rewrite/sections/run-control-brief.md | 86 | brief | 2026-11-07 | graceful stop; BC policy snapshots; averages after warm-up; GPU job queue; bnb queue |
| docs/rewrite/sections/run-fixes-brief.md | 51 | brief | 2026-11-04 | fixes from first BC run |
| docs/rewrite/sections/runs-brief.md | 125 | brief | none | section 11 runs and checkpoints; context handling; boundaries; hand-back |
| docs/rewrite/sections/runs-notes.md | 342 | notes | 2026-11-02 | section 11 runs and checkpoints notes; progress; design; decisions and deviations; checkpoint size and times with rollout carry; throughput with and without checkpoints; exact resume; full test suite; changes outside bnb.runs; requests and open questions; code size; fix round |
| docs/rewrite/sections/runs-spec.md | 266 | spec | 2026-11-02 to 2026-11-08 | packages; serialisation; policy files and training checkpoints; model registry and build record; run directory; command line; memory probe |
| docs/rewrite/sections/runs.md | 208 | decisions | 2026-11-01 to 2026-11-02 | saved things kinds; build record; file format; run directory; command line |
| docs/rewrite/sections/scripted-bc-brief.md | 115 | brief | none | sections 4 and 7; Imugi; BC; boundaries and hand-back |
| docs/rewrite/sections/scripted-bc-notes.md | 531 | notes | 2026-11-02 | Imugi scripted controller and behaviour cloning implementation; ship characteristics from the ship card; combat and mode layers in one fixed-shape pass; Frontline zone assignment by rank; lead distance cap; inverse-CDF sampling; per-view parameters; BehaviourCloning learner and update-time teacher labels; Imugi parameter search and opponent scores; teacher entropy, temperature and probability floor; toy BC test; BC stop rule as lower confidence bound; BC throughput versus PPO; policy-file warm start to RL; behaviour-mutation and acceptance tests |
| docs/rewrite/sections/scripted-bc-spec.md | 254 | spec | 2026-11-01 to 2026-11-05 | Imugi interface; ship card characteristics; Imugi parameters; combat layer; Frontline layer; rungs; BC |
| docs/rewrite/sections/scripted-memory-brief.md | 121 | brief | 2026-11-05 | Imugi per-ship memory; soft hysteresis; dead reckoning of hidden enemies; ladder bottom; parameter search |
| docs/rewrite/sections/scripted-memory-notes.md | 173 | notes | 2026-11-05 | Imugi memory form; search; acceptance; softness audit; open items |
| docs/rewrite/sections/scripted-redesign-brief.md | 124 | brief | none | soft Imugi; logistic ramps; temperature and floor; population round robin; wiring in; acceptance |
| docs/rewrite/sections/scripted-redesign-notes.md | 188 | notes | none | scale floors; floors per decision; population search; acceptance; softness audit |
| docs/rewrite/sections/scripted-softness-brief.md | 85 | brief | 2026-11-04 to 2026-11-05 | Imugi softness; head-to-head players; tempered scripted agents; BC ceiling; softness audit |
| docs/rewrite/sections/scripted-softness-notes.md | 134 | notes | 2026-11-04 to 2026-11-05 | scores against raw Imugi; softness audit; chosen-value probability and entropy; ramps; step-like ramps |
| docs/rewrite/sections/scripted.md | 58 | decisions | 2026-11-01 to 2026-11-05 | section 4 Imugi; training review 4c; author decisions |
| docs/rewrite/sections/seat-symmetry-notes.md | 125 | notes | 2026-11-02 | seat symmetry; mirror test; coverage; cause; fix; before and after comparison |
| docs/rewrite/sections/seed-isolation/progress.md | 60 | log | none | seed isolation; audit scripts re-run; milestones; CPU cost |
| docs/rewrite/sections/ship-card-brief.md | 89 | brief | none | section 1b ship card; boundaries and hand-back |
| docs/rewrite/sections/ship-card-notes.md | 566 | notes | 2026-11-02 | ship card section implementation; per-type scale fields and bleed exponent; fighter and heavy types; flight and combat handles; handle solver (Levenberg-Marquardt); manoeuvre measurements; zero-lift drag and P_min cruise fix; per-type vision range and bullet drag; head-on fraction as frontal armour; time-native bullet life and fire rate; bnb ship-card chart tool; round trip, independence and conditioning tests; tick-rate sensitivity; energy conservation check; Yemong coding ranges from the card; merge of origin/v2 and launch/fire split; fix round |
| docs/rewrite/sections/ship-card-spec.md | 177 | spec | 2026-10-31 to 2026-11-02 | per-type parameters; physics changes; handles; solver; tick rates; reference chart tool; acceptance |
| docs/rewrite/sections/ship-card.md | 132 | decisions | 2026-10-31 | ship card; energy-maneuverability lens; flight handles; combat handles; cruise and store; solve method; acceptance |
| docs/rewrite/sections/simplification/ |  | other | 2026-11-21 | simplification audit scripts; operation counts; learner FLOP and byte counts |
| docs/rewrite/sections/spawn-heading/progress.md | 100 | log | 2026-11-12 | spawn heading; room to turn away from the edge; random headings; rule |
| docs/rewrite/sections/state-deltas-brief.md | 77 | brief | 2026-11-07 | per-tick ship-state change; delta target codes; CPU rollouts |
| docs/rewrite/sections/state-deltas-tables.md | 1975 | results | none | per-tick state change tables; structure shares by player and preset; scripted and calm-fern self-play segments; Frontline, Deathmatch and wide Deathmatch presets; position, velocity, heading, nose rate, angle of attack changes; shield, shield recharge wait, energy, gun cooldown changes; log refractive index (field bands); persistence, dead-reckoning, arc and rotated-velocity predictors; codes at horizons 1, 8 and 32; residual quantiles; boost, reverse, respawn, zone and boundary events |
| docs/rewrite/sections/state-deltas.md | 364 | results | none | one-tick change of ship-state quantities; delta-target coding for the next-state head; current absolute codes; hard bounds from flight_tick for fighter and heavy; power and stick clusters; rare events (shots, hits, boundary ticks); dead reckoning and kick-drift-kick baselines; body frames and polar frame; candidate codes (uniform, symlog, quantile, hierarchy); symlog two-hot with zero centre; bin count sweep; belief horizons; nose rate range; velocity reach in refraction fields; Imugi and calm-fern sample players |
| docs/rewrite/sections/steady-speed/ (other files) |  | other | none | steady-speed run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/steady-speed/progress.md | 237 | log | 2026-11-12 to 2026-11-13 | steady speed; method; 2048 environments; log |
| docs/rewrite/sections/steady-speed/records/ (other files) |  | other | none | steady-speed records run artefacts; per-game or per-record jsonl |
| docs/rewrite/sections/tank-armour/ (other files) |  | other | none | tank-armour run artefacts; python scripts; shell scripts |
| docs/rewrite/sections/tank-armour/confirm/ (other files) |  | other | none | tank-armour confirm run artefacts; result json; run log text |
| docs/rewrite/sections/tank-armour/confirm/fights.md | 22 | results | none | tank armour confirmation fight table; two-circle fights; chooser; same and opposite sense |
| docs/rewrite/sections/tank-armour/confirm/sizes.md | 7 | results | none | tank armour confirmation sizes table; candidates by fleet size |
| docs/rewrite/sections/tank-armour/confirm/table.md | 7 | results | none | tank armour confirmation summary table; 1v1 mutual fights; mixed kills |
| docs/rewrite/sections/tank-armour/images/ (other files) |  | other | none | trajectory and event images with data; result json; png images |
| docs/rewrite/sections/tank-armour/progress.md | 313 | log | none | tune/tank-armour branch; armour cone sharpness card field; head-on fraction and angle scales; dive exchange strength; fight statistics (two circles, post-merge choices); round 1 cone grid; pace compensators; scripted yo-yo stalemate breaker sketch; crowd-only counters for the Yongma; D2 card; 1024-game confirmation 1v1 to 10v10; mirror joust; final state; memory-limited GPU unit launches |
| docs/rewrite/sections/tank-armour/round1/ (other files) |  | other | none | tank-armour round1 run artefacts; result json; run log text |
| docs/rewrite/sections/tank-armour/round1/fights.md | 182 | results | none | tank armour round1 fight table; two-circle fights; chooser; same and opposite sense |
| docs/rewrite/sections/tank-armour/round1/table.md | 47 | results | none | tank armour round1 summary table; 1v1 mutual fights; mixed kills |
| docs/rewrite/sections/tank-armour/round2/ (other files) |  | other | none | tank-armour round2 run artefacts; result json |
| docs/rewrite/sections/tank-armour/round2/fights.md | 106 | results | none | tank armour round2 fight table; two-circle fights; chooser; same and opposite sense |
| docs/rewrite/sections/tank-armour/round2/sizes.md | 28 | results | none | tank armour round2 sizes table; candidates by fleet size |
| docs/rewrite/sections/tank-armour/round2/table.md | 28 | results | none | tank armour round2 summary table; 1v1 mutual fights; mixed kills |
| docs/rewrite/sections/tank-armour/round3/ (other files) |  | other | none | tank-armour round3 run artefacts; result json |
| docs/rewrite/sections/tank-armour/round3/fights.md | 46 | results | none | tank armour round3 fight table; two-circle fights; chooser; same and opposite sense |
| docs/rewrite/sections/tank-armour/round3/sizes.md | 13 | results | none | tank armour round3 sizes table; candidates by fleet size |
| docs/rewrite/sections/tank-armour/round3/table.md | 13 | results | none | tank armour round3 summary table; 1v1 mutual fights; mixed kills |
| docs/rewrite/sections/tank-armour/round4/ (other files) |  | other | none | tank-armour round4 run artefacts; result json |
| docs/rewrite/sections/tank-armour/round4/fights.md | 34 | results | none | tank armour round4 fight table; two-circle fights; chooser; same and opposite sense |
| docs/rewrite/sections/tank-armour/round4/sizes.md | 10 | results | none | tank armour round4 sizes table; candidates by fleet size |
| docs/rewrite/sections/tank-armour/round4/table.md | 10 | results | none | tank armour round4 summary table; 1v1 mutual fights; mixed kills |
| docs/rewrite/sections/temperature-sweep.md | 119 | results | 2026-11-09 to 2026-11-11 | temperature sweep; Elo ratings; entropy; setup; caveats |
| docs/rewrite/sections/temperature-sweep/ (other files) |  | other | none | temperature-sweep run artefacts; result json; per-game or per-record jsonl |
| docs/rewrite/sections/temporal-first-brief.md | 81 | brief | 2026-11-06 | temporal-first trunk; rotary read-out; BC arms C and D; CPU tests; docs |
| docs/rewrite/sections/tournament-slots-brief.md | 72 | brief | 2026-11-06 | offline rating tournament; fixed slots; first real use; GPU machine usage |
| docs/rewrite/sections/training-defaults/progress.md | 107 | log | 2026-11-15 | training defaults; deathmatch outcome kill_lead; mirrored coin-flip lineups; mode mix |
| docs/rewrite/sections/yemong-brief.md | 104 | brief | none | section 5 Yemong; boundaries and hand-back |
| docs/rewrite/sections/yemong-notes.md | 443 | notes | 2026-11-01 | Yemong model implementation; module layout and reference adapter; Carry pytree (Griffin state, convolution buffer, belief); step and unroll equivalence; compile, capture and ONNX export; replay of action belief and spreads; time_since_seen; bf16 autocast option; rotary periods; density target and return scale learner-side; presence radius; speed benchmarks and VRAM; Outputs and Trajectory contract changes; per-view position offset; float32 precision in shifted frame; chunked scan memory; fused projections; replay dtype and act-stage drift; unroll benchmark cells |
| docs/rewrite/sections/yemong-spec.md | 595 | spec | 2026-11-01 to 2026-11-12 | Yemong scope and configuration; ship state code; belief channels; code layout; code primitives; ship-state codec |
| docs/rewrite/sections/yemong.md | 64 | decisions | 2026-10-31 to 2026-11-01 | Yemong model; related decisions outside the model; items settled in the spec |
| docs/rewrite/sections/zero-sum-rewards/progress.md | 52 | log | 2026-11-12 | zero-sum rewards; two reward tests; history of the failures; tests versus rewards |
| docs/rewrite/ship-card.md | 164 | spec | 2026-10-23 | Ship card; flight handles; global shape rules; combat handles; P-51D similarity ratios; ship radius and gun range scaling |
| docs/rewrite/simplification.md | 715 | review | 2026-10-16 to 2026-11-21 | simplification audit; project goals, fixed scope and non-goals; hand-built belief for fogged enemies; distributional outputs and symlog HL-Gauss bins; 236 km position frame; input encoding; auxiliary heads and lookahead; rating lanes; Imugi agent cost; mixed batches; bit-exact reproducibility; dynamic versus fixed shapes; constants as tensors; Triton kernels decision; operation counts per tick and per forward; compile time budgets; real-time frame budget; GPU hours and learner FLOPs; band-aids by root cause R1-R11; unconfirmed defects; v1 versus v2 comparison |
| docs/rewrite/visual-mockups/ (other files) |  | other | none | visual mock-up scenes; html page; png images |
| docs/rewrite/visual-mockups/NOTE-v2.md | 8 | notes | 2026-11-01 | note on the v2 copy of the visual mock-ups |
| docs/rewrite/visual-mockups/README.md | 72 | guide | 2026-10-27 | visual mock-ups; scene files; rules the scenes follow; current state |
| docs/rewrite/yemong-review.md | 2011 | review | 2026-09-30 to 2026-10-22 | Yemong network shape and token inputs; rotary position encoding; Griffin RG-LRU temporal block and memory resets; heads (action, enemy action, next state, value, hex density, counterfactual shot); ghost bullets; HL-Gauss codes and calibrated bin values; belief store and moment decoding; masked-token encoding; per-level critic, GAE and return scale; reward function event blocks; naming convention (ship, team, controller, lineup, view, carry); hex density grid layout and disc size; rollout loop and ego_pass; privileged fog-free critic; league opponents, slots and drain; behaviour cloning teacher; PPO batch geometry, optimizer and host streaming; non-finite gradient handling; evaluation pipeline and leaderboard |
