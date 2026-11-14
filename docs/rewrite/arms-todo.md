# Before the arms: decisions and to-do

This is the tracking document for the work before renting the 5090s for the pairwise arms. It records:
- what is open for the author;
- every decision, with its date;
- the work in progress and what it waits on.

It is updated and committed whenever a decision is made or a task changes. Candidate arms are in [ideas.md](ideas.md), and papers in [references.md](references.md). Started 2026-11-14.

## Open for the author

Ordered for working through:

0. **The split critic:** it's decided for the premium arm. Open: make it the default for every arm (recommended: one shared BC, and no split-against-unsplit arm), and which levels get the 2:1 premium (recommended: damage and kill/death only; objective progress, objective and outcome stay zero-sum).

1. **Arena size for games at other fleet sizes: deferred (2026-11-14)** until the map-size arm shows whether the agent trains on varied maps. Until then evaluations use the game's density rule provisionally, the same for both arms of a pair.
2. **The explicit list of evaluation games:** drafted in [evaluation-suite.md](evaluation-suite.md), for the author to review.
3. **The break-even SE target:** 0.15 ships provisionally (decided 2026-11-14); final after a GPU probe of what 20-a-side batches cost.
4. **The rest of the decision rule:** proposal (the author's idea): decide by a sequential test on the combined win-probability score, as in chess-engine testing (Stockfish's fishtest uses a sequential probability ratio test). Play until the combined score's interval excludes zero or falls inside an indifference zone, with a maximum budget. Big gaps stop early. The reported scalars keep a fixed precision floor for the curves and the paper. The test runs over all conditions at once: one combined score and one interval. Each condition can run as its own steady stream, a separate process writing to the shared pool, and the test re-reads the pooled records. A tie that persists triggers the curve check: if the slopes differ by more than their noise, extend both arms once (+10–20M) and retest; otherwise choose the cheaper or simpler arm. Also open: when to extend both arms, the no-regression gate, and the SE target (from the 2 × 2 seed measurement).
5. **Freezing the game:** what goes in (see "First, before the game freeze"), and the clock-end tuning.
6. **Fields for the counts arm:** removing fields from the game and masking their slots isn't implemented. Only absent zone slots and benched ships are masked. It needs building if field counts vary.
7. **The counts arm:** whether frontline goes down to 3v3; and since the density rule scales maps with fleet size, varying counts also varies map size.
8. **The run budget, the final arm list, and a review of every arm idea:** whether each is settled by theory, a test or an arm, and in what order.
9. **The averaging sweep:** which early arm saves the live weights every update, and which variants to sweep.

## Decisions

### 2026-11-14

- **Seed isolation:** make the fixes now, before the arms. Arms share random streams wherever they're the same and within rounding.
- **Rating allocator:** choose games by expected information as a design distribution, with a limit on distinct players per batch. One shared pool for every arm.
- **Rating prior:** priors never enter the fit, which is plain maximum likelihood. Swept checkpoints are reported as bounds. A new checkpoint's first opponents are placed at the mean of the other runs' checkpoints at the same step.
- **Every checkpoint is rated** (a dense ladder) because the paper needs Elo curves for every arm.
- **Snapshot schedule:** RL snapshots at 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M, then every 5M.
- **BC is fixed at 100M** for every arm. The plateau stop only logs.
- **Deathmatch outcome:** win/loss/tie by the kill difference (`kill_lead`), like a sports result regardless of margin. Training showed no difference from kill share.
- **Fog:** 100% fog in training, BC and the game. Full information is for debugging only. The scripted agent already reads only visible enemies.
- **Lineups:** each slot is a 50/50 coin flip between Yongma and Jujak, the same for both teams.
- **Mode mix:** frontline 6/16, deathmatch 5/16, elimination 5/16 (768 / 640 / 640 of 2048 environments; it was 8, 5, 3).
- **League structure: three sections, each with one job,** within a maximum number of networks loaded at once:
  1. **Self-play:** a fixed 50% (decided 2026-11-14), today's value. The loop test finds no significant cycles, so the game looks mostly transitive.
  2. **League training games:** every other game, by (1−p)² over all members, with no uniform floor.
  3. **Rating lanes (never trained on):** the live agent against the members whose win probability against it is least certain. Members already played often in training are well measured and get few lane games; rarely played members get the rest. **Uncertainty measure (decided):** the variance of p, about p(1−p)/n, where n counts every live-against-member game, training games included, discounted with a half-life of 4 updates (about 4M steps; Claude's choice) so p follows the changing live agent.
- **League sampling:** (1−p)² over all members, following AlphaStar, plus a uniform floor. The cheese agents are ordinary members with no separate weight. The averaged agents (uniform and exponential) stay members; they already are (`bnb.league.pool`).
- **League snapshot schedule:** about 100–200 Elo between league members, read off the RL curve. From rl-v20 that's members at 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130 and 160M; re-derive it after the game freeze.
- **Cheese agents:** zone rusher, turtle, evader, deathball, jouster and kiter. They're on by default, in the league and in the exploit check.
- **Features are on by default.** A setting may turn one off for an ablation.
- **The scripted agent is renamed Imugi** (이무기), after the open branches merge.
- **Break-even ratio:** "N of ours hold 20 Imugi". Imugi is fixed at 20 ships and our count varies. A logistic fit in log ship count gives the crossing with an SE, using adaptive counts.
- **Evaluation decides by generalisation.** The conditions:
  - 5v5, all modes;
  - 1v1 and 2v2, deathmatch and elimination;
  - 10v10, all modes;
  - outnumbered, as the break-even ratio against 20 Imugi;
  - map scales 0.5× and 2×, only if variable maps survive their arm.

  Each condition has its own shared pool. **Scoring (decided):** everything is scored by win probability. For each condition, A's expected score against B, minus 0.5: the logistic of the Elo contrast, or for break-even, A's win rate against 20 Imugi at B's break-even count. Every mode gets an equal share, and every condition an equal share within its mode. No yardstick or soft cap is needed; win probability is bounded and comparable across conditions.
- **The averaging sweep:** an early arm saves the live weights every update. Uniform, exponential at several decays and windowed uniform averages are then built after training and rated in one tournament. The exponential average folds once per update, so this reproduces a training-time average exactly.
- **References:** a list of papers with what we take from each, where it applies in the code, and how it affected a decision ([references.md](references.md)).
- **Early arms:** variable map sizes against fixed (first; it decides whether variable maps stay in the game and the paper, and if they don't, the colour codes and rotary frequencies are re-tuned for fixed sizes); variable ship and field counts against fixed.
- **Deferred to the later review of all arms:** self-play training both sides against one side; the separate critic (too much work for an arm now).
- **Dropped:** ego-centric frame (×N compute); rotation and reflection augmentation (more environments do the same); KL-to-BC penalty (a single scripted teacher has no diversity to keep); greedy evaluation (evaluation samples like training); belief decay constants (41) and the delta/absolute head (40), using the earlier v2 results.

## In progress

| Work | Branch | State |
|---|---|---|
| Seed isolation | `feat/seed-isolation` (a13173da) | Done on the CPU. Waits for the GPU speed check, the GPU tests and the ONNX export test. |
| Rating allocator, shared pool, placement-only prior | `feat/rating-allocator` (6538fa81) | Done on the CPU. Waits for the layout probe on the GPU. |
| Cheese agents | `feat/cheese-agents` (30a5dd66, from seed isolation) | Done on the CPU. Waits for the first exploit check on the GPU. |
| Break-even ratio | `feat/break-even` (66fd2ba0, from seed isolation) | Done on the CPU: a logistic fit in log count; blocks go to the count that most reduces the variance of log N*; stops at SE ≤ 0.3 ships, 8192 games or a bound; per-batch records with resume; mirrored coin-flip lineups (`Arena(game_lineups=True)`). A benched slot costs about as much as an active one, so every break-even game costs a full 20v20. Waits for the first GPU measurement. |
| References list | docs only (dab3a8de) | 26 entries, about 35 papers. Only 5 are verified online so far; the rest are marked "(unverified)". Update the code locations of the loop check and the allocator after the rating branch merges. |
| Seed twin of rl-v20 (`rl-v20-seed1`, seed 1) | laptop GPU, old code | Running; ETA 01:10 (worst 02:15). |

**GPU queue after the twin:**
1. The seed-streams steps-per-second check, old against new. Merge only if throughput holds.
2. The GPU tests and the ONNX export test.
3. The rating layout probe.
4. Merge seed isolation, then the cheese agents, the rating allocator and break-even.
5. The first exploit check and break-even measurement of rl-v20's final checkpoint.
6. Rate the twin against rl-v20 at matching steps (the noise floor).

## First, before the game freeze

- [ ] **Clock ends (58).** Set each mode's free knob so most games end by the objective, some end on the clock, and progress decides the winner:
  - deathmatch: a fun time limit, then tickets to fit it;
  - frontline: the time limit;
  - elimination: the boundary's shrink rate.

  Tune on a fixed reference matchup.
- [ ] **The rest of the freeze:**
  - frontline's minimum team size;
  - bullet-drag units (idea 75);
  - pending flight and balance work;
  - one game-version bump before the first arm.

## Decided, to implement

- [ ] Snapshot schedule: `policy_snapshot_every` takes one number today, so it needs a list plus a tail.
- [ ] Deathmatch outcome `kill_lead` as the default.
- [ ] Mirrored 50/50 coin-flip lineups.
- [ ] Mode mix 6/5/5.
- [ ] League: (1−p)² sampling, and the league snapshot schedule.
- [ ] Saving the live weights every update (for the early arm), and the post-hoc averaging tool.
- [ ] The Imugi rename.
- [ ] BC fixed at 100M, with the plateau stop logging only.
- [ ] One committed settings file per arm, with no hand-typed overrides.
- [ ] The evaluation suite as a command: the conditions above, their pools and the combined score.

## To write down before the first run

- [ ] **The decision rule:**
  - endpoint Elo (mean of the last 2–3 snapshots) in the shared pools, confirmed by a direct head-to-head of the final checkpoints;
  - extensions when the gap is within noise;
  - the no-regression gate;
  - the SE target.
- [ ] **The run budget and the arm list,** costed by class: RL-only forks from one BC, or full BC+RL.
- [ ] **The figures for the paper,** including Elo over steps for every arm.

## Box setup

- [ ] A setup script that creates the volume layout (runs, caches, ratings, repository and virtual environment) and sets `CACHE`, `RUNS` and the uv and Hugging Face cache paths in the shell profile.
- [ ] Supervisor services that resume runs after a reboot.
- [ ] A periodic copy of the volume to the laptop.
- [ ] First jobs: 2 arms × 2 seeds (the noise of the difference between arms), and the steps-per-second check of the seed streams.

## Cheap checks (no arm needed)

- [ ] **Seat symmetry (47).** A probe predicting the viewer's team from the trunk's features should score no better than chance. A mirror match, a checkpoint against itself scored by seat, should be 50/50.
- [ ] **Old log-probabilities (11).** The KL between the rollout's recorded probabilities and the learner's recomputed ones.
- [ ] **DPPO divergence.** The distribution of the total variation between old and new policies under today's clipping, to see what `dppo_delta` = 0.15 would mask.

## Candidate arms (for the later review)

- **Offensive premium (the author, 2026-11-14; split critic decided for it):** three RL forks from one split-critic BC: zero-sum 1:1 throughout, 2:1 throughout, and 2:1 annealed slowly to 1:1. The author's hypothesis: passive play is an equilibrium of zero-sum self-play when aggression is risky and the winning aggressive line is hard to find; a premium on offence makes searching for it worthwhile, and once found, passivity stops being safe. Predictions to check: fewer draws, shorter lifespans and more engagements early in the premium arms; the deciding measure stays win/loss Elo.
- **Original note:** offence weighted 2:1 over defence, against zero-sum 1:1, or annealed slowly from 2:1 to 1:1. The author recalls an earlier test (v1 or v2.1) where it trained to a clearly higher Elo; v2 dropped the premium on 2026-10-30 so the critic's targets never move. The proposed infrastructure is a split critic: each reward level gets an offence value and a defence value (10 outputs instead of 5), and the weights combine their advantages. Changing or annealing the ratio then never moves a critic target, so one BC run serves every ratio and the arms are cheap RL forks. The split would go in before the shared BC, on by default at 1:1.

- Self-play: train both sides, or one side with more environments and "latest" as a league member at a fixed share.
- Clipping against DPPO (before the learning-rate arm).
- The rest of the numbered list in [ideas.md](ideas.md), with the comments from 2026-11-14.
