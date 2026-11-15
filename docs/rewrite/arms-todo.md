# Before the arms: decisions and to-do

This is the tracking document for the work before renting the 5090s for the pairwise arms. It records:
- what is open for the author;
- every decision, with its date;
- the work in progress and what it waits on.

It is updated and committed whenever a decision is made or a task changes. Candidate arms are in [ideas.md](ideas.md), and papers in [references.md](references.md). Started 2026-11-14.

## Open for the author

Updated 2026-11-15 10:30; the conversation behind it is in [briefing.md](briefing.md).

4. **Game freeze:** locked (2026-11-15, session 3) and implemented as **game version 21** on `feat/imugi-boundary`: deathmatch tickets 4·N^(4/3) to the nearest 5 with 10 minutes; elimination a linear 4-minute shrink from 0 (248 s clock), independent spawn centres; Frontline ±2 with 10 minutes, 3v3 minimum; Imugi's moving-edge steer and explore search. **The hashed fingerprint is gone** (it depended on the machine); `tests/sections/game_reference.pt` is a committed reference compared with a tolerance. Next: re-measure the three modes with the frozen settings, then merge into `prep/arms` and `v2`.
6. **The final arm list** ([arm-list.md](arm-list.md)): Yes/Maybe/No chosen (2026-11-15, session 3). Open: which Maybe runs go in, the A10 variant, and the order.
7. **For review (drafts, 2026-11-15):** [decision-rule.md](decision-rule.md) (how every decision is measured; the late-climbing check and δ from the seed pairs are proposals), [figures-brainstorm.md](figures-brainstorm.md) (a wide list to narrow; items needing logging before the runs are marked), [run-order.md](run-order.md) (waves against one chain).
8. **Older unmerged branches** (tuning and evaluation records from 2026-11-10 to 11-13: tune/ships-r9m2, tune/respawn-waves, eval/yongma-lead, eval/r9m2-maps, eval/morning, tune/frontline-design, tune/duel-deny, run/onehead-champion, probe/position-aliasing, perf/gpu-box, feat/nearest-mlp, feat/crossover, work/design-v3 (WIP, failing tests), wip/playground): not merged into v2. Merge, keep as records, or delete?

## Decisions

### 2026-11-14

- **Seed isolation:** make the fixes now, before the arms. Arms share random streams wherever they're the same and within rounding.
- **Rating allocator:** choose games by expected information as a design distribution, with a limit on distinct players per batch. One shared pool for every arm.
- **Rating prior:** priors never enter the fit, which is plain maximum likelihood. Swept checkpoints are reported as bounds. A new checkpoint's first opponents are placed at the mean of the other runs' checkpoints at the same step.
- **Every checkpoint is rated** (a dense ladder) because the paper needs Elo curves for every arm.
- **Snapshot schedule:** RL snapshots at 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M, then every 5M.
- **BC is fixed at 100M** for every arm. The plateau stop only logs.
- **Deathmatch outcome:** win/loss/tie by the kill difference (`kill_lead`), like a sports result regardless of margin. Training showed no difference from kill share.
- **Fog:** 100% fog in training, BC and the game. Full information is for debugging only. Imugi already reads only visible enemies.
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
- **The scripted agent is renamed Imugi** (이무기). Done on `feat/imugi-rename` (2026-11-15): code, docs and the CLI player name (`imugi`, `imugi@T=..`, `cheese:<name>`); the recorded identity strings stay `scripted`, `scripted@T=..`, `scripted:<name>` so every stored record still reads (`bnb.runtime.opponents.stored_name`). Notes in `docs/rewrite/sections/imugi-rename/progress.md`.
- **Break-even ratio:** "N of ours hold 20 Imugi". Imugi is fixed at 20 ships and our count varies. A logistic fit in log ship count gives the crossing with an SE, using adaptive counts.
- **Evaluation decides by generalisation.** The conditions:
  - 5v5, all modes;
  - 1v1 and 2v2, deathmatch and elimination;
  - 10v10, all modes;
  - outnumbered, as the break-even ratio against 20 Imugi;
  - map scales 0.5× and 2×, only if variable maps survive their arm.

  Each condition has its own shared pool. **Scoring (decided):** everything is scored by win probability. For each condition, A's expected score against B, minus 0.5: the logistic of the Elo contrast, or for break-even, A's win rate against 20 Imugi at B's break-even count. Every mode gets an equal share, and every condition an equal share within its mode. No yardstick or soft cap is needed; win probability is bounded and comparable across conditions.
- **The averaging sweep:** an early arm saves the live weights every update. Uniform, exponential at several decays and windowed uniform averages are then built after training and rated in one tournament. The exponential average folds once per update, so this reproduces a training-time average exactly.
- **Split critic for every arm:** each reward level has an offence and a defence value, and one shared BC serves every arm. No split-against-unsplit arm.
- **Offensive premium levels:** 2:1 on damage and kill/death only. Objective progress, objective and outcome stay zero-sum, since a capture premium would reward swapping zones in self-play.
- **The decision is a sequential test with a fixed number of planned looks** (group sequential). The number of looks and the acceptance thresholds are fixed in advance.
- **Decisions don't need the ladder.** Per condition, the two finals play each other directly (win rate minus 0.5), and break-even plays against Imugi. The full Elo ladder (curves) is for the paper.
- **Ladder timing (decided 2026-11-15):** mostly in one batch at the end. Earlier only on an idle GPU, and only when it doesn't compete with compilation on the CPU. On a tie, each arm's last 5 snapshots (Claude's choice, about 20–25M steps) are rated on demand for the slope check.
### 2026-11-15

- **Allocation within a round is cost-aware** (the most interval reduction per GPU-hour). It adds no bias: the score is a fixed-weight sum of per-condition means, each unbiased however many games it gets. A bias would appear only if games were pooled with weights proportional to their counts, which we don't do. The design simulation checks it.
- **The sequential test's numbers come from a pilot plus simulation**, without waiting for the 2 × 2 seed measurement. The pilot needs the evaluation-suite command and the merged branches.
- **No-regression gate:** significantly worse (about 2 SE) in any mode means no win. A 2-against-1 split is a tie under the tie rule; if it's still split, the incumbent stays.
- **Ladder scope:** curves at 5v5 for the arms. For the first proper run (the baseline) and the final winner, the full ladder in every condition (1v1, 2v2, 10v10, all). Fix as many rules as possible before the first run, and the rest from its data.
- **Frontline lock:** re-tune it to a random walk, measured with fields on (the training conditions).
- **Deathmatch clock:** a standard Halo deathmatch time limit (Halo Infinite's team slayer is 50 kills in 12 minutes; to verify), with tickets set from Imugi's deaths per minute so that about 80% of games end before the clock.
- **Elimination shrink:** from Imugi's time to wipe with no shrink, time the shrink so about 80% of games end before the boundary decides, rounded to a nice number.
- **Fields:** build field removal and masking, like benched ships (branch `feat/absent-fields`, in progress).
- **Frontline team sizes:** 1v1 is out (it likely degenerates to rush, defend or duel); 2v2 and up are in, pending the small-team sanity check (mode-timing bench).
- **Mode coupling, first read from existing runs** (gain from bc-v20 at 75M, each run in its own pool):

  | run (trained on) | deathmatch | elimination | frontline |
  |---|---|---|---|
  | rl-v20 (all modes) | +212 | +662 | +1366 |
  | rl-mix-kl (all modes) | +191 | +743 | +1393 |
  | rl-dm (deathmatch only) | +251 | +916 | +22 (peaked +690 at 45M) |
  | rl-kl (deathmatch only) | +205 | +713 | +598 (peaked +873 at 56M) |

  Deathmatch-only training gains as much elimination as the generalist, or more. Frontline gains early, then forgets. There's no elimination-only or frontline-only run, so the reverse directions are unmeasured.
- **Evaluation weights: thirds** (decided 2026-11-15), with the no-regression gate.
- **Challenger rule (decided 2026-11-15):** the challenger, normally the more complex or slower option, must:
  - beat the incumbent past the gating threshold on the overall score;
  - and not lose to it in any single mode by the set amount.
- **Training mix arm (decided 2026-11-15):** 6/5/5 against 8/4/4, as a cheap RL-only arm. 6/5/5 stays the default until it reports.
- **Game fingerprint:** Claude's call. Keep records consistent: one correct fingerprint and version after the merge, and stored tournaments from older versions are never reused across versions.
- **References:** a list of papers with what we take from each, where it applies in the code, and how it affected a decision ([references.md](references.md)).
- **Early arms:** variable map sizes against fixed (first; it decides whether variable maps stay in the game and the paper, and if they don't, the colour codes and rotary frequencies are re-tuned for fixed sizes); variable ship and field counts against fixed.
- **Deferred to the later review of all arms:** self-play training both sides against one side; the separate critic (too much work for an arm now).
- **Frontline break-even (decided 2026-11-15):** wait for the map-size arm (A1). Until then frontline break-even is not scored. *Replaced the same day (session 3): scored regardless of whether maps can change size; `--unscored` is empty by default.*
- **Per-update policy files (decided 2026-11-15):** off by default; on only for the one arm that feeds the averaging sweep. Done in the commit after this one.
- **Fixed-size maps (author, 2026-11-15, firm):** every map is fixed-size (each preset's own arena at every team size) until an agent trains successfully on variable maps (the map-size arm A1); if that fails, fixed-size full stop. Training's map-size sampling and the evaluation's density rule become opt-in for A1 (in progress, `feat/fixed-maps`).
- **Freeze suggestions (author, 2026-11-15; suggestions, not locked in):** frontline ±2 with a 10-minute clock; deathmatch 40 tickets with a 10-minute clock; elimination: the collapsing map should decide games more often (not necessarily by killing directly), so study shrink start times and rates and their effect on mean and median game length (sweep running, `elim-shrink` unit).
- **Fixed maps merged (696a78e9):** training map-size sampling off by default (opt-in `arena_scale` for A1); evaluation and break-even on each preset's own map (`--map-rule density` opt-in); break-even counts up to 30; frontline break-even unscored (`--unscored`), so frontline's third is 5v5 and 10v10 at 1/6 each. The design pilot is being re-run on the preset maps (`evaluate-pilot-fixed`, after the elimination sweeps): its 1v1, 2v2, 10v10 and break-even variances and costs came from density maps.
- **Elimination shrink (author, 2026-11-15, session 2):**
  - The shrink starts at 0 s (no hold). Compare a few linear rates against a few fast-then-slow curves; constant area per second dropped (it spends even less of the shrink below 2.5 km than linear).
  - **The boundary never closes faster than a cap** below cruise speed: 150 m/s at real scale (about half of cruise; `ModeRules.shrink_max_speed`).
  - New laws built on `feat/imugi-boundary`: 2, the power law (1 − s)^p; 3, the corner (linear to a radius at a time, then linear past 0).
  - Track the losing team's deaths 1 to 5 (time and cause) in the bench.
- **Imugi's edge steer in the boundary's moving frame (author, 2026-11-15):** margin k·ρ·f(β), f(β) = √(1 − β²) + β·arccos(−β), β = edge speed / ship speed (exact for a straight edge closing steadily; equal to today's rule for a still edge); outward measured relative to the edge; the edge speed from the last two observations (Imugi's carry), extrapolated as linear. Below 2–4 own turn radii of boundary the steer fades out and the ship fights.
- **Imugi without a known enemy (author, 2026-11-15):** drop "head for the centre" (a coupling: both sides agree to meet in the middle). Candidates being measured: fly straight (the edge turns it back), and a team search (the enemy's spawn, then random points inside the boundary, with cohesion to the team's centre of mass). Changing Imugi changes the BC teacher and the yardstick behind every `imugi`/`scripted` record, so it goes into the freeze's version bump, and every freeze measurement is re-run after it.
- **Imugi's search (author, 2026-11-15):** the explore search with the wide spacing is the default in deathmatch and elimination (`imugi_combat.SEARCH = "explore"`); the centre rule stays as an option. Elimination's spawn centres are drawn per team over a 120° sector of its side, so neither spawn gives the other's away. Measured in [sections/elimination-shrink/results.md](sections/elimination-shrink/results.md) (section 8).
- **Averaging tool merged** (`bnb average`, suite in `sections/averaging/suite.toml`, 14 averages; the run also saves its starting weights so the training exponential average is rebuilt exactly; `bnb runs ema` removed).
- **Maybe arms built before the reference commit (author, 2026-11-15, session 3):** every run in arm-list.md, Yes and Maybe, launches from configuration alone; arms needing code are built behind settings, default off. An audit of what's missing is running.
- **HL-Gauss everywhere (author, 2026-11-15, session 3):** every binned target and input uses HL-Gauss with the paper's default (σ = 0.75 bin widths) unless there is a specific reason not to; above all the critic. Applied: the critic (with 2 padding bins per side) on `feat/split-critic`; the ellipse features, the shot head's speed and index targets, and the state code's linear parts (velocity, nose rate, the scalars; σ = √(0.75² + spread²), decoding removes the floor) on `feat/hl-gauss-codes`. **Kept, with reasons:** the circular colour codes (position, heading, shot displacement and direction: 3–4 bins per level, where 0.75 bins would erase each level's contrast) and the gun cooldown (one bin per tick, a sharp base plus the belief's residual).
- **A18 (position encoding): no** (author, 2026-11-15, session 3).
- **Arm choices (author, 2026-11-15, session 3), in [arm-list.md](arm-list.md):** Yes: R0, R5, A1, A2, A7, A14, A13, A15, A16, A17, A24, R1. Maybe: A4, A5, A6, A8, A9, A10, A19, A22, A23, R3, R4. No: A11, A12, A20, A21, R2, R6. Second seeds for R0 and for the first arm that trains (A1 and A2 may fail). A13 follows A14: if the belief helps, it keeps the next-state head and removes the other auxiliary heads; otherwise it removes all. The GRU for A16 is being built (`feat/gru-temporal`).
- **RL budget per arm (author, 2026-11-15, session 3): 100M**, the same for every pairwise decision (BC stays 100M). From rl-v20's curve: the mean Elo gains about 94% of its 160M gain by 100M; the reward-rebalance round robin crossed at about 70M. Arms whose question is about steps keep their own budgets (A24 from scratch at 2–3×, A23 at equal steps and equal compute; R2 and the publication run are references). To confirm from the dry run's version-21 curve. **Proposed, not decided:** extend both arms (e.g. to 150M) when the loser at 100M is climbing significantly faster in its last 5 snapshots, not only on ties.
- **Cost rate:** $0.50 per 5090 GPU-hour (the author's rough estimate).
- **Old runs (author, 2026-11-15, session 3):** no old run needs to resume; old league states and records may break.
- **Averaging variants (author, 2026-11-15, session 3):** decided after training: a large suite of averages is built from the saved per-update weights and tested then. Nothing to choose before the run.
- **Freeze leftovers closed (author, 2026-11-15, session 3):**
  - **Bullet drag (idea 75):** no change. The real-scale set is in real metres and 5e-4 per metre is about a .50 cal round's; bullets stay above Mach 2 within the 600 m gun range, so no transonic rise.
  - **Flight:** signed off ("the ships feel good").
  - **Cards:** final. To check: a frontal hit that deals no damage still restarts the shield's recharge wait.
  - **vizia** is dropped from the real-scale set.
  - **To do (subagent, `check/freeze-final`):** the shield check; whether deathmatch's respawn follows `tune/respawn-waves` (8 s); a final good-enough check of the modes and the Yongma–Jujak balance with Imugi on both sides; a data report, then a review report for the author.
- **Game fingerprint (author, 2026-11-15, session 3):** the hashed fingerprint goes (it depended on the machine: the cloud container and the laptop disagree at the same commit). It is replaced by a committed reference of each preset's short match, compared with a float tolerance (integers and flags exactly). `VERSION` stays the game's identity. Done on branch `feat/game-reference`: regenerate the reference after a version bump with `uv run --no-sync python -m tests.sections.test_game_version --write`; it was generated on the laptop and not yet checked on another machine.
- **Freeze measurements: Imugi only** (author, 2026-11-15, session 3); no trained-agent runs of the frozen modes.
- **Break-even merged into `feat/imugi-boundary`** (99baa02b): no floor (down to 1 ship; a result at 2 or fewer against 20 is flagged as suspicious), the ratio reported as Imugi's ships per one of ours, lineups per game, a first batch over 13 ratios from 1v20 to 20v1. A fix rode along: the "double the ratio" step could loop on a mixed lineup.
- **League snapshot schedule (author, 2026-11-15, session 3):** every 5M up to 50M, then the interval doubles each time the steps double (10M to 100M, 20M to 200M, 40M to 400M, 80M after): 5, 10, …, 50, 60, …, 100, 120, …, 200, 240, …, 400, 480M. Replaces the rl-v20-derived list. Evidence in [sections/league-schedule/results.md](sections/league-schedule/results.md). Merged into `feat/imugi-boundary` with the pruning removal.
- **League retention (author, 2026-11-15, session 3):** keep every league member; nothing is pruned. The age rule (`bnb.league.pool.retained`) goes. **Implemented and merged** with the schedule (`feat/league-schedule`): `SnapshotSchedule(every=5M, doubling_from=50M)`; the timed-cadence fallback and entry ordinals are removed too, so a league state saved before this change does not resume.
- **Break-even: the weaker side is fixed at 20 and the stronger side's count is varied downward (author, 2026-11-15, session 3).** This replaces "our count up to 30 against 20 Imugi", and the 30 cap goes. The fit is logistic in log(our ships / Imugi ships), one continuous ratio through 20v20; which side shrinks is read from the data (20v20 blocks first). The smaller side's floor is the mode's minimum team (Frontline 3). The score stays A's win rate in B's break-even lineup. No game is larger than 20v20. Branch `feat/break-even-weaker-twenty` (subagent).
- **Game freeze, locked in (author, 2026-11-15, session 3; expected numbers in [sections/freeze-proposals.md](sections/freeze-proposals.md)):**
  - **Deathmatch:** a 10-minute clock. Tickets per team are 4·N^(4/3) rounded to the nearest 5 at every team size, 1v1 included: 5, 10, 15, 25, 35, 85 and 215 for N = 1, 2, 3, 4, 5, 10 and 20. They replace tickets per ship.
  - **Elimination:** the boundary shrinks linearly from 0 s (no hold) to zero over 4 minutes, with a 248 s clock.
  - **Frontline:** ±2 with a 10-minute clock, 3v3 at the smallest, and the lock kept at 6 s/√km (about 14 s at 5v5 on the real-scale map). Capture stays at 20 s for a one-ship lead (not discussed). No centre zone and no healing on capture for now (both stay as ideas).
- **Dropped:** ego-centric frame (×N compute); rotation and reflection augmentation (more environments do the same); KL-to-BC penalty (a single scripted teacher has no diversity to keep); greedy evaluation (evaluation samples like training); belief decay constants (41) and the delta/absolute head (40), using the earlier v2 results.

## In progress

| Work | Branch | State |
|---|---|---|
| Seed isolation | `feat/seed-isolation` (a13173da) | Done on the CPU. Waits for the GPU speed check, the GPU tests and the ONNX export test. |
| Rating allocator, shared pool, placement-only prior | `feat/rating-allocator` (6538fa81, merged) | Merged. The layout probe has been running on the laptop GPU since 07:01 (`rating-layout-probe` unit, performance profile). |
| Cheese agents | `feat/cheese-agents` (30a5dd66, from seed isolation) | Done on the CPU. Waits for the first exploit check on the GPU. |
| Break-even ratio | `feat/break-even` (66fd2ba0, from seed isolation) | Done on the CPU: a logistic fit in log count; blocks go to the count that most reduces the variance of log N*; stops at SE ≤ 0.3 ships, 8192 games or a bound; per-batch records with resume; mirrored coin-flip lineups (`Arena(game_lineups=True)`). A benched slot costs about as much as an active one, so every break-even game costs a full 20v20. Waits for the first GPU measurement. |
| References list | docs only (dab3a8de) | 26 entries, about 35 papers. Only 5 are verified online so far; the rest are marked "(unverified)". Update the code locations of the loop check and the allocator after the rating branch merges. |
| Absent fields | `feat/absent-fields` (7f588608, from seed isolation) | Done on the CPU. Field counts per episode keep the map point-symmetric; absent fields have exactly no physical effect; `Yemong.mask_absent_fields` masks them; 16 new tests pass. Game version raised to 20. Waits for the fused-kernel test with the longer key mask on the GPU. |
| Seed twin of rl-v20 (`rl-v20-seed1`, seed 1) | laptop GPU, old code | Finished 02:11 at 149.9M steps. **Head-to-head against rl-v20** (512 games per mode, old code, win/loss): the twin is consistently weaker. Scores by mode: 50M 0.34 / 0.50 / 0.45, 100M 0.29 / 0.40 / 0.33, 150M 0.41 / 0.39 / 0.32 (frontline / elimination / deathmatch), roughly −70 to −150 Elo. Not yet clean seed noise: the twin also differs in micro-batch rows (96 against 480), host store, GPU (4070 against 5090) and commit, and its policy drift ran lower (0.16–0.22 against 0.33–0.41). **Micro-batch check (2026-11-15, `check/micro-batch` ae5e381c, merged):**
  - **The update is micro-batch independent** within rounding: gradient |Δg|/|g| 6e-7 in float32, 3e-3 in bf16, against 1.4e-2 for bf16 against float32.
  - **The drift metric wasn't.** It read only the first micro-batch, so 48 views at 480 rows against 8 at 96. Fixed: a fixed 48-view sample.
  - **The code between the two runs' commits** (5b85b6c2..df8294f3) adds only default-off options and the fused key-mask kernel.
  - **So the 70–150 Elo gap is most likely seed noise,** with GPU numerics a minor factor. Seeds matter about as much as on the 3090 (about 110 Elo), and pairwise arms must share seeds. |
| Seed-streams speed check | laptop GPU (`seed-speed` unit) | Running since 02:21: 4 updates from bc-v20 on the old tree, then the new. |

**Merged 2026-11-15 (52728ca1):**
- **Branches:** seed isolation (plus a compile fix, 5589e850: stream keys are compile-time constants), cheese agents, break-even, absent fields, rating allocator, frontline lock, mode timing.
- **Game version:** VERSION 20, FINGERPRINT 477cbfae. Continuing an old version's tournament record is now refused, and crossover records include the version.
- **Tests:** 866 CPU and 61 GPU tests pass.
- **Seed-streams speed check passed:** steady rollout 23.1 s on both trees, and learning equal when the CPU is uncontended (238 s).

**GPU queue (old, before the merge):**
1. The seed-streams steps-per-second check, old against new. Merge only if throughput holds.
2. The GPU tests, the ONNX export test, and the fused attention kernel with absent fields' longer key mask.
3. The rating layout probe.
4. Merge seed isolation, then the cheese agents, the rating allocator, break-even and absent fields.
   - **One game version at the merge.** The seed-isolation branch changed the game's draws without updating the recorded game fingerprint (it records e16ea10c but computes 7f35d79c). The absent-fields branch raised the version to 20. Settle the fingerprint once at the merge; the freeze will bump the version again.
   - Fix the failing docstring test on `bench/attention_shares.py`.
5. The first exploit check and break-even measurement of rl-v20's final checkpoint.
6. Rate the twin against rl-v20 at matching steps (the noise floor).

**First exploit check (2026-11-15, laptop GPU; rl-v20's final 171M exponential average against the six cheese agents, 128 games per mode, deathmatch by win/loss):**
- Wins nearly everything: every frontline and deathmatch game against turtle, evader, deathball, jouster and kiter; elimination 0.93 to 1.00 (evader 0.93, the lowest).
- **Flag: the zone rusher in frontline, 0.55 ± 0.04** (67-6-55; 0.48 at 75M). Imugi beats the same rusher 0.75 ± 0.04, and rl-v20 beats Imugi in every 5v5 frontline game. So it's a cycle and a blind spot of the policy, not a degenerate mode. rl-v20 trained without cheese agents in the league; the first arm (cheese on by default) should close it. Check it again on the first arm's final checkpoint.

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

- [x] **Mode timing (`bench/mode-timing` eecbacce; CPU, Imugi on both sides, fields on):**
  - **Deathmatch:** 4.57 deaths per team per minute (sd 1.08), steady after the first minute. Tickets per team for 80% of games ending by tickets: 40 at 10 min, 47 at 12 min, 55 at 15 min (±5). Today's 25 end 99% of games within 10 min.
  - **Elimination, time to wipe** (no shrink): p50 52 s, p80 90 s, p90 120 s. Today's shrink (hold 36 s, clock 180 s) causes only 2.4% of deaths.
  - **Frontline 2v2:** captures are rare (1.9 per game) and 85% of games end on the clock. 3v3 ends by capture 52% of the time. Both measured at the old lock value of 19.
- [x] **Frontline lock re-tuned:** 6 s per √km, game v20.
- [ ] **Freeze values (proposal, 2026-11-15; measured with Imugi, fields on, lock 6, 256 games each with a 30-minute clock):**
  - **Frontline:** win at ±2 (was ±3) with a 12-minute clock.
    - At 5v5, ±2 ends 79% by capture within 10 min and 84% within 12 (median 5.0 min). ±3 ends only 53% within 12 min (median 10.7).
    - Smaller teams are slower: at ±2, 3v3 ends 68% within 12 min, 2v2 36%.
    - Open: frontline's minimum team size of 3v3; whether its clock scales with fleet size.
  - **Deathmatch:** 12-minute clock with 50 tickets per team (about 75–80% end by tickets; Halo-like).
  - **Elimination:** keep the 3-minute clock and start the shrink at about 90 s (the 80th percentile of time to wipe) instead of 36 s.
- [x] **The evaluation-suite command, then the sequential-test pilot** (rl-v20 against rl-mix-kl and against rl-dm at 75M, every condition) and the design simulation. Merged 2026-11-15 (a7f647cc); the results are in [evaluation-suite.md](evaluation-suite.md#design-pilot) (75c215a7).
  - **Design:** K = 4, δ = 0.02 (about 14 Elo), α 0.05, power 0.8. Simulated false-winner rate 3.8%, power 0.81. At most about 0.6–0.85 laptop engine-hours per decision.
  - **Seed noise, not games, limits a decision.** δ will be set from arm A3.
  - **Frontline break-even has no fit:** both arms lose every game at every count, including 20 against 20. That's the 15.6 km map, not the code: rl-v20 wins every 5v5 and 10v10 frontline game against Imugi on the preset map, 26 of 64 at 20v20 on the preset map, and 0 of 64 at 20v20 on 15.6 km. Open for the author.
- [x] The split offence/defence critic, before the shared BC, with advantage weights per part and the premium schedule as a setting (default 1:1). Built on `feat/split-critic` (2026-11-15, not merged), with HL-Gauss value targets; notes in [sections/split-critic/progress.md](sections/split-critic/progress.md).
- [ ] **Front reversal (decided 2026-11-14).** Measure P(reversal) now, on the current game with Imugi on both sides (in progress, `bench/front-reversal`). Measure it again at the end of the first arm, to check the front stays roughly a random walk in self-play. The 0.84 figure is from September (v1); the later lock was meant to make it a random walk.
  - **First result (2026-11-15, CPU, scripted against scripted, fields off; `bench/front-reversal` b18f8cce):** the front is **persistent, not a random walk.**
    - P(reversal) = 0.19 [0.14, 0.25], from 208 pairs of consecutive moves in 64 games.
    - 66% of games are 3-0 sweeps (25% expected for a fair coin).
    - 86% of games end by capture and 14% on the clock (mean 296 s of 600 s).
    - This contradicts the docs' claim that the lock was tuned so the capturing team is no more likely to capture next.
    - **Cause (diagnosed):** the 45 s lock outlasts the capturing team's run to the next zone. The same team's next capture comes a median 36 s later, a reversal 91 s later.
  - **Re-tuned (2026-11-15, game v20, `game/frontline-lock` e5bed5a1):** `lock_seconds_per_root_km` 19 → 6 (about 14 s at 5v5).
    - With fields on and Imugi on both sides: P(reversal) = 0.490 [0.468, 0.512] over 512 games (19 gives 0.29).
    - Still 0.435 at front ±2, so a little momentum near the win.
    - **56% of games end on the 600 s clock** (mean 484 s); for the clock tuning.
  - Re-tuning the lock is a game-freeze item.
- [x] Snapshot schedules (32d1622a): RL at 1–20M then every 5M; league members 5…160M then every 30M.
- [x] Deathmatch outcome `kill_lead` as the default (d93e8498; the rating score keeps its recorded name `win_loss`).
- [x] Mirrored 50/50 coin-flip lineups in training (10856169, `TrainConfig.lineup_draw`, on by default).
- [x] Mode mix 6/5/5 (fa7bf3ba).
- [x] League: (1−p)² over every member, no floor (3df94c64; p(1−p) with a floor stays as an ablation). Merged with the items above in d81c133e.
- [ ] Saving the live weights every update (for the early arm), and the post-hoc averaging tool.
- [x] The Imugi rename (575546e0): `imugi`, `imugi@T=..` and `cheese:<name>` on the command line; recorded identities stay `scripted…`, so old records still read. Old play flags `--scripted0/1` keep their names.
- [x] BC fixed at 100M, plateau stop logging only (85ffcd72). A chain's BC stage takes min(combined budget, 100M).
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

- **Training mix:** frontline/deathmatch/elimination 6/5/5 against 8/4/4, as RL-only forks from the shared BC. Deathmatch and elimination share skill (rl-dm gained more elimination than the generalist), and frontline is the distinct skill still climbing.

- **Offensive premium (the author, 2026-11-14; split critic decided for it):** three RL forks from one split-critic BC: zero-sum 1:1 throughout, 2:1 throughout, and 2:1 annealed slowly to 1:1. The author's hypothesis: passive play is an equilibrium of zero-sum self-play when aggression is risky and the winning aggressive line is hard to find; a premium on offence makes searching for it worthwhile, and once found, passivity stops being safe. Predictions to check: fewer draws, shorter lifespans and more engagements early in the premium arms; the deciding measure stays win/loss Elo.
- **Original note:** offence weighted 2:1 over defence, against zero-sum 1:1, or annealed slowly from 2:1 to 1:1. The author recalls an earlier test (v1 or v2.1) where it trained to a clearly higher Elo; v2 dropped the premium on 2026-10-30 so the critic's targets never move. The proposed infrastructure is a split critic: each reward level gets an offence value and a defence value (10 outputs instead of 5), and the weights combine their advantages. Changing or annealing the ratio then never moves a critic target, so one BC run serves every ratio and the arms are cheap RL forks. The split would go in before the shared BC, on by default at 1:1.

- Self-play: train both sides, or one side with more environments and "latest" as a league member at a fixed share.
- Clipping against DPPO (before the learning-rate arm).
- The rest of the numbered list in [ideas.md](ideas.md), with the comments from 2026-11-14.
