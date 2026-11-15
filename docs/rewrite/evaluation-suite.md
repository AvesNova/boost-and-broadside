# Evaluation suite

These are the games that decide each pairwise arm, and the Elo ladder we build for the paper. `bnb evaluate` (`bnb.league.evaluation`) plays the decision part; this page is its reference. Items marked **(open)** are still to be decided; **(provisional)** marks a working choice until a test settles it. The decisions behind it are in [arms-todo.md](arms-todo.md), and the command's build and pilot notes in [sections/evaluation-suite/progress.md](sections/evaluation-suite/progress.md).

**Decisions use win probability, never Elo.** Every condition is scored by how often one arm's final checkpoint wins. Elo appears only in the ladder (the curves), which is rated separately and doesn't decide anything.

## Players and play

- **Decision players:** each arm's final checkpoint (its exponential-average weights, the ones we ship).
- **Play:** actions are sampled exactly as in training (temperature 1, never greedy), with fog of war on in every game.
- **Lineups:** each slot is a 50/50 coin flip between Yongma and Jujak (`--lineup-types`), the same for both teams, drawn from the keyed stream `evaluation_lineup/<condition>` by the seed and the batch, so both arms see the same lineups.
- **Seats** alternate from game to game within each block.
- **Games are keyed by batch number,** and each condition's numbers continue past its recorded batches, so a resumed or parallel process never replays a game.

## Conditions

| # | Name | Mode | Teams | Map |
|---|---|---|---|---|
| 1 | `frontline_5v5` | frontline | 5v5 | preset |
| 2 | `deathmatch_5v5` | deathmatch | 5v5 | preset |
| 3 | `elimination_5v5` | elimination | 5v5 | preset |
| 4 | `deathmatch_1v1` | deathmatch | 1v1 | preset |
| 5 | `deathmatch_2v2` | deathmatch | 2v2 | preset |
| 6 | `elimination_1v1` | elimination | 1v1 | preset |
| 7 | `elimination_2v2` | elimination | 2v2 | preset |
| 8 | `frontline_10v10` | frontline | 10v10 | preset |
| 9 | `deathmatch_10v10` | deathmatch | 10v10 | preset |
| 10 | `elimination_10v10` | elimination | 10v10 | preset |
| 11 | `frontline_break_even` | frontline | break-even | preset |
| 12 | `deathmatch_break_even` | deathmatch | break-even | preset |
| 13 | `elimination_break_even` | elimination | break-even | preset |
| 14–19 | — | all | 5v5 at map scales 0.5× and 2× | scaled (**only if variable maps survive their arm**; not in the command) |

- **Games:** conditions 1–10 are A against B. The base games are the real-scale presets (`--presets`), and Imugi is Imugi (`--opponent imugi`).
- **Map rule:** every condition plays its preset's own arena radius at every team size, 1v1 to break-even's (`--map-rule preset`, the default). This is the author's rule of 2026-11-15: every map is fixed-size until an agent has trained successfully on variable-size maps (the map-size arm, A1). The density rule (`--map-rule density`: `api.arena_radius_for`, the radius scaling with the fleet down to each mode's minimum; break-even at the opponent's 20) is an opt-in setting for that arm. Both arms of a pair always use the same rule.
- **Every break-even is scored, frontline's included,** whether or not maps can change size (the author's decision, 2026-11-15, replacing the earlier wait for the map-size arm). `--unscored` names conditions to leave out (none by default); a left-out condition is neither played nor waited on, and its mode's weight is shared among the rest.
- **Break-even** reuses `bnb crossover`'s fit and allocation (`bnb.league.crossover`). A lineup is a pair (the arm's ships, Imugi's ships): the weaker side fields 20 and the stronger side's count varies downward, to the mode's smallest team at least (Frontline 3), so no side ever fields more than 20. Which side is the weaker comes from each arm's own games, starting at 20 against 20. One engine plays both arms at 20 a side: half of each batch's blocks are A against Imugi, half B, each at the lineups its own fit asks for, on the same lineups of ship types and maps.

## Score

1. **Per condition, a win probability minus 0.5:**
   - **Head-to-head conditions (1–10):** A's mean score against B (a win is 1, a draw ½), minus 0.5. It's positive when A is better. Its variance is the games' sample variance over their number.
   - **Break-even (11–13):** each arm's fit is a logistic curve of its win probability on the log ratio r = log(its ships / Imugi's ships), one axis across both sides of parity (20 against 20 is r = 0). Its break-even lineup is where the curve crosses one half. A's score is its win probability against Imugi in B's break-even lineup, minus 0.5, and symmetrically for B: ½ (p_A(r*_B) − 0.5) − ½ (p_B(r*_A) − 0.5). The curve depends on a lineup only through r, so the score is defined whether the two arms are on the same side of parity or on opposite sides (A at 15 against 20, B at 20 against 16, say). Its variance comes from the two fits' covariances by the delta method, with the fits taken as independent.
2. **Per mode:** the mean over that mode's conditions, each weighted equally.
   - frontline: 1 and 8 (and 11 once it is scored again);
   - deathmatch: 2, 4, 5, 9 and 12;
   - elimination: 3, 6, 7, 10 and 13.
3. **Overall:** the mean of the three modes, each weighted equally (thirds). This is one number with one standard error: the weighted sum of the conditions' variances.

**The weights stay fixed** whatever the games. Each condition's estimate is unbiased however many games it has, so allocating games by cost moves no weight. A bias would appear only if games were pooled with weights proportional to their counts, which the score never does. A test checks it by simulation (`test_the_cost_aware_allocation_leaves_the_score_unbiased`).

## Decision

- **A group-sequential test on the overall score** (`bnb.league.sequential`). At a look, Z = score / SE, and the information fraction is t = SE⁻² / I_max.
- **Boundaries.** A wins at Z ≥ u_k, B wins at Z ≤ −u_k, and the test stops with a tie at |Z| ≤ l_k. Otherwise it continues.
  - u_k spends a two-sided α (`--alpha`, 0.05) by an O'Brien-Fleming-type function (Lan and DeMets), at the looks' actual fractions.
  - l_k spends β (`--beta`, 0.2) the same way under a true difference of ±δ (`--delta`, the indifference zone).
  - The last look always decides (l = u). It comes at t ≥ 1, or once the budget is spent (`--budget-hours` of measured engine time, every process's).
- **I_max** is the information at which the planned looks (`--looks` K, at t = k / K) spend α and β exactly: I_max = (η_δ / δ)², with η_δ found by bisection.
- **Outcomes:** A wins, B wins, or a tie.
- **No-regression gate:** a mode where a player's score is at least `--gate-stderrs` (2) standard errors below zero. A winner caught by the gate is held to a tie (a 2-against-1 split goes to the tie rule; if it's still split, the incumbent stays).
- **Tie rule:** rate each arm's last **5** snapshots on demand, against each other and the two finals (about the last 20–25M steps). If the slopes differ by more than about 2 SE, extend both arms once, by 10–20M steps, and run the test again. Otherwise choose the cheaper or simpler arm. (Not in the command.)
- **Settings (open until the pilot below is reviewed):** K, α, β, δ and the budget.

## Allocation

- **First,** every scored condition plays `--initial-batches` (1). A scored break-even condition keeps playing until both arms' fits cross, at most `--initial-batch-limit` (8) batches, after which the suite stops with the reason.
- **Toward each planned look:** the next look needs the overall variance 1 / (t_next · I_max). Batches are added one at a time where the next batch most reduces the overall variance per engine second, until the variance is reached.
  - **Per-game variance:** from the condition's games. A head-to-head condition's is shrunk toward ¼ by one pseudo-game, for the allocation only.
  - **Cost:** the measured engine seconds per game. An engine's first batch, which compiles, is left out where there are others.
- **The targets are recomputed** from the pooled records before every batch. The look is taken when no condition is short.

## Records and processes

One directory per comparison:

- `settings.json`{.plain}: the players (name, path and identity, the SHA-256 of the weights) and the settings. Written by the first process, and checked by every later one (another policy, game version or setting is refused).
- `batches/<writer>.jsonl`{.plain}: every batch, with its condition, number, seconds, games, whether its engine compiled in it, and each block's full outcome (wins, draws, losses; break-even's also the arm and its lineup, both sides' counts).
- `looks/<writer>.jsonl`{.plain}: every look, with its fraction, score, SE, Z, boundaries, the test's decision, the gate and the decision after it.

A writer is one process (host, process id and start time), and only it appends to its files, as in the shared rating pool. Readers merge every file, keeping the first of a repeated (condition, batch) or look number. So each condition may run as its own process (`--play CONDITION`): a process plays its own conditions toward the look, waits (`--poll-seconds`) while another's are short, and whichever process finds the look's information reached takes the look.

## Command

<!-- not-runnable: needs CUDA and two policies -->
```bash
# one process for the whole suite
uv run --no-sync bnb evaluate A_POLICY B_POLICY --out evaluations/a-against-b
# or one process per condition, sharing the directory
uv run --no-sync bnb evaluate A_POLICY B_POLICY --out evaluations/a-against-b --play deathmatch_1v1
# the scores and the test at a look now (--peek: without recording the look)
uv run --no-sync bnb evaluate decide evaluations/a-against-b --peek
# a pilot: every condition to 512 games, no looks; then the design from pilots
uv run --no-sync bnb evaluate A_POLICY B_POLICY --out pilot/close --games 512
uv run --no-sync bnb evaluate design pilot/close pilot/far --looks 2 3 4 5 --simulations 400
```

Each batch prints a line with the overall score so far and the next look's ETA as a clock time (and the budget's end). `decide` prints the per-condition table (score, SE, games, per-game variance, seconds per game), the per-mode and overall scores with their SEs, and the look.

## Design pilot

rl-v20 against rl-mix-kl (a close pair, 1024 games per head-to-head condition) and against rl-dm (a far pair, 512), every arm's 75M exponential average, on the laptop's RTX 4070 in the performance profile, 2026-11-15. The pilot played the density rule's maps at every size but 5v5, before the fixed-map rule; its variances and costs at those sizes stand until a pilot on the preset maps replaces them.

**Per condition (close pair):**

| Condition | Score (A − ½) | SE | Variance per game | Seconds per game |
|---|---|---|---|---|
| frontline 5v5 | +0.046 | 0.015 | 0.228 | 0.66 |
| deathmatch 5v5 | +0.098 | 0.015 | 0.241 | 0.48 |
| elimination 5v5 | +0.024 | 0.016 | 0.249 | 0.14 |
| deathmatch 1v1 | +0.219 | 0.013 | 0.159 | 0.45 |
| deathmatch 2v2 | +0.028 | 0.016 | 0.247 | 0.50 |
| elimination 1v1 | +0.013 | 0.016 | 0.250 | 0.07 |
| elimination 2v2 | +0.014 | 0.016 | 0.250 | 0.10 |
| frontline 10v10 | −0.149 | 0.014 | 0.188 | 0.91 |
| deathmatch 10v10 | +0.049 | 0.016 | 0.248 | 0.63 |
| elimination 10v10 | −0.007 | 0.016 | 0.250 | 0.23 |
| frontline break-even | no fit | | | 0.53 |
| deathmatch break-even | +0.316 | 0.047 | 1.15 | 0.50 |
| elimination break-even | +0.300 | 0.042 | 0.90 | 0.17 |

**The design** (the close pair's variances; frontline break-even, scored at the time, stood in at the mean of the other two break-even variances): δ = 0.02 (52% against 50%, about 14 Elo), α = 0.05, power 0.8.

- **Cost:** at most about 0.6 engine-hours on the laptop for any K from 2 to 5. The first batch of every condition costs about 0.7 hours by itself, so the expected cost is 0.7 to 0.85 hours whatever the difference.
- **Simulated** (400 repeats of the whole procedure, with estimated variances, batch-sized steps and the cost-aware allocation):
  - K = 3: a false winner in 2.8% of repeats, and power 0.80 at δ;
  - K = 4: a false winner in 3.8%, and power 0.81.
  - The estimate at the stop runs about 5% high (0.0416 to 0.0421 for a true 0.04). That's the usual bias of stopping early, not the allocation's.
- **K:** 4, the default. It costs nothing over 3 here and stops earlier on large differences.
- **Seed noise limits a decision, not games.** At δ = 0.02 the test resolves about 14 Elo, while two seeds of one configuration differ by about 110 Elo. δ will be set from the seed pairs (arm A3); the game budget is not the constraint.

**Frontline break-even has no fit.** Both arms lost all 32 games at every count from 6 to 20, including 20 against 20. The cause is the map, not the code. Head-to-head against Imugi in frontline, rl-v20 won:

- 5v5 and 10v10 on the preset map (7.8 km): every game;
- 10v10 on the density map (11 km): 58 of 64;
- 14v14 on 13 km: 15 of 64, though it outkilled Imugi (61 kills against 52 per game);
- 20v20 on the preset map: 26 of 64, with 14 draws;
- 20v20 on 15.6 km: none of 64, with about 8 kills against 27 deaths per game, and every game over in about 2 minutes.

The pilot's break-even played on the 20-a-side density map. For an agent trained on one map size, frontline break-even there measured transfer to a map twice the size, not skill. Break-even now plays the preset map (the map rule above), and frontline break-even is left out of the score until the map-size arm reports.

**Break-even's lineups.** The weaker side is fixed at 20 and the stronger side's count varies downward (the author's decision, 2026-11-15), so an arm weaker than Imugi at parity crosses at Imugi's count below 20 rather than at ours above it (in the pilot, at 20 at most, rl-dm in deathmatch won 5 of 32 at 20 against 20; in frontline on the preset map rl-v20 is about even with 20 Imugi at 20 ships, 26 wins and 14 draws in 64). Every break-even game is 20 a side, whichever side's spare slots are benched. If an arm's fit has no crossing after `initial_batch_limit` batches, the run stops with an error.

## The ladder (for the paper, not for decisions)

- **What:** Elo curves for every arm, in the shared pool, at standard 5v5 per mode. Every saved snapshot is rated, to about ±40–50 Elo.
- **When:** mostly in one batch at the end, when the allocator can plan over every checkpoint at once. Earlier only on an idle GPU, and only when it doesn't compete with compilation on the CPU.
- **Curves at the other conditions:** for the arms, curves at 5v5 only. For the first proper run (the baseline) and the final winner, the full ladder in every condition (decided 2026-11-15).

## Reported but not scored

- **Exploit check:** each final checkpoint against the six cheese agents in every mode (zone rusher, turtle, evader, deathball, jouster, kiter). A loss to one is flagged for the author.
- **Loop check:** goodness of fit, and a likelihood-ratio test against mElo, in the ladder's pools.
- **Seat symmetry:** the mirror match, a checkpoint against itself, scored by seat.
- **Front reversal:** P(reversal) in frontline, after the first arm, to check the front stays roughly a random walk.
