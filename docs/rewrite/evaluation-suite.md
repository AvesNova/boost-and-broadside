# Evaluation suite (draft for review)

These are the games that decide each pairwise arm, and the Elo ladder we build for the paper. It is a draft dated 2026-11-15, for the author to review. Items marked **(open)** are still to be decided; **(provisional)** marks a working choice until a test settles it. The decisions behind it are in [arms-todo.md](arms-todo.md).

**Decisions use win probability, never Elo.** Every condition is scored by how often one arm's final checkpoint wins. Elo appears only in the ladder (the curves), which is rated separately and doesn't decide anything.

## Players and play

- **Decision players:** each arm's final checkpoint (its exponential-average weights, the ones we ship).
- **Play:** actions are sampled exactly as in training (temperature 1, never greedy), with fog of war on in every game.
- **Lineups:** each slot is a 50/50 coin flip between Yongma and Jujak, the same for both teams, drawn from the keyed stream so both arms see the same lineups.
- **Seats** alternate evenly within each block of games.

## Conditions

| # | Mode | Teams | Map | Games |
|---|---|---|---|---|
| 1 | frontline | 5v5 | preset | A final against B final |
| 2 | deathmatch | 5v5 | preset | A against B |
| 3 | elimination | 5v5 | preset | A against B |
| 4 | deathmatch | 1v1 | density rule **(provisional)** | A against B |
| 5 | deathmatch | 2v2 | density rule **(provisional)** | A against B |
| 6 | elimination | 1v1 | density rule **(provisional)** | A against B |
| 7 | elimination | 2v2 | density rule **(provisional)** | A against B |
| 8 | frontline | 10v10 | density rule **(provisional)** | A against B |
| 9 | deathmatch | 10v10 | density rule **(provisional)** | A against B |
| 10 | elimination | 10v10 | density rule **(provisional)** | A against B |
| 11 | frontline | break-even: each arm against 20 Imugi | density rule at 20 **(provisional)** | each arm against Imugi, our count varied |
| 12 | deathmatch | break-even | density rule at 20 **(provisional)** | each arm against Imugi |
| 13 | elimination | break-even | density rule at 20 **(provisional)** | each arm against Imugi |
| 14–19 | all | 5v5 at map scales 0.5× and 2× | scaled | A against B (**only if variable maps survive their arm**) |

**Map rule (provisional):** every size other than 5v5 uses the game's density rule (`api.arena_radius_for`). The radius scales with the fleet, down to each mode's minimum. The final choice between this and the preset's 5v5 map is deferred until the map-size arm shows whether the agent trains on varied maps. Both arms of a pair always use the same rule.

## Score

1. **Per condition, a win probability minus 0.5:**
   - **Head-to-head conditions (1–10, 14–19):** A's score against B (a win is 1, a draw ½), minus 0.5. It's positive when A is better.
   - **Break-even (11–13):** each arm's win probability against 20 Imugi when fielding the other arm's break-even count N*, from each arm's fitted curve. Symmetrised: ½ (A's win rate at N*_B − 0.5) − ½ (B's win rate at N*_A − 0.5).
2. **Per mode:** the mean over that mode's conditions, each weighted equally.
   - frontline: 1, 8 and 11 (plus 14–19 if they apply);
   - deathmatch: 2, 4, 5, 9 and 12;
   - elimination: 3, 6, 7, 10 and 13.
3. **Overall:** the mean of the three modes, each weighted equally. This is one number with one interval.

## Decision

- **A group-sequential test on the overall score,** with a fixed number of planned looks and acceptance thresholds set in advance **(open: the number of looks, the thresholds, the indifference zone, the maximum budget)**.
- **Each round** adds games to the conditions where the overall score's uncertainty comes from, per unit of cost. A 1v1 game is cheap; a 20-a-side break-even game is not.
- **Each condition runs as its own steady stream,** a separate process writing to the shared records, and the test reads the pooled records at each look.
- **Outcomes:** A wins, B wins, or a tie (the interval falls inside the indifference zone).
- **Tie rule:** rate each arm's last **5** snapshots on demand, against each other and the two finals (about the last 20–25M steps). If the slopes differ by more than about 2 SE, extend both arms once, by 10–20M steps, and run the test again. Otherwise choose the cheaper or simpler arm.
- **No-regression gate (open):** whether a winner may be clearly worse than the loser in any single mode.

## The ladder (for the paper, not for decisions)

- **What:** Elo curves for every arm, in the shared pool, at standard 5v5 per mode. Every saved snapshot is rated, to about ±40–50 Elo.
- **When:** mostly in one batch at the end, when the allocator can plan over every checkpoint at once. Earlier only on an idle GPU, and only when it doesn't compete with compilation on the CPU.
- **Curves at the other conditions (open):** whether the paper also wants curves at 1v1, 2v2 and 10v10, or only the finals there.

## Reported but not scored

- **Exploit check:** each final checkpoint against the six cheese agents in every mode (zone rusher, turtle, evader, deathball, jouster, kiter). A loss to one is flagged for the author.
- **Loop check:** goodness of fit, and a likelihood-ratio test against mElo, in the ladder's pools.
- **Seat symmetry:** the mirror match, a checkpoint against itself, scored by seat.
- **Front reversal:** P(reversal) in frontline, after the first arm, to check the front stays roughly a random walk.

## Cost (to measure)

- Games per second at each condition's size come from the rating layout probe on the GPU.
- Break-even games cost a full 20v20 each, because a benched slot costs about as much as an active one.
- **Break-even precision:** the SE of N* stops at 0.15 ships **(provisional)**, to be set from a GPU probe of what 20-a-side batches cost.
