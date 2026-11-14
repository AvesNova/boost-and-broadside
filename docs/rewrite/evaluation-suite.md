# Evaluation suite (draft for review)

This is the list of evaluation games used to pick the winner of each pairwise arm. It is a draft dated 2026-11-14, for the author to review. Items marked **(open)** are still to be decided; **(provisional)** marks a working choice until a test settles it. The decisions behind it are in [arms-todo.md](arms-todo.md).

## What is rated

- **Players.** Every RL snapshot of both arms (the dense ladder), plus the BC handoff checkpoint each arm starts from. Snapshots are at 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M steps, then every 5M.
  - The weights rated are the **exponential average**, the ones we ship.
  - The **final checkpoint** is the mean of the last 2–3 snapshots' ratings, confirmed by direct games.
- **Play.** Actions are sampled exactly as in training: temperature 1, never greedy.
- **Fog of war** is on in every game.
- **Lineups.** Each slot is a 50/50 coin flip between Yongma and Jujak, the same for both teams, drawn from the keyed stream so every arm sees the same lineups.
- **Seats** alternate evenly within each block of games.
- **Pools.** Each condition has its own shared rating pool, holding every arm's checkpoints. The allocator chooses the games.

## Conditions

| # | Mode | Teams | Map | Measured as |
|---|---|---|---|---|
| 1 | frontline | 5v5 | preset | Elo in the condition's pool |
| 2 | deathmatch | 5v5 | preset | Elo |
| 3 | elimination | 5v5 | preset | Elo |
| 4 | deathmatch | 1v1 | mode minimum (density rule) | Elo |
| 5 | deathmatch | 2v2 | density rule | Elo |
| 6 | elimination | 1v1 | mode minimum (density rule) | Elo |
| 7 | elimination | 2v2 | density rule | Elo |
| 8 | frontline | 10v10 | density rule **(provisional)** | Elo |
| 9 | deathmatch | 10v10 | density rule **(provisional)** | Elo |
| 10 | elimination | 10v10 | density rule **(provisional)** | Elo |
| 11 | frontline | break-even: ours against 20 Imugi | density rule at 20 **(provisional)** | N*, the ship count where the score crosses 0.5 |
| 12 | deathmatch | break-even | density rule at 20 **(provisional)** | N* |
| 13 | elimination | break-even | density rule at 20 **(provisional)** | N* |
| 14–19 | all | map scales 0.5× and 2× at 5v5 | scaled | Elo (**only if variable maps survive their arm**) |

**Map rule (provisional):** the game's density rule (`api.arena_radius_for`): the radius scales with fleet size, down to each mode's minimum. The final choice between this and the preset's 5v5 map is deferred until the map-size arm shows whether the agent trains on varied maps. Both arms of any pair always use the same rule, so comparisons stay fair.

## Score

1. **Per condition:** A's expected score against B, minus 0.5.
   - For Elo conditions, this is the logistic of the Elo contrast between the two arms' final checkpoints.
   - For break-even, it is A's win rate against 20 Imugi when fielding B's break-even count N*_B, read off A's fitted curve. The difference is symmetrised: half of A's win rate at N*_B minus 0.5, minus half of B's at N*_A minus 0.5.
2. **Per mode:** the mean over that mode's conditions, each weighted equally.
   - frontline: 1, 8 and 11 (plus 14–19 if they apply);
   - deathmatch: 2, 4, 5, 9 and 12;
   - elimination: 3, 6, 7, 10 and 13.
3. **Overall:** the mean of the three modes, each weighted equally.
4. **Decision:** follows the decision rule. The SE target and the no-regression gate are **(open)**.

## Precision

- **Ladder:** every checkpoint is rated to about ±40–50 Elo, enough for the curves.
- **Decision contrasts:** the two finals' contrast in every condition, with a target from the 2 × 2 seed measurement **(open)**.
- **Break-even:** stops at an SE of N* of 0.15 ships **(open; 0.3 was the agent's default)**, or 8192 games per mode, or a final bound.

## Reported but not scored

- **Exploit check:** each final checkpoint against the six cheese agents in every mode (zone rusher, turtle, evader, deathball, jouster, kiter). A loss to one is flagged for the author.
- **Loop check:** goodness of fit, and a likelihood-ratio test against mElo, in every pool.
- **Seat symmetry:** the mirror match, a checkpoint against itself, scored by seat.

## Cost (to measure)

- **Unknowns:** games per second at each condition's size come from the rating layout probe on the GPU. Break-even games cost a full 20v20 each, because a benched slot costs about as much as an active one.
- **Ladder:** checkpoints are only rated in the 5v5 conditions (1–3) **(proposal)**. The other conditions rate the finals plus a few anchor checkpoints from each arm. This keeps the load about 2–3× a single pool rather than 13×.
