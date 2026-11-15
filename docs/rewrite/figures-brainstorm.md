# Figures: a wide first list (for narrowing with the author)

Every plot or figure the paper, the README or the project page might want, grouped by the question it answers. Nothing is chosen yet. For each, **data** says where the numbers come from:
- **ladder:** the rated snapshots;
- **suite:** the evaluation games;
- **training log:** metrics.jsonl;
- **replay:** recorded games;
- **bench:** the game benches.

**When** marks what has to be switched on before the runs, since some data can't be recovered afterwards.

Marks: ★ my guess at a core figure. ⚠ needs logging or saving that isn't on today.

## A. The game (what the agents play)

1. ★ **The three modes at a glance:** one still or small multiple per mode (deathmatch, elimination with the boundary, Frontline's ring of zones), with ships, fields and zones labelled. Data: replay.
2. **The two ships:** Yongma and Jujak, with their card numbers (turn rate, speed, climb, guns, armour). Their flight envelopes (turn rate against speed) on one chart. Data: ship card.
3. **Time-lapse trajectories** of one fight: tracks of every ship, coloured by team, with kills marked. Data: replay.
4. **Frontline's front over a game:** the front's position against time, captures marked, for several games. Data: replay or bench.
5. **Elimination's shrinking boundary:** radius against time with deaths overlaid, and the share of deaths caused by the boundary. Data: bench.
6. **Game length distributions per mode** under the frozen rules, from the freeze check. Data: bench.
7. **Kill aspects:** a polar histogram of the attacker's angle on the target at the kill (head-on, flank, tail), per ship matchup. Data: replay or bench.
8. **Fog of war:** what one agent sees against the true state at one moment. Data: replay.

## B. The model (Yemong)

9. ★ **Architecture diagram:** tokens (ships, fields, zones, global), codes, spatial and temporal blocks, heads, and the belief loop.
10. **The input codes:** how a scalar becomes an HL-Gauss code; the multi-level position colour code; the rotary bands. Illustrative.
11. **Belief under fog:** a hidden enemy's predicted position and spread over time against its true track. Data: replay with the belief decoded. ⚠ needs belief outputs saved in replays.
12. **Attention maps:** who attends to whom in a fight (the attention-share bench exists). Data: replay plus a bench.
13. **Auxiliary heads' accuracy over training:** next-state, shot and geometry losses against steps. Data: training log.

## C. Training dynamics

14. ★ **Elo against steps, BC then RL,** for the baseline (R0): one curve per mode plus the overall, with the BC→RL handoff marked. Data: ladder.
15. ★ **Elo against steps for every arm,** small multiples, each arm against its incumbent with the decision marked. Data: ladder.
16. **Elo against wall-clock and against GPU-hours,** for compute claims (width at equal compute). Data: ladder plus the log's timings.
17. **Win rate against Imugi over training,** per mode: the most intuitive curve for a reader. Data: ladder or suite.
18. **Break-even ratio over training:** "holds X times its number" against steps. Data: break-even measurements at snapshots. ⚠ break-even at several snapshots costs extra games.
19. **Losses and diagnostics:**
    - policy, value per level and part, entropy, KL, clip fraction, explained variance;
    - gradient shares per head.

    Data: training log.
20. **Behaviour over training:** kills per minute, game length, mutual head-on share, time spent near the boundary, captures per game. Data: the rollout's game statistics. ⚠ check they're logged per mode.
21. **The league:**
    - members over time (the schedule);
    - the live agent's win probability against each member (a heatmap: member × step);
    - sampling weights.

    Data: league records.
22. **The averaging sweep:** the Elo of each average (uniform, exponential half-lives, windows) against the live weights. Data: R5's tournament.

## D. The arms (results)

23. ★ **The decision table as a figure:** each arm's overall score with its interval against the indifference zone, coloured by outcome (win, tie, loss, failed). A forest plot. Data: suite.
24. ★ **Per-mode and per-condition results** for each arm: a heatmap of conditions × arms. Data: suite.
25. **Ablation summary:** a bar or dot chart of the change in Elo when each component is removed or replaced (A13–A17, R1). Data: suite or ladder.
26. **The offensive premium (A7):** Elo plus behaviour (draw rate, lifespan, engagements early in training) for 1:1, 2:1 and annealed. Data: ladder plus training log.
27. **Seed noise (A3):** the two seed pairs' curves overlaid, and their score against δ. Data: ladder plus suite.
28. **RL from scratch against BC→RL (A24):** curves on one axis, steps to reach the same Elo. Data: ladder.
29. **MLP against Yemong (R1):** curves, plus generalisation to 1v1, 2v2 and 10v10. Data: ladder plus suite.
30. **Map sizes (A1):** Elo against map scale for the fixed-map and variable-map agents. Data: suite's scaled conditions.
31. **Team sizes (A2):** Elo or win rate at 1v1 through 10v10, plus break-even, for fixed-count and variable-count agents. Data: suite.

## E. Generalisation and robustness

32. ★ **Generalisation across team sizes:** the final champion's win rate against Imugi at 1v1, 2v2, 5v5, 10v10 and 20v20. Data: suite.
33. **Outnumbered:** the break-even curve, win probability against the ships ratio, for the champion and R0. Data: break-even fits.
34. **Exploitability (R3):** the exploiter's win rate against the frozen final over its training. Data: R3's ladder.
35. **Cheese agents:** the champion's win rate against each of the six cheese agents, and over training. Data: exploit checks at snapshots. ⚠ only the final is checked today.
36. **Specialists against the generalist (R4):** per-mode Elo. Data: suite.
37. **Mode transfer:** a matrix of the Elo gained in each mode from training on another (from rl-dm and similar). Data: ladder.

## F. Behaviour (what the agents learned)

38. **Learned manoeuvres:** a gallery of short clips or trajectory plots (scissors, one-circle and two-circle fights, boom-and-zoom), Imugi against the champion. Data: replay.
39. **Aspect and energy:** kill aspects and energy at engagement for Imugi against the champion; does the energy fighter fight as an energy fighter? Data: replay.
40. **Team play:** distance to the nearest ally over time, focus fire (simultaneous attackers per target), and zone presence in Frontline. Data: replay or rollout statistics.
41. **Strategy embedding:** a 2-D map of game-level behaviour statistics for many agents (snapshots, arms, cheese, Imugi), to show diversity. Data: suite games' statistics.
42. **A failure gallery:** what failed arms do (A1 or A2 if they don't train). Data: replay.

## G. Compute and engineering

43. **Throughput:** steps per second for BC and RL, rollout against learning, on the 4070, 3090 and 5090. Data: logs.
44. **Cost per arm:** GPU-hours by arm class, and the total program cost. Data: arm review.
45. **The sequential test in action:** Z against information fraction with the boundaries, for a few decisions. Data: suite looks.
46. **Allocation:** games per condition against variance and cost for one decision. Data: suite.

## H. Project page or README extras

47. A short video loop of the champion against Imugi.
48. A "how good is it" headline: the champion holds X times its number of Imugi.
49. Ship-card visuals.

## Before the runs: what must be on

Must be logged or saved, because it can't be recovered later:
- **11:** belief outputs in replays.
- **18:** break-even at several snapshots (it can be run afterwards from the saved snapshots, so it isn't strictly needed now).
- **20:** per-mode behaviour statistics in the training log.
- **21:** the league's per-member win probabilities over time.
- **35:** exploit checks at snapshots (they can also be run afterwards from snapshots).

Recoverable afterwards: anything from snapshots (they're all kept) and from the evaluation records.
