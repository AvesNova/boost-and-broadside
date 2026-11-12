# Balance across fleet sizes: progress

Branch `tune/balance-sizes` (from integrate/next-runs 9f690a7b, the frozen r9 card). The task:
Yongma (energy) and Jujak (turn) balanced at every fleet size, 1v1 to 50v50 (energy v turn
0.40-0.60, ideally 0.45-0.55, at 1v1, 2v2, 5v5, 10v10; spot check 20v20), keeping mutual
head-on near 0, trades low, life 45-300 s, 0.2-1.4 kills a minute a ship, and the scripted
agent's strength. Hard stop 22:00 KST. Power profile: **performance** (read 19:12).

## Tools

- `sizes.py`: candidates (a card variant and a player: a league champion with doctrine
  fields overridden) in self-play at each fleet size, one compiled loop a size (300 s
  deathmatch games, the arena by the density rule, radius ~ sqrt(n / 5)); writes each size's
  rows as it finishes. E v T is the score of the Yongma side (win = more kills; draws half);
  the kill share (Yongma's share of the game's kills) is reported beside it because the win
  score sharpens with the kills a game (a 0.44 kill share is a 0.19 score at 20v20, 68 kills
  a game).
- `trajectories.py`: a few games recorded tick by tick, uncompiled, the 1v1 fights measured
  from the tracks and PNG plots (path coloured by time light to dark, Yongma blue, Jujak
  orange; merges, hits on the target, the kill).
- `handles.py`: a card variant in handle space (the ship file's handles re-solved); 4.6 min a
  solve on the CPU, so the search uses card factors and only the final card is solved.
- **Bug fixed** in `bench/game_loop.py`: `pace` divided by 10 ships at every size, so the
  game-loop's 1v1 and 2v2 "kills a minute a ship" were 5x and 2.5x too low (its 1v1 0.05 is
  0.25). `Loop.play` now records the ships a game.

## 19:15-19:24 KST: baseline (frozen card, r9), every size

`baseline/sizes.{json,md}`, 1024 games a lineup (10v10 512, 20v20 256), 300 s:

| Size | E v T score | E v T kill share | Turn mirror | Energy mirror | Kills m / o / f / t (E v T) | Trades | Life s (E v T / JJ / YY) | Kills/min/ship (E v T) |
|---|---|---|---|---|---|---|---|---|
| 1v1 | **0.013 ± 0.003** | 0.195 | 0.498 | 0.505 | 0.16 / 0.14 / 0.41 / 0.30 | 0.00 | 168 / 152 / **1076** | 0.34 |
| 2v2 | **0.364 ± 0.013** | 0.459 | 0.507 | 0.509 | 0.10 / 0.17 / 0.53 / 0.20 | 0.01 | 111 / 95 / 221 | 0.50 |
| 5v5 | 0.524 ± 0.014 | 0.501 | 0.490 | 0.515 | 0.06 / 0.17 / 0.53 / 0.24 | 0.04 | 110 / 91 / 161 | 0.50 |
| 10v10 | 0.518 ± 0.021 | 0.496 | 0.489 | 0.500 | 0.05 / 0.17 / 0.51 / 0.27 | 0.08 | 125 / 103 / 219 | 0.45 |
| 20v20 | **0.189 ± 0.024** | 0.444 | 0.559 ± 0.030 | 0.447 ± 0.030 | 0.05 / 0.16 / 0.50 / 0.30 | 0.12 | 158 / 124 / 374 | 0.36 |

- Balanced at 5v5 and 10v10 only. **Yongma loses every duel** (kill share 0.20: it dies ~3
  times a 300 s game and kills under once), loses 2v2 (0.36), and **loses 20v20 again**
  (kill share 0.44, a 0.19 score at 68 kills a game). The 20v20 mirrors are off 0.5 by ~2 SE
  (0.559 / 0.447, 256 games): worth a larger check before reading much into 20v20.
- Self-play pace is in range at every size except the 1v1 energy mirror (life 1076 s, 0.53
  kills a game: two Yongmas rarely finish each other) and its 20v20 (374 s).

**Fight shape, 1v1** (the tally's engagements, 2 turn diameters):

| 1v1 | Merges a kill mean / median / first-pass / p90 | Circles turned a ship | Merges 1-circle / 2-circle / scissors | Kills m / o / f / t | Engagement to death median s |
|---|---|---|---|---|---|
| energy v turn | 2.84 / 3 / 0.26 / 6 | 2.25 | 0.11 / 0.70 / 0.20 | 0.16 / 0.14 / 0.41 / 0.30 | 50 |
| turn mirror | 2.18 / 2 / 0.44 / 4 | 1.83 | 0.16 / 0.80 / 0.04 | **0.81** / 0.17 / 0.02 / 0.00 | 38 |
| energy mirror | 3.60 / 3 / 0.22 / 9 | 3.11 | 0.04 / 0.93 / 0.03 | 0.40 / **0.52** / 0.08 / 0.00 | 65 |

- The mixed duel already has the wanted spread (mean 2.8, median 3, a quarter first-pass,
  p90 6) and real turning (2.3 circles a ship, mostly two-circle, a fifth scissors), but
  one-sided: the Jujak out-turns the Yongma and sits on its tail (flank + tail 0.71).
- **The same-type duels are jousts**: Jujak v Jujak kills are 81% mutual head-on, Yongma v
  Yongma 92% mutual + offset and rarely finish (life 1076 s). In a crowd a third ship ends
  the fight from the flank; alone, two equal ships meet nose to nose.
- In crowds a kill is 0.4-0.9 merges (first-pass 0.8-0.95): the target of ~3 is met only in
  duels.
