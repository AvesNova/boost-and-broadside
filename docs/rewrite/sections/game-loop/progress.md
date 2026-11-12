# Game loop: progress

Branch tune/game-loop (from feat/bullet-damage 8995ad73). The task: the doctrines and the
ship cards of Yongma (energy) and Jujak (turn) tuned on the GPU, compiled, with no
recompilation inside the loop; the scripted agent for Elo against a league, the cards for
energy v turn within 55/45 and a mix of aspects with mutual head-on near zero; pace as
kills per minute in a 5v5 deathmatch with respawns. Hard stop 20:00 KST.

## 2026-11-12 14:45-15:05 KST: the harness, round 0 baselines

**Harness** `bench/game_loop.py` (`Loop`, `Job`, `run`, `summary`, `ratings`,
`distinctness`): `real_scale_deathmatch` 5v5 with respawns (10 s), no fields, no zones,
7800 m, fixed-length games (120 s here) that all end on the same tick; one
`torch.compile(fullgraph=True)` tick (scripted agent for both teams + move + resolve +
perceive + every statistic) on the GPU. Card variants are rows of a small type table (K
Yongma rows, K Jujak rows) and the doctrines a per-view table (2B, 2K, WIDTH), the agent's
temperature per view: all written in place, so a new sweep of the same B and K never
recompiles (checked: one compile per process, ~45-75 s).

- Throughput (RTX laptop, performance profile, 60-120 s games): B=512 0.54M env-ticks/s,
  1024 1.0M, 2048 1.7M, 8192 1.77M (491 games/s of 120 s; 0.23 GiB). GPU-bound from
  B ~2048; **a 10-minute round is ~290k games of 120 s**: the laptop is enough, no need
  for a 5090.
- Statistics per environment: kills by side, by killer and victim type and aspect; damage
  taken by victim side, type and aspect; passes before each kill (histogram), first-merge
  kills, kills by the pair's first hitter, TTK (from the victim's first hit, 1 s bins),
  P(win | game's first hit); post-merge turning fights (3 s after a merge ends, both noses
  turning > 0.05 rad/s for half of it: same turn sense two-circle, opposite one-circle,
  both reversing twice scissors). Win = more kills in the game, draws half.
- League (`league.py` here): `off` (no doctrines, the anchor at 1000), `hand` (round 1's
  doctrines), `random` (temperature inf), `hot` (temperature 3); champions join as rounds
  go. Elo = `bnb.league.ratings.fit_ratings` over the mixed lineups (both orders).
- Distinctness: the energy and the turn doctrine on the same observations (every 150
  ticks of a mixed game, the memoryless agent), symmetric KL of the power + stick marginals
  per living own ship, and the share with the same most likely (power, stick).
- Elimination dropped: deathmatch with fixed-length games is cheaper on a fixed-shape GPU
  graph (no compaction, no timeouts) and gives pace directly.

**Round 0** (base card = ship files with hit damage x 0.65; 120 s games; 2048 games a
self-play row, 512 a league pair and order; full table `round0-table.md`, data
`round0.json`):

| Players | Lineup | A scores | Kills/min mean (median) | Damage m / o / f / t | Kills m / o / f / t | Passes a kill | First-merge kills | P(win \| first hit) | Circles 1 / 2 / sc |
|---|---|---|---|---|---|---|---|---|---|
| off v off | energy v turn | 0.546 ± 0.010 | 5.58 (5.50) | 0.50 / 0.21 / 0.25 / 0.03 | 0.32 / 0.27 / 0.36 / 0.06 | 0.57 | 0.53 | 0.60 | 0.32 / 0.67 / 0.00 |
| off v off | turn mirror | 0.501 | 5.58 | 0.58 / 0.19 / 0.20 / 0.03 | 0.41 / 0.24 / 0.30 / 0.05 | 0.66 | 0.58 | 0.59 | 0.35 / 0.64 / 0.00 |
| off v off | energy mirror | 0.489 | 5.62 | 0.46 / 0.23 / 0.28 / 0.04 | 0.28 / 0.27 / 0.38 / 0.07 | 0.53 | 0.48 | 0.58 | 0.29 / 0.71 / 0.00 |
| off v off | mixed | 0.494 | 5.63 | 0.50 / 0.21 / 0.25 / 0.04 | 0.31 / 0.27 / 0.36 / 0.06 | 0.57 | 0.52 | 0.59 | 0.32 / 0.68 / 0.00 |
| hand v hand | energy v turn | 0.118 ± 0.006 | 2.69 (2.50) | 0.05 / 0.15 / 0.09 / 0.71 | 0.03 / 0.13 / 0.12 / 0.72 | 0.21 | 0.14 | 0.57 | 0.39 / 0.54 / 0.07 |
| hand v hand | turn mirror | 0.489 | 2.27 | 0.01 / 0.05 / 0.22 / 0.72 | 0.00 / 0.01 / 0.15 / 0.85 | 0.29 | 0.01 | 0.57 | 0.20 / 0.79 / 0.01 |
| hand v hand | energy mirror | 0.507 | 5.21 | 0.39 / 0.23 / 0.22 / 0.16 | 0.23 / 0.27 / 0.27 / 0.23 | 0.49 | 0.44 | 0.57 | 0.15 / 0.09 / 0.75 |
| hand v hand | mixed | 0.495 | 3.10 | 0.23 / 0.18 / 0.20 / 0.39 | 0.09 / 0.17 / 0.23 / 0.50 | 0.32 | 0.23 | 0.57 | 0.16 / 0.61 / 0.24 |

League Elo (mixed, both orders, 1024 games a pair): off 1000, hot 1006 ± 10, **hand 516 ±
17** (off beats hand 0.96-0.99 in deathmatch), random 98 ± 24. Hand distinctness:
symmetric KL 3.37 nats (Yongma ships) / 3.20 (Jujak ships), agreement 0.63 / 0.60.

- Without doctrines it is still jousting (mutual head-on half the damage, a kill in 0.57
  passes, 0.53 in the first merge); energy v turn 0.55 for Yongma. The hand doctrines give
  the wanted aspect mix (tail 0.7) but halve the pace, lose the energy v turn matchup 0.12
  for Yongma, and lose to off almost always: the doctrine as hand-set is not strength.
- "hot" at temperature 3 is no weaker than off: not a useful null reference; random is.

**Next**: the doctrine search on the GPU against the league (both types at once,
candidates in one batch), then the card sweep on the champion.
