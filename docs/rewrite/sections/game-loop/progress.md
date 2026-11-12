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

## 2026-11-12 15:02-15:30 KST: round 1 (doctrine search; card sweep 1)

**Doctrine search** (`doctrine_round.py --round 1`, the (1 + lambda) mutation search of
the doctrine arena on the GPU, both types at once, 10 + 10 children a generation, 32 games
per candidate x opponent x lineup, 6 lineups (mixed and reversed weighted half), against
off and hand, from hand and from off; ~30 s a generation; data `round1/`):

- From hand: 0.28 -> 0.72 in one generation (strength cut, extend/slash/lag kept partly),
  then a plateau ~0.71-0.72; the later promotions were selection noise (the promoted
  parent re-scored lower: 0.723 -> 0.688). **Fixed** in the search: one promotion at a
  time, on probation against the incumbent on fresh games.
- From off: no child beat off by 2 SE in 12 generations.
- **Elo** (mixed, both orders, 512 games a pair and order): hot 1001 ± 7, off 1000,
  r1-off 997 ± 7, r1-hand 994 ± 7, hand 513 ± 12, random 191 ± 17. **No doctrine is
  stronger than none** on this card: the searches found off (r1-off: KL 0.016, agreement
  0.99, collapsed) or something off-like that stays distinct (r1-hand: KL 1.25 / 0.14,
  agreement 0.85 / 0.97). r1-off joined the league as `r1`, r1-hand as `r1-hand`.
- Next for the search: the ES (`es_search`: antithetic pairs over both doctrines in a
  log/linear space, normalised gradient step, probation) and the per-game kill share as
  the objective (lower variance than win/loss; `kill_score`).

**Card sweep 1** (`card_round.py --round 1 --player r1-hand`; 16 variants as rows of one
type table, r1-hand self-play in 5 lineups x 1024 games and r1-hand v off; 544 s at K = 16:
`per_ship`'s select over 32 types makes a tick ~2.3x slower than K = 1; data
`round1/cards.{json,md}`). f = head-on fraction scale (of 0.2 x the card's 0.667), a = the
half-strength angle scale (of 42 degrees); kill shares mutual / offset / flank / tail
summed over the lineups:

| Variant | v off kill share | E v T (Yongma) | Kills/min | Kills m / o / f / t | Damage m / o / f / t | Passes a kill | Card head-on passes YY / JJ / YJ / JY |
|---|---|---|---|---|---|---|---|
| f1 a1 (base) | 0.500 | 0.561 ± 0.014 | 5.57 | 0.33 / 0.26 / 0.35 / 0.06 | 0.51 / 0.21 / 0.24 / 0.04 | 0.58 | 3.46 / 3.33 / 2.84 / 4.04 |
| f1 a0.25 | 0.503 | 0.492 | 5.88 | 0.36 / 0.31 / 0.29 / 0.04 | 0.49 / 0.28 / 0.21 / 0.02 | 0.62 | same |
| f0.65 a1 | 0.503 | 0.522 | 5.20 | 0.21 / 0.26 / 0.44 / 0.09 | 0.38 / 0.24 / 0.33 / 0.06 | 0.50 | 5.36 / 5.13 / 4.36 / 6.24 |
| f0.65 a0.25 | 0.498 | 0.470 | 5.62 | 0.26 / 0.34 / 0.34 / 0.05 | 0.37 / 0.33 / 0.27 / 0.03 | 0.56 | same |
| f0.4 a1 | 0.486 | 0.452 | 4.88 | 0.13 / 0.25 / 0.51 / 0.11 | 0.25 / 0.26 / 0.41 / 0.08 | 0.42 | 8.71 / 8.33 / 7.12 / 10.17 |
| f0.4 a0.25 | 0.500 | 0.444 | 5.47 | 0.21 / 0.37 / 0.37 / 0.06 | 0.26 / 0.38 / 0.31 / 0.04 | 0.54 | same |
| f1 a1 slow | 0.501 | 0.532 | 5.34 | 0.30 / 0.26 / 0.37 / 0.07 | 0.50 / 0.20 / 0.26 / 0.05 | 0.55 | 4.06 / 3.91 / 3.33 / 4.74 |
| f0.65 a0.5 Y explosive 0.5 | 0.501 | 0.500 | 5.50 | 0.25 / 0.32 / 0.38 / 0.06 | 0.37 / 0.30 / 0.29 / 0.04 | 0.53 | 5.36 / ... |
| f0.65 a0.5 Jujak damage x0.95 | 0.501 | 0.496 | 5.43 | 0.23 / 0.32 / 0.38 / 0.06 | 0.37 / 0.30 / 0.30 / 0.04 | 0.53 | 5.36 / 5.41 / ... |

(slow = muzzle x 0.8 with the reference speed, spread x 1.5, hit radius x 0.85.)

- Frontal armour moves mutual head-on into the flank (kills 0.33 -> 0.13 at f0.4) and a
  narrower cone into the offset head-on; tail stays 0.04-0.11 under these off-like
  doctrines. More armour costs Yongma (energy v turn 0.56 -> 0.44): Yongma's edge is the
  joust. Yongma's explosive share 0.5 or Jujak's damage x 0.95 rebalance it to 0.50.
- Head-on TTK on the card is 1 / (damage x fraction): f < 1 alone overshoots 3-4 passes
  (5-10). So round 2 trades them: fraction down, damage up (d x f ~ 1).
- Measured passes a kill stay 0.4-0.6 whatever the card: kills come in the first merge's
  turn, not from repeated head-on passes. The card's head-on TTK is not what the agent
  plays.
- The doctrine value (r1-hand v off) is 0.49-0.51 on every card: no card here makes the
  doctrine pay.

## 2026-11-12 15:29-15:36 KST: round 2 card sweep (fraction x damage at a constant head-on TTK)

`card_round.py --round 2 --player r1 hand` (8 variants, K = 8: 401 s; `round2/cards.*`).
s = head-on fraction scale, d = hit damage factor (on the x 0.65 base), a = cone scale;
d x s = 1 keeps the card's head-on TTK (3.3-4.0 passes; Yongma on Jujak 2.8).

| Variant | Player | v off kill share | E v T | Kills/min | Kills m / o / f / t | Damage m / o / f / t | Passes a kill |
|---|---|---|---|---|---|---|---|
| s1 d1 a1 | r1 (off-like) | 0.506 | 0.541 | 5.60 | 0.33 / 0.26 / 0.35 / 0.06 | 0.52 / 0.21 / 0.24 / 0.03 | 0.59 |
| s1 d1 a1 | hand | 0.235 | 0.118 | 3.18 | 0.08 / 0.14 / 0.20 / 0.58 | 0.16 / 0.15 / 0.19 / 0.50 | 0.32 |
| s0.5 d2 a1 | r1 | 0.501 | 0.556 | 5.80 | 0.36 / 0.27 / 0.33 / 0.04 | 0.50 / 0.22 / 0.25 / 0.03 | 0.59 |
| s0.5 d2 a0.5 | r1 | 0.500 | 0.502 | 5.89 | 0.37 / 0.30 / 0.30 / 0.04 | 0.47 / 0.27 / 0.23 / 0.02 | 0.60 |
| s0.35 d2.86 a0.25 | r1 | 0.504 | 0.480 | 6.03 | 0.37 / 0.33 / 0.27 / 0.03 | 0.44 / 0.32 / 0.22 / 0.02 | 0.62 |
| s0.35 d2.86 a0.25 | hand | 0.200 | 0.184 | 3.83 | 0.11 / 0.24 / 0.18 / 0.47 | 0.15 / 0.25 / 0.19 / 0.41 | 0.40 |
| s0.5 d1.5 a0.5 | r1 | 0.497 | 0.506 | 5.66 | 0.29 / 0.31 / 0.34 / 0.05 | 0.41 / 0.29 / 0.27 / 0.03 | 0.56 |
| s0.5 d2 a0.5 slow | r1 | 0.499 | 0.498 | 5.74 | 0.34 / 0.30 / 0.32 / 0.04 | 0.48 / 0.26 / 0.24 / 0.03 | 0.59 |

- At a constant head-on damage, raising the side damage does not cut mutual head-on (0.33
  -> 0.37 of kills): the off-like agent jousts whatever the card, and the mix follows the
  head-on damage itself (round 1: fraction 0.4 -> mutual 0.13-0.21). Pace rises a little
  with damage (5.6 -> 6.0 kills/min).
- The hand doctrine keeps its tail-heavy mix on every card and loses to off on every card
  (kill share 0.20-0.24): no card here makes it pay.
- **So the mix needs an agent that avoids the joust because it pays**: the head-on
  damage down (fraction 0.5, d = 1: card 5.4 head-on passes, but the aimed fire's damage
  per hit is ~1.9x the card's mean closeness, so the measured joust is about half that)
  and a search that can find strength in flank and tail play.
- Card for round 2's search: s0.5 a0.5 d1, Yongma explosive share 0.5 (round 1: balances
  energy v turn to 0.50 at fraction 0.65).

**Round 2 search** (running, 15:36): the ES over both doctrines and the combat/memory
scripted parameters (27 numbers) at once, from hand and from r1-hand, against off, hand,
r1, r1-hand; 16 antithetic pairs, 16 games per candidate x opponent x lineup, kill share.

## 2026-11-12 15:36-16:02 KST: round 2 (ES over doctrines + scripted parameters)

Card: s0.5 a0.5 d1, Yongma explosive share 0.5 (`round2/card.json`). ES (`es_search`, 16
antithetic pairs, sigma 0.15, 16 games a candidate x opponent x lineup, kill share, ~35-67
s a generation, 15 generations a start) over both doctrines and 27 scripted parameters,
against off, hand, r1, r1-hand.

- From hand: kill share 0.28 -> 0.68 (by opponent at gen 8: off 0.63, r1 0.61, r1-hand
  0.61, hand 0.81). From r1-hand: 0.56 -> 0.66.
- **Elo** (mixed, both orders, 512 games a pair and order): **r2-hand 1296 ± 8**,
  r2-r1-hand 1240 ± 7, hot 1017, r1-hand 1007, r1 1002, off 1000, hand 506, random 108.
  The first real gain: +300 Elo over the anchor. r2-hand joined the league as `r2`.
- What moved: mostly the scripted parameters: boost while slower than 1.87 corner speeds
  (was 0.93: fly fast), shoot within 1.24 gun ranges (was 1.47), aim gate wider (0.78 ->
  0.99 target sizes), side commitment 3.2 -> 4.2 stall angles. Doctrines: Yongma extend
  0.95, slash 0.97, Jujak bleed 0.55, lag 0.32, deny 0.63, but both gated by
  in_advantage centres 1.47 / 1.85 (rarely active).
- **Distinctness collapsed**: r2's energy v turn doctrine KL 0.0001 / 0.0005 (agreement
  0.999): the doctrines are written distinct but gated off; the strength is in the shared
  parameters. r2-r1-hand keeps 0.61 on Yongma ships. The next search penalises collapse.
- **r2 self-play aspect mix (the wanted shape)**: kills mutual / offset / flank / tail
  0.07-0.11 / 0.25-0.27 / 0.52-0.55 / 0.11-0.13 in every lineup; damage mutual 0.14-0.18.
  Pace 5.7-6.2 kills/min (mean ~ median). But **energy v turn 0.335 ± 0.010** (Yongma
  loses) and passes a kill 0.30-0.39.
- r2 v off: 0.73 (energy v turn), 0.87 (turn mirror), 0.76 (energy mirror), 0.84-0.86
  (mixed).

**Next**: round 3 cards on r2 (running, 16:02: fraction 0.35-0.65, cone 0.25-1, Yongma
explosive 0 / 0.5, damage x 1.25, slow bullets) to rebalance energy v turn; then an ES with
a distinctness penalty from r2.

**Stop plan** (the author's 20:00-21:00 window): last search round ends by ~19:15; the
final measurement (tight error bars, 300 s pace games) 19:15-19:50; freeze and commit by
~20:00. Expected stop **~20:00 KST**, later only if a final run is in flight (hard limit
21:00).
