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

## 2026-11-12 16:02-16:08 KST: round 3 cards (on r2)

`card_round.py --round 3 --player r2` (8 variants, 245 s; `round3/cards.*`):

| Variant | r2 v off kill share (score) | E v T (Yongma) | Kills/min mean (median) | Kills m / o / f / t | Damage m / o / f / t | Passes a kill | First-merge |
|---|---|---|---|---|---|---|---|
| s0.5 a0.5 Yx0.5 (round 2) | 0.639 (0.856) | 0.345 ± 0.014 | 6.02 (6.00) | 0.09 / 0.26 / 0.54 / 0.12 | 0.16 / 0.27 / 0.48 / 0.09 | 0.35 | 0.26 |
| s0.65 | 0.636 (0.853) | 0.326 | 6.09 (6.00) | 0.10 / 0.26 / 0.52 / 0.11 | 0.19 / 0.26 / 0.46 / 0.09 | 0.37 | 0.28 |
| s0.35 | 0.657 (0.851) | 0.332 | 5.94 (6.00) | 0.07 / 0.26 / 0.55 / 0.12 | 0.12 / 0.27 / 0.51 / 0.10 | 0.33 | 0.24 |
| a1 | 0.648 (0.850) | 0.330 | 5.67 (6.00) | 0.08 / 0.19 / 0.59 / 0.14 | 0.16 / 0.20 / 0.52 / 0.11 | 0.31 | 0.20 |
| a0.25 | 0.632 (0.839) | 0.338 | 6.24 (6.50) | 0.10 / 0.30 / 0.49 / 0.11 | 0.16 / 0.30 / 0.45 / 0.09 | 0.38 | 0.31 |
| Yongma explosive 0 | 0.649 (0.862) | 0.332 | 6.03 (6.00) | 0.09 / 0.26 / 0.54 / 0.12 | 0.16 / 0.27 / 0.48 / 0.09 | 0.35 | 0.26 |
| damage x 1.25 | 0.631 (0.852) | 0.334 | 6.26 (6.50) | 0.11 / 0.27 / 0.52 / 0.11 | 0.18 / 0.27 / 0.47 / 0.08 | 0.37 | 0.29 |
| slow bullets | 0.676 (0.887) | 0.291 | 5.67 (5.50) | 0.07 / 0.26 / 0.54 / 0.13 | 0.16 / 0.26 / 0.48 / 0.10 | 0.34 | 0.24 |

- Under r2 the mix is now set by the agent, not the card: every variant gives mutual
  0.07-0.11, offset 0.19-0.30, flank 0.49-0.59, tail 0.11-0.14 of kills; the cone moves
  offset against flank (a0.25: 0.30 / 0.49; a1: 0.19 / 0.59).
- **Energy v turn 0.29-0.35 on every one**: no armour, cone, damage or explosive change
  rebalances it; r2's shared parameters favour the turn fighter. Round 5 tries
  type-specific levers (Yongma damage / shield up, Jujak's down).
- The coordinator's check, the **trade fraction** (deaths with an enemy death within 0.5
  s), is now in the tick (`deaths`, `traded`; `trade_fraction` in every table from round
  4 on).

**Round 4** (running, 16:08, ETA ~16:40): the ES from r2 and from hand, with the
distinctness penalty (0.1 x max(0, 1 - KL / 0.5 nats), KL on 8 snapshots x 4096 views of
the parent's mixed games), against off, hand, r1, r1-hand, r2.

## 2026-11-12 16:08-16:32 KST: round 4 (ES with the distinctness penalty)

Same card as round 2. ES from r2 and from hand, doctrines + parameters, the distinctness
penalty (floor 0.5 nats, weight 0.1 kill share), against off, hand, r1, r1-hand, r2.

- From r2: KL 0.000 -> 0.65 in 5 generations while the kill share rose 0.532 -> 0.652
  (penalised; r2 itself scores 0.532 penalised = 0.632 raw). Then probation kept failing
  and the step halved to nothing (fixed for later rounds: a passed probation grows the
  step 1.5x). From hand: KL 3.37 -> 0.24, kill share 0.28 -> 0.44: far behind.
- **Elo** (mixed, both orders, 512 a pair and order): **r4-r2 1294 ± 7**, r2 1279 ± 7,
  hot 1005, r1 1003, off 1000, r1-hand 999, r4-hand 909, hand 478, random 98.
- **Did it improve on the champion?** Elo +15 ± 10 over r2: no significant gain, but the
  same strength with **distinct doctrines**: KL 0.21 on Yongma ships / 1.71 on Jujak ships
  (agreement 0.95 / 0.77), against r2's 0.0001 / 0.0005. r4-r2 joined the league as `r4`.
- **Aspect mix** (r4 self-play, all lineups): kills mutual 0.07-0.10, offset 0.25-0.27,
  flank 0.52-0.55, tail 0.10-0.14 (r2's within 0.01). Damage mutual 0.14-0.18.
- **Trade fraction** (first measurement): **0.07-0.09** in every self-play lineup and 0.07-0.09
  against off: decisive fights, inside the author's 10-20%.
- Energy v turn still **0.336 ± 0.010** (Yongma loses): round 5 (running) tries type
  levers (Yongma damage x1.15 / x1.3, shield x1.15, Jujak damage x0.85, shield x0.87,
  Yongma frontal armour x0.7) with the pace statistics added (gap between deaths, life
  length, kills per minute per ship, spawn to first hit, first hit and kill).

## 2026-11-12 16:32-16:37 KST: round 5 cards (balance levers, on r4)

`card_round.py --round 5 --player r4` (275 s; `round5/cards.*`); the round-2 card plus one
type lever each. Pace columns now measured (120 s games from a cold start, so the first
engagement is in every game).

| Variant | E v T (Yongma) | Kills m / o / f / t | Trades | Life s | Death gap mean (median) s | Kills/min/ship | Spawn to first hit s | Game's first hit / kill s |
|---|---|---|---|---|---|---|---|---|
| round-2 card | 0.313 ± 0.013 | 0.08 / 0.26 / 0.54 / 0.12 | 0.08 | 89 | 7.7 (4.7) | 0.61 | 37.5 | 25.0 / 29.6 |
| Y damage x1.15 | 0.362 | 0.09 / 0.26 / 0.53 / 0.12 | 0.08 | 88 | 7.6 (4.7) | 0.62 | 37.5 | 25.1 / 29.6 |
| Y damage x1.3 | 0.386 | 0.09 / 0.26 / 0.53 / 0.11 | 0.08 | 87 | 7.6 (4.5) | 0.62 | 37.5 | 25.0 / 29.5 |
| J damage x0.85 | 0.340 | 0.08 / 0.25 / 0.55 / 0.13 | 0.08 | 91 | 7.8 (4.7) | 0.60 | 37.5 | 25.5 / 30.3 |
| Y shield x1.15 | 0.376 | 0.08 / 0.26 / 0.54 / 0.13 | 0.08 | 91 | 7.9 (4.9) | 0.60 | 37.5 | 24.6 / 29.6 |
| Y shield + damage x1.15 | 0.409 | 0.08 / 0.26 / 0.54 / 0.12 | 0.08 | 89 | 7.7 (4.7) | 0.61 | 37.5 | 25.0 / 29.6 |
| J shield x0.87 | 0.359 | 0.09 / 0.26 / 0.54 / 0.12 | 0.08 | 88 | 7.7 (4.7) | 0.62 | 37.5 | 25.3 / 29.5 |
| Y frontal fraction x0.7 | 0.319 | 0.08 / 0.26 / 0.55 / 0.12 | 0.08 | 90 | 7.8 (4.7) | 0.60 | 37.5 | 25.5 / 29.9 |

- Each lever moves energy v turn by only +0.03-0.07; the best pair (Yongma shield and
  damage x1.15) reaches 0.41. The mix, trades (0.08) and pace do not move. Round 6
  (running) stacks them: Yongma shield/damage x1.2-1.4 with Jujak shield x0.8-0.9.
- Pace inside the author's bands: life 87-91 s (band 45-300 s), 0.60-0.62 kills a minute
  a ship (band 0.2-1.4); the gap between deaths 7.7 s mean, 4.7 s median (whole-second
  bins); a ship's spawn to its first hit 37.5 s median (the 7800 m map's approach).

## 2026-11-12 16:37-16:40 KST: round 6 cards (stacked balance levers, on r4)

| Variant (on the round-2 card) | r4 v off kill share | E v T (Yongma) | Kills m / o / f / t | Trades | Life s | Death gap mean (median) | Kills/min/ship |
|---|---|---|---|---|---|---|---|
| Y shield, damage x1.15 | 0.644 | 0.371 ± 0.014 | 0.09 / 0.26 / 0.54 / 0.12 | 0.08 | 89 | 7.8 (4.7) | 0.61 |
| Y x1.3, x1.3 | 0.648 | 0.479 | 0.09 / 0.25 / 0.54 / 0.12 | 0.08 | 89 | 7.7 (4.7) | 0.61 |
| Y x1.15, x1.15; J shield x0.87 | 0.632 | 0.453 | 0.09 / 0.26 / 0.53 / 0.12 | 0.08 | 88 | 7.7 (4.7) | 0.62 |
| Y x1.3, x1.15; J shield x0.87 | 0.642 | 0.493 | 0.09 / 0.25 / 0.54 / 0.12 | 0.08 | 89 | 7.8 (4.7) | 0.61 |
| Y x1.2, x1.2; J x0.9, x0.9 | 0.644 | 0.490 | 0.09 / 0.25 / 0.54 / 0.12 | 0.08 | 89 | 7.7 (4.7) | 0.61 |
| **Y x1.3, x1.3; J shield x0.87** | 0.631 | **0.503** | 0.09 / 0.26 / 0.54 / 0.12 | 0.08 | 88 | 7.7 (4.7) | 0.62 |
| Y shield x1.4, damage x1.3; J shield x0.87 | 0.629 | 0.555 | 0.09 / 0.25 / 0.54 / 0.12 | 0.08 | 89 | 7.8 (4.7) | 0.61 |
| Y x1.3, x1.3; J shield x0.8, damage x0.9 | 0.635 | 0.581 | 0.09 / 0.25 / 0.54 / 0.12 | 0.08 | 88 | 7.7 (4.7) | 0.62 |

- Energy v turn now spans 0.37-0.58 and the mix, trades and pace stay put. **Card for
  round 7: Yongma shield x1.3 and hit damage x1.3, Jujak shield x0.87** (on the round-2
  card: fraction x0.5, cone x0.5, Yongma explosive 0.5): energy v turn 0.503 ± 0.014.
  The energy fighter becomes the heavier, harder-hitting type (the P-47 / Fw 190 side of
  the archetype), the turn fighter the lighter.
- **Round 7** (running 16:40, ETA ~17:05): ES from r4 on this card, distinctness on,
  20 generations, against off, hand, r1, r1-hand, r2, r4.

## How a round works (plainly)

- **Doctrine/parameter round** (`doctrine_round.py`, `es_search`): one search state, the
  champion's 2 doctrines (29 numbers each) and 31 scripted parameters. A generation is
  **33-34 candidates** (the parent, the incumbent on probation, 16 antithetic pairs of
  random perturbations), each playing **16 games per league opponent per lineup** (6
  lineups, 5-7 opponents: ~500-700 games a candidate) as side A; **all candidates' games
  run batched together on the GPU** (8192 games per compiled batch, ~13-20k games a
  generation, 35-70 s). Score = the candidate's share of the kills, lineups weighted (mixed
  1/2, single-type 1/2), minus the distinctness penalty; SE ~0.005-0.008 per candidate.
  The gradient comes from the pairs' differences; the parent moves; the move is kept only
  if the parent's fresh games beat the incumbent's. 15-20 generations a start.
- At the round's end: **Elo** over the whole league (every pair, mixed + mixed reversed,
  512 games a pair and order: ±6-8 Elo), distinctness, and the champion's self-play table
  (2048 games a lineup) and against off (1024).
- **Card round** (`card_round.py`): **8 card variants** (16 in round 1) as rows of one
  type table, **all in one batched run**: each variant x 5 lineups x 1024 games of the
  champion's self-play plus 1024 games against off (49k games, 2-5 min). Error bars:
  energy v turn ±0.014; mix shares ±0.01.
- Batching more variants per run is possible (each variant is two rows of the type table);
  the tick slows with the type count (per_ship's select: K = 16 was ~2.3x K = 1), so 8-16
  per run is the sweet spot. A random sample of card vectors in one tournament would fit
  in the same harness; not done yet.

## 2026-11-12 16:40-17:03 KST: round 7 (ES on the balanced card)

Card: round-2 card + Yongma shield x1.3 and damage x1.3, Jujak shield x0.87.

- ES from r4, 20 generations: kill share 0.615 -> 0.626 (penalised), KL 0.82.
- **Elo**: **r7-r4 1291 ± 7**, r4 1280 ± 7, r2 1263 ± 7, hot 1003, off 1000, r1-hand 992,
  r1 992, hand 506, random 108. **Improvement over r4: +11 ± 10 Elo: not significant.**
  The ES has converged on this neighbourhood (rounds 4 and 7 both plateau after ~6
  generations). r7 joined the league.
- Distinctness: KL 0.28 (Yongma ships) / 2.02 (Jujak ships), agreement 0.94 / 0.73.
- **Energy v turn 0.509 ± 0.010** (target met); turn mirror 0.488, energy mirror 0.502,
  mixed 0.485 / 0.486 (seats fair within 1.5 SE).
- Mix (self-play kills): mutual 0.07-0.12, offset 0.25-0.28, flank 0.51-0.54, tail
  0.10-0.13. Trades 0.07-0.10. Life 82-94 s, gap between deaths 7.2-8.1 s (median
  4.5-5.5), first hit 37.5 s after spawn.

## 17:03 KST: the author's round-4/5 feedback, done

- Engagement statistics in the tick (two definitions: within the pair's gun range; within
  two full-stick turn diameters; ends 3 s apart or on a death): engagements, kills in them,
  **merges per kill** (closing -> least distance -> opening by 30 m; mean, median,
  first-pass share = 0 or 1 merges, 90th percentile), merges per engagement, time from
  start to first damage and to death, length, circles turned per ship, and each merge's
  kind (one-circle: opposite turn senses; two-circle: same; scissors: both reversed since
  the last merge).
- Fleet sizes: `Loop(ships_per_team=1, 2, 5)`, the arena by the density rule (7800 m x
  sqrt(n / 5)); lineups cut to size. Measured in the final.
- **Fog off** in every run from round 8 on (`api.reset`'s fogged flag); earlier rounds ran
  fogged. Team deathmatch's early win (25 kills) disabled: every game runs the clock.
- A first CPU look (r4, 60 s, 4 games): **merges per kill 0.1-0.7, first-pass share ~1.0**:
  kills come on the first pass. The author wants ~3 with spread: that is a lethality
  question. **Round 8** (running, 17:03): lethality on the round-7 card (damage x0.35-0.7,
  shields x1.5-2, damage x0.5 with more frontal armour or a faster recharge).

## 2026-11-12 17:03-17:10 KST: round 8 (lethality, on r7)

**Fog off is not neutral.** The first run of round 8 had fog off (the author's suggestion);
on the unchanged round-7 card r7's energy v turn went 0.507 -> **0.702**, mutual head-on
kills 0.09 -> 0.21, r7 v off 0.861 -> 0.729 (`round8/cards.*`). r7's parameters were
tuned fogged and the game differs materially without fog, so **fog is back on** for every
run (the release game's), and round 8 was rerun fogged (`round8/cards-fog.*`; 177 s):

| Variant (on the round-7 card, both types) | r7 v off kill share | E v T | Kills m / o / f / t | Trades | Life s | Death gap mean (median) | Kills/min/ship | Merges a kill, gun range: mean / median / first-pass / p90 | ... two turn diameters | Engagement to death median s (gun range; 2 TD) | Circles turned (gun; 2 TD) | Merges 1-circle / 2-circle / scissors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round-7 card | 0.646 | 0.507 ± 0.014 | 0.09 / 0.26 / 0.53 / 0.12 | 0.08 | 87 | 7.6 (4.5) | 0.62 | 0.05 / 0 / 1.00 / 0 | 0.28 / 0 / 0.97 / 1 | 1.5; 11.9 | 0.27; 0.60 | 0.33 / 0.64 / 0.03 |
| damage x0.5 | 0.670 | 0.550 | 0.05 / 0.21 / 0.57 / 0.16 | 0.06 | 104 | 8.8 (5.9) | 0.53 | 0.09 / 0 / 0.99 / 0 | 0.42 / 0 / 0.93 / 1 | 2.3; 13.7 | 0.34; 0.78 | 0.32 / 0.62 / 0.05 |
| damage x0.35 | 0.669 | 0.614 | 0.04 / 0.18 / 0.56 / 0.21 | 0.05 | 123 | 9.9 (7.1) | 0.46 | 0.13 / 0 / 0.98 / 0.8 | 0.56 / 0 / 0.89 / 1.8 | 2.5; 15.5 | 0.40; 0.93 | 0.32 / 0.61 / 0.07 |
| damage x0.25 | 0.660 | 0.664 | 0.03 / 0.15 / 0.54 / 0.27 | 0.04 | 154 | 11.7 (9.1) | 0.38 | 0.19 / 0 / 0.96 / 1.0 | 0.76 / 0.2 / 0.82 / 2.0 | 2.5; 18.7 | 0.47; 1.12 | 0.32 / 0.59 / 0.10 |
| damage x0.15 | 0.616 | 0.597 | 0.02 / 0.12 / 0.46 / 0.40 | 0.03 | 244 | 15.6 (12.5) | 0.25 | 0.42 / 0 / 0.89 / 1.6 | 1.27 / 1.0 / 0.64 / 3.0 | 3.9; 28.9 | 0.59; 1.43 | 0.32 / 0.55 / 0.13 |
| damage x0.1 | 0.601 | 0.531 | 0.02 / 0.09 / 0.41 / 0.48 | 0.02 | 398 | 19.3 (15.5) | 0.17 | 0.67 / 0 / 0.81 / 2.0 | 1.74 / 1.8 / 0.49 / 3.8 | 6.7; 38.1 | 0.68; 1.61 | 0.32 / 0.52 / 0.15 |
| damage x0.25, fraction x0.7 | 0.652 | 0.659 | 0.03 / 0.16 / 0.54 / 0.28 | 0.04 | 154 | 11.7 (9.1) | 0.38 | 0.20 / 0 / 0.96 / 1.0 | 0.76 / 0.4 / 0.81 / 2.0 | 2.5; 18.7 | 0.47; 1.13 | 0.32 / 0.58 / 0.10 |
| damage x0.25, cone x2 | 0.644 | 0.604 | 0.03 / 0.08 / 0.55 / 0.34 | 0.04 | 176 | 12.9 (10.1) | 0.33 | 0.26 / 0 / 0.94 / 1.0 | 0.89 / 1.0 / 0.77 / 2.4 | 3.1; 20.7 | 0.51; 1.23 | 0.31 / 0.58 / 0.10 |

- **At today's lethality kills come on the first pass** (first-pass share 0.97-1.00; a
  kill 1.5 s into a gun-range engagement). Lowering the damage lengthens fights, raises
  the tail share (0.12 -> 0.48), lowers trades (0.08 -> 0.02) and mutual head-on (0.09 ->
  0.02), and shifts energy v turn toward Yongma (0.51 -> 0.66 at x0.25).
- **Merges a kill near 3 is not reachable inside the pace band with this agent**: at x0.15
  (life 244 s, 0.25 kills/min/ship, the band's slow end) the mean is 1.27 (2 turn
  diameters; median 1, p90 3, first-pass 0.64); at x0.1 (life 398 s, outside the band)
  1.74. The agent kills from the flank and tail after one pass rather than over repeated
  passes; more passes need the agent to make repeated passes pay (or the game to make the
  first one not decide), not only less damage.
- The head-on TTK target (3-4 passes on the card) pulls the other way: less damage at the
  same frontal armour means more head-on passes on the card (x0.3: ~20 card passes).
- **Round 9** (running, 17:10): ES from r7 on damage x0.3 (Yongma's x0.85 more, as at
  lower damage Yongma gains): life ~140 s, tail ~0.25, to re-tune the agent for slower
  fights; then a card round to rebalance.

## 17:22 KST: the author's regime idea (noted; not run yet)

- The distinctness measure now also splits by regime against the nearest seen enemy:
  own specific energy (v^2/2 + store) ahead or behind, and inside or outside two own
  full-stick turn radii (`distinctness`; the final run reports it). A CPU smoke on r7
  (4 games, 30 s) already shows the split is informative: Jujak ships differ most when
  energy-ahead and inside (KL 7.0, agreement 0.33), Yongma ships least when outside
  (0.3, 0.95); samples tiny.
- **Can today's doctrine express a selector?** Partly. The gate (`strength x
  in_advantage`) is a logistic in the *card* ratios (turn, climb, speed, range), so it
  depends on the matchup but not the situation; only extend's re-engage rule reads the
  live energy ratio, and bleed / deny read the nose angles and range. There is no switch
  between two whole control sets (energy-fight and turn-fight) by energy state and
  relative position, with the type moving the boundary. That **needs code**: a logistic
  selector in the live energy ratio and the range in turn radii (per-type centres) that
  blends two shared element sets. One rebuild; the ES then tunes the boundaries.

## 2026-11-12 17:10-17:37 KST: round 9 (ES on damage x0.3)

Card: round-7 card with both types' damage x0.3 and Yongma's x0.85 more (`round9/card.json`).

- ES from r7, 16 generations: 0.609 -> 0.636 (penalised kill share); KL 1.1.
- **Elo** (this card): **r9-r7 1234 ± 6**, r7 1215 ± 6, r4 1208, r2 1206, hot 1006, off
  1000, r1 991, r1-hand 984, hand 633, random 516. **Improvement over r7: +19 ± 8 Elo
  (~2.3 SE): small but real.** The whole scale compresses at low lethality (random 108 ->
  516: fewer kills a game, more draws), so r9 v off is 0.65-0.88 by lineup rather than
  0.75-0.91. r9 joined the league.
- Distinctness: KL 0.08 on Yongma ships (agreement 0.97), 2.59 on Jujak ships (0.66).
- r9 self-play: energy v turn **0.579 ± 0.010** (Yongma ahead: round 10 rebalances); kills
  mutual 0.03-0.05, offset 0.16-0.19, flank 0.51-0.59, **tail 0.17-0.31**; trades
  0.04-0.06; life 115-189 s; gap between deaths 9.3-13.7 s.

## 17:37-17:40 KST: round 10 cards (balance at damage x0.3, on r9)

| Variant (on the round-9 card) | r9 v off kill share | E v T | Kills m / o / f / t | Trades | Life s | Kills/min/ship | Merges a kill, 2 turn diameters: mean / median / first-pass / p90 |
|---|---|---|---|---|---|---|---|
| round-9 card | 0.668 | 0.618 ± 0.014 | 0.04 / 0.16 / 0.55 / 0.25 | 0.05 | 141 | 0.41 | 0.69 / 0 / 0.84 / 2.0 |
| **Y damage x0.9** | 0.660 | **0.508 ± 0.015** | 0.03 / 0.16 / 0.55 / 0.25 | 0.04 | 148 | 0.40 | 0.72 / 0.4 / 0.83 / 2.0 |
| Y damage x0.8 | 0.672 | 0.469 | 0.03 / 0.16 / 0.55 / 0.26 | 0.04 | 157 | 0.38 | 0.78 / 0.6 / 0.81 / 2.0 |
| Y damage x0.7 | 0.670 | 0.343 | 0.03 / 0.15 / 0.55 / 0.27 | 0.04 | 168 | 0.37 | 0.82 / 0.8 / 0.79 / 2.0 |
| J damage x1.15 | 0.680 | 0.499 | 0.03 / 0.17 / 0.56 / 0.24 | 0.05 | 137 | 0.42 | 0.66 / 0 / 0.85 / 1.8 |
| J damage x1.3 | 0.671 | 0.412 | 0.04 / 0.18 / 0.56 / 0.22 | 0.05 | 132 | 0.44 | 0.62 / 0 / 0.86 / 1.8 |
| Y shield x0.9 | 0.672 | 0.526 | 0.04 / 0.17 / 0.56 / 0.23 | 0.05 | 135 | 0.42 | 0.65 / 0 / 0.85 / 2.0 |
| Y x0.85, J x1.1 damage | 0.678 | 0.457 | 0.03 / 0.16 / 0.55 / 0.25 | 0.04 | 148 | 0.40 | 0.71 / 0.4 / 0.83 / 1.8 |

- Damage is the clean balance knob here (about 0.11 of energy v turn per 10% of Yongma's
  damage). **Card for round 11 and the final: Yongma damage x0.9 more (energy v turn
  0.508)**; both types' damage factors now ~0.30 of the ship files' x 0.65 (Yongma
  0.298, Jujak 0.300), shields Yongma x1.3, Jujak x0.87.
- **Round 11** (running, 17:40, ETA ~18:08): ES from r9 on this card. Then the final
  measurement (~18:10-18:45), the freeze, and the report. **Expected stop ~19:15 KST.**

## 17:40-17:57 KST: round 11 (ES from r9 on the final card): no move

8 generations (108 s each: 7 league opponents), every probation failed (0 passed); the
step fell to 0.01 and the parent stayed r9. **No improvement over r9**: the ES has
converged around this champion; further Elo needs the method changed (a regime selector
in code, a wider/random restart population, or longer games at this lethality), not more
generations. Stopped at 17:57; the final measurement (r9 on the round-11 card, every
league pair 1024 games a pair and order, 300 s pace games at 5v5, 2v2, 1v1) started
17:57, ETA ~18:25.
