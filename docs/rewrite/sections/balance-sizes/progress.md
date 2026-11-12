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

The 1v1 pictures (`images/frozen-1v1-*.png`):

- Mixed duel (`frozen-1v1-evt-1.png`, `-evt-3.png`): a real turning fight. Both ships circle,
  and the Jujak tightens inside and holds the Yongma's tail with a stream of hits until it
  dies. The Yongma never extends.
- Jujak mirror (`frozen-1v1-jj-0.png`): **loop and joust**. A head-on pass, mirror-image
  turn-backs, and another head-on pass, until a nose-on hit stream kills one of them.
- Yongma mirror (`frozen-1v1-yy-0.png`): orbiting, 7.7 circles over 201 s before a kill.
- Under fog, ships that lose sight of each other fly back through the map centre, so the
  long straight legs cross at (0, 0) ("flower petals", `frozen-1v1-evt-3.png`). Noted as the
  fog search, not a fault.
- 2v2 / 5v5 (`frozen-2v2-evt-*.png`): every ship converges on the centre; the first kill
  comes at the first merge.

## 19:27-19:36 KST: round 1 (Yongma's doctrine; four cards)

`round1/sizes.{json,md}`, 512 games a size (10v10 256), E v T only; score / kill share:

| Card \| doctrine | 1v1 | 2v2 | 5v5 | 10v10 |
|---|---|---|---|---|
| frozen \| r9 | 0.01 / 0.195 | 0.40 / 0.469 | 0.56 / 0.506 | 0.51 / 0.497 |
| frozen \| D1 (Yongma's gate open v Jujak) | 0.09 / 0.189 | **0.01 / 0.074** | **0.00 / 0.079** | 0.00 / 0.071 |
| Y engine x1.2 \| r9 | 0.02 / 0.203 | 0.41 / 0.479 | 0.62 / 0.518 | 0.63 / 0.514 |
| Y engine x1.2 \| D1 | **0.14** / 0.224 | 0.01 / 0.082 | 0.00 / 0.092 | 0.00 / 0.082 |
| Y drag x0.8 \| r9 | 0.05 / 0.169 | 0.54 / 0.501 | 0.74 / 0.541 | 0.76 / 0.539 |
| J induced x1.3 \| r9 | 0.11 / 0.228 | 0.62 / 0.522 | 0.82 / 0.555 | 0.82 / 0.555 |

(D2-D4, more extension, were worse than D1 everywhere.)

- **The stored doctrines are gated off on their own ships.** Yongma's gate is a logistic in
  (own / target turn rate)^1.86 with its centre at 1.3. Against Jujak that ratio is 0.63,
  which puts the gate at about 0.0005, and against Yongma it is about 0.06. Jujak's gate
  (centre 1.68) is closed too. So r9's ships fly the parameters alone, which explains the
  game loop's Yongma KL of 0.08.
- Opening the gate (extend, slash and lag against Jujak) is a **disaster in teams**: the
  extending Yongma is shot in the tail (tail kills 0.86-0.93), and its 2v2-10v10 score falls
  to about 0. In the duel it helps a little (0.01 to 0.09, or 0.14 with engine x1.2). At
  cruise 167 against 158 m/s the Yongma cannot outrun a Jujak that turns 29% faster, so
  extension does not buy separation. Not carried forward.
- The card levers move every size together: the kill-share gap from 1v1 to 5v5 stays at
  about 0.3.

## 19:37-19:42 KST: round 2 (16 single card levers, sizes 1, 2, 5)

`round2/sizes.{json,md}`, 512 games; the change in Yongma's kill share from the frozen card:

| Lever | 1v1 score / share | 5v5 score / share | Δ share 1v1 | Δ 2v2 | Δ 5v5 |
|---|---|---|---|---|---|
| frozen | 0.01 / 0.200 | 0.52 / 0.498 | | | |
| **J induced drag x1.6** | **0.66 / 0.543** | 0.96 / 0.594 | **+0.343** | +0.135 | +0.095 |
| Y induced drag x0.7 | 0.19 / 0.281 | 0.83 / 0.562 | +0.081 | +0.074 | +0.063 |
| Y damage x1.3 | 0.03 / 0.271 | 0.74 / 0.534 | +0.071 | +0.037 | +0.036 |
| J shield x0.8 | 0.04 / 0.269 | 0.71 / 0.531 | +0.069 | +0.028 | +0.033 |
| Y shield x1.3 | 0.04 / 0.203 | 0.75 / 0.542 | **+0.003** | +0.033 | +0.044 |
| Y engine x1.5 | 0.02 / 0.166 | 0.71 / 0.532 | **-0.034** | +0.035 | +0.033 |
| Y static thrust x1.5 | 0.03 / 0.148 | 0.65 / 0.522 | -0.052 | +0.025 | +0.024 |
| Y lift x1.2 / J lift x0.85 | 0.02-0.03 | 0.50-0.53 | +0.026 / -0.020 | 0 | 0 |
| head-on fraction, recharge, nose frequency, J bullet life | | | -0.02 to +0.01 | 0 to +0.02 | 0 to +0.02 |

- **The duel is decided by the energy bleed in the turn.** Jujak's induced drag moves the
  duel 3.6 times as far as the crowd. The other levers that favour duels over crowds are Y
  damage and J shield, at about 2:1.
- **Yongma's shield is a crowd-only lever.** It does nothing in the duel and +0.044 at 5v5,
  so it can pull the crowds back without touching the duel. More engine power or static
  thrust *hurts* the Yongma's duel: a faster ship flies wider circles.
- The lift coefficient (the corner, instantaneous turn) barely matters. The sustained turn,
  which bleed sets, is what counts.
- **Round 3** (running, 19:43, ETA 19:58 (worst 20:05)): J induced x1.3 / 1.45 / 1.6 x Y
  shield x1 / 0.8 / 0.65, at 1v1, 2v2, 5v5 and 10v10, energy v turn and both mirrors.

## 19:43-19:54 KST: round 3 (Jujak's bleed x Yongma's shield; with the mirrors)

`round3/sizes.{json,md}`, 512 games a size and lineup (10v10 256). The cells give E v T
score / kill share, then the mirrors' mutual head-on kill share (JJ, YY) and life:

| Card | 1v1 | 2v2 | 5v5 | 10v10 | Worst \|E v T - 0.5\| | YY life 1v1 |
|---|---|---|---|---|---|---|
| frozen | 0.01 / 0.196 | 0.33 / 0.453 | 0.56 / 0.508 | 0.53 / 0.500 | 0.49 | 1001 s |
| J induced x1.3, Y shield x0.65 | 0.04 / 0.246 | 0.35 / 0.459 | 0.49 / 0.500 | 0.51 / 0.499 | 0.46 | 275 s |
| J induced x1.45, Y shield x0.65 | 0.21 / 0.359 | 0.53 / 0.500 | 0.70 / 0.525 | 0.63 / 0.514 | 0.29 | 273 s |
| J induced x1.6, Y shield x1.0 | 0.65 / 0.543 | 0.86 / 0.598 | 0.92 / 0.590 | 0.94 / 0.589 | 0.44 | 927 s |
| J induced x1.6, Y shield x0.8 | 0.62 / 0.526 | 0.73 / 0.552 | 0.86 / 0.560 | 0.83 / 0.552 | 0.36 | 433 s |
| **J induced x1.6, Y shield x0.65** | **0.58 / 0.499** | 0.60 / 0.518 | 0.76 / 0.538 | 0.70 / 0.523 | **0.26** | **280 s** |
| J induced x1.6, Y shield x0.8, Y damage x0.85 | 0.58 / 0.505 | 0.64 / 0.533 | 0.80 / 0.542 | 0.74 / 0.532 | 0.30 | 825 s |
| J induced x1.45, Y shield x0.8, Y engine x0.85 | 0.13 / 0.279 | 0.53 / 0.506 | 0.75 / 0.533 | 0.70 / 0.524 | 0.37 | 359 s |

- The duel switches between Jujak induced x1.45 and x1.6: from there the Jujak's sustained
  turn no longer beats the Yongma's.
- **The duel's shape changes with it.** On the frozen card the mixed 1v1 kills are 0.16 /
  0.13 / 0.41 / 0.30 (m / o / f / t), a turning fight. At J induced x1.6 they are 0.33 /
  0.59 / 0.08 / 0.00: with neither ship able to out-turn the other, the duel becomes a joust,
  like the mirrors.
- Yongma's shield pulls the crowds back (5v5 0.92 to 0.76 from x1 to x0.65) and fixes the
  Yongma mirror's stalemate (1v1 life 927 s to 280 s, inside the 300 s band).
- The Jujak mirror's 1v1 is 0.76-0.80 mutual head-on on every card: card balance does not
  touch the joust.
- Best of the round: J induced x1.6, Y shield x0.65 (worst size 0.26 off 0.5, against 0.49
  frozen; 5v5 still 0.76). This beats the frozen card on the size objective, but 5v5 and
  10v10 are outside the band.
- **Round 4** (running 19:54, ETA 20:06 (worst 20:12)): Y shield x0.4-0.65 against J induced
  x1.5-1.7, Y bullet life, damage and static thrust. **Mirror round** (1v1 and 2v2, running
  alongside): head-on armour (fraction x0.5 / 0.33 / 0.05, angle x1.5) and the deny element
  opened on Jujak or on both types, against the joust.

## 19:54-19:59 KST: round 4 (the shield and bleed grid): **new champion card**

`round4/sizes.{json,md}`, 512 games (10v10 256), E v T score / kill share:

| Card | 1v1 | 2v2 | 5v5 | 10v10 | Worst \|E v T - 0.5\| |
|---|---|---|---|---|---|
| J induced x1.6, Y shield x0.65 | 0.58 / 0.507 | 0.63 / 0.527 | 0.78 / 0.542 | 0.75 / 0.526 | 0.28 |
| J induced x1.6, Y shield x0.5 | 0.52 / 0.484 | 0.50 / 0.490 | 0.67 / 0.522 | 0.50 / 0.500 | 0.17 |
| J induced x1.6, Y shield x0.4 | 0.38 / 0.440 | 0.42 / 0.474 | 0.51 / 0.499 | 0.40 / 0.491 | 0.12 |
| J induced x1.5, Y shield x0.5 | 0.24 / 0.389 | 0.42 / 0.480 | 0.62 / 0.515 | 0.54 / 0.500 | 0.26 |
| J induced x1.55, Y shield x0.45 | 0.32 / 0.400 | 0.43 / 0.479 | 0.56 / 0.507 | 0.43 / 0.492 | 0.18 |
| **J induced x1.7, Y shield x0.5** | **0.57 / 0.500** | **0.47 / 0.490** | **0.50 / 0.501** | **0.45 / 0.489** | **0.07** |
| J induced x1.7, Y shield x0.4 | 0.53 / 0.498 | 0.34 / 0.458 | 0.43 / 0.489 | 0.30 / 0.471 | 0.20 |
| J induced x1.6, Y shield x0.5, Y bullet life x0.85 | 0.53 / 0.478 | 0.48 / 0.490 | 0.58 / 0.511 | 0.48 / 0.488 | 0.08 |
| J induced x1.6, Y shield x0.65, Y bullet life x0.8 | 0.58 / 0.501 | 0.61 / 0.520 | 0.69 / 0.524 | 0.62 / 0.510 | 0.19 |
| J induced x1.6, Y shield x0.5, Y damage x0.9 | 0.47 / 0.459 | 0.46 / 0.485 | 0.50 / 0.500 | 0.40 / 0.485 | 0.10 |
| J induced x1.6, Y shield x0.5, Y static thrust x0.8 | 0.10 / 0.345 | 0.39 / 0.471 | 0.55 / 0.506 | 0.52 / 0.498 | 0.40 |

- **Beat the champion**: J induced x1.7, Y shield x0.5 is within 0.40-0.60 at every size
  (worst 0.07 off 0.5; the frozen card's worst is 0.49, at 1v1). The margin is many SE
  (±0.02 a cell). It needs confirming at 1024 games, which round 5 does.
- Less Yongma static thrust ruins the duel (0.10): its low-speed acceleration is what lets
  it hold energy in the duel.

## 19:55-19:59 KST: mirror round (1v1 and 2v2, on J induced x1.6, Y shield x0.65)

`mirror1/sizes.{json,md}`, 512 games. 1v1 cells: kills m / o / f / t, life, merges a kill
mean / median / first-pass / p90, circles turned a ship:

| Variant | E v T 1v1 / 2v2 | JJ 1v1: kills; life; merges; circles | YY 1v1: kills; life |
|---|---|---|---|
| base | 0.56 / 0.62 | 0.77 / 0.19 / 0.04 / 0; 141 s; 1.8 / 2 / 0.47 / 4; 1.6 | 0.44 / 0.45 / 0.11 / 0; 283 s |
| head-on fraction x0.5 (both) | 0.51 / 0.59 | 0.63 / 0.31 / 0.06 / 0; 255 s; 3.3 / 3 / 0.39 / 7; 2.9 | 0.06 / 0.77 / 0.17 / 0; 910 s |
| head-on fraction x0.33 | 0.52 / 0.62 | 0.35 / 0.58 / 0.07 / 0; 738 s; 3.8 / 3 / 0.26 / 7; 3.3 | 0.03 / 0.79 / 0.18 / 0; 1057 s |
| head-on fraction x0.05 | 0.51 / 0.60 | 0.11 / 0.76 / 0.13 / 0; **1852 s** | 0.01 / 0.73 / 0.26 / 0; 1446 s |
| head-on angle x1.5 | 0.57 / 0.64 | 0.79 / ...; 147 s (no change) | 0.45 / ...; 310 s |
| Jujak's doctrine opened (all of it) | 0.79 / 0.97 | 0 / 0 / 0 / **1.00**; **23 618 s** (broken) | unchanged |
| Jujak's deny opened (deny only) | 0.62 / 0.77 | **0.02** / 0.93 / 0.05 / 0; 375 s; 4.2 / 4 / 0.17 / 8; 3.9 | unchanged |
| **deny opened on both types** | 0.64 / 0.66 | **0.02** / 0.92 / 0.06 / 0; 380 s; 4.2 / 4 / 0.17 / 8; 3.9 | **0.00** / 0.82 / 0.18 / 0; 582 s |
| head-on fraction x0.33 + deny both | 0.59 / 0.67 | 0.01 / 0.95 / 0.05 / 0; 485 s | 0.00 / 0.75 / 0.25 / 0; 778 s |

- **Head-on armour does not stop the joust.** Mutual kills turn into offset kills (the
  nose-on hits become hits just off the nose), and life explodes, because nothing else
  kills in a duel of equals.
- **The deny element**, already in the stored doctrines but gated off, ends mutual head-on
  in duels: 0.77 to 0.02 (JJ) and 0.44 to 0.00 (YY). Ships turn off the line when the
  enemy's nose is on them. The duel gets longer and rounder: 4.2 merges a kill (median 4,
  p90 8, first-pass 0.17) and 3.9 circles a ship. But the kills are still offset (0.82-0.93),
  not flank or tail, and life rises to 380 / 582 s, above the 300 s band.
- In duels between equal turners nobody reaches the tail: there is no move that gains
  angle at equal energy, as the coordinator suspected. That is a doctrine gap (a break or
  reversal to force an overshoot), not a card issue. Not built here.
- Opening Jujak's whole stored doctrine breaks the mirror: its extend element makes both
  ships run, giving 1v1 life of 23 618 s.

## 20:00-20:13 KST: round 5 (confirm at 1024 games; the deny element): **champion confirmed**

`round5/sizes.{json,md}`, 1024 games a size and lineup (10v10 512), E v T score / kill
share, then the mirrors (JJ, YY): mutual head-on kill share and life:

| Card \| doctrine | 1v1 | 2v2 | 5v5 | 10v10 | Worst \|E v T - 0.5\| |
|---|---|---|---|---|---|
| **J induced x1.6, Y shield x0.5, Y bullet life x0.85 \| r9** | **0.51** / 0.480; JJ 0.76, 142 s; YY 0.51, 206 s | **0.53** / 0.500 | **0.55** / 0.505 | **0.47** / 0.492 | **0.05** |
| J induced x1.7, Y shield x0.5 \| r9 | 0.58 / 0.513; JJ 0.78, 145 s; YY 0.52, 200 s | 0.43 / 0.481 | 0.55 / 0.505 | 0.40 / 0.486 | 0.10 |
| J induced x1.6, Y shield x0.5, Y bullet life x0.85 \| deny | 0.61; JJ 0.02, 368 s; YY 0.01, 535 s | 0.53 | 0.65 | 0.60 | 0.15 |
| J induced x1.7, Y shield x0.5 \| deny | 0.80; JJ 0.02, 384 s; YY 0.00, 480 s | 0.63 | 0.75 | 0.74 | 0.30 |
| J induced x1.6, Y shield x0.5, Y damage x0.9 \| deny | 0.60 | 0.52 | 0.66 | 0.54 | 0.16 |

E v T SE: ±0.01 at 1v1, ±0.014 at 2v2 and 5v5, ±0.02 at 10v10.

- **Champion: J induced x1.6, Y shield x0.5, Y bullet life x0.85, flown by r9.** Every size
  is within 0.45-0.55: 0.51 / 0.53 / 0.55 / 0.47. The frozen card's worst size was 0.49 off
  0.5 (1v1 0.01); this card's worst is 0.05.
- **Strength**: the agent is r9 unchanged, so the card changes nothing the agent knows.
  Against r9 on the new card it scores 0.503 / 0.482 (mixed / reversed, 5v5, 1024 games).
- **The deny element costs strength.** Played against r9 at 5v5 it scores **0.25-0.32**
  (about -150 Elo). It ends the mirror joust, but the ships turn away from shots they could
  have traded. **Not carried forward**; a weaker or shorter-range deny is being measured
  (`deny2/`).
- Pace on the champion: life 91-142 s (1v1-10v10 self-play, both mirrors included) and the
  YY 1v1 206 s, all within 45-300 s. Kills a minute a ship 0.53-0.59 at 5v5. Trades
  0.02-0.06.
