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

## 20:10-20:17 KST: deny strength sweep (`deny2/`), on the champion card

Deny opened on both types at a lower weight, a narrower angle or a shorter range. 1v1 JJ
mutual head-on and life; strength is the score against r9 at 5v5 (mixed / reversed, 512
games, ±0.02):

| Deny | JJ 1v1 mutual, life | YY 1v1 mutual, life | v r9 (5v5) |
|---|---|---|---|
| off (r9) | 0.77, 141 s | 0.53, 215 s | 0.52 / 0.53 |
| 0.8, angle 0.39 rad | 0.02, 375 s | 0.01, 543 s | 0.30 / 0.27 |
| 0.4 | 0.42, 564 s | 0.08, 462 s | 0.35 / 0.33 |
| 0.2 | 0.54, 291 s | 0.13, 475 s | 0.39 / 0.43 |
| 0.8, angle 0.2 | 0.42, 506 s | 0.09, 513 s | 0.36 / 0.30 |
| 0.8, range 1.5 gun ranges | 0.02, 377 s | 0.01, 505 s | 0.29 / 0.30 |
| 0.4, range 1.5 | 0.39, 526 s | 0.07, 482 s | 0.34 / 0.31 |

- **Every deny costs strength**: even at weight 0.2 it scores 0.39-0.43 against r9. Only
  the full deny ends the joust, and it costs about 150 Elo. A weak deny halves mutual
  head-on but doubles the duel's length.
- **The Jujak mirror joust is the main remaining fight-shape issue.** With the scripted
  agent, two equal ships in a duel meet nose to nose. The fix is a duel-only behaviour (a
  break or reversal that forces an overshoot, or a deny gated by how many enemies are
  near), which needs code. Left open.

**Incident, 20:16 (fixed):** the freeze's handle solve (`freeze.py`) wrote the Yongma ship
file at 20:16:05 while round 6 and the trajectory job were running, and those read the
card from the files when a size starts. Round 6's 5v5 and 10v10 rows and the trajectory
job's later lineups were therefore on a doubled-down card. They were stopped, the file
was restored, and round 6's 5v5 and 10v10 rerun (`round6/sizes-5-10.*`). The 1v1 and 2v2
rows and the deny sweep started before 20:16 and are clean. From here the freeze writes
only after every GPU run has ended.

## 20:13-20:24 KST: the archetype question; rounds 6-7 (card B: Jujak keeps its sustained edge)

Characteristics (`characteristics.py`; sustained turn = settled full-stick turn):

| Card | Ship | Sustained turn rad/s | Cruise m/s | Top m/s | Corner m/s | Shield | Hit damage | Gun range m |
|---|---|---|---|---|---|---|---|---|
| frozen | Yongma | 0.241 | 167 | 422 | 144 | 142 | 3.34 | 612 |
| frozen | Jujak | 0.311 | 158 | 350 | 125 | 80 | 2.67 | 607 |
| A (J induced x1.6, Y shield x0.5, Y bullet life x0.85) | Yongma | 0.241 | 167 | 422 | 144 | 71 | 3.34 | 536 |
| A | Jujak | **0.232** | **132** | 348 | 125 | 80 | 2.67 | 594 |
| B base (J induced x1.5, Y shield x0.35, Y damage x1.4, J shield x0.75) | Yongma | 0.241 | 167 | 422 | 144 | 50 | 4.68 | 612 |
| B base | Jujak | 0.242 | 139 | 349 | 125 | 60 | 2.67 | 598 |

- **Card A inverts the sustained turn**: the Jujak bleeds to 0.232 rad/s, under the
  Yongma's 0.241, and cruises at 132 m/s against 167. It keeps the better instantaneous
  turn (corner 125 against 144 m/s, lift coefficient +34%) and the lower stall speed. The
  coordinator flags this as likely unacceptable ("Jujak = turn fighter" was decided), so
  A stays a named alternative, not the only result.
- **Card B direction**: Jujak induced x1.45-1.5 keeps it at or just above the Yongma's
  sustained turn (0.242-0.248). Its 1v1 deficit is paid with Yongma damage up and Jujak
  shield down, the duel-weighted levers (about 2:1, round 2).

Round 6 (`round6/sizes.json`, 1v1 and 2v2 rows only; its 5v5 and 10v10 were contaminated
and the rerun was cancelled for round 8): J induced 1.45-1.55 with Y damage x1.0-1.3 and
J shield x0.85-1.0 gives 1v1 0.15-0.40 and 2v2 0.39-0.54. Not enough duel.

Round 7 (`round7/sizes.{json,md}`, 512 games, E v T):

| Card | 1v1 | 2v2 | 5v5 | 10v10 |
|---|---|---|---|---|
| J1.5, Y shield x0.4, Y damage x1.4, J shield x0.75 | 0.50 | 0.65 | 0.84 | 0.86 |
| J1.5, Ys x0.35, Yd x1.4, Js x0.75 | 0.46 | 0.60 | 0.76 | 0.79 |
| J1.5, Ys x0.3, Yd x1.6, Js x0.65 | 0.53 | 0.70 | 0.86 | 0.89 |
| J1.45, Ys x0.35, Yd x1.5, Js x0.7 | 0.48 | 0.69 | 0.83 | 0.83 |
| J1.45, Ys x0.4, Yd x1.4, Js x0.65 | 0.53 | 0.68 | 0.88 | 0.88 |

- With Jujak's sustained edge kept, balancing the duel with damage and shield **spills into
  the crowds** (5v5 0.76-0.91): those levers are only 2:1 duel-weighted, and the Yongma's
  shield, the crowd-only counterweight, is already down to x0.3-0.4.
- **Round 8** (running 20:25, ETA 20:37 (worst 20:45)): the B base (J1.5, Ys x0.35, Yd x1.4,
  Js x0.75) with crowd reducers from round 2: Yongma engine down (round 2: x1.5 gave -0.034
  in the duel and +0.033 at 5v5, so less engine should help the duel and cool the crowds),
  Jujak nose frequency up, and Yongma bullet life down.

## 20:25-20:28 KST: round 8 (crowd reducers on the B base): B does not reach the band

`round8/sizes.{json,md}`, 512 games, E v T:

| B base (J1.5, Ys x0.35, Yd x1.4, Js x0.75) plus | 1v1 | 2v2 | 5v5 | 10v10 | Worst |
|---|---|---|---|---|---|
| nothing | 0.47 | 0.61 | 0.80 | 0.80 | 0.30 |
| **Y bullet life x0.85** | **0.45** | **0.56** | **0.73** | **0.72** | **0.23** |
| J nose frequency x1.3 | 0.38 | 0.53 | 0.75 | 0.77 | 0.27 |
| Y engine x0.75 | 0.26 | 0.49 | 0.57 | 0.61 | 0.24 |
| Y engine x0.75, J nose x1.3 | 0.21 | 0.42 | 0.56 | 0.51 | 0.29 |
| Y engine x0.6 | 0.21 | 0.35 | 0.41 | 0.39 | 0.29 |
| Y shield x0.3, Y engine x0.75 | 0.19 | 0.38 | 0.48 | 0.48 | 0.31 |
| Y engine x0.5 | 0.18 | 0.30 | 0.25 | 0.18 | 0.32 |

- Less Yongma engine cools the crowds, but on this base it **also costs the duel** (0.47
  to 0.26 at x0.75). That is the opposite of round 2's sign on the frozen card: there the
  Yongma lost the turn fight anyway, and here engine power is its sustained turn margin.
- **No B variant gets within 0.42-0.58 at every size.** The best, B with Y bullet life
  x0.85, is 0.45 / 0.56 / 0.73 / 0.72. With the sustained turns at parity, 1v1 needs the
  Yongma's damage up and the Jujak's shield down, and those levers also win the crowds.
- **Conclusion (with this agent):** a duel goes to the better sustained turner. A card
  balanced from 1v1 to 10v10 needs the sustained turns at parity *and* the toughness gap
  closed (card A: J 0.232 v Y 0.241, Yongma shield 142 to 71). Keeping Jujak's sustained
  edge (B: 0.242 v 0.241, essentially parity already) leaves the crowds to the Yongma
  (0.72-0.73). An archetype-true card at every size needs the energy fighter's duel tool
  (extend and return at speed) to work in the agent. Round 1 showed the stored extend
  element loses badly in teams, so that needs a duel-aware doctrine (code), not card
  tuning.
- Final measurement (running 20:28, ETA 20:55 (worst 21:10)): frozen, A and B at 1024
  games, 1v1 to 20v20, every lineup, against r9.

## 20:28-20:44 KST: final measurement (`final/`): frozen, candidate A and the best B

1024 games a size and lineup (10v10 512, 20v20 256), 300 s, flown by r9. Cells give E v T
score / kill share; JJ and YY are the mirrors' mutual head-on kill share and life:

| Card | 1v1 | 2v2 | 5v5 | 10v10 | 20v20 |
|---|---|---|---|---|---|
| frozen | **0.01** / 0.186; JJ 0.80, 156 s; YY 0.32, **927 s** | 0.37 / 0.464 | 0.54 / 0.505 | 0.49 / 0.493 | **0.19** / 0.441 |
| **A** (J induced x1.6, Y shield x0.5, Y bullet life x0.85) | **0.50** / 0.473; JJ 0.76, 144 s; YY 0.51, 207 s | **0.47** / 0.492 | **0.57** / 0.507 | **0.47** / 0.493 | **0.12** / 0.429 |
| B (J induced x1.5, J shield x0.75, Y shield x0.35, Y damage x1.4, Y bullet life x0.85) | 0.43 / 0.461; JJ 0.78, 110 s; YY 0.65, 130 s | 0.57 / 0.512 | 0.76 / 0.532 | 0.75 / 0.527 | 0.27 / 0.470 |

SE: ±0.01 (1v1), ±0.015 (2v2, 5v5), ±0.02 (10v10), ±0.03 (20v20). Strength against r9
(5v5, mixed / reversed): frozen 0.50 / 0.51, A 0.49 / 0.51, B 0.49 / 0.47.

E v T pace and shape (self-play):

| Size | Card | Kills/min/ship | Life s | Trades | Kills m / o / f / t | Merges a kill mean / median / first-pass / p90 | Circles a ship |
|---|---|---|---|---|---|---|---|
| 1v1 | frozen | 0.33 | 172 | 0.00 | 0.15 / 0.12 / 0.42 / 0.31 | 2.9 / 3 / 0.23 / 6 | 2.3 |
| 1v1 | A | 0.26 | 225 | 0.00 | 0.43 / 0.50 / 0.07 / 0.00 | 2.6 / 2 / 0.44 / 6 | 2.3 |
| 1v1 | B | 0.49 | 111 | 0.02 | 0.59 / 0.33 / 0.08 / 0.00 | 1.2 / 1 / 0.79 / 3 | 1.1 |
| 5v5 | frozen | 0.51 | 109 | 0.04 | 0.07 / 0.16 / 0.54 / 0.24 | 0.64 / 0 / 0.86 / 2 | 1.1 |
| 5v5 | A | 0.55 | 99 | 0.05 | 0.08 / 0.19 / 0.55 / 0.18 | 0.35 / 0 / 0.95 / 1 | 0.8 |
| 5v5 | B | 0.63 | 85 | 0.07 | 0.13 / 0.23 / 0.51 / 0.14 | 0.25 / 0 / 0.98 / 1 | 0.6 |
| 20v20 | A | 0.37 | 152 | 0.13 | 0.06 / 0.20 / 0.51 / 0.23 | 0.34 / 0 / 0.94 / 1 | 0.9 |

- **A holds 0.47-0.57 from 1v1 to 10v10**, with no strength cost, pace in band, and the
  Yongma mirror's stalemate gone (927 s to 207 s). Its costs: the inverted sustained turn,
  jousting duels (mixed 1v1 0.43 mutual, JJ 0.76), and fewer merges a kill in crowds
  (5v5 0.35).
- **20v20 fails on every card**, frozen included: Yongma's kill share is 0.43-0.47, a
  0.12-0.27 score at ~70 kills a game. Something about the 20v20 game (the arena twice
  as wide, longer approaches, or the agent's view of 40 ships) favours Jujak independently
  of the card. Not diagnosed; 50v50 not run.
- **Candidate A is saved as a play map**: `src/bnb/game/maps/balance-a.json` (the real-scale
  deathmatch with A's card fields as the map's tuning; the ship files and the play default
  stay the frozen card). Launch: `bnb play --preset balance-a`.

## 20:44-20:51 KST: path run 1 (`paths1/`): anti-joust parameters and head-on mechanics, on A

The author's paths 3 and 6, plus scripted parameters near the joust. 512 games; 1v1 and
5v5; strength is the score against r9 at 5v5 (mixed / reversed, ±0.02):

| On card A | E v T 1v1 / 5v5 | JJ 1v1 mutual, life | YY 1v1 mutual, life | v r9 |
|---|---|---|---|---|
| A | 0.49 / 0.59 | 0.78, 141 s | 0.51, 207 s | 0.47 / 0.50 |
| shoot range 0.7 gun ranges (r9 1.04) | 0.50 / 0.58 | 0.79, 150 s | 0.53, 256 s | **0.36 / 0.38** |
| shoot range 0.5 | 0.53 / 0.53 | 0.82, 163 s | 0.54, 327 s | **0.13 / 0.10** |
| aim cone 0.6 target sizes (r9 1.01) | 0.55 / 0.59 | 0.77 | 0.54 | 0.48 / 0.49 |
| aim cone 2.0 | 0.49 / 0.57 | 0.75 | 0.51 | 0.49 / 0.52 |
| reverse range 0.5 (r9 0.21) | 0.70 / 0.76 | 0.78 | 0.59 | 0.45 / 0.42 |
| side commit 2.5 / 10 stall angles (r9 5.0) | 0.53 / 0.55; 0.53 / 0.56 | 0.74; 0.80 | 0.49; 0.54 | 0.44 / 0.47; 0.50 / 0.50 |
| aggression 1.5 (r9 2.84) | 0.51 / 0.61 | 0.78 | 0.52 | 0.53 / 0.50 |
| lead cap 2 gun ranges (r9 4.78) | 0.54 / 0.55 | 0.77 | 0.50 | 0.51 / 0.54 |
| head-on angle x0.5, both | 0.52 / 0.62 | 0.74 | 0.51 | 0.52 / 0.45 |
| **Jujak head-on fraction x0.5** | 0.40 / 0.49 | **0.64, 256 s** | 0.50, 224 s | 0.48 / 0.49 |

- **No fire-discipline or geometry parameter moves the joust** (JJ 0.74-0.82). Holding fire
  to a shorter range only loses strength (0.10-0.38 against r9). The joust is how two
  equal ships under this agent meet, not a shooting habit.
- Per-type frontal armour (Jujak's head-on fraction x0.5) is the only card lever that bites
  (JJ 0.78 to 0.64, life still 256 s). It also costs Yongma's duel (0.49 to 0.40), because
  nose-on is how A's Yongma wins.
- Path 4 (an ES with a mirror-joust penalty) was not run. The ES (`doctrine_round.py`)
  plays 5v5 only and would need a 1v1 lineup in its loop; with the deny element's -150 Elo
  as the measured cost of anti-joust behaviour, a duel-gated deny (code) looks more
  promising. Left open.
- Candidate B is saved as the play map `balance-b` (`bnb play --preset balance-b`).

## 20:52-20:57 KST: path run 2 (`paths2/`): top-end speed and Yongma's range

512 games, E v T:

| Card | 1v1 | 2v2 | 5v5 | 10v10 |
|---|---|---|---|---|
| J1.5 Js x0.75 \| Ys x0.35 Yd x1.4, Y parasitic drag x0.7 | 0.48 | 0.64 | 0.82 | 0.88 |
| J1.4 \| Ys x0.5, Y parasitic drag x0.6 | 0.10 | 0.33 | 0.53 | 0.54 |
| J1.3 \| Ys x0.5, Y parasitic drag x0.5 | 0.03 | 0.27 | 0.45 | 0.39 |
| J1.45 \| Ys x0.4, Y parasitic drag x0.6, Y engine x1.3 | 0.33 | 0.42 | 0.63 | 0.59 |
| J1.5 Js x0.75 \| Ys x0.35 Yd x1.4, Y bullet life x0.7 | 0.39 | 0.54 | 0.70 | 0.64 |
| J1.5 Js x0.75 \| Ys x0.4 Yd x1.4, Y bullet life x0.6 | 0.43 | 0.52 | 0.68 | 0.69 |

- **Top-end speed does not win duels** (path 2). A Yongma with half the parasitic drag
  still loses the 1v1 to a gentler Jujak (x1.3: 0.03, x1.4: 0.10), and its speed lifts the
  crowds. With this agent the Yongma never uses speed to disengage and re-engage, so speed
  only widens its circles.
- **Yongma's range is a crowd lever** (path 1's "range"): bullet life x0.6-0.7 cuts 5v5 from
  0.80 to 0.68-0.70 and the duel less (0.47 to 0.39-0.43). Path run 3 pushes it (x0.5).

## 20:57-21:05 KST: path run 3 (`paths3/`): range (C) and head-on (D) flavours

512 games, E v T, with the Jujak mirror's 1v1 mutual head-on and life:

| Card | 1v1 | 2v2 | 5v5 | 10v10 | JJ 1v1 |
|---|---|---|---|---|---|
| C: J1.5 Js x0.75 \| Ys x0.4 Yd x1.4 Yb x0.5 | 0.37 | 0.48 | 0.60 | 0.52 | 0.77, 109 s |
| C: J1.5 Js x0.8 \| Ys x0.45 Yd x1.5 Yb x0.5 | 0.39 | 0.53 | 0.65 | 0.49 | 0.78, 114 s |
| C: J1.5 Js x0.75 \| Ys x0.4 Yd x1.5 Yb x0.55 | 0.43 | 0.56 | 0.72 | 0.63 | 0.79, 110 s |
| C: J1.45 Js x0.75 \| Ys x0.4 Yd x1.5 Yb x0.5 | 0.30 | 0.45 | 0.55 | 0.48 | 0.80, 109 s |
| D: A + J fraction x0.5, Y damage x1.1 | 0.40 | 0.43 | 0.55 | 0.50 | 0.63, 259 s |
| **D: A + both fractions x0.5, Y damage x1.1** | **0.50** | **0.51** | 0.64 | 0.58 | **0.64, 255 s** |
| D: A + J fraction x0.4, Y damage x1.2 | 0.41 | 0.45 | 0.63 | 0.53 | 0.58, 431 s |

- **C (range)**: a short-ranged, hard-hitting, fragile Yongma (bullet life x0.5: gun range
  about 300 m against Jujak's 600; damage x1.4-1.5; shield x0.4) against a Jujak that keeps
  its sustained edge. The crowds come close (5v5 0.55-0.65, 10v10 0.48-0.55), but the duel
  stays at 0.30-0.43.
- **D (head-on)**: halving the frontal fraction on both types (nose-on hits do 15% damage,
  not 30%) makes the joust pay half. Mixed 1v1 mutual kills fall from 0.43 to 0.21, JJ
  1v1 from 0.76 to 0.64 with life 255 s (in band), and 5v5 JJ from 0.08 to 0.04.
- Path run 4 (running 21:05, ETA 21:25 (worst 21:35)) refines C and D (Yongma range and
  shield for D's crowds; damage and shield for C's duel) with both mirrors and strength.

## 21:05-21:13 KST: path run 4 (`paths4/`): C and D refined; 20v20 spot check

512 games, E v T, both mirrors (1v1 mutual head-on kill share and life), strength (5v5
against r9, mixed / reversed):

| Card | 1v1 | 2v2 | 5v5 | 10v10 | JJ 1v1 | YY 1v1 | v r9 |
|---|---|---|---|---|---|---|---|
| A (reference) | 0.49 | 0.48 | 0.61 | 0.47 | 0.76, 144 s | 0.48, 219 s | 0.50 / 0.48 |
| **D**: A + both head-on fractions x0.5, Y damage x1.1, Y bullet life x0.75 | **0.51** | **0.50** | **0.58** | **0.54** | **0.66, 262 s** | **0.21**, 482 s | 0.51 / 0.54 |
| D: A + both fractions x0.5, Y damage x1.1, Y shield x0.45 | 0.52 | 0.47 | 0.59 | 0.49 | 0.67, 253 s | 0.25, 342 s | 0.53 / 0.46 |
| D: A + both fractions x0.5, Y damage x1.15, Y bullet life x0.7 | 0.52 | 0.54 | 0.59 | 0.51 | 0.65, 253 s | 0.17, 455 s | 0.49 / 0.50 |
| C: J1.5 Js x0.8 \| Ys x0.4, Yd x1.6, Yb x0.45 | 0.33 | 0.48 | 0.60 | 0.45 | 0.78, 116 s | 0.65, 195 s | 0.50 / 0.53 |
| **C**: J1.5 Js x0.8 \| Ys x0.45, Yd x1.6, Yb x0.45 | **0.41** | **0.53** | 0.64 | **0.51** | 0.77, 115 s | 0.62, 203 s | 0.51 / 0.53 |
| C: J1.5 Js x0.75 \| Ys x0.4, Yd x1.6, Yb x0.45 | 0.37 | 0.49 | 0.62 | 0.47 | 0.77, 110 s | 0.66, 190 s | 0.47 / 0.50 |

20v20 spot check (256 games): D 0.10, C 0.07, the same Jujak lean as every card (frozen
0.19, A 0.12, B 0.27). A diagnosis is running (`diag20/`: the frozen card at 20v20 in the
10v10 arena, without fog, and in 600 s games).

## The playtest set (opt-in maps; the play default stays the frozen card)

Each candidate is a saved project map: the real-scale deathmatch game with the
candidate's Yongma and Jujak card fields as the map's tuning
(`src/bnb/game/maps/balance-*.json`, written by `make_map.py`). The ship files are
untouched, and all four candidates fly r9's doctrines and parameters, so the scripted
agent is the same everywhere. Ship files in handle form for A and B are in `cards/A/` and
`cards/B/` (the variant's flown handles; the Jujak re-solve does not converge, so its
[card] is the variant card itself).

| | A (fallback) | B | C | D |
|---|---|---|---|---|
| Launch | `bnb play --preset balance-a` | `--preset balance-b` | `--preset balance-c` | `--preset balance-d` |
| Card (factors on the frozen card) | J induced drag x1.6; Y shield x0.5, bullet life x0.85 | J induced x1.5, shield x0.75; Y shield x0.35, damage x1.4, bullet life x0.85 | J induced x1.5, shield x0.8; Y shield x0.45, damage x1.6, bullet life x0.45 | A + head-on fraction x0.5 on both; Y damage x1.1, bullet life x0.75 |
| Balanced through | energy (Jujak bleeds) and Yongma's toughness | damage and shield | range: a short-ranged heavy-hitting Yongma | head-on mechanics on top of A |
| Sustained turn Y / J (rad/s) | 0.241 / **0.232** | 0.241 / 0.242 | 0.241 / 0.242 | 0.241 / **0.232** |
| Cruise Y / J (m/s); top | 167 / 132; 422 / 348 | 167 / 139; 422 / 349 | 167 / 139; 422 / 349 | 167 / 132; 422 / 348 |
| Shield Y / J; hit damage Y / J | 71 / 80; 3.34 / 2.67 | 50 / 60; 4.68 / 2.67 | 64 / 64; 5.35 / 2.67 | 71 / 80; 3.68 / 2.67 |
| Gun range Y / J (m) | 536 / 594 | 536 / 598 | **286** / 598 | 483 / 594 |
| E v T 1v1 / 2v2 / 5v5 / 10v10 | **0.50 / 0.47 / 0.57 / 0.47** | 0.43 / 0.57 / **0.76 / 0.75** | **0.39** / 0.51 / **0.61** / 0.49 | **0.53 / 0.52 / 0.56 / 0.49** |
| 20v20 | 0.12 | 0.27 | 0.07 | 0.10 |
| Strength v r9 (5v5) | 0.49 / 0.51 | 0.49 / 0.47 | 0.51 / 0.53 | 0.51 / 0.54 |
| Mixed 1v1 kills m / o / f / t | 0.43 / 0.50 / 0.07 / 0 | 0.59 / 0.33 / 0.08 / 0 | 0.54 / 0.36 / 0.10 / 0 | **0.17** / 0.74 / 0.08 / 0 |
| JJ 1v1 mutual, life | 0.76, 144 s | 0.78, 110 s | 0.77, 115 s | **0.66, 262 s** |
| YY 1v1 mutual, life | 0.51, 207 s | 0.65, 130 s | 0.62, 203 s | **0.21**, 482 s |
| Merges a kill, mixed 1v1 (mean / median / first-pass / p90) | 2.6 / 2 / 0.43 / 7 | 1.2 / 1 / 0.79 / 3 | 2.1 / 1 / 0.58 / 6 | **3.6 / 3 / 0.28 / 8** |
| Life s (1v1-10v10, all lineups) | 91-224 | 81-130 | 86-203 | 96-308; YY 1v1 482 |
| Kills/min/ship; trades | 0.26-0.59; 0-0.11 | 0.43-0.66; 0.02-0.14 | 0.28-0.63; 0.01-0.13 | 0.12-0.57; 0-0.10 |
| Source (games) | `final/` 1024 | `final/` 1024 | `confirm/` 1024 (balance), `paths4/` 512 (the rest) | `confirm/` 1024 (balance), `paths4/` 512 (the rest) |

How each plays:

- **A, the fallback (energy).** The Jujak keeps the tighter instantaneous turn (corner 125
  against 144 m/s, lift +34%), but it bleeds: below corner speed it cannot hold a turn, and
  it cruises at 132 m/s. The Yongma is faster everywhere and now holds the better sustained
  turn, at half its old shield. In a duel the Yongma takes wider, faster loops and meets
  the Jujak nose to nose. Mixed duels are jousts with some turning (2.3 circles a ship),
  and the Jujak mirror is loop-and-joust (`images/A-1v1-evt-2.png`,
  `images/A-1v1-jj-1.png`). The archetype is inverted in sustained turn.
- **B (damage and shield).** The Jujak keeps a hair of sustained edge (0.242 against 0.241)
  and the Yongma becomes a glass cannon (shield 50, damage x1.4). Duels are short
  head-on trades (1.2 merges a kill, 79% first-pass), and in crowds the Yongma's damage
  wins (5v5 0.76). Fast, lethal, jousty (`images/B-1v1-evt-0.png`). Out of band in crowds.
- **C (range).** The same Jujak as B; the Yongma is a short-ranged brawler (286 m, damage
  x1.6, shield 64). It must close inside the Jujak's guns to hurt it; the duels are
  two-circle fights ending nose to nose (8 circles in `images/C-1v1-evt-0.png`). Near the
  band except 1v1 (0.39) and 5v5 (0.61): the duel and the crowd pull its numbers apart.
- **D (head-on).** A's ships with half the frontal fraction on both (a nose-on hit does
  15% damage, not 30%) and a slightly harder-hitting, shorter-ranged Yongma. The joust
  stops paying, so duels go round: 3.6 merges a kill (median 3, first-pass 0.28, p90 8),
  mixed 1v1 mutual head-on 0.17, JJ 0.66. The fights are long turning fights, some up to
  11 circles (`images/D-1v1-evt-0.png`), and the Yongma mirror's life is 482 s. **D is the
  only candidate that meets the balance and moves the fight shape**, at the cost of A's
  inverted sustained turn and long duels.

**Ranking of the paths** (what moved what):

1. **Jujak's energy bleed** (induced drag): the only lever that moves the duel much more than
   the crowd (3.6:1). The duel flips at sustained-turn parity (x1.45-1.6).
2. **Yongma's shield**: crowd-only, which makes it the counterweight to anything that
   helps the Yongma's duel.
3. **Head-on fraction (both types)**: halves the joust's pay. Mixed duel mutual kills
   0.43 to 0.17 and JJ 0.76 to 0.66 in D, at no strength cost; mirror life rises.
4. **Yongma's range** (bullet life): mostly a crowd lever (5v5 -0.1 to -0.15 for x0.5-0.6).
5. **Damage and Jujak shield**: duel-weighted only 2:1, so they spill into the crowds.
6. **The deny element**: ends the joust (0.02), but costs about 150 Elo at every weight that
   matters.
7. **No effect, or harmful**: top-end speed and engine (they widen the Yongma's circles and
   lose the duel), extension doctrines (tail-shot in teams), fire discipline (costs
   strength), aim cone, side commit, aggression, lead cap, head-on angle.

None of the candidates keeps Jujak's sustained edge *and* holds 0.42-0.58 at every size:
B and C come closest and lean Yongma at 5v5. Every card fails 20v20 (0.07-0.27), frozen
included.

## 21:17-21:24 KST: why 20v20 fails (`diag20/`): the arena's size, not the fleet

The frozen card, E v T, 256 games (±0.03), 300 s unless noted:

| Game | Arena radius | E v T |
|---|---|---|
| 10v10, the density rule | 11.0 km | 0.49-0.53 |
| **10v10 in the 20v20 arena** | 15.6 km | **0.13** |
| 20v20, the density rule | 15.6 km | 0.19 |
| **20v20 in the 10v10 arena** | 11.0 km | **0.66** |
| 20v20, no fog | 15.6 km | 0.41 |
| 20v20, 600 s games | 15.6 km | 0.11 |

- **The energy v turn result follows the arena's radius**, not the ship count: the same
  10v10 goes from even to 0.13 in the wider arena, and 20v20 goes from 0.19 to 0.66 in the
  narrower one. Fog is part of it (0.19 to 0.41 without fog). Longer games make it worse,
  so it is not a slow start. A likely mechanism: in a wide arena under fog, ships arrive
  and search one by one (the fog search flies back through the centre), and the scattered
  fights are duels, which the Jujak wins. Not tested further.
- **For the zero-shot plan this matters more than the fleet size.** Map size moves the
  balance as much as 1v1 against 5v5 does. The density rule ties the two together, so a
  "20v20 balance" is really a map-size balance. The size table should be read per arena
  radius, and the map-size generalization (training on smaller and larger maps) will
  meet the same lean.

**Confirm at 1024 games (`confirm/`, 21:25-21:28):** D 0.53 / 0.52 / 0.56 / 0.49 (JJ 1v1
0.64 mutual, 256 s; YY 0.20, 484 s); C 0.39 / 0.51 / 0.61 / 0.49 (JJ 0.78). D holds the
band at every size from 1v1 to 10v10; C misses at 1v1 and 5v5 by about 0.1.
