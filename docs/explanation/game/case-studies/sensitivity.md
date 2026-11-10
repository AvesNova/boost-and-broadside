---
covers: [bench/case_study.py]
---

# Sensitivity: the Elo of every ship knob, and trade-off pairs

What each value of a ship is worth in rating, one at a time, around one airframe, and whether
two changes that should cancel do. Everything is the [case-study harness](../case-studies.md)
on the La-7 (the design base; its own variant is the baseline, 0 Elo) on the current real-scale
deathmatch set: every variant a type slot, every ship of a team the slot's type, the scripted
agent on both sides, the league's A-optimal tournament over every pair. Elo is against the
unchanged La-7 of the same run, ± one standard error. 112,896 games in all.

The runs ([guns](sensitivity-guns.json), [armour and body](sensitivity-armour.json),
[flight and store](sensitivity-flight.json), [the steep values in small steps](sensitivity-fine.json),
[pairs](sensitivity-pairs.json)) each hold the baseline, so each has its own anchor;
reports are in the run directories beside the specs.

## One at a time

A knob here is a design knob (`bnb.game.ship_design.KNOBS`) or a handle the design search does
not move. "Elo per doubling" is the weighted least-squares slope through the baseline over every
point of the knob and, after the slash, over the points within 12% of it only (where it was
measured there); the last column is the change that moves the rating about 25 Elo near the
baseline, from the second slope. Where that column is empty the knob is flat or a step
(see the notes).

| muzzle speed | x0.9 -181 ± 8; x0.95 -98 ± 8; x0.98 -31 ± 14; x1.02 +50 ± 13; x1.05 +114 ± 8; x1.1 +210 ± 9 | +1364 / +1364 | 1.3% |
| damage per second | x0.9 -148 ± 8; x0.95 -115 ± 8; x0.975 -12 ± 13; x1.025 +28 ± 13; x1.05 +4 ± 8; x1.075 +162 ± 15; x1.1 +154 ± 8 | +1024 / +1024 | 1.7% |
| 5 Hz, damage/s held |  -196 ± 8 | | |
| 6 Hz, damage/s held |  -128 ± 8 | | |
| 15 Hz, damage/s held |  +64 ± 8 | | |
| 15 Hz, damage/bullet held |  +491 ± 12 | | |
| 6 Hz, damage/bullet held |  -941 ± 20 | | |
| dispersion | x0.5 -3 ± 8; x0.7 +5 ± 8; x1.4 -2 ± 8; x2 -8 ± 8 | -3 / +nan | nan% |
| shot cost | x0.5 -62 ± 8; x0.7 -20 ± 8; x1.4 +11 ± 8; x2 +18 ± 8 | +38 / +nan | nan% |
| shield | x0.9 -133 ± 8; x0.95 -129 ± 8; x0.97 -5 ± 14; x0.98 +5 ± 14; x1.02 +29 ± 13; x1.03 +9 ± 14; x1.05 +132 ± 8; x1.1 +138 ± 8 | +1076 / +1076 | 1.6% |
| shield recharge rate | x0.5 -15 ± 8; x0.7 +4 ± 8; x1.4 +17 ± 8; x2 +24 ± 8 | +18 / +nan | nan% |
| shield recharge delay | x0.5 +15 ± 8; x0.7 +15 ± 8; x1.4 -26 ± 8; x2 -29 ± 8 | -26 / +nan | nan% |
| head-on fraction | x0.5 +1282 ± 20; x0.75 +621 ± 12; x0.9 +119 ± 14; x0.95 +142 ± 14; x1.05 -13 ± 14; x1.1 -132 ± 14; x1.5 -387 ± 10; x2 -587 ± 13 | -820 / -903 | 1.9% |
| hit radius | x0.8 +136 ± 8; x0.9 +50 ± 8; x0.95 +52 ± 14; x1.05 +0 ± 14; x1.1 -47 ± 8; x1.25 -89 ± 8 | -348 / -339 | 5.1% |
| spawn energy | x0.5 -24 ± 8; x2 -12 ± 8 | +6 / +nan | nan% |
| stall speed | x0.8 -107 ± 17; x0.9 -76 ± 16; x0.95 -31 ± 14; x1.05 +27 ± 14; x1.1 +65 ± 16; x1.25 +111 ± 17 | +368 / +465 | 3.7% |
| cruise speed | x0.8 -34 ± 17; x0.9 -1 ± 17; x1.1 +4 ± 16; x1.25 +16 ± 17 | +67 / +17 | 102.0% |
| turn rate | x0.9 -98 ± 17; x0.95 -48 ± 16; x0.975 -23 ± 13; x1.025 +39 ± 14; x1.05 +38 ± 17; x1.1 +110 ± 17 | +700 / +700 | 2.5% |
| climb rate | x0.5 -1 ± 17; x0.7 +8 ± 17; x1.4 -11 ± 16; x2 -31 ± 17 | -16 / +nan | nan% |
| store | x0.5 -76 ± 17; x0.7 -4 ± 16; x1.4 +10 ± 17; x2 +27 ± 16 | +43 / +nan | nan% |
| half-lift drag | x0.5 +55 ± 17; x0.7 +37 ± 17; x1.4 -30 ± 16; x2 -74 ± 17 | -65 / +nan | nan% |
| engine knee | x0.6 -9 ± 17; x1.4 +11 ± 16; x1.8 +15 ± 16 | +16 / +nan | nan% |
| coast exchange | x0.5 +21 ± 16; x2 -47 ± 16 | -34 / +nan | nan% |

Reading the rows:

- **Fire-rate rows.** "damage/s held" keeps the damage per second (the damage per hit follows):
  5 Hz -196, 6 Hz -128, 15 Hz +64, about +110 to +200 Elo a doubling of the rate. "damage/bullet
  held" changes the damage per second with the rate (15 Hz with 1.5 times the damage per second
  of 10 Hz: +491; 6 Hz with 0.6 times: -941); those two rows are the damage per second's
  steepness, not the fire rate's. Only whole-hertz divisors of 30 were used.
- **Shot cost.** The rows are multipliers of `shots_per_store`, so "x2" is half the cost per
  shot and "x0.5" twice the cost.
- **Steps.** The shield, the damage per second and the head-on fraction are not smooth near the
  baseline: they move in steps of about 130 to 150 Elo with plateaus of ±10 between them. The
  shield is flat from x0.97 to x1.03 (-5 to +29), -130 at x0.9 and x0.95, +130 at x1.05 and
  x1.1: one step between x0.95 and x0.97, one between x1.03 and x1.05. The damage per second
  is -115 at x0.95, about 0 from x0.975 to x1.05, +160 at x1.075 and x1.1. The head-on fraction
  is +120 to +140 at x0.9 and x0.95, -13 at x1.05, -132 at x1.1, then -387 and -587 at x1.5 and x2
  (and +621, +1282 at x0.75, x0.5). These are the hits a kill takes (`ceil(shield / hit
  damage)`), as in the [armour study](../case-studies.md#shield-recharge-and-damage-per-hit);
  the slopes per doubling are averages over steps, and no change of less than the step's width
  moves a rating inside a plateau.
- **Cliffs.** The store is flat from x0.7 up (-4, +10, +27) and -76 at x0.5; the climb rate is
  flat to x1.4 (x2: -31); the engine knee does nothing (-9 to +15).
- **Asymmetric.** The half-lift drag: x0.5 +55, x2 -74 (about 65 a doubling, close to
  symmetric); the coast exchange: x0.5 +21, x2 -47; the hit radius: x0.8 +136, x1.25 -89 (a
  smaller radius gains more than a larger one loses); the stall speed: x0.8 -107, x1.25 +111
  (a faster stall is worth more in rating at x1.1 to x1.25 than expected from the small steps
  alone: x1.05 gives +27, x1.1 +65, x1.25 +111).
- **Direction.** More stall speed, turn rate, muzzle speed, damage, shield, store and cruise are
  better; a larger hit radius, head-on fraction, half-lift drag, coast exchange and recharge
  delay are worse. The climb rate (the store's initial charge over the set level) is slightly
  negative, the spawn energy is not monotone (both x0.5 and x2 lose, -24 and -12: within about
  two standard errors of zero each).

### Ranked, in Elo per doubling

| Elo per doubling | Knobs |
|---|---|
| about +1,400 | muzzle speed (steep and smooth: 1.3% is 25 Elo) |
| +1,000 to +1,100 (steps) | shield, damage per second (1.6 to 1.7% is a step of 130 to 150) |
| about -820 to -900 (steps) | head-on fraction |
| +700 | sustained turn rate (2.5% is 25 Elo) |
| +370 to +465 | stall speed (about 4%) |
| about -350 | hit radius (5%) |
| about +110 to +200 | fire rate at a constant damage per second |
| about 40 to 70 | half-lift drag (-65), cruise speed (+67, but +17 within 12%), store (+43, a cliff below x0.7), shot cost (+38) |
| under 35 | coast exchange (-34), recharge delay (-26), recharge rate (+18), climb rate (-16), engine knee (+16), spawn energy (about 0), dispersion (-3) |

### The recommended step for about ±25 Elo

| Knob | Step |
|---|---|
| muzzle speed | 1.3% |
| turn rate | 2.5% |
| stall speed | 4% |
| hit radius | 5% |
| half-lift drag | about 30% (x1.3 or x0.77) |
| coast exchange | about 65% (x1.65) |
| shot cost | x1.6 on the shots per store |
| recharge delay | about x2 |
| shield, damage per second, head-on fraction | no ±25 step: a step is 130 to 150 Elo (shield 4 to 5% up, 4% down; damage per second +6% up, -4% down); a finer one has to move the hit damage across a threshold of the opponents' |
| fire rate | no finer than the next whole divisor of 30 (a step of about 60 to 130 Elo) |
| store, climb, knee, dispersion, recharge rate, spawn energy, cruise | none: they move the rating by under 30 Elo across a factor of two or more (store: not below x0.7); good for making types feel different at the same rating |

The rounded steps carry the slopes' uncertainty (the standard errors are 8 to 17 Elo per
point): near the baseline a step of "25 Elo" is a coin toss against a ±12 reading.

## Trade-off pairs

Pairs of changes, one worth about +50 (or +100 to +150) Elo and one about -50, chosen from the
phase-1 readings (linearly interpolated on the log of each factor between the baseline and
the tried values), mixed across guns, flight and body; the baseline and ten of the single
changes with them: 29 slots, every pair played, 18,432 games, standard error 11 on every slot,
at least 64 games a pair ([run](sensitivity-pairs/report.md)).

The prediction is the sum of the two single readings.

| Pair (A up, B down) | Predicted from phase 1 | Measured |
|---|---|---|
| muzzle x1.02 + turn x0.95 | +2 | -3 ± 11 |
| muzzle x1.02 + stall x0.92 | -7 | +5 ± 11 |
| muzzle x1.02 + hit radius x1.1 | +3 | -11 ± 11 |
| turn x1.03 + muzzle x0.97 | -15 | -33 ± 11 |
| turn x1.03 + hit radius x1.1 | -9 | -24 ± 11 |
| turn x1.03 + half-lift drag x1.7 | -16 | -37 ± 11 |
| stall x1.08 + half-lift drag x1.7 | -4 | +3 ± 11 |
| stall x1.08 + turn x0.95 | +2 | -24 ± 11 |
| hit radius x0.95 + muzzle x0.97 | -1 | -43 ± 11 |
| hit radius x0.95 + stall x0.92 | -6 | -29 ± 11 |
| 15 Hz + turn x0.95 | +16 | +23 ± 11 |
| 15 Hz + hit radius x1.1 | +17 | +0 ± 11 |
| muzzle x1.05 + turn x0.9 | +16 | +31 ± 11 |
| turn x1.1 + muzzle x0.95 | +12 | -75 ± 11 |
| hit radius x0.8 + turn x0.9 | +38 | +9 ± 11 |
| shield x1.05 + head-on x1.1 | -0 | -9 ± 11 |
| damage/s x1.1 + shield x0.95 | +25 | +13 ± 11 |
| stall x1.1 + hit radius x1.25 | -24 | -76 ± 11 |

Additivity:

- **Against the phase-1 prediction**: the mean of measured less predicted is -18 Elo and the
  RMS 30 over 18 pairs. With every reading carrying about 8 to 17 Elo of error, the sum of two
  has about 15 to 20: the RMS is 1.5 to 2 times that, and the sign is mostly negative. The pairs
  are a little worse than the sum.
- **Against the pair's own two singles in the same run** (10 pairs): measured less sum, mean
  +10, RMS 21 (the singles run a few Elo lower than the readings: below).
- **The large misses** (measured less predicted): turn x1.1 with muzzle x0.95, -87 (about
  five standard errors of the pair's 17); stall x1.1 with hit radius x1.25, -53; hit radius
  x0.95 with muzzle x0.97, -42; hit radius x0.8 with turn x0.9, -29. Every miss beyond 25 is
  negative; the two largest are the largest changes. 13 of 18 pairs are within 25 Elo of the
  prediction and 14 of 18 fall below it. The pairs run from -76 to +31 Elo (the +50/-50 ones
  from -43 to +23).
- The model "the sum of the readings" is good to about ±30 Elo for changes of about 50 and
  worse for about 100 each; a built type needs a rating check.
- The single changes in this field read -30 to +10 lower than in phase 1 for the small steps
  (muzzle x1.02 +30 against +50, hit radius x0.95 +24 against +52, half-lift drag x0.5 +28
  against +55, muzzle x0.97 -82 against -53): the readings' interpolation and the 11 Elo
  standard error between them.

| Single knob | Phase-1 reading (interpolated) | Measured in this field |
|---|---|---|
| muzzle x1.02 | +50 | +30 ± 11 |
| muzzle x0.97 | -53 | -82 ± 11 |
| turn x1.03 | +38 | +22 ± 11 |
| turn x0.95 | -48 | -43 ± 11 |
| hit radius x0.95 | +52 | +24 ± 11 |
| hit radius x1.1 | -47 | -55 ± 11 |
| stall x1.08 | +50 | +52 ± 11 |
| stall x0.92 | -58 | -65 ± 11 |
| half-lift drag x0.5 | +55 | +28 ± 11 |
| half-lift drag x1.7 | -54 | -56 ± 11 |

### Cycles and the cyclic share

71 majority 3-cycles among the 29 slots (a pair of 64 to 700 games, so many scores are
noise-level). Their weakest edges are low: one cycle has every edge at 0.57 or more
(muzzle x1.02 with turn x0.95, -3 Elo, beats 15 Hz with hit radius x1.1, 0, at 0.578; that beats
hit radius x0.8 with turn x0.9, +9, at 0.594; and that beats the first at 0.578). The
three ratings are within 12 Elo of each other, the edges 1.3 to 1.5 standard errors above one half
(64 to 100 games a pair), so it is the kind of cycle that 29 near-equal slots produce by chance.
The next strongest cycles' weakest edges are 0.562 and below.

The league's cross-validated model (`bnb.league.cyclic.cross_validate`, 5 folds) picks a
transitive one (probit link, k = 0): there is no cyclic term, the cyclic share is 0. A
one-dimensional cyclic fit on a Bradley-Terry world with the same games (20 simulations, the
penalty 3.2) gives a cyclic share of 0.13 on average and up to 0.19 from noise alone, so the
counts show nothing beyond that. In this airframe, with one change in each, the knob space of
the La-7 is transitive: the changes that cancel in Elo cancel in head-to-head too, to the
tournament's resolution (about 0.54 for 64 games to 0.52 for a few hundred).

## What this means for tuning

- **Three knobs are the whole rating**: muzzle speed, the damage and shield's thresholds, and
  turn rate (with the stall speed and the hit radius a step behind). The others are free to
  within 40 Elo across a factor of two: the dispersion, the recharge pair, the knee, the
  climb, the spawn energy, the store above x0.7, the cruise speed. They are the character
  levers: they can differ by a lot at the same rating.
- **Use the smooth knobs for the fine tuning** (muzzle speed 1.3%, turn 2.5%, stall 4%, hit radius
  5% per 25 Elo) and the step knobs (shield, damage, head-on fraction) for coarse moves and for
  the discrete interactions that make cycles. A shield or damage step of 130 to 150 Elo is
  also a hits-to-kill change for every opponent, so it is not a free +130 against a field of
  other types (the readings here are against the La-7's own shield and damage).
- **Cancelling pairs work to about ±30 Elo** (RMS 30 for the sum, with a negative bias of about
  18): good enough to propose a design, not to leave it unrated. The ±100 trade of turn for muzzle
  speed fell short by 87; the small ones with the stall speed and the half-lift drag went
  close to the sum.
- **Every reading is one airframe.** The La-7's other values (its shield of 110 against its
  hit damage of 34.8, its turn and its small hit radius) set where the steps sit; the
  [four-airframe gun study](../case-studies.md#guns-on-four-airframes) found the muzzle
  speed worth the same everywhere and the damage dependent on the airframe, and nothing here
  says the other slopes carry over.
- **Odd or missing.** The slot names of the shot-cost rows are the handle's multiplier, not
  the cost's. The climb rate and the store are tied (the store sets the spawn energy, the
  shot cost and the climb time as the design search's solver does); no knob was infeasible. The
  hit radius, the head-on fraction, the spawn energy, the half-lift drag and the coast exchange
  are handles, not search knobs; the head-on fraction is the steepest handle of the game
  (-900 a doubling: 0.3 to 0.15 is +1,282 Elo), and a ship's rating moved by it alone is a
  reason to hold it fixed across the set. The cruise speed reads +16 at x1.25 in a scripted
  agent that does not choose its fights by speed; that may change with the learned agents.
