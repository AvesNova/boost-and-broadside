# Look-ahead personas, round 2: the persona test as it will be evaluated

(Summary: to be written at the end.)

Author's request 2026-11-27 ~20:15: 1v1, 4v4 and 16v16, always with fields and fog, longer
match lengths allowed. Branch `personas-2` from core 2770a376, laptop RTX 4070 (performance
profile), GPU shared under the lock with the league agent. Scripts in this folder (`clock.py`,
`skill2.py`, `analyse2.py`, `crowd.py`, `surrogate.py`, `tune.py`, `blur2.py`, the `*.sh`
loops); full tables in `tables-2.md`; numbers in `clock-2.json`, `clock-2b.json`,
`skill-2.json`, `analysis-2.json`, `crowd-2.json`, `screen-2.json`; raw games in
`~/avesnova/boost-and-broadside/data/lookahead-personas-2/` (SQLite stores per step). Every
game here is the game's defaults (4,200 m arena, 10 fields, fog, the standard spawn) except
the clock. Every agent is the look-ahead agent (`lean.scripted`). Elo is lean.eval's
Bradley-Terry fit.

## Measurements

### 1. Match length per size

Pairings: the yardstick mirror and the yardstick against six round-1 personas (persona14, 21,
27, 00, 53, 44), under a long cap. A game shorter than the cap ended by a wipe. The agent does
not read the clock, so the share of games wiped before a clock T is read off these lengths for
any T.

| size | games | cap | p50 / p80 / p90 / p95 | at the cap | draw wipes | wiped by 60 s | by 90 s | by 120 s | by 300 s | mirror: by 60 s / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1v1 | 7168 | 600 s | 75 / 204 / 395 / 600 s | 6.5% | 5.1% | 39.7% | 57.3% | 66.7% | 86.5% | 38.5% / 600 s |
| 2v2 | 3584 | 300 s | 60 / 94 / 131 / 191 s | 2.5% | 0.5% | 50.1% | 77.8% | 87.9% | 97.5% | 53.3% / 129 s |
| 4v4 | 3584 | 300 s | 49 / 66 / 81 / 105 s | 1.5% | 0.1% | 72.3% | 92.9% | 95.9% | 98.5% | 72.1% / 80 s |
| 8v8 | 2688 | 300 s | 48 / 61 / 71 / 82 s | 0.7% | 0.1% | 79.0% | 96.7% | 98.4% | 99.3% | 76.8% / 71 s |
| 16v16 | 2688 | 300 s | 53 / 66 / 76 / 86 s | 0.3% | 0.0% | 67.7% | 96.3% | 98.8% | 99.7% | 63.0% / 80 s |

- At today's clock (60 s) the wipe ends 40% of 1v1 games, 50% of 2v2, 72% of 4v4, 79% of 8v8
  and 68% of 16v16; the rest are decided by the clock rule (living ships, then shield).
- 4v4 to 16v16: about 90% of games are wiped by 71-81 s; 93-97% by 90 s.
- 1v1 has a long tail: median 75 s, p90 395 s, and 6.5% of games still running at 600 s (the
  yardstick mirror 11.5%). The mirror reaches 86.5% at 450 s and 88.5% at 600 s: 90% by
  elimination is not reachable for the mirror at any practical clock. 5.1% of 1v1 games end
  with both ships dead (a draw wipe).

**Study clocks** (p90 of the pooled wipe times, rounded up to 10 s): 1v1 400 s (12,000 ticks),
2v2 140 s (4,200), 4v4 90 s (2,700), 8v8 80 s (2,400), 16v16 80 s (2,400). Pooled share wiped
before them: 1v1 90.2%, 2v2 91.2%, 4v4 92.9%, 8v8 94.6%, 16v16 92.9%.

### 2. Skill and behaviour per size

Round 1's 106 agents (the 42 one-row box edges and the 64 spread-1 personas, the same unit
points), each a team of one agent against a yardstick team, sides balanced, games added
adaptively to a target score standard error: 0.015 at 1v1 and 4v4 (131,072 games each),
0.02 at 16v16 (65,536 games; 16v16 plays about 20-22 games/s here, so 0.015 would have taken
about 100 minutes). Pure persona teams at every size (no mixed lineups).

| size | personas | games (all agents) | mean Elo | sd observed | sd without error | mean error | min / 10% / median / 90% / max | above 1000 | best | draws mean / max | games at the clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1v1 | 64 | 131072 | 984 | 71 | 71 | 10 | 801 / 875 / 989 / 1072 / 1090 | 30 | persona24 | 8.7% / 11.9% | 11.9% |
| 4v4 | 64 | 131072 | 978 | 58 | 57 | 10 | 812 / 902 / 981 / 1039 / 1103 | 27 | persona38 | 0.1% / 0.5% | 6.2% |
| 16v16 | 64 | 65536 | 973 | 156 | 155 | 17 | 497 / 770 / 978 / 1150 / 1350 | 29 | persona15 | 0.0% / 0.2% | 6.8% |

- The skill spread is real at every size and largest at 16v16: sd 71 Elo (1v1), 57 (4v4),
  155 (16v16), measurement error removed.
- A persona's 4v4 and 16v16 Elo agree (r +0.90); its 1v1 Elo is unrelated to its 4v4 Elo
  (+0.08) and slightly opposed to its 16v16 Elo (-0.27).
- Draws are 8.7% of 1v1 games on average (at most 11.9%), under 0.5% at the team sizes; 6-12%
  of games reach the study clock.
- Under fog and fields round 1's 1v1 (fog off, no fields) picture changes: the 1v1 sd is 71
  (round 1: 154), and the yardstick flies at a full store (0.91) and corner speed (1.00) and
  fires 13.9 shots per ship-minute (round 1: 0.79, 0.63, 44.7).

**What drives skill** (weighted least squares of Elo on the unit coordinates, the 106
agents; the slope is the Elo change across the spread-1 box; R^2 0.85 / 0.85 / 0.88,
leave-one-out 0.72 / 0.75 / 0.79; a quadratic per row adds little: 0.90 / 0.94 / 0.97,
leave-one-out 0.68 / 0.81 / 0.89):

| row | 1v1 | 4v4 | 16v16 |
|---|---|---|---|
| threat_weight | +9 +- 11 | -29 +- 9 | -74 +- 16 |
| angles_weight | +138 +- 11 | -39 +- 9 | -222 +- 15 |
| energy_weight | -37 +- 12 | +62 +- 10 | +198 +- 18 |
| aggression | +86 +- 12 | -55 +- 9 | -186 +- 16 |
| speed_weight | +17 +- 11 | +101 +- 9 | +176 +- 15 |
| collision_weight | -47 +- 11 | -41 +- 9 | -78 +- 15 |
| boundary_weight | -6 +- 12 | -9 +- 9 | -2 +- 16 |
| field_weight | +12 +- 11 | +9 +- 9 | +14 +- 16 |
| horizon | +12 +- 12 | -69 +- 10 | -216 +- 16 |
| preferred_speed | -39 +- 13 | +17 +- 10 | +81 +- 17 |
| reply_temperature | -67 +- 12 | -2 +- 9 | +73 +- 15 |
| caution | -1 +- 12 | +12 +- 10 | +33 +- 16 |
| temperature | -104 +- 11 | -50 +- 9 | -83 +- 15 |
| stickiness | +25 +- 12 | +2 +- 9 | +29 +- 15 |
| store_weight | -27 +- 11 | -38 +- 9 | -102 +- 16 |
| power_horizon | +20 +- 13 | +18 +- 10 | -4 +- 17 |
| power_temperature | +15 +- 11 | -1 +- 9 | -16 +- 15 |
| power_stickiness | -22 +- 12 | -20 +- 10 | +6 +- 17 |
| disengaged_energy_weight | -43 +- 11 | +1 +- 9 | +4 +- 15 |
| seek_weight | +4 +- 12 | +8 +- 10 | -3 +- 17 |
| fire_tolerance | -0 +- 12 | +71 +- 9 | +184 +- 16 |
| R^2 / leave-one-out R^2 / points | 0.85 / 0.72 / 106 | 0.85 / 0.75 / 106 | 0.91 / 0.83 / 106 |

- **Size-dependent** (slope 1v1 to 16v16 differs by z > 4): angles weight, aggression, energy
  weight, speed weight, horizon, fire tolerance, reply temperature, preferred speed, store
  weight, threat weight. In every one the 4v4 slope lies between the 1v1 and 16v16 slopes,
  about 40-50% of the way; 16v16 amplifies 4v4.
- **Sign flips** (significant both ways): the angles weight (+138 per box at 1v1, -231 at
  16v16), aggression (+86 / -182), the energy weight (-37 / +193), the reply temperature (-67
  / +87) and the preferred speed (-39 / +86). The duel wants angles, aggression, a cold reply
  model and a lower preferred speed; the crowd wants energy, speed, a short horizon, a looser
  trigger and less angles and aggression.
- Size-dependent in strength only (flat at 1v1): fire tolerance (0 / +71 / +200), horizon (+12
  / -69 / -205), speed weight (+17 / +101 / +179), store weight (-27 / -38 / -98), threat
  weight (+9 / -29 / -66).
- **Not size-dependent:** the flight temperature (colder is better at every size: -104 / -50
  / -86), the collision weight (-47 / -41 / -77), the power rows, stickiness, caution.
- **Inert re-check** (box-edge Elo within about 30 of 1000 and slopes within 2 SE at every
  size): boundary weight, seek weight and field weight. The disengaged energy weight is not
  inert under fog in 1v1 (-43 +- 11 per box; edges 1029 / 982), inert at 4v4 and 16v16.

**Style per size** (lean.eval.style over each persona's games against the yardstick;
tables-2.md has every counter):

- The fight changes with size: the yardstick's store falls from 0.91 (1v1) to 0.62 (4v4) and
  0.42 (16v16), dive share rises 0.02 / 0.18 / 0.29, climb falls 0.46 / 0.38 / 0.25, shots per
  ship-minute 13.9 / 74 / 105, hits per ship-minute 3.7 / 18.4 / 24.6.
- Personas differ mostly in energy management (store 0.72-0.96 at 1v1, 0.30-0.85 at 4v4,
  0.16-0.75 at 16v16; climb and dive shares likewise); the fight geometry hardly moves
  (turning share sd 0.011-0.049, two-circle share sd 0.005-0.056).
- What goes with winning: at 1v1 the two-circle share (r +0.62) and hits (+0.85), slower is
  slightly better (speed r -0.35); at 16v16 speed (+0.71), hit range (+0.61), the two-circle
  share (+0.60); a lower store goes with winning at every size (r -0.27 to -0.32).

### 3. Crowd-dependent parameters

Ten rows (`parameters.CROWDED`: angles weight, energy weight, aggression, speed weight,
collision weight, horizon, preferred speed, temperature, reply temperature, fire tolerance)
got a duel value (the row) and a crowd value (the row x `<row>_crowd`), read as row x
ratio^w with the crowd weight w = (m - 1) / (m - 1 + `crowd_half`), m the crowd measure
(`crowd_measure`), so a duel reads the row itself. All are fixed rows, default ratio 1 (off),
so `SAMPLED`, `personas`, `box_point` and `ScriptedConfig` are unchanged; `PARAMETERS` and
`vector` are 12 rows longer. The store keeps its own crowd rule (`store_crowd`).

The three measures the agent can see, on the yardstick mirror (every 15 ticks, living ships;
mean / share with m <= 1 / mean weight at crowd_half 1):

| size | near (2 gun ranges) | known enemies | own living ships |
|---|---|---|---|
| 1v1 | 0.28 / 1.00 / 0.00 | 0.29 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| 2v2 | 0.67 / 0.84 / 0.08 | 0.92 / 0.75 / 0.12 | 1.87 / 0.13 / 0.43 |
| 4v4 | 1.48 / 0.55 / 0.27 | 2.33 / 0.26 / 0.46 | 3.49 / 0.04 / 0.68 |
| 8v8 | 2.29 / 0.34 / 0.43 | 5.33 / 0.05 / 0.76 | 6.52 / 0.01 / 0.82 |
| 16v16 | 2.75 / 0.22 / 0.53 | 11.12 / 0.02 / 0.87 | 12.26 / 0.00 / 0.90 |

Choosing the measure: the per-size fits were read as an additive surrogate (`surrogate.py`;
its predicted Elo is an extrapolation and not trusted in size), and its best point per measure
was played with fresh games against the old yardstick and head to head (SE 0.02 per pairing,
one Bradley-Terry fit per size over all pairings):

| candidate | 1v1 | 2v2 | 4v4 | 8v8 | 16v16 | mean 1v1/4v4/16v16 |
|---|---|---|---|---|---|---|
| plain | 845 +- 9 | 967 +- 9 | 1276 +- 10 | 1666 +- 21 | 2078 +- 63 | 1400 |
| plain-half | 977 +- 13 | 1035 +- 13 | 1207 +- 16 | 1400 +- 23 | 1671 +- 47 | 1285 |
| crowd-own | 989 +- 9 | 1131 +- 9 | 1310 +- 11 | 1643 +- 21 | 2019 +- 62 | 1439 |
| crowd-known | 1084 +- 9 | 1146 +- 9 | 1299 +- 11 | 1617 +- 21 | 2020 +- 62 | 1467 |
| crowd-near | 1139 +- 9 | 1122 +- 9 | 1262 +- 10 | 1571 +- 21 | 1975 +- 62 | 1459 |
| crowd-own-half | 1179 +- 15 | 1113 +- 14 | 1191 +- 15 | 1403 +- 23 | 1712 +- 52 | 1361 |

Games per size (all pairings): 8192 / 8192 / 8192 / 8192 / 8192. plain-half and crowd-own-half are those points moved halfway back to the old yardstick (in the search coordinates). The old yardstick is at 1000.

- Every crowd candidate beats the best single point (plain) at 1v1 (by 140-290 Elo) and 2v2
  (150-180) and stays within 60 Elo of it at 4v4-16v16.
- Mean over 1v1, 4v4, 16v16: known enemies 1468, near enemies 1459, own ships 1439, plain
  1400. Over all five sizes: known 1433, own 1418, near 1414, plain 1366.
- The chosen measure: **known enemies** (`crowd_measure` 1: seen, or hidden within the roam
  cutoff), `crowd_half` 1. Its weight is 0 in a duel, about 0.46 at 4v4, 0.87 at 16v16.
- The old yardstick saturates at 8v8 and 16v16 (these candidates score 95-99.7% against it):
  Elo above about 1500 there rests on the head-to-head games.

### 4. Tuning

(CMA-ES results: to be written.)

### 5. Re-centring

(To be written.)

## Recommendations (not measured; for the author)

(To be written.)
