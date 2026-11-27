# Look-ahead personas, round 2: the persona test as it will be evaluated

## Summary (2026-11-28 02:15)

Measured (sections 1-7; laptop RTX 4070, GPU shared with the league agent):

- **Clocks.** At today's 60 s a wipe ends 40% / 72% / 68% of 1v1 / 4v4 / 16v16 games. Study
  clocks (p90 of wipe times): 1v1 400 s, 2v2 140 s, 4v4 90 s, 8v8 80 s, 16v16 80 s (90-95% of
  games wiped before them). 1v1 has a fog tail: the yardstick mirror never reaches 90%.
- **Skill spread** (64 spread-1 personas around the old yardstick): sd 71 / 57 / 155 Elo at
  1v1 / 4v4 / 16v16. 4v4 and 16v16 Elo agree (r +0.90); 1v1 is its own game (r +0.08, -0.27).
- **Size-dependent rows**: angles weight, aggression, energy weight, reply temperature,
  preferred speed flip sign 1v1 -> 16v16; speed weight, horizon, fire tolerance, store and
  threat weight grow with size; 4v4 sits about halfway. Temperature: colder wins at every size.
- **Crowd rows** (new, off by default): duel and crowd values interpolated by a crowd weight;
  measure chosen by games: **known enemies**, crowd_half 1. The crowd configurations beat the
  best single point by 120-290 Elo at 1v1 and 2v2 and match it at 4v4-16v16.
- **Skill surface**: one Bradley-Terry fit on 627,456 games (517 agents), f quadratic with
  products per size; 82.7% of the explainable deviance held out (additive: 75.9%). Near-
  additive (ST - S1 <= 0.02); largest coupling angles x energy at 16v16 (+103 Elo per box^2),
  angles x aggression at 1v1 and 4v4 (-58, -43). The top is flat (five directions with
  curvature within +-3.2 Elo per box^2) and pressed against the explored range in seven rows.
- **Robust centre** (centroid of the 50-Elo region): in a fresh 16-agent ladder 1278 / 1482 /
  2335 against the old yardstick at 1v1 / 4v4 / 16v16, within 38 / 11 / 28 Elo of the best
  tested agent; the old yardstick is 722 / 518 / about -335 with the centre at 1000. The CMA's
  best point (stopped at generation 24): 1277 / 1476 / 2325.
- **Ladder test**: the adaptive ladder (137 games per pair) and a 1v1 round robin (1,024 per
  pair) agree in order (Kendall tau 1.00) and within 0.8 SE per agent; no cycles anywhere.
- **Behaviour** around the centre is moved by the preferred speed, the store weight, the
  power stickiness, the temperatures and stickiness; the skill rows (fire tolerance,
  aggression, threat, collision, reply temperature) barely move the label. Label blur
  1.0-1.7% / 3.8-6.1% / 13-19% at spreads 0.25 / 0.5 / 1. One degenerate mode: stickiness over
  about 8 x the temperature stalls team play (Elo near the floor at 4v4 / 16v16, normal at 1v1).

Proposed (for the author; section "Recommendations"): the robust centre as the new
yardstick (`candidates/robust-centre.json`), the persona box `proposal-2.json` (inert rows
held, narrowed where skill falls off), BC sampling at half that box with a 50-Elo and a
stickiness / temperature <= 5 rule, five league personas (league1, 2, 3, 5, 6) at the
centre's level with JS 0.02-0.11 between them, a 90 s game clock for team games.


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

### 4. CMA-ES search (stopped on the author's redirect)

`tune.py cma`: 19 rows (boundary and seek left out) + 5 crowd ratios + store_crowd, popsize 12,
192 games per candidate and size; generations 1-16 rated against crowd-known, 17-24 against
the generation-16 mean (at 16v16 the population scored 85-90% against crowd-known). Stopped
after 24 generations at 00:42 (author: generation 1 was already good enough; over-optimising
risks a brittle point). Table in `tables-2.md`; journal `data/.../cma/cma.jsonl`.

- Population mean against crowd-known: 982 (generation 1) -> 1176 (16); against the
  generation-16 mean: 981 (17) -> 1004 (24). Sigma 0.138 -> 0.183 (never contracted).
- In the ladder (section 6), the CMA's best evaluated point (cma-top1) is 1277 / 1476 / 2325
  against the old yardstick, the generation-16 mean 1229 / 1433 / 2284.

### 5. The skill surface

**Data** (`surface.py`): every round-2 game with one game version, the study clocks, fields
and fog, equal sizes 1, 4, 16: the step-2 skill games, the screen, the CMA evaluations, the
around-crowd-known games (90 agents against crowd-known: every row at u = -0.5 and 1.5 of its
box around crowd-known, and 48 spread-1 personas around it, SE 0.03 / 0.03 / 0.04) and two
rounds of active sampling (24 points near the top, each against crowd-known and the
generation-16 mean, 128 games per pairing and size). In all: 517 agents, 1,638 pairings,
627,456 games (1v1 233,728, 4v4 233,728, 16v16 160,000). Round 1's games are left out: other
conditions (1v1 without fog and fields, 60 s clock) and an older game version; its 64
personas are round 2's population anyway.

**Model**: Bradley-Terry on the pairings' outcomes, P(A beats B) = sigmoid(f(A) - f(B)), f a
quadratic (21 linear, 21 squared, 210 products) in each agent's effective unit coordinates at
that size (a crowded row's value is duel x ratio^w_s, w_s the size's mean crowd weight of the
agent's measure; the store weight is scaled by store_crowd's mean factor), ridge (lambda 100 on
the squares and products, 10 on the linear terms). Laplace covariance for the uncertainty.

| model (5-fold CV over agents, references kept in training) | held-out deviance explained (coin to saturated) |
|---|---|
| per size, linear + squares + products, lambda 100 | 82.7% |
| shared quadratic + per-size linear and squares, lambda 100 | 81.6% |
| per size, additive (no products), best lambda (fit on 559k games) | 75.9% |

Per size with products fits best, so no GP. Checks: the first surface (before active
sampling) predicted the scores of active round 1 with r 0.81 / 0.95 / 0.93 (1v1 / 4v4 /
16v16), RMSE 0.08 / 0.07 / 0.12 in score against a binomial SE of 0.044 (so the surface's own
error is about 0.06-0.11 in score, 40-80 Elo, near the top). Against the ladder's Elo (16
agents, fresh games) the final surface has r 0.98 / 0.99 / 1.00 over all agents; within the
top group (11 agents above crowd-known) r 0.46 / 0.61 / 0.86: the surface ranks the near-equal
top agents only roughly.

**Where skill peaks** (mean over the three sizes, inside the data's 2-98% range per size;
relative Elo, offsets arbitrary): the robust centre (the centroid of a uniform sample of the
region within 50 Elo of the peak, 960 samples) is 41 below the peak on the surface (632
against 673; per size -43 / -48 / -32), with a smaller predicted SD (10 against 15). The top is
flat in many directions: of 28 directions at the centre, five have curvatures within +-3.2
Elo per box^2 (caution, the power horizon, the crowd rows of angles, energy, aggression and the
store, the field weight) while the steepest is -301 (energy weight against angles weight and
the energy crowd ratio), -179 (stickiness against preferred speed and temperature) and -137
(preferred speed with stickiness).

Seven rows sit at the edge of the data at the centre (threat weight, collision weight, horizon,
temperature at the low edge; speed weight, power temperature, fire tolerance at the high edge,
all about half a box past round 1's box): the surface would go further than any game has.

The rows (old yardstick / robust centre; full table with the peak and the region in
`tables-2-surface.md`):

| row | old | centre | row | old | centre |
|---|---|---|---|---|---|
| threat_weight | 1 | 0.573 | caution | 0 | 1.66 |
| angles_weight | 0.3 | 0.509 | temperature | 0.05 | 0.0223 |
| energy_weight | 0.25 | 0.305 | stickiness | 0.05 | 0.0437 |
| aggression | 0.5 | 0.924 | store_weight | 0.5 | 0.283 |
| speed_weight | 0.3 | 0.623 | power_horizon | 3 | 5.13 |
| collision_weight | 3 | 1.63 | power_temperature | 0.05 | 0.105 |
| horizon | 1.5 | 0.806 | power_stickiness | 0.3 | 0.200 |
| preferred_speed | 0.9 | 0.870 | disengaged_energy_weight | 0.5 | 0.358 |
| reply_temperature | 0.2 | 0.0857 | fire_tolerance | 1 | 1.85 |
| boundary / field / seek | 2 / 0.3 / 0.1 | 1.98 / 0.40 / 0.078 | crowd measure, half | - | known enemies, 1 |

Crowd ratios at the centre (crowd value / duel value): angles 0.52, energy 0.65, aggression
0.16, speed 2.0, preferred speed 1.44, reply temperature 6.1; store_crowd 0.38.

**Couplings** (largest products, Elo per box^2, SE 12-14): 1v1 angles x aggression -58, speed
x reply temperature -57, angles x speed +49, speed x temperature -44; 4v4 angles x aggression
-43, energy x stickiness -42, angles x energy +41, energy x speed -41; 16v16 angles x energy
+103, energy x speed -67, energy x horizon +55, caution x stickiness +54. Interactions are small
against the main effects: on the surface ST - S1 is at most 0.02 for every row and size.

**Sobol indices of skill on the surface** (spread-1 box around the old yardstick, uniform),
first order; ST within 0.02 of it:

| 1v1 | S1 | 4v4 | S1 | 16v16 | S1 |
|---|---|---|---|---|---|
| angles_weight | 0.33 | speed_weight | 0.22 | horizon | 0.16 |
| temperature | 0.16 | fire_tolerance | 0.15 | fire_tolerance | 0.15 |
| aggression | 0.15 | horizon | 0.10 | angles_weight | 0.13 |
| energy_weight | 0.07 | collision_weight | 0.08 | speed_weight | 0.12 |
| reply_temperature | 0.07 | energy_weight | 0.08 | energy_weight | 0.11 |
| (Elo sd over the box) | 66 | | 67 | aggression | 0.10 (sd 172) |

**Working ranges** (`tables-2-surface.md`: per row and size the slice through the centre
where f is within 50 and 100 Elo of the slice's best). Narrow: horizon (16v16: about 0.13 box
either side), fire tolerance (16v16 within 50 Elo only from u 1.12), temperature (1v1 within 50
only below u -0.09), threat and collision weight (low end of the data). Wide (flat over the
data): caution, power horizon, power temperature, power stickiness, disengaged energy, store
weight, and boundary, field and seek (inert, as before).

**Degenerate play** (the 90 around-crowd-known agents; flags: shots under half the centre's,
draws over 20%, speed under 0.8 x the centre's, Elo under 800): one failure mode, team
stalling from **stickiness**: the stickiness at 0.29 with the temperature 0.027 (ratio 10.5;
u = 1.5 around crowd-known) and two spread-1 personas with ratios 8.7 and 9.0 fire under half
the centre's shots and lose almost every team game (4v4 and 16v16 Elo near -75, the floor of
the fit; persona26 492 at 4v4), while their 1v1 Elo is normal (946, 981). Other losses are
plain weakness without a degenerate signature: the energy weight at u 1.5 (786 at 4v4, 507 at
16v16), the preferred speed at u -0.5 (555 at 16v16, slow 0.82 corner speeds). No agent drew
over 20%; ramming cannot be read (hits count bullets and collisions together).

### 6. Validation: the adaptive ladder and a round robin

16 agents (the robust centre, the surface peak, cma-top1 and cma-top2, the generation-16
mean, the old yardstick, crowd-known, round-1 personas 14, 27, 00, and six league candidates),
every pair, fresh store, adaptive (lean.eval `adaptive`) to a score SE of 0.05 per pair: 16,384
games per size (137 per pair). Elo with the old yardstick at 1000 (full tables with style in
`tables-2-ladder.md`):

| agent | 1v1 | 4v4 | 16v16 |
|---|---|---|---|
| robust-centre | 1278 +- 13 | 1482 +- 17 | 2335 +- 80 |
| surface-peak | 1285 | 1493 | 2363 |
| cma-top1 | 1277 | 1476 | 2325 |
| cma-top2 | 1316 | 1439 | 2232 |
| cma-mean-g16 | 1229 | 1433 | 2284 |
| league1-6 (range) | 1193-1290 | 1445-1481 | 2304-2353 |
| crowd-known | 1171 | 1260 | 1940 |
| r1-persona14 / 27 / 00 | 1079 / 1033 / 843 | 929 / 1008 / 1035 | 794 / 1045 / 1246 |
| old yardstick | 1000 | 1000 | 1000 |

- The robust centre is 11 Elo below the best agent at 4v4 (the surface peak), 28 at 16v16
  (the surface peak; SE 80 there) and 38 at 1v1 (cma-top2, which is 43 and 103 below the
  centre at 4v4 and 16v16). It is above cma-top1 and the generation-16 mean at every size.
- With the robust centre at 1000, the old yardstick is 722 (1v1), 518 (4v4) and about -335
  (16v16; the old yardstick wins almost no 16v16 game against the top group, so that number
  rests on the chain through crowd-known and the round-1 personas).
- **Round robin** (1v1, 6 of the agents, 1,024 games per pair, 15,360 games): Elo cma-top1 1276,
  generation-16 mean 1239, crowd-known 1169, persona14 1075, persona27 1045, yardstick 1000 (SE
  7). The ladder (137 games per pair) gives the same order (Kendall tau +1.00) and every
  rating within 0.8 combined SE of the round robin's (differences +0.1, -0.6, +0.1, +0.3, -0.8
  SE): the ladder's reported SEs are not too small; with five comparisons they may be
  somewhat conservative.
- **Non-transitivity**: no cycle (A > B > C > A with every edge 2 SE from even) in the round
  robin or in any size's ladder.
- Ladder cost: 16,384 games per size for 120 pairs at SE 0.05 (1v1 about 2 minutes, 4v4 about
  2.5, 16v16 about 14 on the laptop).

### 7. Behaviour around the robust centre

State sets around the robust centre (`blur2.py states`: games with sampled actions, each
team the centre (1/3) or spread-1 personas around it (2/3), fields, fog, the study clocks):
12,012 / 21,494 / 21,999 ship-states (1v1 / 4v4 / 16v16). Full tables in `tables-2-behaviour.md`.

**Which rows change behaviour** (Saltelli, base 32: 736 configurations of the spread-1 box
around the centre per size; Sobol indices of the mean per-state JS from the centre; base 32
is coarse, so the intervals are wide, ST +- up to 0.6 for the largest):

| row | 1v1 S1 / ST | 4v4 S1 / ST | 16v16 S1 / ST |
|---|---|---|---|
| preferred_speed | 0.53 / 1.10 | 0.64 / 1.13 | 0.19 / 0.58 |
| store_weight | 0.07 / 0.34 | 0.06 / 0.22 | -0.03 / 0.08 |
| power_stickiness | -0.03 / 0.10 | -0.04 / 0.08 | 0.07 / 0.31 |
| temperature | -0.03 / 0.10 | 0.01 / 0.06 | -0.09 / 0.18 |
| stickiness | 0.02 / 0.06 | 0.02 / 0.05 | 0.01 / 0.17 |
| horizon | -0.03 / 0.02 | -0.06 / 0.04 | -0.17 / 0.19 |
| speed_weight | -0.04 / 0.03 | -0.01 / 0.03 | -0.03 / 0.15 |
| power_temperature | 0.02 / 0.01 | 0.02 / 0.02 | -0.10 / 0.18 |
| angles_weight | 0.00 / 0.00 | 0.00 / 0.01 | 0.05 / 0.14 |

- Around the centre (cold flight softmax, temperature 0.022) the label is moved mostly by the
  preferred speed, then the store weight and the power stickiness (the power head), the two
  temperatures and stickiness; at 16v16 also the horizon, the speed and angles weights. ST
  well above S1 for these rows: their effects on the label interact.
- Rows that drive skill but hardly move the label (ST <= 0.01 at every size): fire tolerance,
  aggression, threat weight, collision weight, reply temperature, caution. Skill and behaviour
  are carried by different rows: the skill rows mostly act on rare decisions (the trigger,
  engagement choice) that the per-state JS barely sees.
- Inert for behaviour as for skill: boundary, seek and field weight (ST 0.00-0.004).

**Label blur** (64 personas, the same unit points at each spread, around the centre; boundary,
seek and field held; spreads of round 1's widths):

| size | centre entropy | blur at spread 0.25 | 0.5 | 1 |
|---|---|---|---|---|
| 1v1 | 1.040 | 0.018 (1.7%) | 0.064 (6.0%) | 0.212 (18.8%) |
| 4v4 | 0.893 | 0.016 (1.7%) | 0.057 (6.1%) | 0.187 (19.2%) |
| 16v16 | 0.828 | 0.009 (1.0%) | 0.033 (3.8%) | 0.118 (13.3%) |

The blur is about twice round 1's (round 1 at spread 1: 9-12%): the colder centre makes each
persona's labels sharper, so the same spread moves them further apart. Still about quadratic
in the spread.

**The league candidates' behaviour** (pairwise mean JS on these states): between the six
candidates 0.02-0.08 (1v1), 0.02-0.10 (4v4), 0.02-0.11 (16v16); each against the robust centre
0.01-0.07; against the old yardstick 0.05-0.11 (1v1) and 0.21-0.28 (4v4, 16v16). league2 is the
most distinct in team games (0.07-0.11 from the others), league1 and league5 in 1v1; league4 is
closest to the centre (0.011-0.018 at the team sizes).

## Recommendations (not measured; for the author)

- **Game clock.** Today's 60 s ends only 40% (1v1), 72% (4v4) and 68% (16v16) of games by a
  wipe. 90 s gives 93-97% at 4v4-16v16 and is the clock I would recommend for team games; 1v1
  needs about 400 s for 90% (fog loses contact; the mirror never reaches 90%), so a 1v1 clock
  is a choice between long games and many clock decisions. One shared default: 90 s, with the
  1v1 study clock kept for 1v1 evaluation.
- **Crowd rows.** Keep them (off by default on this branch until the author decides): every
  crowd candidate beat the best single point at 1v1 and 2v2 by 120-290 Elo, and the top region
  of the surface uses them (angles and aggression down in crowds, speed, preferred speed and
  the reply temperature up). The shared measure: the known enemies, crowd_half 1. Rows that
  the data never moved in crowds (collision, horizon, temperature, fire tolerance) can be
  dropped from `CROWDED` (6 of 10 kept).
- **Proposed yardstick: the robust centre** (`candidates/robust-centre.json`). It is within
  11 / 28 / 38 Elo of the best tested agent at 4v4 / 16v16 / 1v1, in a flat region; it is not the
  sharp peak. Caveat: seven of its rows sit at the edge of the explored range, so a later
  round might move them further (threat, collision, horizon, temperature lower; speed weight,
  power temperature, fire tolerance higher).
- **Persona box** (`proposal-2.json`): centred on the robust centre; boundary, seek and field
  weight held (inert at every size); per row a half-width of half the narrowest 50-Elo slice
  over the sizes, clipped to 0.1-0.5 of round 1's box: horizon 0.078 (log units; round 1 0.3),
  fire tolerance 0.135, temperature 0.204, threat 0.186, collision 0.192, energy 0.242, speed
  0.283, reply temperature 0.322, stickiness 0.358, angles 0.381, aggression 0.492, preferred
  speed 0.133; caution, store, power rows and disengaged energy keep round 1's widths.
- **BC good region**: draw on that box, reject draws whose surface mean over the three sizes
  is more than 50 Elo below the centre's, and reject stickiness / temperature > 5 (the stalling
  failure). Label blur (measured around the centre on round 1's widths): 1.0-1.7% of the label
  entropy at spread 0.25, 3.8-6.1% at 0.5, 13-19% at 1. The proposed box is about 0.5-1 of round
  1's widths per row, so it sits between the 0.5 and 1 figures; for BC I would sample it at
  half its width (blur roughly 2-4%) unless the student is told the persona.
- **League personas** (`candidates/league1.json` ... `league6.json`): six points of the 50-Elo
  region, picked far apart in rows that change behaviour. In the ladder they are within 97 Elo
  of each other at 1v1 (1193-1290), 36 at 4v4 and 49 at 16v16, all at the level of the
  centre. Their style differs little at 4v4 and 16v16 (store 0.15-0.23 and 0.05-0.10); at 1v1
  league1 and league5 are the distinct ones (store 0.71 / 0.69, speed 0.95 / 0.93, turning 0.55
  / 0.58 against the others' 0.56-0.66, 0.81-0.88, 0.64-0.66). Their labels differ by a JS of 0.02-0.11 from each other (as much as a typical spread-1
  persona from the centre, a third to a half of the old yardstick's distance from the centre
  in team games). Proposed league set (5): league1, league2, league3, league5, league6 (league4
  dropped: closest to the centre and to league5). If the league needs
  clearly different styles at equal Elo in team games, the top region alone does not provide
  them: the store, climb and dive regimes that separate round-1 personas (store 0.16-0.75 at
  16v16) all lie well below the top in skill.
