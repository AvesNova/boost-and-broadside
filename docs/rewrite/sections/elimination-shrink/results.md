# Elimination shrink, Imugi's edge steer and its search (2026-11-15)

Branch `feat/imugi-boundary`. Run: `bench/elimination_search_run.sh` (phases search, edge, laws, interaction, nocap, deathmatch); tables: `bench/elimination_search_report.py data/*.jsonl`.

**Setup:** Imugi on both sides, 5v5, fields on, mirrored lineups, fog on, seed 61, 128 games per row (64 for deathmatch), each environment's first game. CPU, eager, about 2–9 minutes per row on this container's 4 cores. The map is the preset's 7.8 km (elimination) or 4.2 km (deathmatch).

**What changed** (decided 2026-11-15, see arms-todo.md):
- **Shrink laws** (`bnb.game.layout`): law 2, the power law r = R(1 − s)^p; law 3, the corner (linear to a radius at a time, then linear past 0). Every law is capped at `shrink_max_speed`, 150 m/s at real scale (about half of cruise): the radius is max(law, R − 150 m/s × t).
- **Imugi's edge steer in the edge's moving frame** (`imugi_combat.context`): margin k·ρ·f(β) with f(β) = √(1 − β²) + β·arccos(−β), β = edge speed / ship speed, "outward" relative to the edge, and the steer fading out between 4 and 2 turn radii of boundary. The edge's speed comes from Imugi's carry (the last two observations). For a still edge it is exactly the old rule.
- **Imugi without a known enemy** (`imugi_combat.SEARCH`): "centre" (the old rule: head for the centre, hold near it), "straight" (fly on; the edge steer turns it back), "waypoint" (the team heads for the enemy's spawn, the image of its own through the centre, then random points in 60% of the boundary, each ship pulled toward its team's centre of mass beyond 1–3 gun ranges).
- **The bench** now records every ship's first death (its tick, whether the boundary killed it, its distance from the centre). It includes the final kill, which the old score trace missed.

**Columns:** ended, the share of games ending by a wipe; mean, median, p80, p90, game length in s; hit1, the median first enemy hit (s); bdry, the share of all deaths the boundary caused (no shooter); kills/min, deaths per team per minute. Second table: the losing team's k-th death, median/p90 in s, [the boundary's share of those deaths].

## 1. Search without a known enemy (no shrink, 300 s clock)

```
label            games ended  mean   med   p80   p90  hit1  bdry kills/min
search-centre      128  0.95    78    54    99   147    15  0.01      2.63
search-straight    128  0.91   108    77   162   298    15  0.06      1.88
search-waypoint    128  0.96    87    69   112   172    15  0.00      2.38

loser's k-th death: median/p90 s [boundary share]
search-centre      18/35  [0.00]    25/49  [0.00]    33/66  [0.01]    43/76  [0.00]    52/119 [0.03]
search-straight    23/66  [0.04]    35/76  [0.05]    44/100 [0.06]    53/129 [0.06]    70/169 [0.09]
search-waypoint    19/36  [0.00]    25/57  [0.00]    35/75  [0.00]    45/97  [0.00]    67/134 [0.01]
```

- **The teams meet at once in all three:** the first hit comes at 15 s. That is the spawns' doing, not vision (a ship sees 1 km at real scale): with the old spawns every team knows where the other is and starts flying at it. The ships spawn nose toward the enemy area's centre, so "straight" flies there too, and "waypoint" heads for the enemy's spawn by construction. All three were handshakes at the start (corrected in section 8, with independent spawn centres).
- **"Straight"** scatters the fight: the first death comes later (23 s) with a long tail (p90 298 s, 9% of games unfinished at 300 s), and 6% of deaths are ships flying into the still edge. It is the most "uncoupled" and the worst for game length.
- **"Waypoint"** keeps the first four deaths where "centre" has them (19 / 25 / 35 / 45 s) and slows only the last one (67 s against 52 s): the hunt for the last ship takes longer without the agreed meeting point. Its tail (p90 172 s) is between the two. No boundary deaths.

## 2. The moving-frame edge steer (linear shrink from 0 over 60 s, the fastest linear)

```
label            games ended  mean   med   p80   p90  hit1  bdry kills/min
edge-still         128  1.00    47    49    55    57    19  0.42      4.74
edge-moving        128  1.00    49    52    57    58    19  0.33      4.56

loser's k-th death: median/p90 s [boundary share]
edge-still         26/46  [0.23]    32/48  [0.25]    39/51  [0.33]    44/54  [0.52]    49/57  [0.60]
edge-moving        27/46  [0.15]    35/50  [0.29]    41/53  [0.30]    47/56  [0.38]    52/58  [0.48]
```

Both with the waypoint search. The moving-frame steer cuts the boundary's share of deaths from 42% to 33% (of first deaths, 23% to 15%), at an edge speed of 130 m/s. The rest is unavoidable at this speed: once the boundary is a few km across, the fight happens at its edge.

## 3. Shrink laws (start 0, cap 150 m/s, moving-frame steer, waypoint search)

```
label            games ended  mean   med   p80   p90  hit1  bdry kills/min
linear-60          128  1.00    49    52    57    58    19  0.33      4.56
linear-90          128  1.00    56    52    73    81    15  0.13      3.81
linear-120         128  1.00    60    52    80    96    15  0.07      3.51
linear-180         128  1.00    67    58    90   119    15  0.03      3.09
power2-120         128  1.00    56    53    70    80    19  0.15      3.84
power3-120         128  1.00    53    53    63    69    21  0.26      4.08
corner0.32-120     128  1.00    58    55    70    83    21  0.18      3.74
corner0.32-180     128  1.00    60    55    77    93    21  0.16      3.54
corner0.256-180    128  1.00    59    53    73    84    21  0.20      3.68

loser's k-th death: median/p90 s [boundary share]
linear-60          27/46  [0.15]    35/50  [0.29]    41/53  [0.30]    47/56  [0.38]    52/58  [0.48]
linear-90          19/45  [0.03]    26/64  [0.09]    34/70  [0.14]    45/74  [0.15]    53/81  [0.23]
linear-120         19/38  [0.01]    25/53  [0.02]    35/72  [0.07]    45/84  [0.06]    52/96  [0.12]
linear-180         19/36  [0.00]    26/55  [0.02]    35/76  [0.04]    47/90  [0.02]    58/119 [0.05]
power2-120         25/44  [0.05]    32/53  [0.07]    40/61  [0.12]    45/70  [0.19]    53/80  [0.22]
power3-120         28/45  [0.12]    36/50  [0.20]    42/56  [0.23]    48/63  [0.32]    53/69  [0.30]
corner0.32-120     28/41  [0.09]    36/50  [0.12]    42/62  [0.15]    49/69  [0.19]    55/83  [0.21]
corner0.32-180     28/45  [0.09]    36/53  [0.13]    42/64  [0.12]    48/77  [0.14]    55/93  [0.23]
corner0.256-180    28/41  [0.10]    36/50  [0.16]    40/60  [0.20]    47/74  [0.15]    53/84  [0.21]
```

(The capped corner laws reach only about 3.3 km at 30 s, not their corner: the cap holds the first segment at 150 m/s.)

- **The median hardly moves (52–58 s) in any setting.** Every shrink here works on the tail: p90 from 119 s (linear 180) to 58 s (linear 60).
- **The trade-off is between the tail and boundary kills, and linear is on or ahead of the curved laws' frontier:**
  - linear-90 (p90 81 s, 13% boundary) against power2-120 (80 s, 15%) and corner0.256-180 (84 s, 20%);
  - linear-120 (96 s, 7%) against corner0.32-180 (93 s, 16%).
- **Why fast-then-slow doesn't help with Imugi:** the fast first phase reaches the ships at their spawns (0.875 of the radius). They spend the opening steering away from the edge, so the first hit comes later (19–21 s against 15 s) and the first death later (25–28 s against 19 s), with 5–12% of first deaths by the boundary. The game is then where it would have been anyway, with more boundary deaths on top.
- **Caveat:** this is Imugi. A trained agent that anticipates the edge will lose fewer ships to it, which would move every curved law's boundary share down. Whether it moves them below linear's can only be seen with trained agents.

## 4. The search under a mid shrink (linear 120 s)

```
label                games ended  mean   med   p80   p90  hit1  bdry kills/min
linear-120-centre      128  1.00    60    56    82    97    15  0.06      3.52
linear-120 (waypoint)  128  1.00    60    52    80    96    15  0.07      3.51
linear-120-straight    128  1.00    70    67    93   105    16  0.15      3.11

loser's k-th death: median/p90 s [boundary share]
linear-120-centre     20/39  [0.01]    30/58  [0.02]    39/68  [0.04]    46/84  [0.10]    56/97  [0.13]
linear-120 (waypoint) 19/38  [0.01]    25/53  [0.02]    35/72  [0.07]    45/84  [0.06]    52/96  [0.12]
linear-120-straight   23/53  [0.07]    36/74  [0.07]    44/82  [0.12]    53/93  [0.16]    68/105 [0.23]
```

With a shrink, the waypoint search and the old centre rule play the same game: the shrinking boundary does the meeting. "Straight" stays worse: later, more spread-out deaths and twice the boundary's share.

## 5. The curved laws without the speed cap

```
label                 games ended  mean   med   p80   p90  hit1  bdry kills/min
power3-120 (cap 150)    128  1.00    53    53    63    69    21  0.26      4.08
power3-120-nocap        128  1.00    53    52    63    69    20  0.27      4.16
corner0.32-120 (cap)    128  1.00    58    55    70    83    21  0.18      3.74
corner0.32-120-nocap    128  1.00    55    51    69    87    21  0.21      3.82
```

The cap changes almost nothing (the uncapped power-3 law starts at 195 m/s, the corner at 177 m/s). It is a safety rule, not a design lever, at these speeds.

## 6. Deathmatch kill rate by search (64 games each, 600 s, tickets off, 4.2 km)

| search | deaths per team per minute | sd over games | first hit (median) |
|---|---|---|---|
| centre | 4.69 | 1.13 | 4.6 s |
| straight | 4.34 | 0.98 | 3.9 s |
| waypoint | 4.62 | 1.13 | 5.1 s |

The standard error of each mean is about 0.14, so centre and waypoint are the same, and straight is about 0.35 lower (2.5 SE). The deathmatch map is small (4.2 km) and its spawn groups are random over it, so the search barely matters. The ticket calculation (4.57 per team per minute, 40 tickets for 80% of games ending by tickets within 10 minutes) stands with the waypoint search.

## 7. Summary and recommendation

1. **Search without a known enemy: "waypoint".** It removes the agreed meeting point and changes little else:
   - elimination without a shrink: the first four deaths as before; the last one later (67 s against 52 s median), because the last ship has to be found;
   - elimination with a shrink: the same game as the centre rule;
   - deathmatch: the same kill rate.

   "Straight" is clearly worse: scattered fights, long tails, ships hitting the edge.
2. **Edge steer: keep the moving-frame steer.** It cuts the boundary's share of deaths by about a fifth at the fastest linear shrink and is identical to the old rule for a still edge.
3. **Shrink law: linear, starting at 0.** With Imugi, no fast-then-slow curve beats a linear shrink of matching tail: the fast phase reaches the ships at their spawns, delays the first fight, and adds boundary deaths without shortening the median. The duration sets the trade-off between the tail and boundary kills:

   | shrink | p90 game | boundary share of deaths | last death by the boundary |
   |---|---|---|---|
   | none (300 s clock) | 172 s | 0% | 1% |
   | linear 180 s | 119 s | 3% | 5% |
   | linear 120 s | 96 s | 7% | 12% |
   | linear 90 s | 81 s | 13% | 23% |
   | linear 60 s | 58 s | 33% | 48% |

   For "the map decides games more often, not necessarily by killing", linear 120 s (clock 128 s) or linear 90 s (clock 98 s) are the candidates; linear 120 s keeps boundary kills at 7%.
4. **The speed cap (150 m/s)** never binds on the linear candidates and barely matters for the curved ones. Keep it as a safety rule.
5. **Caveat:** all of this is Imugi. A trained agent that anticipates the edge and searches better would shift the boundary shares down and the tails in. Re-measure with rl-v20 self-play when a GPU is available (open item 5).

## 9. Other team sizes on the fixed maps

Imugi, waypoint search, old spawns. Elimination with linear 120 s from 0 (7.8 km, 128 games); deathmatch with tickets off and the preset's 600 s clock (4.2 km; 64 games, 32 at 10v10).

```
label            games ended  mean   med   p80   p90  hit1  bdry kills/min
size-elim-1v1      128  1.00    78    82   106   110    21  0.48      0.38
size-elim-2v2      128  1.00    65    66    91    96    16  0.13      1.08
(5v5: linear-120)  128  1.00    60    52    80    96    15  0.07      3.51
size-elim-10v10    128  1.00    60    54    78   100    14  0.04      7.74
```

- **1v1 elimination:** the boundary kills in 48% of games. A lone duel rarely resolves before the map closes.
- **10v10:** the losing team's deaths come steadily from 16 s to 54 s (median), few by the boundary (4%).

**Deathmatch kill rate grows faster than the team size:**

| teams | deaths per team per minute | per ship per minute | tickets for 70% / 80% / 90% of games ending by tickets within 10 min |
|---|---|---|---|
| 1v1 | 0.40 (sd 0.16) | 0.40 | 4 / 4 / 3 |
| 2v2 | 1.29 (sd 0.38) | 0.65 | 13 / 11 / 8 |
| 5v5 | 4.69 (sd 1.13) | 0.94 | 41 / 36 / 32 (this run; 40 at 80% in the first mode-timing run) |
| 10v10 | 10.48 (sd 2.45) | 1.05 | 91 / 87 / 79 |

Tickets per ship (`kills_per_ship` × team size, as today) would leave small games mostly on the clock: 8 tickets at 1v1 would take a median 20 minutes. The 80% column fits **T ≈ 4 · N^(4/3)** (4, 10, 34, 87 for N = 1, 2, 5, 10). Proposal: set deathmatch tickets per team by that rule (or a table) instead of per ship. To re-measure with the explore search and trained agents.
