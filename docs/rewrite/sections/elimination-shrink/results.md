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

- **The teams see each other almost at once:** the first hit comes at 15 s in all three. Spawns are within vision range (3 km) of the middle. So the no-target rule decides little at the start; it decides how the last ships are found.
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

## 4. Pending

- The search under a mid shrink (linear 120 with centre and straight; waypoint is in section 3).
- The curved laws without the speed cap.
- Deathmatch kill rate by search (sets the tickets).
