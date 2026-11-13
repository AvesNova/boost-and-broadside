---
covers: [bench/ship_type_tournament.py]
---

# The ship-type tournament

> The researched types this page rates left the game on 2026-11-13; the real-scale set
> holds Yongma, Jujak and vizia. The page and its data are kept as a record; the types'
> research and last cards are reference data ([The real planes](real-planes.md)).

Which real-scale type wins most in a deathmatch, as the scripted agent flies it. Every
type of the [real-scale set](real-scale.md) played every other in real_scale_deathmatch:
five ships of type A against five of type B, both teams flown by the scripted agent at
T = 1 with its real-scale deathmatch set, fog on, 128 games a pair with each type on each
side in half of them. That is 351 pairs and 44,928 games, about ten minutes on the laptop
GPU (performance profile; 13 s a batch of 1,152 games once compiled).

It rates a type as a scripted agent tuned on the P-51D flies it, not the type's ceiling:
a type whose strength needs another way of fighting (a slashing energy fighter, a
one-pass gunship) is rated by how well this agent's way suits it.

## How

`bench/ship_type_tournament.py play` runs the league's match engine
([Published ratings](../league.md#published-ratings)) with one player per type
(`bnb.league.matches.ship_type_players`) and the balanced allocation, a full round robin;
each batch is appended to the record as it finishes. `report` fits the ratings and writes
the tables; `analyse` the correlations below. The data are in
[real-scale-tournament/](real-scale-tournament/ratings.csv): batches.jsonl (the record),
ratings.csv, pairs.csv (every ordered pair's wins, draws, losses and score),
tournament.json, and analysis.json.

The ratings are the Bradley–Terry fit over every game (draws as half a win, one prior game
against the anchor), on the Elo scale with the P-51D at 1000. The interval is 1.96
standard errors, relative to the P-51D (whose own is zero). 128 games a pair was chosen
from the per-game variance: at even odds a type's 3,328 games pin it to about +-6 points,
and most pairs are far from even, so the errors came out at 14 to 87 points (the largest
for the types that almost never win). Draws were 1.4% of the games.

## The ratings

Kills are the type's kills (enemies its hits killed), deaths its own ships' (any cause);
the hit share is the shots that damaged an enemy; the mean speed is over its living
ship-ticks.

| Rank | Type | Elo (95% interval) | Score | W / D / L | Kills / deaths a game | Kill ratio | Shots a game | Hit share | Mean speed m/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Tempest V | 1734 (1699 to 1769) | 0.937 | 3100 / 36 / 192 | 22.96 / 13.35 | 1.72 | 1576 | 0.153 | 102 |
| 2 | P-47D | 1703 (1669 to 1737) | 0.926 | 3065 / 34 / 229 | 22.78 / 12.34 | 1.85 | 1681 | 0.181 | 107 |
| 3 | Fw 190A | 1591 (1560 to 1622) | 0.880 | 2905 / 46 / 377 | 22.12 / 13.22 | 1.67 | 1158 | 0.180 | 114 |
| 4 | Mosquito | 1459 (1430 to 1487) | 0.811 | 2677 / 47 / 604 | 22.85 / 15.37 | 1.49 | 1168 | 0.155 | 106 |
| 5 | Zerstörer | 1414 (1386 to 1442) | 0.786 | 2588 / 54 / 686 | 22.09 / 15.98 | 1.38 | 1068 | 0.162 | 105 |
| 6 | Me 262 | 1375 (1347 to 1402) | 0.762 | 2501 / 70 / 757 | 21.23 / 15.47 | 1.37 | 791 | 0.112 | 130 |
| 7 | P-38L | 1364 (1337 to 1392) | 0.756 | 2493 / 45 / 790 | 21.38 / 14.77 | 1.45 | 1759 | 0.209 | 120 |
| 8 | Hellcat | 1353 (1326 to 1381) | 0.749 | 2469 / 47 / 812 | 21.37 / 15.14 | 1.41 | 1799 | 0.218 | 95 |
| 9 | Shrike | 1348 (1320 to 1375) | 0.745 | 2446 / 70 / 812 | 21.01 / 15.91 | 1.32 | 1178 | 0.190 | 108 |
| 10 | Corsair | 1231 (1205 to 1258) | 0.673 | 2212 / 56 / 1060 | 20.96 / 16.43 | 1.28 | 1749 | 0.221 | 99 |
| 11 | Spitfire IX | 1102 (1076 to 1129) | 0.594 | 1952 / 47 / 1329 | 20.64 / 17.61 | 1.17 | 1333 | 0.196 | 105 |
| 12 | Spitfire XIV | 1094 (1067 to 1120) | 0.588 | 1935 / 46 / 1347 | 20.35 / 17.52 | 1.16 | 1322 | 0.195 | 108 |
| 13 | Heavy | 1086 (1059 to 1112) | 0.584 | 1914 / 56 / 1358 | 19.31 / 16.22 | 1.19 | 1382 | 0.204 | 113 |
| 14 | P-51D | 1000 (1000 to 1000) | 0.534 | 1753 / 47 / 1528 | 19.61 / 17.91 | 1.09 | 1828 | 0.207 | 104 |
| 15 | Ki-84 | 854 (827 to 882) | 0.454 | 1485 / 55 / 1788 | 17.51 / 17.93 | 0.98 | 1748 | 0.232 | 109 |
| 16 | Fw 190D | 833 (806 to 861) | 0.444 | 1434 / 86 / 1808 | 15.95 / 16.96 | 0.94 | 1266 | 0.226 | 113 |
| 17 | Yak-3 | 740 (710 to 769) | 0.398 | 1293 / 66 / 1969 | 16.32 / 18.47 | 0.88 | 1913 | 0.246 | 112 |
| 18 | Me 163 | 730 (700 to 759) | 0.394 | 1276 / 70 / 1982 | 16.56 / 18.68 | 0.89 | 771 | 0.159 | 134 |
| 19 | Bf 109K | 622 (591 to 654) | 0.348 | 1134 / 48 / 2146 | 15.16 / 19.30 | 0.79 | 949 | 0.191 | 116 |
| 20 | La-7 | 463 (427 to 499) | 0.291 | 943 / 49 / 2336 | 13.41 / 18.59 | 0.72 | 1440 | 0.245 | 110 |
| 21 | Hammerhead | 203 (160 to 247) | 0.213 | 685 / 48 / 2595 | 9.75 / 19.24 | 0.51 | 494 | 0.200 | 111 |
| 22 | Zero | 52 (5 to 98) | 0.169 | 550 / 28 / 2750 | 13.00 / 22.06 | 0.59 | 1326 | 0.232 | 103 |
| 23 | Anvil | 46 (-1 to 92) | 0.168 | 540 / 37 / 2751 | 8.70 / 19.80 | 0.44 | 285 | 0.208 | 101 |
| 24 | Albatross | 19 (-29 to 66) | 0.160 | 523 / 21 / 2784 | 11.54 / 21.93 | 0.53 | 1458 | 0.255 | 84 |
| 25 | Wren | -398 (-460 to -335) | 0.068 | 221 / 11 / 3096 | 9.64 / 23.29 | 0.41 | 2080 | 0.296 | 98 |
| 26 | Lantern | -409 (-472 to -346) | 0.066 | 210 / 20 / 3098 | 7.80 / 22.07 | 0.35 | 1961 | 0.280 | 101 |
| 27 | Firecracker | -1126 (-1297 to -956) | 0.001 | 1 / 6 / 3321 | 0.82 / 18.12 | 0.05 | 212 | 0.105 | 103 |

The ratings span 2,900 points, far wider than the standard game's two types. At the top
are the hard-hitting, well-armoured fighters (Tempest V, P-47D, Fw 190A), then the twins
and heavy-gun types (Mosquito, Zerstörer, Me 262, P-38L); the light-gunned agile types are
at the bottom (Zero, Wren, Lantern), with the Firecracker last: one win in 3,328 games,
its 31 kJ store of propellant spent before it can fight. The P-51D, the type the agent was
tuned on, is 14th. Every type flies far below its cruise (mean speeds 84 to 134 m/s): the
agent fights a turning, energy-bleeding fight.

**How well one number describes it.** The Elo is a summary, not the whole matrix: in 70
of the 351 pairs the observed score differs from the fit's prediction by more than three
standard errors (about one would by chance), by 0.04 on average and up to 0.35. The
largest: the Hellcat scores 0.07 against the Zerstörer (the fit says 0.41), the Me 262
0.44 against the P-47D (0.13) but only 0.53 against the Spitfire IX (0.83), the Me 163
0.31 against the Me 262 (0.02), and the Albatross 0.79 against the Zero (0.45) but 0.18
against the Anvil (0.46). pairs.csv has every pair.

## What predicts the rating

With 27 types, each handle and a few dimensionless groups against the Elo
(`bench/ship_type_tournament.py analyse`): the weight of fire (fire rate x damage a shot,
in damage per second), the speed range (cruise over stall), the turn load (sustained turn
rate x cruise over g), the climb over cruise, the shots a type needs to kill a P-51D
(100 shield over its damage a shot), the kill-time ratio (its weight of fire over the
P-51D's, times 100 over its shield) and the target over the cone (hit radius over the
cone's width at its gun range). The handles are the ship files'; the climb rate is the one
the combat economy was set from (for the heavy placeholder, an approximation).

| Feature | Pearson r | Spearman rho |
|---|---|---|
| weight_of_fire | +0.53 | +0.78 |
| max_shield | +0.64 | +0.65 |
| fire_rate | +0.37 | +0.57 |
| muzzle_speed | +0.37 | +0.50 |
| kill_time_ratio | +0.29 | +0.50 |
| speed_range | -0.52 | -0.44 |
| hit_radius | +0.30 | +0.40 |
| shots_to_kill_p51 | -0.40 | -0.37 |
| low_speed_acceleration | -0.48 | -0.36 |
| stall_speed | +0.36 | +0.36 |
| gun_range | +0.37 | +0.35 |
| turn_load | -0.34 | -0.35 |
| bullet_spread | +0.39 | +0.31 |
| hit_damage | +0.05 | +0.31 |
| sustained_turn_rate | -0.24 | -0.27 |
| climb_over_cruise | -0.31 | -0.25 |
| climb_rate | -0.32 | -0.23 |
| target_over_cone | -0.24 | -0.21 |
| nose_reversal_time | +0.17 | +0.21 |
| stall_angle | +0.10 | +0.19 |
| store | +0.17 | -0.10 |
| cruise_speed | -0.24 | -0.06 |
| zero_lift_drag | +0.00 | +0.05 |
| g_limit | -0.18 | -0.04 |
| gun_range_over_turn_radius | +0.03 | +0.00 |

Leave-one-out RMSE (Elo): the mean 740; ridge on all 25 features 461 (penalty 6.4); best 1-feature 606 (max_shield); best 2-feature 417 (fire_rate, shots_to_kill_p51); best 3-feature 318 (speed_range, kill_time_ratio, target_over_cone). The best one- or two-feature model on shuffled Elo (100 shuffles): median 0.92 of the mean's error, 5th percentile 0.83; ridge on shuffled Elo: median 1.00, 5th percentile 0.92.

The Firecracker's -1126 dominates the errors over all 27 types, so the models were also fitted
without it (`analyse --exclude Firecracker`, analysis-without-firecracker.json):
leave-one-out RMSE of the mean 641; ridge on all 25 features 247 (0.39 of the mean's); best
one feature 542 (the shots to kill a P-51D), best two 275 (fire rate and shots to kill),
best three 226 (fire rate, gun range, shots to kill). On shuffled Elo the best one- or
two-feature model reaches 0.80 of the mean's error at its 5th percentile, ridge 0.90.

So it predicts well beyond noise, and the guns carry it. The rank correlations are
strongest for the weight of fire (+0.78) and the shield (+0.65); the flight handles are
weak (the turn rate -0.27, the climb rate -0.23, the cruise -0.06). Without the
Firecracker a two-variable fit, Elo = 109 + 161 x fire rate (Hz) - 248 x shots to kill a
P-51D, explains 83% of the variance: a type does well when it fires often and needs few
hits (the Me 262: 10 Hz, one hit), and badly when it needs many (the Wren and the Lantern:
twelve hits at 15 Hz). Heavy single shots at a low rate do not make up for it (the Anvil:
2 Hz, one hit, 46). With a quarter of a P-51's hit share and three times its shots, the
agent sprays, which favours a dense, heavy fire over a nimble airframe. This is a fit to
27 points with up to three features chosen from 25, so its exact coefficients will not
survive a new set; the direction (guns over flight, for this agent) is the robust part.
