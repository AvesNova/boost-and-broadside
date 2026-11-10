---
covers: [src/bnb/league/cyclic.py, bench/scripted_search.py, bench/scripted_compare.py]
---

# Balancing the real-scale set

The [ship-type tournament](real-scale-tournament.md) rated the 27 real-scale types over a
2,900-point spread, and the guns explained most of it: the weight of fire and the shots a
type needs to kill a P-51D. This page balances the set in rounds, each of four steps:

1. tune the ships;
2. search the scripted agent's real-scale sets again, each mode against its current
   champion, on mixed lineups of the set's types;
3. rate the types in a tournament;
4. find which values explain what is left of the spread, and move one if it is loosely
   known.

The rounds stop when the ratings follow well-documented values, or when a change is within
the noise. Each round's data are in `real-scale-balance/`
(`round-N/`).

## The rules the rounds follow

**Equal damage per second.** Every type deals about the P-51D's 360 per second
(`bnb.game.real_scale.balanced_damage_per_second`):

    damage per second = 360 x (w / 3.68 kg/s)^0.05, kept within 5% of 360,

w being the type's weight of fire (for the heavy placeholder, the weight of fire its
placeholder damage would mean). The exponent keeps a trace of the real ordering: twice the
P-51D's weight of fire gives 3.5% more damage per second, and the band stops the Me 262's
13.6 kg/s at +5% (378) and the Wren's and Lantern's 1.36 kg/s at -5% (342). Each type keeps
its real fire rate, so a slower-firing type hits harder a shot: the Anvil's 2 Hz cannon
177 a shot, the 10 Hz cannon types about 36, the 15 Hz .50 batteries about 24. The shot cost
is unchanged (continuous fire costs every type the same multiple of its own climb charge).

**Leeway by confidence.** A value the research holds well (a "good" or "medium" datum in
`bnb.game.real_scale_aircraft`) stays where it is. A loosely known one (low, estimated,
invented or a placeholder) may move within its plausible range when a round's analysis
points at it. Every move is an entry of the balance file (`bnb.game.real_scale.BALANCE_FILE`,
`src/bnb/game/real_scale_balance.json`), and the ship file's note for the handle records the
old value, the new and the reason; the equal-damage rule's note does the same for the hit
damage.

**The decision rule.** After each round's tournament, `bench/balance_round.py`'s `propose`
stage sets the next round's moves without a judgement call:

- *The shield step, for every type.* A researched type's shield is 100 x (1 + step x (its
  survivability rating - 2)). The ratings (1 to 5) are estimates and the step (0.2 at the
  start: shields 80 to 160) is a proposal, so the step is the first loosely known value to
  move. When the Spearman correlation of the shield with the round's Elo is at least 0.4,
  the step falls by 0.05 (by 0.10 at 0.7 or more); at -0.4 or less it rises as much; it
  stays within 0 and 0.2.
- *A loosely known outlier.* A type with no researched datum (the eight invented designs and
  the heavy placeholder) rated more than 200 Elo, and more than three standard errors, from
  the median has its own shield moved 20 towards the median's side, within the set's 80 to
  160. A researched aircraft is never moved on its own.
- *Stop* when neither fires, or after round 4. A type whose shield is already at its bound
  and still an outlier is listed for the author: its next lever is a flight value (such as
  the Firecracker's store), which needs a solve and a decision.

**The agent's searches.** `bench/scripted_search.py` searches each mode's set
(real_scale_deathmatch, real_scale_frontline) to win that mode, against that mode's current
champion (`--champion`), on mixed lineups (`--mixed`): every game draws each seat's type
from the set, and both teams fly the same draw seat for seat, so neither side is favoured
by its ships. Four types are held out of the draws (Spitfire XIV, Ki-84, Zerstörer,
Lantern) to check that a set generalises. Half of each candidate's 64 games a generation
are against the champion, the rest against other candidates; candidates play at a
temperature drawn per game from 1, 1.5, 2 and 3, the champion at 1. The ratings are
anchored on the champion. A champion that every candidate beats would hold the ratings
by the prior (the light retune's plateau rule stopped early for that reason), so when the
median candidate scores 0.75 against the champion, the generation's top candidate becomes
the champion and the plateau history starts again. The search stops when the median
rating has not risen 10 points in 5 generations, or at its generation bound.

A round's new champion is the best-rated survivor of the last generation (one of the
eight kept from the generation before, so rated on two sets of fresh games), checked
against the previous champion by `bench/scripted_compare.py`: 1,024 fresh games at T = 1
on mixed lineups of every type, and 512 on mixed lineups of the held-out types alone.

**The tournament.** `bench/ship_type_tournament.py play --allocation a_optimal`: the
league's tournament as designed, each batch's pairs chosen by A-optimal value, until every
type's standard error (against the P-51D, the anchor at 1000) is under 25 Elo, or 100,000
games (about 20 minutes). Twenty-five is the league's own target, a quarter of the
100-point gap the rounds care about. A type that almost never wins (the Firecracker) carries
little information a game and holds the largest error; the budget, not the target, ends the
tournament while it is in the field. A-optimal play goes to the pairs near even odds, and
the intransitivity checks need every pair, so a balanced top-up then gives every pair with
fewer than 64 games one block of 64 (the reason the extra coverage is gathered).

**Rock, paper, scissors.** One Elo a type assumes the results are transitive. Three checks
say how far they are not (`bench/ship_type_tournament.py cyclic`, `bnb.league.cyclic`):

- the cycles by majority: the triples where i beats j, j beats k and k beats i, each in
  more than half their games (who wins, not by how much);
- the pairs whose observed score is more than three standard errors off a fit's
  prediction;
- held-out likelihood: each pair's games are dealt into five folds, and each model is
  fitted on four and scored on the fifth.

The models, from fewest assumptions added: Bradley–Terry (the logit link); the same
ratings with another link from rating gap to win probability, Thurstone's probit (lighter
tails) or a power link, sigmoid of about sign(z) |z|^gamma with gamma fitted (it stretches or
squeezes the large margins, so a margin that one Elo gets wrong without any reversal is
fitted without a cycle); and the multidimensional Elo mElo-2k (Balduzzi et al. 2018,
"Re-evaluating Evaluation"), which adds to each type a 2k-vector c, so the log-odds of i
beating j become r_i - r_j + c_i' W c_j, W the block-diagonal of k quarter-turn rotations:
k cyclic games on top of the rating, its vectors under an L2 penalty chosen with k by the
same folds. A cyclic part is kept only when it predicts held-out games better than the best
link without one, by more than twice the difference's standard error. The page then gives
both parts: the transitive rating, and each type's cyclic strength (the length of its
vector on the Elo scale; two types' cyclic term is the product of their strengths and the
sine of the angle between them).

"Cyclic" here is in the sense of the decomposition (a part of the log-odds no rating
difference can make), not necessarily a reversal of who wins: a pair whose margin is too
wide or too narrow for its two ratings, with the same winner, is cyclic in this sense when
no link explains it.

On the light retune's round robin (before any balancing, 44,928 games over all 351
pairs): one triple goes round by majority (Fw 190D over Yak-3 0.84, Yak-3 over Me 163 0.62,
Me 163 over Fw 190D 0.53). Bradley–Terry leaves 70 pairs more than three standard errors
off. Held out, the power link (gamma 0.61: the large margins squeezed) predicts 0.0047 nats
a game better than Bradley–Terry (standard error 0.0005), the probit 0.0058 worse; mElo
with k = 1 0.0166 better and k = 3 0.0210 (0.0010), leaving 6 pairs off. So the deviations
are in the margins, not in who wins, and the link explains about a quarter of them: the
rest is a margin pattern no single rating per type reproduces. Beside the 2,900-point
spread it is small: the cyclic part's root-mean-square term is 200 Elo against the
transitive part's 940 (4% of the summed squares).

## Round 1

**The ships.** The equal-damage rule: every type's damage per second went from its weight
of fire's (133 for the Wren and the Lantern to 1,334 for the Me 262) to 342 to 378, at its
own fire rate. Nothing else moved. The shot cost is unchanged.

**The agent.** Each mode's search played the re-armed set from its champion, the light
retune's set (32 candidates of 64 games a generation, half against the champion).

| Mode | Generations | Champion promotions in the search | Candidate | Against the previous champion, mixed lineups (1,024 games) | Held-out types (512 games) | Adopted |
|---|---|---|---|---|---|---|
| deathmatch | 25 | none | 24:6 | 0.607 +- 0.015 | 0.572 +- 0.021 | yes |
| Frontline | 27 | at generation 6 | 26:1 | 0.951 +- 0.005 | 0.961 +- 0.006 | yes |

The deathmatch search found a set only a little better than its champion (the median
candidate stayed about 50 points below it at mixed temperatures; the plateau rule ended it
after 25 generations). The Frontline search beat its champion in six generations (the
median candidate scored 0.78 against it), promoted its top candidate and went on 20 more.
Self-play of the new and the previous sets on mixed lineups (1,024 games each, T = 1):

| Mode | Set | Kills a game | Kills a minute | Shots a game | Hit share | Front moves a minute | Drawn |
|---|---|---|---|---|---|---|---|
| deathmatch | previous | 42.5 | 5.88 | 2,901 | 0.22 | | 4.0% |
| deathmatch | new | 44.5 | 6.38 | 2,648 | 0.26 | | 3.2% |
| Frontline | previous | 48.1 | 4.84 | 2,726 | 0.26 | 0.34 | 44.7% |
| Frontline | new | 42.9 | 4.45 | 1,825 | 0.32 | 0.69 | 26.1% |

The new Frontline set captures twice as often as the light retune's and draws less: winning
Frontline against a strong champion, on every type, no longer converges on fighting at the
expense of the front. It is also more aggressive (aggression 3.4 to 5.6), fights closer
(combat radius 4.8 to 3.2 gun ranges) and holds wider of the centre. Both sets still shoot
from beyond a gun range (the shot logistic's centre 1.61 gun ranges in deathmatch, 1.14 in
Frontline, from 1.71 and 1.90), by the author's decision not to change the shooting rule.

**The ratings.** 111,744 games in real_scale_deathmatch: 99,072 A-optimal (the largest
standard error then 26.6, the Firecracker's; every other type's under 25), then 12,672 in
the top-up over the 186 pairs short of 64 games. Draws were 1.7%.

| Rank | Type | Elo (95% interval) | Score | W / D / L | Kills / deaths a game | Kill ratio | Shots a game | Hit share | Mean speed m/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P-47D | 1671 (1653 to 1689) | 0.845 | 4002 / 108 / 690 | 23.14 / 18.33 | 1.26 | 1680 | 0.274 | 112 |
| 2 | Hellcat | 1534 (1518 to 1550) | 0.721 | 3457 / 98 / 1309 | 22.97 / 19.27 | 1.19 | 1713 | 0.281 | 100 |
| 3 | P-38L | 1440 (1425 to 1455) | 0.672 | 3297 / 119 / 1576 | 22.67 / 19.54 | 1.16 | 1641 | 0.279 | 123 |
| 4 | Tempest V | 1408 (1393 to 1423) | 0.657 | 3141 / 106 / 1617 | 22.64 / 20.35 | 1.11 | 1598 | 0.275 | 108 |
| 5 | Fw 190A | 1386 (1371 to 1400) | 0.663 | 3333 / 121 / 1666 | 21.74 / 19.76 | 1.10 | 1146 | 0.256 | 118 |
| 6 | Corsair | 1375 (1361 to 1390) | 0.658 | 3365 / 91 / 1728 | 22.76 / 19.66 | 1.16 | 1637 | 0.268 | 104 |
| 7 | Heavy | 1319 (1305 to 1332) | 0.653 | 3466 / 90 / 1820 | 22.54 / 19.89 | 1.13 | 1167 | 0.255 | 116 |
| 8 | Shrike | 1243 (1230 to 1256) | 0.594 | 3058 / 118 / 2072 | 22.14 / 21.33 | 1.04 | 1097 | 0.258 | 115 |
| 9 | Zerstörer | 1142 (1129 to 1155) | 0.525 | 2549 / 78 / 2301 | 21.55 / 20.73 | 1.04 | 1039 | 0.254 | 109 |
| 10 | Yak-3 | 1119 (1107 to 1132) | 0.517 | 2389 / 117 / 2230 | 21.45 / 21.50 | 1.00 | 1584 | 0.258 | 117 |
| 11 | Fw 190D | 1074 (1061 to 1087) | 0.519 | 2125 / 197 / 1966 | 20.83 / 20.86 | 1.00 | 1094 | 0.249 | 118 |
| 12 | Spitfire IX | 1071 (1058 to 1084) | 0.523 | 2130 / 160 / 1934 | 22.78 / 22.17 | 1.03 | 1139 | 0.250 | 108 |
| 13 | La-7 | 1061 (1048 to 1074) | 0.528 | 2185 / 91 / 1948 | 21.31 / 21.23 | 1.00 | 1128 | 0.246 | 116 |
| 14 | Spitfire XIV | 1057 (1044 to 1070) | 0.529 | 2161 / 148 / 1915 | 22.61 / 21.99 | 1.03 | 1140 | 0.244 | 112 |
| 15 | Mosquito | 1051 (1039 to 1064) | 0.531 | 2209 / 68 / 1947 | 22.23 / 21.55 | 1.03 | 1090 | 0.245 | 109 |
| 16 | P-51D | 1000 (1000 to 1000) | 0.512 | 11142 / 440 / 10626 | 22.23 / 22.80 | 0.97 | 1558 | 0.246 | 110 |
| 17 | Ki-84 | 980 (967 to 993) | 0.511 | 2280 / 81 / 2183 | 21.87 / 21.77 | 1.00 | 1439 | 0.259 | 113 |
| 18 | Albatross | 849 (837 to 862) | 0.526 | 3394 / 80 / 3054 | 22.44 / 19.75 | 1.14 | 1125 | 0.226 | 85 |
| 19 | Wren | 792 (780 to 804) | 0.535 | 4158 / 31 / 3619 | 22.48 / 21.90 | 1.03 | 1517 | 0.252 | 99 |
| 20 | Lantern | 689 (677 to 701) | 0.469 | 5088 / 154 / 5766 | 21.45 / 21.13 | 1.01 | 1568 | 0.246 | 105 |
| 21 | Zero | 478 (463 to 492) | 0.434 | 4709 / 72 / 6163 | 19.98 / 22.31 | 0.90 | 1063 | 0.253 | 103 |
| 22 | Me 262 | 339 (323 to 355) | 0.401 | 4112 / 354 / 6222 | 16.07 / 19.40 | 0.83 | 835 | 0.222 | 127 |
| 23 | Anvil | 182 (164 to 200) | 0.348 | 2394 / 196 / 4578 | 17.41 / 21.01 | 0.83 | 256 | 0.260 | 103 |
| 24 | Bf 109K | 64 (44 to 83) | 0.252 | 1455 / 127 / 4434 | 15.85 / 22.09 | 0.72 | 875 | 0.243 | 114 |
| 25 | Hammerhead | -1 (-22 to 19) | 0.237 | 1088 / 189 / 3715 | 14.31 / 20.23 | 0.71 | 469 | 0.252 | 109 |
| 26 | Me 163 | -43 (-63 to -23) | 0.839 | 27145 / 290 / 5077 | 16.29 / 8.46 | 1.92 | 740 | 0.216 | 109 |
| 27 | Firecracker | -1129 (-1181 to -1077) | 0.002 | 22 / 56 / 27698 | 4.31 / 17.36 | 0.25 | 205 | 0.182 | 101 |

Equal damage per second narrowed the field but did not even it: the standard deviation of
the ratings over the types fell from 713 to 611 (617 to 480 without the Firecracker), the
range without the Firecracker from 2,143 to 1,714. The leaders are now the armoured types
(P-47D and Hellcat, shield 160; P-38L, Tempest V, Fw 190A, Corsair, 140), and the bottom the
types whose guns fire slowly or slowly travel: the Me 262, the Bf 109K, the Me 163, the
Hammerhead and the Firecracker all fire 520 m/s rounds; the Anvil fires twice a second.
The Firecracker still wins 22 of 27,776 games: its store of 31 kJ/kg, not its gun, is what
it lacks.

**Intransitivity.** Nine triples now go round by majority, every one through the Mosquito
(with the Zerstörer, the La-7, the Fw 190D, the Yak-3 and the Spitfires), the weakest edge of
the strongest 0.62. Bradley–Terry leaves 71 pairs more than three standard errors off. A
different link does not help: the power link fits gamma 0.93 and gains 0.00007 nats a game
held out (standard error 0.00005), the probit loses 0.0017. mElo with k = 3 gains 0.0102
(0.0005) over the best link and leaves 2 pairs off, so the cyclic part is supported. It is
small beside the transitive part: its root-mean-square log-odds term is 159 Elo against
797 (3.8% of the summed squares). The types with the largest cyclic strength are the Me 163,
the Zero and the Bf 109K (25 on the scale above); the largest cyclic terms are the Albatross
over the Zero and the Me 163 over the Zerstörer (about 460 and 430 Elo beyond their ratings).
cyclic.json holds each type's transitive Elo and cyclic vector.

**What explains the ratings.** The strongest rank correlations with the Elo are now the fire
rate (+0.71), the shots a type needs to kill a P-51D (+0.71: the 15 Hz types, with their
smaller shots, need more), the shield (+0.69), the bullet spread (+0.65) and the muzzle speed
(+0.59; Pearson +0.83). The best single predictor is the muzzle speed (leave-one-out error
381 against the mean's 635), the best pair the muzzle speed and the kill-time ratio (282),
ridge on all 25 features 220; on shuffled ratings the best one- or two-feature model reaches
0.80 of the mean's error at its 5th percentile, so these are well beyond noise. With equal
damage per second the guns still lead, now through how the damage arrives: often (15 Hz
beats 10 Hz at equal damage per second) and fast (the 520 m/s cannon lose). Fire rates and
muzzle speeds are the guns' researched values and stay; the shield, from estimated
survivability ratings, is the loosely known value the rule moves.

**The decision for round 2** (`round-1/proposal.json`, applied to the balance file and the
ship files): the shield step falls from 0.2 to 0.15 (the shield's correlation +0.69), so the
researched types' shields span 85 to 145; and six loosely known types move 20 towards the
median: the Heavy 160 to 140 (rated +262 from the median), the Albatross 100 to 120 (-208),
the Wren and the Lantern 80 to 100 (-265, -368), the Hammerhead 120 to 140 (-1,058) and the
Firecracker 80 to 100 (-2,186). The Anvil (-875) is at its bound and listed for the author,
as are two levers the rule does not touch: the fire-rate granularity and the muzzle speed,
both researched, which now carry most of the spread.

## Round 2

**The ships.** The round-1 decision, applied: the shield step fell from 0.2 to 0.15, so the
researched types' shields span 85 to 145, and six loosely known types moved 20 towards the
median: the Heavy 160 to 140, the Albatross 100 to 120, the Wren and the Lantern 80 to 100,
the Hammerhead 120 to 140 and the Firecracker 80 to 100. Nothing else moved; the Anvil's shield
followed the step, 160 to 145. The ships stage had nothing left to change (the Hammerhead's flight
solve still misses its low-speed acceleration and sustained turn rate, as before; a shield is
not a flight handle).

**The agent.** Each mode searched from its champion, the round-1 set.

| Mode | Generations | Champion promotions in the search | Candidate | Against the previous champion, mixed lineups (1,024 games) | Held-out types (512 games) | Adopted |
|---|---|---|---|---|---|---|
| deathmatch | 33 | none | 32:2 | 0.566 +- 0.015 | 0.555 +- 0.022 | yes |
| Frontline | 26 | none | 25:2 | 0.520 +- 0.013 | 0.592 +- 0.018 | no |

The deathmatch candidate is 4.3 standard errors above one half on mixed lineups (0.555 on the
held-out types): adopted. The Frontline candidate is 1.5
standard errors above one half on mixed lineups (0.592 on the held-out types), short of the
two the rule asks, so the round-1 Frontline set stays the champion. Self-play of the new and
the previous sets on mixed lineups (1,024 games each, T = 1):

| Mode | Set | Kills a game | Kills a minute | Shots a game | Hit share | Front moves a minute | Drawn |
|---|---|---|---|---|---|---|---|
| deathmatch | previous | 44.4 | 6.45 | 2,582 | 0.26 | | 2.9% |
| deathmatch | new | 45.0 | 6.93 | 2,284 | 0.29 | | 3.7% |
| Frontline | previous | 43.9 | 4.52 | 1,836 | 0.32 | 0.69 | 34.1% |
| Frontline | candidate (not adopted) | 50.8 | 5.17 | 2,239 | 0.32 | 0.56 | 32.0% |

**The ratings.** 64,512 games in real_scale_deathmatch: 52,992 A-optimal (every type's
standard error under 25 Elo after 46 batches; the largest then 24.6, the Firecracker's), then
11,520 in the top-up over the 174 pairs short of 64 games. Draws were 1.8%.

| Rank | Type | Elo (95% interval) | Score | W / D / L | Kills / deaths a game | Kill ratio | Shots a game | Hit share | Mean speed m/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | P-47D | 1548 (1527 to 1569) | 0.860 | 2896 / 45 / 451 | 23.39 / 18.44 | 1.27 | 1478 | 0.285 | 121 |
| 2 | Hellcat | 1419 (1401 to 1438) | 0.752 | 2529 / 46 / 817 | 23.25 / 19.40 | 1.20 | 1486 | 0.295 | 108 |
| 3 | P-38L | 1351 (1333 to 1368) | 0.724 | 2512 / 71 / 937 | 23.04 / 19.71 | 1.17 | 1413 | 0.293 | 133 |
| 4 | Fw 190A | 1276 (1259 to 1293) | 0.679 | 2305 / 81 / 1070 | 22.11 / 20.22 | 1.09 | 977 | 0.282 | 127 |
| 5 | Corsair | 1271 (1254 to 1288) | 0.671 | 2208 / 50 / 1070 | 23.02 / 20.48 | 1.12 | 1411 | 0.297 | 112 |
| 6 | Tempest V | 1255 (1239 to 1272) | 0.662 | 2303 / 58 / 1159 | 22.83 / 20.90 | 1.09 | 1352 | 0.294 | 115 |
| 7 | Shrike | 1106 (1090 to 1122) | 0.573 | 1654 / 66 / 1224 | 22.17 / 21.41 | 1.04 | 945 | 0.287 | 123 |
| 8 | Yak-3 | 1071 (1055 to 1088) | 0.567 | 1559 / 73 / 1184 | 22.27 / 20.92 | 1.06 | 1371 | 0.294 | 124 |
| 9 | Heavy | 1054 (1038 to 1070) | 0.563 | 1522 / 53 / 1177 | 22.49 / 20.86 | 1.08 | 1053 | 0.262 | 125 |
| 10 | Zerstörer | 1030 (1014 to 1047) | 0.550 | 1500 / 28 / 1224 | 21.90 / 21.07 | 1.04 | 933 | 0.273 | 117 |
| 11 | Spitfire IX | 1017 (1001 to 1034) | 0.559 | 1495 / 86 / 1171 | 22.78 / 21.36 | 1.07 | 1014 | 0.275 | 115 |
| 12 | Spitfire XIV | 1007 (991 to 1024) | 0.544 | 1416 / 90 / 1182 | 22.48 / 21.38 | 1.05 | 1009 | 0.273 | 119 |
| 13 | P-51D | 1000 (1000 to 1000) | 0.508 | 6398 / 263 / 6203 | 22.60 / 22.28 | 1.01 | 1389 | 0.284 | 118 |
| 14 | La-7 | 986 (970 to 1002) | 0.521 | 1402 / 63 / 1287 | 21.58 / 21.02 | 1.03 | 980 | 0.279 | 123 |
| 15 | Fw 190D | 986 (970 to 1002) | 0.520 | 1386 / 89 / 1277 | 21.41 / 20.67 | 1.04 | 940 | 0.282 | 126 |
| 16 | Albatross | 983 (967 to 999) | 0.520 | 1420 / 23 / 1309 | 22.96 / 21.83 | 1.05 | 1029 | 0.281 | 93 |
| 17 | Mosquito | 946 (930 to 961) | 0.607 | 2270 / 47 / 1459 | 22.88 / 20.02 | 1.14 | 974 | 0.257 | 118 |
| 18 | Wren | 944 (928 to 959) | 0.594 | 2142 / 49 / 1457 | 23.29 / 20.99 | 1.11 | 1364 | 0.292 | 106 |
| 19 | Ki-84 | 881 (867 to 895) | 0.614 | 3187 / 67 / 1994 | 22.78 / 21.66 | 1.05 | 1245 | 0.283 | 122 |
| 20 | Lantern | 868 (854 to 882) | 0.605 | 3276 / 103 / 2125 | 23.01 / 20.95 | 1.10 | 1386 | 0.268 | 116 |
| 21 | Zero | 460 (443 to 476) | 0.362 | 5106 / 165 / 9065 | 19.60 / 22.95 | 0.85 | 951 | 0.276 | 109 |
| 22 | Me 262 | 210 (188 to 232) | 0.366 | 1881 / 128 / 3303 | 17.73 / 20.15 | 0.88 | 755 | 0.264 | 137 |
| 23 | Hammerhead | 185 (162 to 207) | 0.346 | 1786 / 153 / 3437 | 17.76 / 21.35 | 0.83 | 433 | 0.274 | 121 |
| 24 | Bf 109K | 19 (-7 to 45) | 0.301 | 1271 / 82 / 2999 | 16.50 / 20.33 | 0.81 | 808 | 0.268 | 118 |
| 25 | Me 163 | -6 (-33 to 20) | 0.225 | 768 / 102 / 2778 | 15.08 / 20.90 | 0.72 | 733 | 0.273 | 139 |
| 26 | Anvil | -84 (-109 to -58) | 0.670 | 7135 / 139 / 3478 | 12.94 / 11.10 | 1.17 | 214 | 0.242 | 91 |
| 27 | Firecracker | -867 (-914 to -820) | 0.009 | 46 / 58 / 8536 | 3.45 / 14.68 | 0.23 | 170 | 0.238 | 100 |

The standard deviation of the ratings over the types fell from 611 to 549 (480 to 447 without
the Firecracker), the range without the Firecracker from 1,714 to 1,631. The armoured types
still lead (P-47D, Hellcat, P-38L, Fw 190A, Corsair, Tempest V), and the bottom is still the
520 m/s guns, the Anvil and the Firecracker. The Anvil is the odd row: it wins 67% of its
games and is rated -84, because 6,592 of its 10,752 games are against the Firecracker (won
99%), which the A-optimal pairs favour, while it loses to the Me 262 (1%) and the Hammerhead
(12%) it met 896 times each; its cyclic part below is the largest. The Heavy, the Albatross, the Wren and the Lantern, moved towards the median,
now sit at 868 to 1054. The Hammerhead (+186) and the Firecracker (+262), whose shields
rose, gained; the Me 262 (-129) and the Bf 109K (-45), not moved, fell.

**Intransitivity.** Thirty-six triples go round by majority (nine in round 1); most go through
the Mosquito or the Zerstörer, the weakest edge of the strongest 0.65 (Spitfire IX, Mosquito,
Zerstörer). Bradley-Terry leaves 54 pairs more than three standard errors off; the power
link (gamma 0.94) 51 and gains 0.00006 nats a game held out, nothing; the probit loses
0.0025. mElo with k = 2 and a power link (gamma 1.04) gains 0.0105 (0.0006) over the best
link without a cycle and leaves 2 pairs off, so the cyclic part is supported. It is
small beside the transitive part: its root-mean-square log-odds term is 209 Elo against 733
(7.5% of the summed squares; round 1: 159 against 797). The types with the largest cyclic
strength are the Anvil (49.5), the Firecracker (26.3), the Hellcat, the P-47D and the Zero
(about 23); the largest cyclic terms are the Anvil over the Hellcat and over the Bf 109K
(about 930 Elo beyond their ratings) and over the Zerstörer (810). The Anvil's transitive
rating is -536, not -84.

**What explains the ratings.** The strongest rank correlations with the Elo are the fire rate
(+0.76), the shots a type needs to kill a P-51D (+0.76), the bullet spread (+0.65), the
target over the firing cone (-0.63), the muzzle speed and the gun range (+0.55 each) and the
shield (+0.54, from +0.69; Pearson +0.36). The best single predictor is the muzzle speed
(leave-one-out error 297 against the mean's 570), the best pair the shield and the muzzle
speed (271), the best three those and the fire rate (215), ridge on all 25 features 184; on
shuffled ratings the best one- or two-feature model reaches 0.78 of the mean's error at its
5th percentile, so these are beyond noise. The shield enters the best models only
beside the muzzle speed: the guns still decide the order.

**The decision for round 3** (`round-2/proposal.json`, applied to the balance file and the
ship files): the shield step falls from 0.15 to 0.10 (the shield's correlation +0.54), so
the researched types' shields span 90 to 130; and three loosely known types move 20 towards
the median: the Hammerhead 140 to 160 (rated -802 from the median 986), the Anvil 145 to 160
(-1,070) and the Firecracker 100 to 120 (-1,854).

## Running a round

    uv run --no-sync python -m bench.balance_round --round N

runs the round's six stages in order, each resuming from what is on disk (`--stages` runs
some of them): `ships` rebuilds the set with the balance file's moves; `search` searches each
mode against its champion (at most 40 generations; the median rating's plateau over 10);
`adopt` checks each mode's candidate (the best-rated survivor of its last generation)
against the champion, 1,024 games at T = 1 on mixed lineups and 512 on the held-out types,
and adopts it when its score is more than two standard errors above one half
(`src/bnb/controllers/scripted_real_scale.json` holds the champions); `tournament` runs the
A-optimal tournament to 25 Elo, then the top-up; `analyse` writes the ratings, the
correlations and the intransitivity checks (`tables.txt` holds them as tables); `propose`
applies the decision rule. The search logs go to `runs/real-scale-balance/round-N/`; the
rest to `real-scale-balance/round-N/` beside this page.
