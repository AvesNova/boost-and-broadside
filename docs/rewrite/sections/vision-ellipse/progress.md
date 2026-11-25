# Sight ellipse: progress

- 22:57 tried: card column vision_eccentricity (default 0), core.sight_pairs with the
  ellipse test, fog tests, replay --vision-eccentricity / --draw-vision, the sandbox tool's
  outlines and hidden-ship hollow. worked: fog tests 11 passed; quick tier on lean 326 passed
  (first commit); game_ops: step +6 compute ops (1161 -> 1167 no fields, 1499 -> 1505 with
  fields), deal +6. next: measurements e 0 vs 0.5 (CPU, 512 games per condition), replays, the
  compile check once bnb-nose is done.

## Measurements (CPU, 512 yardstick games per condition, paired deals; data in
## ~/avesnova/boost-and-broadside/data/vision-ellipse-2026-11-25/, measure.py beside this file)

Quartiles [25, 50, 75]. "First" = each team's first sighting (pooled); "later" = each ship's first
sighting after the deal (ships not in sight at the deal). Angles in degrees, 0 = nose-on.

| condition | first dist m | later dist m | later obs off nose | later tgt off nose | merge noses | merge miss m | first spotter wins (decisive n) | mean s | W0/W1/D, wipes |
|---|---|---|---|---|---|---|---|---|---|
| e0 1v1 no fields | 1490/1495/1498 | 1493/1495/1498 | 10/28/54 | 10/28/54 | 177/178/179 | 2.1/3.9/5.9 | none (sight symmetric) | 58.7 | 164/170/178, 12 |
| e0.5 1v1 no fields | 1736/2413/2818 | 2076/2599/2877 | 16/32/56 | 8/12/40 | 177/178/179 | 2.3/4.1/6.0 | 44% (284) | 58.7 | 165/173/174, 14 |
| e0 1v1 fields | 829/1297/1494 | 857/1361/1495 | 10/29/62 | same | 175/178/179 | 2.4/4.9/7.8 | none | 49.6 | 220/206/86, 177 |
| e0.5 1v1 fields | 854/1364/2067 | 917/1429/2145 | 11/32/64 | 9/26/65 | 175/178/179 | 2.6/4.9/7.3 | 60% (173) | 49.8 | 213/219/80, 173 |
| e0 4v4 no fields | 514/737/1068 (91% at the deal) | 1494/1496/1498 | 29/53/88 | 6/10/31 | 165/178/179 | 2.6/4.9/20 | none | 35.2 | 238/271/3, 469 |
| e0.5 4v4 no fields | 514/737/1082 (97% at the deal) | 1867/2402/2769 | 23/40/66 | 9/16/56 | 164/178/179 | 2.4/4.7/22 | 54% (26) | 37.1 | 239/269/4, 449 |
| e0 4v4 fields | 476/773/1140 (78% at the deal) | 863/1364/1496 | 26/57/96 | 5/12/45 | 143/177/179 | 3.3/6.3/89 | none | 38.7 | 282/227/3, 436 |
| e0.5 4v4 fields | 488/797/1214 (82% at the deal) | 894/1426/2110 | 24/49/81 | 6/16/59 | 145/177/179 | 3.0/6.1/77 | 54% (91) | 39.7 | 272/238/2, 421 |

Merge share: 98% (1v1 no fields), 89% (1v1 fields), 100% (4v4), equal at e 0 and 0.5.

- At e 0 sight is symmetric (equal ranges), so neither team ever spots first: the who-spotted-first
  question only exists with the ellipse.
- The ellipse moves sightings out ahead (1v1 no fields median 1,495 -> 2,413 m) and mostly
  nose-on for the target too (later sightings, target off nose median 28 -> 12 deg), but the
  first merge is the same: head-on (noses ~178 deg) with a miss of a few metres at e 0 and 0.5.
  The yardstick does not use earlier sight to set up a different pass; game length and results
  barely move.
- First spotter: 44% of decisive 1v1 games without fields (n 284), 60% with fields (n 173),
  54% in 4v4 (n 26 without fields, 91 with). No consistent first-spotter advantage for the
  yardstick.
- 4v4: most first sightings happen at the deal (ships spawn in sight), so "later" is the useful
  column there.

- 23:05 PAUSED (coordinator: laptop must stay quiet). Done: code, tests, replay + sandbox
  tool, docs, replays (1v1 no fields and 4v4 fields, seeds 0-1, e 0.5), 7 of 8 measurement
  conditions. Remaining (heavy): e 0.5 4v4 fields measurement (`measure.py --eccentricity 0.5`,
  ~2 min CPU for all four; only that file is missing), the compile check as unit
  bnb-vision-ellipse (game_ops CPU count already: step +6 compute ops), the quick tier on the
  final tree (it passed, 326, on the first commit; fog and tool tests pass on the final one),
  `git merge core` if it moved (it had not at 22:59), the core-decisions entry's check line.

- 00:56 RESUMED (CPU only). tried: e 0.5 4v4 fields measurement (nice 10), quick tier
  on the final tree, merge core. worked: measurement done (table above complete); quick tier
  326 passed; ruff clean; core had not moved (nothing to merge). next: the compile check as
  unit bnb-vision-ellipse once the coordinator says the GPU is free.
