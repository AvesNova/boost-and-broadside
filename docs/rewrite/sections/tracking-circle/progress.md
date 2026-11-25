# Tracking circle: progress

- 06:45 tried: GameConfig.fog_tracking (default on), core.tracked_by_enemy (ticks since seen 0
  and known: no new state), sight_pairs(state, tracked) with the circle reach p / (1 - e) for
  tracked targets (torch.where, square-root free), perceive(state, revealed, tracking).
  worked: 5 new fog tests (16 in test_fog.py), each fails with tracking stubbed out; quick tier
  lean/game + lean/scripted 143 passed; ruff clean. next: measure.py tracking option and the
  16 conditions (CPU), compile check (unit bnb-tracking), replays, NOTES + core-decisions.

- 06:49 tried: measure.py with --tracking on|off and --vision-range (CPU, 4 processes, nice 10,
  MemoryMax 8G scope), 256 games per condition, e 0.5, p 1500 and 850. worked: all 16 done in
  about 2 min. next: compile check as unit bnb-tracking (started 06:51, ETA 06:55, worst 07:00),
  then replays.

## Measurements (CPU, 256 yardstick games per condition, paired deals: the same seed with
## tracking off and on; data in ~/avesnova/boost-and-broadside/data/tracking-circle-2026-11-26/)

Visibility is the game's own (ticks since seen 0). A contact starts when an enemy turns visible
(or is visible at the deal); a loss is a visible enemy turning hidden while it and some ship of
its enemy team are alive; a hidden spell runs from a loss to the next visible tick (spells cut
by death or the game's end left out of the mean). First sighting distance: quartiles of each
team's first ellipse sighting (the closest new pair). Off / on.

| p, condition | first sighting m (off = on) | contacts per game | losses per game | losses per contact | hidden s mean | draws | mean s | kills per min |
|---|---|---|---|---|---|---|---|---|
| 1500, 1v1 no fields | 1690/2418/2798 | 2.25 / 2.00 | 0.25 / 0.00 | 0.110 / 0 | 0.62 / none | 38 / 38 | 30.2 / 30.4 | 1.91 / 1.89 |
| 1500, 1v1 fields | 848/1328/2014 | 7.53 / 7.34 | 5.59 / 5.38 | 0.742 / 0.733 | 1.05 / 1.07 | 25 / 23 | 29.2 / 29.5 | 1.96 / 1.94 |
| 1500, 4v4 no fields | 532/823/1148 | 15.64 / 9.48 | 6.13 / 1.54 | 0.392 / 0.163 | 2.75 / 5.68 | 3 / 1 | 27.9 / 27.0 | 11.71 / 11.99 |
| 1500, 4v4 fields | 495/835/1302 (on: 1290 upper) | 38.43 / 33.19 | 28.12 / 23.89 | 0.732 / 0.720 | 1.26 / 1.30 | 1 / 4 | 32.4 / 32.4 | 10.14 / 10.12 |
| 850, 1v1 no fields | 1059/1431/1650 | 7.54 / 2.00 | 5.65 / 0.00 | 0.750 / 0 | 2.10 / none | 27 / 36 | 30.5 / 29.2 | 1.83 / 1.94 |
| 850, 1v1 fields | 704/1025/1414 | 10.61 / 7.09 | 8.69 / 5.14 | 0.820 / 0.726 | 1.25 / 1.14 | 19 / 21 | 29.6 / 29.8 | 1.84 / 1.82 |
| 850, 4v4 no fields | 532/912/1294 | 29.08 / 11.16 | 19.35 / 2.99 | 0.666 / 0.268 | 1.21 / 4.17 | 3 / 1 | 25.0 / 25.4 | 13.31 / 13.11 |
| 850, 4v4 fields | 477/800/1187 (on: 1185 upper) | 54.91 / 37.07 | 44.28 / 28.01 | 0.806 / 0.756 | 0.92 / 1.15 | 5 / 3 | 31.0 / 30.9 | 10.25 / 10.30 |

Hidden spells, quartiles s (off / on): 850 1v1 no fields 0.2/2.9/3.4 / none; 850 4v4 no fields
0.1/0.2/1.2 / 1.3/2.9/6.1; 1500 4v4 no fields 0.4/1.4/3.8 / 2.2/4.8/8.6; with fields the median is
0.2-0.3 s in every condition, on and off (lines flickering across field cores).

- Without fields, tracking holds a 1v1 for good once seen: 0 losses in 256 games at both ranges
  (the yardstick's fight never opens beyond p / (1 - e), 3,000 or 1,700 m). In 4v4 it cuts losses
  per game by 75% (1500) and 85% (850); the losses left are long separations (mean hidden spell
  4-6 s against 1-3 s off), the short flickers are gone.
- With fields, most losses are field cores cutting the line (median spell 0.2 s); tracking keeps
  the core test, so it removes fewer: 4% (1500 1v1) to 41% (850 1v1) of losses per game.
- First sightings are unchanged (they come before any tracking); draws, mean length and kills
  per minute move by no more than the paired noise (e.g. 850 1v1 no fields draws 27 -> 36,
  length 30.5 -> 29.2 s; every other change under 1 s and 0.3 kills per minute). The yardstick
  plays by its team view, so the game it plays barely changes; what changes is what a learner
  would see.

- 06:54 compile check (unit bnb-tracking, laptop 4070, performance profile, load under 3 at the
  start): 4 graphs, 5,938 traced operations (core 17e6cf3b today: 5,930, +8), 0 graph breaks,
  cold 224.6 s (core 222.4 s), "within the baseline", EXIT 0. act 1,821, advance 2,291,
  minibatch_loss 1,080, slot_actions 746.
- 06:56 replays (p 850, e 0.5, tracking on, --draw-vision), seeds 0 and 1: 1v1 no fields and
  4v4 with fields, in ~/avesnova/boost-and-broadside/data/replays-2026-11-26-tracking/ (with
  tick-300 stills of the 1v1). The drawing shows each observer's tracking circle, fainter than
  its ellipse, while its team tracks an enemy. Found and fixed: replay.py's --vision-range and
  --vision-eccentricity each rebuilt the card from the cache, so the second undid the first
  (no effect on the earlier e-only replays). The longer title now touches the "Red N" score.
- 06:58 final tree: quick tier lean/game + lean/scripted 143 passed, lean/tool passed (see commit), ruff clean.
