# Calibration 15: old and new numbers

Queue item 15, 2026-11-28. Everything re-measured under the core rules of items 12 and 13
(nose-cone armour, the circle deflector, the sight egg 1,500 / 850 / 850 m; the shrinking zone,
no clock, every game ending by elimination, longest game 3,934 ticks). Units stay physical:
nothing is rescaled by the arena.

How it was measured (`calibrate.py`): the default scripted agent `ScriptedConfig()` on both
teams (persona spread 0, `lean.learn.rollout.scripted_actions`), the training game
(`GameConfig(training_spawn=True)`: drawn fields, fog, the zone), finished games re-dealt, on
the laptop RTX 4070, performance profile, load about 1:

| Lineup | Environments x ticks | Episodes | Episode ticks median / p99 / max | Seconds |
|---|---|---|---|---|
| 4v4 (the recipe) | 1,024 x 8,000 | 5,439 | 1,301 / 2,777 / 3,524 | 327 |
| 1v1 | 2,048 x 8,000 | 8,758 | 1,510 / 3,486 / 3,641 | 311 |
| 16v16 | 256 x 8,000 | 1,278 | 1,372 / 2,821 / 3,527 | 429 |

Value returns: each living ship's discounted Monte Carlo return per level (the learner's
discounts) for every tick of the episodes that ended inside the run. With no clock every game
ends, so this is a perfect critic's target for any GAE lambda (the lambda-return of the true
values is the Monte Carlo return); the outcome runs on through a death (it is paid to the dead
too), which is option C's target. Next-state targets on every ship alive before and after a
step. Geometry targets and the scaled features every 8th tick on the first 256 environments,
both teams' views, the rows the team sees (the learner's `observed_rows`). Hit outcomes: every
bullet that ended. `analyse.py` gives the occupancy (each target to its nearest centre in bin
space; "99% in" = bins holding 99% of the mass, largest first; the largest bin's share;
`analysis-before.txt` has every line). Data: `data/team{1,4,16}.json` (exact percentiles over
every row); the subsamples (`.npz`, 0.6 GB each) are kept out of git, in
`~/avesnova/boost-and-broadside/data/calibrate-15/` (with copies of the JSON: `analyse.py` reads that folder).

## Value supports (lean/learn/config.py)

Rule unchanged: symmetric, +-the largest measured extreme (now over the three lineups), rounded
out; knees by the event rule (unchanged: hit 8 points, kill 2.83 units).

| Level | Old | New | Extremes 4v4 / 1v1 / 16v16 | 4v4 0.1% / 99.9% | 4v4 99% in, largest (old -> new) |
|---|---|---|---|---|---|
| damage | +-5.4, knee 0.0109 | +-5.6, knee 0.0109 | -2.97..4.15 / +-2.52 / -4.13..5.59 | -1.59 / 1.60 | 42 -> 42, 35.2% -> 35.3% |
| kill_death | +-7.5, knee 0.141 | +-8.2, knee 0.141 | -4.80..8.19 / +-2.83 / -4.81..7.06 | -2.80 / 3.02 | 42 -> 41, 35.9% -> 36.0% |
| outcome | +-10, linear | +-10, linear (unchanged) | +-10 | -9.98 / 9.99 | 30, 6.1% |

The largest damage and kill_death bin is the one at 0 (quiet ticks, every discounted future
event far off); 1v1 puts 67% there. The outcome's middle bins are empty by construction: with
no clock |return| >= 10 x 0.9997^3,934 = 3.1 except draws.

Value coefficient (author 2026-11-28 ~18:15, literature-loss-balance.md item 2): 0.05 -> 0.25.

## Next-state columns (lean/model/policy.py NEXT_STATE_COLUMNS)

Rule changed: +-the largest |0.1st| or |99.9th| percentile over the three lineups, rounded out
(was +-the measured extreme: the extremes are collisions and deflections, and the old ranges
left a quarter to a third of the bins empty); the rest go into the buffer columns (at most
0.03% beyond the outermost centre). Knees: the flat-core data rule applied where it improves
the occupancy (energy); elsewhere kept (the rule's along 2.8e-3 and speed 0.166 change
nothing, across 2.0e-5 makes the occupancy worse, heading has no flat core).

| Column | Old range, knee | New range, knee | 0.1% / 99.9% 4v4; 1v1; 16v16 | 4v4 99% in, largest (old -> new) |
|---|---|---|---|---|
| along (m) | +-1.89, 2.4e-3 | +-0.53, 2.4e-3 | -0.352/0.325; -0.524/0.507; -0.340/0.300 | 35 -> 43, 13.3% -> 10.4% |
| across (m) | +-1.82, 0.0244 | +-0.60, 0.0244 | -0.477/0.481; -0.591/0.588; -0.476/0.474 | 33 -> 44, 6.9% -> 5.7% |
| heading (rad) | +-0.100, 7.7e-5 | +-0.039, 7.7e-5 | -0.0385/0.0385; -0.0378/0.0379; -0.0342/0.0343 | 45 -> 51, 3.6% -> 3.2% |
| speed (m/s) | +-5.39, 0.156 | +-3.6, 0.156 | -3.37/1.83; -2.51/1.59; -3.58/1.95 | 39 -> 44, 6.5% -> 5.8% |
| energy (store) | +-976, 21.8 | +-460, 0.18 | -380/439; -409/422; -380/455 | 35 -> 47, 72.0% -> 58.4% (56% exactly 0) |
| shield (points) | -90..0, 1 | -100..0, 1 | min -97.4; -90.2; -98.2 (99.2% exactly 0) | 1, 99.3% (irreducible) |

## Geometry heads (lean/model/geometry.py)

Rule: the 99.9% interval over the three lineups, rounded out; knees traded against balance.
Occupancy: 99% in / largest bin, 4v4 ; 1v1 ; 16v16.

| Target | Old | New | 0.1% / 50% / 99.9% (4v4) | Old occupancy | New occupancy |
|---|---|---|---|---|---|
| range (m) | 32, 0..6,700, knee 100 | unchanged | 22 / 778 / 6,448 | 23/32 6.9%; 15/32 10.6%; 22/32 11.9% | same |
| range rate (m/s) | 32, +-1,000, knee 50 | unchanged | -745 / -76 / 597 (1v1 -889) | 21 9.8%; 21 6.4%; 21 9.7% | same |
| line-of-sight rate (rad/s) | 32, +-6, knee 0.05 | unchanged (+-9 for 1v1's 8.6 tried: worse at 4v4 and 16v16) | -5.9 / 0 / 5.9 | 22 8.7%; 25 9.3%; 20 12.3% | same |
| energy difference | 32, +-33,000, linear | unchanged | -32,820 / 0 / 32,810 | 26 17.2%; 22 26.8%; 26 12.2% | same |
| train angle | 32, circle, uniform | unchanged | uniform | 32 7.2%; 32 8.0%; 32 5.4% | same |
| gun error | 32, circle, knee 0.02 | unchanged | | 30 8.6%; 31 9.6%; 29 9.2% | same |
| field edge (m) | 32, -2,000..7,000, knee 50 | 32, -1,500..6,200, knee 200 | -1,424 / 2,107 / 6,068 | 21 21.2%; 21 21.5%; 21 21.5% | 23 15.4%; 23 15.3%; 23 15.5% |
| map edge (m) | 16, -4,200..300, knee 100 | 32, -3,900..300, knee 1,000 | -3,831 / -1,516 / 60 | 6/16 37.9%; 6/16 44.1%; 5/16 46.4% | 19/32 8.3%; 20/32 15.1%; 19/32 10.9% |

Every non-circular support has 6 buffer bins (3 each end), so 6 "empty" bins are by
construction. The map edge's bin count went 16 -> 32: the zone brings every ship to the edge,
and at 16 bins (10 across the range) the best knee still held 99% in 8 bins with 21-38% in
one (linear 21.1% / 42.2% / 24.9% at 4v4 / 1v1 / 16v16; knee 1,000 at 16 bins 22.7% / 37.7% /
29.0%). New innermost bins: field edge 50 m (was 19 m; the band is 160 m), map edge 77 m (the
ramp is 300 m).

## Hit-class prior (lean/model/policy.py ModelConfig.hit_class_frequencies)

| | Bullets | Missed | Enemy | Team-mate |
|---|---|---|---|---|
| Old (2026-11-28 ~04:45, item 8a: 48 x 4,200 ticks, CPU, the egg deflector) | 1.6M | 0.9865 | 0.0110 | 0.0025 |
| New, 4v4 (the prior) | 65.0M | 0.9863 | 0.0109 | 0.0028 |
| 1v1 | 32.6M | 0.9958 | 0.0042 | 0 |
| 16v16 | 65.0M | 0.9792 | 0.0141 | 0.0067 |

## Feature scales (lean/game/core.py observe)

No change: no feature has most values above 10 or below 0.01 in scaled units (table in
lean/game/NOTES.md "Feature scales"). Noted, not changed: the angle of attack (linear / 0.25
rad) reaches 3.2 at the 99th percentile (4.0 max) with the post-stall levels; the boundary
scalar sits at its clip (1,000 m inside) in most rows until the zone closes in (p1 0.18 at 4v4);
constant columns (gun cooldown, field band, the zone centres, the next radius) are as designed.

## The smoke's value-support warning

`lean.train --stage all --bc-updates 2 --updates 1` (2026-11-28 19:31, load 3.2): EXIT 0, no
recompile; `narrow_outcome` on BC update 0 and RL update 0 (whose critic is the hand-off EMA
of two BC updates, nearly the fresh weights), not on BC update 1; no clamped warning on any
level. Not a support problem: a near-untrained critic reads about 0 everywhere, so the
outcome's lambda-return target is 10 (gamma lambda)^k = 10 x 0.9703^k at k ticks before a game
end and about 0 elsewhere. It leaves the middle bin (|target| > 0.18) only within about 130
ticks of an end (a tenth of a median 4v4 game of 1,301 ticks), and the bins far from 0 each
take only a tick or two of that, below the monitor's 0.1% per bin, so the count of occupied
bins sits near the monitor's threshold of 8. The perfect critic's targets (Monte Carlo above)
fill 30 bins at 4v4, 36 at 1v1; the warning should stop once the critic reads the outcome.
