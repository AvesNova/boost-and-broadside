# Mode timing: game length of Imugi (2026-11-15)

Measured to set clocks and tickets. Imugi plays both seats, at its default
temperature, fog on, the preset's fields on (10), mirrored lineups (each slot the same ship
type on both teams; environments alternate the slot patterns Yongma,Jujak,... and
Jujak,Yongma,...; these runs alternated them every environment, `bench.mode_dynamics` now
alternates them every pair of environments, both seats of a pair on one pattern). Each environment's first game only. CPU, eager (`bench.mode_dynamics
--eager --fields`; the runs are in `bench/mode_timing_run.sh`, the tables below are
`bench/mode_timing_report.py` over `data/*.jsonl`, full output in `report.txt`). Definitions
are in that script's docstring.

## Values the real-scale presets use

| | real_scale_deathmatch | real_scale_elimination | real_scale_frontline |
|---|---|---|---|
| arena radius | 4200 m (overridden from 7800 m) | 7800 m | 7800 m |
| clock (`episode_seconds`) | 600 s (2 x 300) | 180 s (2 x 90) | 600 s (2 x 300) |
| respawn wait | 10 s (preset override) | none | 8 s (2 x 4), then the team's wave (every 20 s) |
| win | `kills_per_ship` 5 (default) x 5 = 25 kills | wipe (kills_per_ship 1 x 5) | front +/-3 captures |
| other | | shrink: hold 36 s, end lead 8 s, law 0 (linear) | capture 20 s, lock 19 s/sqrt(km) |

`rules.py`: `CLOCKS`, `REAL_SCALE_CLOCK = 2`, `REAL_SCALE_DEATHMATCH_ARENA`; the 36 s hold,
20 s capture and 20 s wave come from `ships/real-scale/rules.toml [mode]`; the 8 s lead and
law 0 are the `ModeRules` defaults.

## 1. Deathmatch kill rate

80 games, 5v5, 4200 m, 10 fields, respawn 10 s, tickets off, a 900 s clock (a 15-minute
game: the longest candidate limit; the preset's own clock is 600 s).

Deaths per minute (a death scores for the other team, whatever its cause):

| | mean | sd | 95% interval of the mean |
|---|---|---|---|
| per team (160 team-games) | 4.57 | 1.08 | [4.40, 4.74] |
| both teams, per game (80) | 9.14 | 2.11 | [8.68, 9.60] |

Per-game total over 15 min: 65 to 190 deaths (mean 137, sd about 32), so the game-to-game
spread is wide: the slowest game has half the mean's deaths. First death 12.7 s in (mean;
median 9.9 s).

Time profile (both teams, deaths per game-minute, mean over the 80 games): 8.4 in the first
minute, then 9.0 to 9.6 in every later minute, no trend through minute 15 (per-minute sd over
games 2.6 to 3.0, so the per-minute means carry about +/-0.3 of noise). The opening is
about 8% slow; after it the rate is steady.

### Tickets for a given share of games ended by tickets

A game ends when a team's kills reach T (the tickets of the other team run out), so a
game's end is the earlier of its two teams' T-th point. Read directly from the 80 games'
point times (no rate model). T is kills (equivalently tickets per team);
`kills_per_ship` = T / 5. The table gives the largest T whose share ended before the limit
is at least the target (shares are steps of 1/80 = 1.25 points, and the curve is flat
near the targets, so a few tickets move the share little).

| time limit | 70% by tickets | 80% by tickets | 90% by tickets |
|---|---|---|---|
| 10 min | 42 (75%) | 40 (80%) | 30 (93%) |
| 12 min | 51 (70%) | 47 (81%) | 38 (90%) |
| 15 min | 65 (70%) | 55 (82%) | 48 (90%) |

(Realised share in brackets; the target's T above lies between the listed one and the
next integer.) Shares of games ended by tickets, by T and limit, for orientation:

| T | 10 min | 12 min | 15 min |
|---|---|---|---|
| 25 (today's) | 0.99 | 1.00 | 1.00 |
| 30 | 0.93 | 0.97 | 1.00 |
| 35 | 0.86 | 0.96 | 1.00 |
| 40 | 0.80 | 0.85 | 0.97 |

Uncertainty: with 80 games a share near 0.8 has a binomial standard error of about 0.045, so
the T for a target moves by roughly +/-5 tickets at the 80% row (more at 70%, because the
slowest games are spread over a long range). Today's 25 kills end 99% of games before 10
minutes: the 600 s clock is rarely reached.

## 2. Elimination: time until one team is fully dead

128 games each, 5v5, 7800 m, 10 fields, no respawn, lineup alternating as above.

| | no shrink (hold 10^5 s, clock 600 s) | today's shrink (hold 36 s, clock 180 s) |
|---|---|---|
| games ended by a wipe | 124 | 128 |
| never ended | 4 (3.1%) | 0 |
| p5 | 32 s | 32 s |
| p10 | 38 s | 38 s |
| p25 | 43 s | 43 s |
| p50 | 52 s | 52 s |
| p75 | 75 s | 70 s |
| p80 | 90 s | 90 s |
| p90 | 120 s | 126 s |
| mean (sd) of games that ended | 68 s (54 s) | 64 s (33 s) |
| deaths by the boundary (no shooter) | 3 of 834 (0.4%) | 21 of 880 (2.4%) |
| games with a boundary death | 3 (2.3%) | 14 (10.9%) |

Percentiles are over all 128 games, a never-ended game counting as longer than any
percentile shown. The "no shrink" run is not quite shrink-free: the shrink's hold is clamped
to one tick before its end (`bnb.game.layout`), so with a long hold the boundary still
collapses in one tick 8 s before the clock (592 s) and kills whoever is left. Four games
ended there (592.2 to 593 s); they are counted as never ended, and their deaths are left
out. The longest wiped games of this run ended at 363 s and 456 s.

The two runs share seeds, so they are the same games until the shrink begins at 36 s; the
percentiles up to p50 coincide for that reason, not as two independent samples. Without the
shrink 3% of games do not finish in 10 minutes. With today's shrink every game ends,
the latest at 153 s. Share of deaths caused by the boundary with today's shrink: 2.4%
(21 of 880); without it 0.4% (ships leaving the fixed edge). A death with no shooter is
counted as the boundary's, so a ship the boundary kills a tick before an enemy would have is
counted as the boundary's. 10.9% of games with the shrink contain a boundary death.

## 3. Frontline at small team sizes

128 games each, the arena by the density rule (`api.arena_radius_for`, a smaller game on
the preset's map, 10 fields, floors keep the zone ring): 2v2 at 4933.2 m, 3v3 at 6041.9 m;
real-scale rules (capture 20 s, threshold 3, 600 s clock, respawn 8 s and 20 s waves).
Definitions as in `bench/front_reversal.py` (a move is a capture; P(reversal) pools
consecutive move pairs; intervals: Wilson and a game-clustered bootstrap).

| | 2v2 | 3v3 |
|---|---|---|
| moves per game (95%) | 1.86 [1.62, 2.10] | 3.55 [3.25, 3.85] |
| games with no move | 29 (23%) | 0 |
| mean game length | 566 s [549, 583] (median 600) | 469 s [440, 498] (median 570) |
| ended by capture | 19 (14.8%, Wilson [9.7, 22.0]%) | 67 (52.3%, [43.7, 60.8]%) |
| ended on the clock | 109 | 61 |
| move pairs, reversals | 139, 31 | 326, 91 |
| P(reversal) Wilson | 0.223 [0.162, 0.299] | 0.279 [0.233, 0.330] |
| P(reversal) bootstrap | [0.146, 0.300] | [0.235, 0.323] |

P(reversal) is well below 0.5 in both: the front persists in the direction of the last
capture, unlike a random walk. Moves per game and the capture share rise with team size; at 2v2 the
game is mostly decided by the clock (85%) and 23% of games have no capture at all in ten
minutes; at 3v3 about half end by capture. This is a sanity check, not a verdict; I did not
test why 2v2 captures are rare.

## Caveats

- Imugi only, mirrored lineups: both teams are equally strong, so deathmatch
  per-team rates are symmetric by construction and a trained agent's kill rate will differ.
- Deathmatch has 80 games (CPU time); the first 32 ran while the machine was loaded and the
  rest alone, with identical settings. The runs are first games per environment; a game
  that ended early in an environment is not replaced.
- Games ended by tickets would change nothing before the end (no feedback from a ticket
  lead to the agents), so reading the end from the tickets-off games is exact for the scripted
  agent.

## 4. Elimination: game length by map size and by shrink (2026-11-15, GPU)

Imugi on both seats, 5v5, fields on, mirrored lineups, 256 games per row, seed 61. Runs: `bench/elimination_static_run.sh`, `bench/elimination_shrink_run.sh`; tables: `bench/elimination_static_report.py`, `bench/elimination_shrink_report.py` over `data/elimination_static_radius.jsonl` and `data/elimination_shrink_sweep.jsonl` (the run scripts write a combined data.jsonl; renamed here). Spawns, fields and the boundary ramp scale with the radius.

**Fixed map, no shrink (900 s clock):**

```
radius |  mean median  p80  p90 | never ended | deaths/min (both) | boundary share
  7800 |    91     58  106  156 |        0.00 |               4.5 |          0.007
  6000 |    74     54   87  122 |        0.00 |               5.6 |          0.005
  4500 |    71     51   82  115 |        0.00 |               5.9 |          0.019
  3500 |    56     47   70   98 |        0.00 |               7.7 |          0.047
  2500 |    61     42   73  101 |        0.00 |               7.4 |          0.088
  1800 |    45     37   56   77 |        0.00 |              10.5 |          0.171
  1200 |    31     26   44   67 |        0.00 |              15.9 |          0.198
   800 |    18     17   25   29 |        0.00 |              26.7 |          0.229
   500 |    12     11   16   19 |        0.00 |              42.0 |          0.359
   300 |     7      6    8   10 |        0.00 |              78.8 |          0.400
   100 |     3      3    4    4 |        0.00 |             188.2 |          0.325
```

**Shrink sweep** (start = shrink_hold_seconds, dur = its duration; clock = start + dur + 8 s; 7800 m):

```
start dur clock |  mean median  p80  p90 | ended before start | radius<75% at end | <50% | boundary deaths | games w/ one | clock ends
    0  60    68 |    46     47   54   56 |               0.00 |              1.00 | 0.95 |           0.332 |         0.70 |       0.00
    0  90    98 |    54     54   70   78 |               0.00 |              1.00 | 0.65 |           0.150 |         0.37 |       0.00
    0 135   143 |    59     51   83   99 |               0.00 |              0.91 | 0.28 |           0.045 |         0.12 |       0.00
    0 180   188 |    66     54   90  129 |               0.00 |              0.69 | 0.20 |           0.033 |         0.14 |       0.00
   18  60    86 |    53     55   67   71 |               0.01 |              0.97 | 0.59 |           0.181 |         0.38 |       0.00
   18  90   116 |    60     56   81   92 |               0.01 |              0.78 | 0.38 |           0.093 |         0.24 |       0.00
   18 135   161 |    64     55   88  115 |               0.01 |              0.57 | 0.21 |           0.052 |         0.14 |       0.00
   18 180   206 |    70     58  100  132 |               0.01 |              0.41 | 0.16 |           0.033 |         0.12 |       0.00
   36  60   104 |    60     56   82   87 |               0.09 |              0.59 | 0.38 |           0.125 |         0.29 |       0.00
   36  90   134 |    63     57   91  106 |               0.09 |              0.48 | 0.26 |           0.053 |         0.15 |       0.00
   36 135   179 |    68     57   95  125 |               0.09 |              0.37 | 0.18 |           0.050 |         0.13 |       0.00
   36 180   224 |    71     58   96  148 |               0.09 |              0.28 | 0.12 |           0.023 |         0.08 |       0.00
   60  60   128 |    66     58   96  107 |               0.52 |              0.37 | 0.23 |           0.060 |         0.17 |       0.00
   60  90   158 |    69     58  105  124 |               0.52 |              0.29 | 0.20 |           0.050 |         0.14 |       0.00
   60 135   203 |    72     58  106  134 |               0.52 |              0.23 | 0.14 |           0.026 |         0.11 |       0.00
   60 180   248 |    74     58  105  146 |               0.52 |              0.19 | 0.09 |           0.019 |         0.09 |       0.00
   90  60   158 |    71     58  109  127 |               0.74 |              0.22 | 0.16 |           0.054 |         0.14 |       0.00
   90  90   188 |    74     58  114  147 |               0.74 |              0.20 | 0.12 |           0.030 |         0.11 |       0.00
   90 135   233 |    77     58  111  157 |               0.74 |              0.16 | 0.09 |           0.023 |         0.09 |       0.00
   90 180   278 |    78     58  114  156 |               0.74 |              0.12 | 0.06 |           0.009 |         0.05 |       0.00
```

- Down to about 2.5 km the radius hardly matters: deaths per minute 4.5 to 7.4, median 58 to 42 s. Below about 2 km the kill rate rises roughly as one over the radius, and the boundary's share of deaths goes from under 5% to 17-40%.
- So a linear shrink from 7.8 km acts only in its last quarter or so: the median stays at 51-58 s in every swept setting except start 0 with a 60 s shrink (47 s, a third of deaths by the boundary). The shrink moves the tail (p80, p90) and the mean.
