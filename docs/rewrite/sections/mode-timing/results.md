# Mode timing: game length of the scripted agent (2026-11-15)

Measured to set clocks and tickets. The scripted agent plays both seats, at its default
temperature, fog on, the preset's fields on (10), mirrored lineups (each slot the same ship
type on both teams; environments alternate the slot patterns Yongma,Jujak,... and
Jujak,Yongma,...). Each environment's first game only. CPU, eager (`bench.mode_dynamics
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

- Scripted agent only, mirrored lineups: both teams are equally strong, so deathmatch
  per-team rates are symmetric by construction and a trained agent's kill rate will differ.
- Deathmatch has 80 games (CPU time); the first 32 ran while the machine was loaded and the
  rest alone, with identical settings. The runs are first games per environment; a game
  that ended early in an environment is not replaced.
- Games ended by tickets would change nothing before the end (no feedback from a ticket
  lead to the agents), so reading the end from the tickets-off games is exact for the scripted
  agent.
