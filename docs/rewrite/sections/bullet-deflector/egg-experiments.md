# The egg deflector in play (2026-11-27 morning, branch `deflector-lookahead`)

Proposal measurements; nothing in core is tuned here. Scripts: `egg_play.py` (play cells),
`egg_tables.py` (these tables), `egg_ttk.py` (time to kill from the workbench), `egg_replay.py`
(replays), `deflector_views.py` (hit widths, bullet paths). Records: `egg_runs/`.

## Setup

- The look-ahead scripted agent (round 3 rows, `lead_error` 0.25) on both teams, compiled, on
  the laptop's 4070. 1v1: fog off, no fields, 512 games; 4v4: GameConfig() (fields, fog), 256
  games; 60 s of play per cell (1,800 ticks), finished games re-dealt; seed 0.
- Cells: the hit model's rows (`core.with_hit_model`, the ship card too): "off" = the egg's
  code with no turn (straight paths; the same as GameConfig(deflector="off")); "egg" = the
  default (45 deg, early hump, egg 53.3/24/24 m). Nose cone off unless "nosecone" (the old
  rule: deflector off, nose cone on). Spread: today's 3.08 m/s per axis (1.5 mrad at the 2,084
  m/s launch speed), 5, 10, 20 mrad.
- Columns: kills per minute (per game); TTK = a ship's last full shield to its death; draws
  (share of games ended); hit chance per real bullet (its flight ended on a ship); shots per
  kill (real bullets ended per death); friendly fire (share of real bullet hits on a
  team-mate); head-on kills (the killer within 30 deg of the victim's nose); fire share (of
  living ship-ticks); store (mean, of full); power use (coast/dive/climb); the agent's joint
  entropy (nats); the nearest enemy's distance at a shot (median); stall (< 1.1 stall speed);
  in enemy egg (a ship's centre inside an enemy's egg); enemy < 30 m. 1v1 also: passes (local
  minima of the distance under the gun range, 1 s apart), hits per pass, passes with no hit,
  hits in control (shooter in range, nose within 30 deg of the target, behind its beam), merge
  turn (after a pass with both noses within 60 deg of the other: until the heading changed by
  90 deg, median), hits by aspect (bullet path within 30 deg of the target's nose / side /
  within 30 deg of its tail).

## The geometry (workbench, the game's own step)

Hit width (aim offsets that hit), target at cruise, 300 m: off 24 m at every aspect; egg 45 deg
4.7 m astern and side, 2.1 m head-on (the game's table for a still ship: 4.8 / 4.8 / 2.0; the
sketch: 4.75 / 4.75 / 2.05); 30 deg 7.1 / 6.8 / 3.1; 60 deg 3.5 / 3.5 / 1.5
(`workbench/apparent_size.png`, `workbench/bullet_paths.png`). The differences from the sketch
(0.1 m) come from the target's motion over the tick and the drag's slow curving of the relative
path; the core taper changes the damage near the edge, not the width.

Time to kill with perfect aim from astern (workbench; shield / expected damage + 1 / hit chance
shots at 30 shots/s; no recharge): `workbench/egg_ttk.md`. The author's band (0.4 s floor
close, at most about 1.0 s for a well-aimed burst at range):

| hit damage | spread | TTK 100 m | TTK 500 m | in band |
|---|---|---|---|---|
| 7 | 1.5 mrad (today) | 0.48 | 0.68 | yes |
| 8.5 | 1.5 mrad | 0.40 | 0.57 | yes |
| 8.5 | 5 mrad | 0.40 | 0.94 | yes |
| 10.5 (today) | 1.5 mrad | 0.33 | 0.47 | below the floor |
| 10.5 | 5 mrad | 0.33 | 0.77 | below the floor |
| 7 | 5 mrad | 0.48 | 1.13 | over at range |
| any | 10-20 mrad | | 1.1-4.0 | over at range |

So the candidates are hit damage about 8.5 with a spread up to about 5 mrad (10 to 12 hits to
kill), or 7 with today's spread. Today's 10.5 is 10 hits, 0.33 s with every shot hitting.
With a 3 mrad aim error on top the 500 m times grow 1.5-2x (egg_ttk.md); from the side the
times are within a few per cent of astern, head-on (2 m wide) about 1.4x.

## Play: the spread grid

| cell | kills/min | TTK median s | TTK p25-p75 s | draws | hit chance/bullet | shots/kill | friendly fire | head-on kills | fire share | store | power coast/dive/climb | entropy | fire dist median m | stall | in enemy egg | enemy <30 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1v1-off-today | 3.25 | 0.80 | 0.37-4.53 | 0.02 | 0.508 | 35 | 0.000 | 0.45 | 0.04 | 0.52 | 0.63/0.14/0.23 | 1.10 | 279 | 0.000 | 0.001 | 0.000 |
| 1v1-off-5mrad | 3.29 | 0.80 | 0.40-4.70 | 0.01 | 0.522 | 35 | 0.000 | 0.49 | 0.04 | 0.51 | 0.63/0.14/0.23 | 1.10 | 282 | 0.000 | 0.000 | 0.000 |
| 1v1-off-10mrad | 3.23 | 1.07 | 0.43-4.87 | 0.01 | 0.521 | 34 | 0.000 | 0.45 | 0.04 | 0.51 | 0.63/0.14/0.23 | 1.10 | 281 | 0.001 | 0.001 | 0.000 |
| 1v1-off-20mrad | 3.23 | 3.07 | 0.53-5.13 | 0.01 | 0.524 | 35 | 0.000 | 0.47 | 0.04 | 0.51 | 0.63/0.14/0.23 | 1.10 | 279 | 0.001 | 0.001 | 0.000 |
| 1v1-egg-today | 1.45 | 7.80 | 4.20-12.90 | 0.01 | 0.252 | 59 | 0.000 | 0.00 | 0.03 | 0.45 | 0.53/0.25/0.22 | 1.01 | 221 | 0.002 | 0.002 | 0.001 |
| 1v1-egg-5mrad | 1.43 | 7.65 | 4.04-11.76 | 0.01 | 0.272 | 56 | 0.000 | 0.00 | 0.02 | 0.43 | 0.53/0.26/0.22 | 1.01 | 213 | 0.003 | 0.002 | 0.002 |
| 1v1-egg-10mrad | 1.43 | 7.63 | 3.90-11.77 | 0.01 | 0.296 | 50 | 0.000 | 0.00 | 0.02 | 0.43 | 0.54/0.25/0.21 | 1.01 | 196 | 0.002 | 0.002 | 0.001 |
| 1v1-egg-20mrad | 1.39 | 6.72 | 3.63-11.19 | 0.01 | 0.289 | 50 | 0.000 | 0.00 | 0.02 | 0.44 | 0.53/0.25/0.22 | 1.01 | 188 | 0.003 | 0.002 | 0.001 |
| 4v4-off-today | 11.13 | 0.62 | 0.37-3.59 | 0.01 | 0.397 | 39 | 0.008 | 0.45 | 0.04 | 0.55 | 0.54/0.17/0.29 | 1.01 | 299 | 0.004 | 0.001 | 0.000 |
| 4v4-off-5mrad | 11.09 | 0.73 | 0.40-4.20 | 0.01 | 0.391 | 40 | 0.008 | 0.45 | 0.04 | 0.56 | 0.53/0.17/0.29 | 1.00 | 301 | 0.004 | 0.001 | 0.000 |
| 4v4-off-10mrad | 10.72 | 0.87 | 0.43-4.50 | 0.01 | 0.388 | 41 | 0.009 | 0.44 | 0.04 | 0.56 | 0.55/0.16/0.29 | 1.00 | 308 | 0.004 | 0.001 | 0.000 |
| 4v4-off-20mrad | 11.17 | 0.97 | 0.50-4.50 | 0.01 | 0.397 | 40 | 0.009 | 0.44 | 0.04 | 0.55 | 0.54/0.17/0.29 | 1.01 | 302 | 0.004 | 0.001 | 0.001 |
| 4v4-egg-today | 6.57 | 7.57 | 2.53-14.43 | 0.00 | 0.238 | 62 | 0.017 | 0.10 | 0.04 | 0.45 | 0.47/0.28/0.25 | 0.80 | 199 | 0.020 | 0.009 | 0.007 |
| 4v4-egg-5mrad | 6.66 | 7.27 | 2.61-13.97 | 0.01 | 0.243 | 60 | 0.017 | 0.10 | 0.04 | 0.46 | 0.47/0.27/0.26 | 0.81 | 192 | 0.017 | 0.008 | 0.006 |
| 4v4-egg-10mrad | 6.56 | 7.20 | 3.00-14.07 | 0.00 | 0.245 | 61 | 0.015 | 0.09 | 0.03 | 0.46 | 0.47/0.27/0.26 | 0.80 | 185 | 0.019 | 0.009 | 0.007 |
| 4v4-egg-20mrad | 6.35 | 7.95 | 3.27-15.66 | 0.00 | 0.247 | 59 | 0.019 | 0.09 | 0.03 | 0.44 | 0.47/0.28/0.25 | 0.79 | 169 | 0.023 | 0.009 | 0.007 |

| cell | passes | hits/pass | no-hit passes | hits in control | merge turn 90 s | hits head-on/side/rear |
|---|---|---|---|---|---|---|
| 1v1-off-today | 3211 | 9.09 | 0.15 | 0.06 | 1.87 | 0.67/0.32/0.00 |
| 1v1-off-5mrad | 3214 | 9.49 | 0.13 | 0.05 | 1.83 | 0.71/0.29/0.00 |
| 1v1-off-10mrad | 3180 | 9.24 | 0.12 | 0.06 | 1.83 | 0.69/0.31/0.00 |
| 1v1-off-20mrad | 3190 | 9.43 | 0.09 | 0.05 | 1.80 | 0.72/0.28/0.00 |
| 1v1-egg-today | 4085 | 2.72 | 0.43 | 0.70 | 2.20 | 0.09/0.57/0.34 |
| 1v1-egg-5mrad | 4098 | 2.72 | 0.44 | 0.68 | 2.17 | 0.10/0.56/0.34 |
| 1v1-egg-10mrad | 4096 | 2.68 | 0.44 | 0.71 | 2.20 | 0.08/0.55/0.37 |
| 1v1-egg-20mrad | 4059 | 2.56 | 0.47 | 0.78 | 2.17 | 0.05/0.51/0.44 |

## Play: largest turn, hit damage, egg front, the old rule

| cell | kills/min | TTK median s | TTK p25-p75 s | draws | hit chance/bullet | shots/kill | friendly fire | head-on kills | fire share | store | power coast/dive/climb | entropy | fire dist median m | stall | in enemy egg | enemy <30 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1v1-egg30-today | 1.56 | 7.00 | 3.85-10.55 | 0.00 | 0.371 | 42 | 0.000 | 0.02 | 0.02 | 0.45 | 0.54/0.24/0.22 | 1.02 | 205 | 0.004 | 0.002 | 0.001 |
| 1v1-egg60-today | 1.23 | 8.83 | 4.13-13.98 | 0.02 | 0.191 | 79 | 0.000 | 0.00 | 0.03 | 0.43 | 0.53/0.26/0.22 | 1.00 | 230 | 0.001 | 0.003 | 0.001 |
| 1v1-egg-today-damage7 | 1.23 | 9.93 | 5.08-15.26 | 0.01 | 0.253 | 88 | 0.000 | 0.00 | 0.03 | 0.43 | 0.52/0.25/0.22 | 1.00 | 215 | 0.002 | 0.003 | 0.001 |
| 1v1-egg-today-damage14 | 1.48 | 7.03 | 3.58-11.43 | 0.00 | 0.249 | 48 | 0.000 | 0.01 | 0.02 | 0.45 | 0.54/0.24/0.22 | 1.02 | 219 | 0.002 | 0.002 | 0.001 |
| 4v4-egg30-today | 7.79 | 5.47 | 1.63-11.40 | 0.01 | 0.286 | 52 | 0.012 | 0.11 | 0.04 | 0.48 | 0.49/0.25/0.26 | 0.84 | 203 | 0.010 | 0.005 | 0.004 |
| 4v4-egg60-today | 5.87 | 8.93 | 3.14-17.86 | 0.00 | 0.205 | 71 | 0.021 | 0.10 | 0.04 | 0.45 | 0.47/0.28/0.25 | 0.78 | 201 | 0.027 | 0.011 | 0.009 |
| 4v4-egg-today-damage7 | 5.95 | 9.78 | 4.28-18.43 | 0.00 | 0.264 | 80 | 0.014 | 0.10 | 0.04 | 0.45 | 0.46/0.28/0.25 | 0.79 | 185 | 0.026 | 0.010 | 0.008 |
| 4v4-egg-today-damage14 | 7.12 | 5.63 | 1.57-12.10 | 0.01 | 0.219 | 53 | 0.015 | 0.11 | 0.03 | 0.47 | 0.48/0.26/0.25 | 0.82 | 209 | 0.014 | 0.007 | 0.005 |
| 1v1-egg-today-front24 | 1.53 | 8.03 | 4.37-12.73 | 0.00 | 0.287 | 55 | 0.000 | 0.03 | 0.03 | 0.45 | 0.54/0.24/0.22 | 1.01 | 222 | 0.002 | 0.001 | 0.001 |
| 1v1-egg-today-front80 | 1.39 | 8.13 | 4.10-12.83 | 0.01 | 0.247 | 62 | 0.000 | 0.00 | 0.03 | 0.44 | 0.53/0.25/0.22 | 1.00 | 222 | 0.002 | 0.006 | 0.001 |
| 4v4-egg-today-front24 | 7.10 | 6.82 | 1.90-13.86 | 0.00 | 0.265 | 57 | 0.013 | 0.16 | 0.04 | 0.48 | 0.49/0.25/0.26 | 0.83 | 203 | 0.013 | 0.003 | 0.005 |
| 4v4-egg-today-front80 | 6.39 | 7.00 | 2.57-14.43 | 0.00 | 0.232 | 64 | 0.014 | 0.09 | 0.04 | 0.46 | 0.47/0.27/0.26 | 0.81 | 201 | 0.019 | 0.013 | 0.006 |
| 1v1-off-today-nosecone | 1.85 | 4.10 | 3.37-6.80 | 0.00 | 0.528 | 27 | 0.000 | 0.01 | 0.02 | 0.47 | 0.54/0.23/0.23 | 1.03 | 106 | 0.008 | 0.004 | 0.003 |
| 4v4-off-today-nosecone | 8.84 | 2.70 | 0.48-5.90 | 0.00 | 0.374 | 40 | 0.023 | 0.02 | 0.03 | 0.51 | 0.50/0.22/0.27 | 0.91 | 207 | 0.010 | 0.004 | 0.003 |

| cell | passes | hits/pass | no-hit passes | hits in control | merge turn 90 s | hits head-on/side/rear |
|---|---|---|---|---|---|---|
| 1v1-egg30-today | 3945 | 3.14 | 0.37 | 0.53 | 2.20 | 0.15/0.65/0.19 |
| 1v1-egg60-today | 4087 | 2.33 | 0.50 | 0.78 | 2.17 | 0.07/0.50/0.43 |
| 1v1-egg-today-damage7 | 4066 | 3.43 | 0.44 | 0.75 | 2.17 | 0.07/0.56/0.38 |
| 1v1-egg-today-damage14 | 3984 | 2.27 | 0.44 | 0.63 | 2.17 | 0.12/0.59/0.29 |
| 1v1-egg-today-front24 | 3983 | 3.10 | 0.38 | 0.57 | 2.17 | 0.18/0.54/0.28 |
| 1v1-egg-today-front80 | 4064 | 2.66 | 0.45 | 0.70 | 2.17 | 0.07/0.59/0.34 |
| 1v1-off-today-nosecone | 3862 | 3.50 | 0.43 | 0.39 | 2.10 | 0.00/0.93/0.07 |

## God mode (shields refilled every tick; nobody dies; games end at the 60 s clock)

| cell | kills/min | TTK median s | TTK p25-p75 s | draws | hit chance/bullet | shots/kill | friendly fire | head-on kills | fire share | store | power coast/dive/climb | entropy | fire dist median m | stall | in enemy egg | enemy <30 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| god-1v1-off | 0.00 | - | - | 0.80 | 0.752 | 265082 | 0.000 | 0.00 | 0.15 | 0.24 | 0.42/0.40/0.18 | 0.91 | 278 | 0.045 | 0.011 | 0.008 |
| god-1v1-egg | 0.00 | - | - | 0.89 | 0.275 | 244727 | 0.000 | 0.00 | 0.13 | 0.39 | 0.50/0.29/0.21 | 0.90 | 223 | 0.003 | 0.017 | 0.012 |
| god-4v4-off | 0.00 | - | - | 0.25 | 0.643 | 813128 | 0.045 | 0.00 | 0.25 | 0.29 | 0.41/0.37/0.22 | 0.65 | 120 | 0.210 | 0.068 | 0.060 |
| god-4v4-egg | 0.00 | - | - | 0.53 | 0.468 | 351876 | 0.049 | 0.00 | 0.10 | 0.32 | 0.42/0.36/0.22 | 0.62 | 93 | 0.157 | 0.068 | 0.059 |

| cell | passes | hits/pass | no-hit passes | hits in control | merge turn 90 s | hits head-on/side/rear |
|---|---|---|---|---|---|---|
| god-1v1-off | 3856 | 51.73 | 0.18 | 0.80 | 2.57 | 0.09/0.52/0.39 |
| god-1v1-egg | 4400 | 15.30 | 0.26 | 0.95 | 2.23 | 0.01/0.42/0.57 |

## Store weight by situation (lower priority; author question 6)

`store_crowd` k (a new look-ahead row, 0 = off, the default): the store weight x (1 + k) /
(1 + k x the scored enemies within 2 gun ranges). "crowd" = store weight 0.5, k 1: 0.5 with
one enemy near, 0.33 with two, 0.2 with four. Duels with the egg on (the default game),
`scripted-lookahead/measure.py --mode duel` (seats alternated), 256 games x 180 s re-dealt;
1v1 fog off no fields, 4v4 fields and fog. Win share of decided games (standard error about
1.4 points in 1v1, 1.7 in 4v4):

| duel | 1v1 wins / draws / losses | 1v1 win share | 4v4 wins / draws / losses | 4v4 win share |
|---|---|---|---|---|
| crowd vs fixed 0.25 | 722 / 13 / 473 | 0.60 | 439 / 2 / 454 | 0.49 |
| crowd vs fixed 0.40 | 667 / 17 / 532 | 0.56 | 479 / 3 / 413 | 0.54 |
| fixed 0.40 vs fixed 0.25 | 701 / 12 / 542 | 0.56 | 431 / 4 / 499 | 0.46 |

The situational weight takes the 1v1 gain of a larger store weight (0.60 against 0.25) and
gives none of it back in 4v4 (0.49, even), where a fixed 0.40 loses (0.46). Mean store 0.60
(1v1) and 0.54 (4v4) of full for the duel's two sides together. Not adopted (decision for the
author); `store_crowd` stays 0.
