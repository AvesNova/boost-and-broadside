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

## Repeat and the agent's lead error

Seed 1 (the same cells, another deal and draw) reproduces seed 0: kills per minute 1v1 off
3.30 / egg 1.44 / old rule 1.92 (seed 0: 3.25 / 1.45 / 1.85), 4v4 11.16 / 6.41 / 8.82 (11.13 /
6.57 / 8.84); hit chance per bullet within 0.01; head-on kills 0.46 / 0.00 / 0.01 in 1v1.
Records: egg_runs/seed1.

`lead_error` 0 and 1 (the agent's expected lead error in its trigger's hit chance; 0.25 the
default) change nothing measurable: 1v1 egg kills per minute 1.45 / 1.45 / 1.42, hit chance per
bullet 0.251 / 0.252 / 0.249; with 20 mrad 1.38 / 1.39 / 1.36 (egg_runs/lead0, lead1). The
trigger's hit chance is near 1 inside 300 m for today's spread whatever the lead error; the
misses come from the aim gate (below).

## The aim gate (`fire_tolerance`)

Why the hit chance per real bullet stays near 0.25 under the egg at every spread: the trigger
fires half the time at an aim error of `fire_tolerance` (1.0) target sizes, and firing costs
nothing, so it sprays. A tighter gate raises the hit chance and lowers the kill rate (fewer
shots, more passes with no hit):

| fire_tolerance | 1v1 egg kills/min | hit chance | shots/kill | 4v4 egg kills/min | hit chance | 1v1 off kills/min | 1v1 off hit chance |
|---|---|---|---|---|---|---|---|
| 1.0 (default) | 1.45 | 0.252 | 59 | 6.57 | 0.238 | 3.25 | 0.508 |
| 0.5 | 1.25 | 0.333 | 45 | 5.84 | 0.320 | 2.97 | 0.812 |
| 0.25 | 1.11 | 0.453 | 33 | 4.96 | 0.414 | 2.33 | 0.923 |

(today's spread; the 20 mrad rows move the same way: egg_runs/tol05, tol025). With free
shots, spraying wins kills; the aim gate is not the egg's limit.

A duel of `fire_tolerance` 0.5 against the default 1.0 (egg on, 256 games x 180 s): 1v1 612 /
15 / 636 (0.49, even); 4v4 319 / 2 / 588 (0.35): the tighter gate loses the team fight.

## What works

- **The head-on joust is gone.** With the nose cone's rule off (today's default) and no
  deflector, the agent jousts: 45% of kills (1v1 and 4v4) are head-on, 67-72% of 1v1 hits
  come within 30 deg of the target's nose, hits in control are 5-6%, the median time to kill
  is 0.6-0.8 s and a 1v1 can end at the merge (replay 1v1_off_today seed 0: 2.7 s). With the
  egg: head-on kills 0.00 in 1v1 and 0.09-0.11 in 4v4 (crossing shots in the melee), head-on
  hits 5-10%, hits in control 0.68-0.80 (god mode 0.95), rear hits 34-46%. The egg does with
  one geometric rule what the nose cone did with a damage rule (old rule: head-on kills 0.01,
  hits in control 0.37-0.39, 93% of hits from the side).
- **Fights last.** Kills per minute fall by about half (1v1 3.25 -> 1.45, 4v4 11.1 -> 6.6; old
  rule 1.85 / 8.8); the median time from a full shield to death rises from 0.6-0.9 s (off) and
  2.5-4.1 s (old rule) to 7-8 s. Draws stay at 0-2%. Merge turn (to 90 deg) 1.8-1.9 s (off) ->
  2.2 s (egg).
- **Nothing degenerate measured under the egg:** firing share 2-4% of ship-ticks (never-firing
  would be 0; a spray would be near 1); stall < 2.5% (4v4); a ship's centre inside an enemy's
  egg 0.2% (1v1) to 0.9% (4v4), an enemy within 30 m 0.1-0.7%: no hugging, no ramming; the
  agent's entropy 1.0 (1v1) and 0.8 (4v4) nats, as off; store 0.43-0.46 and power use about
  0.53/0.25/0.22 (1v1), unchanged by spread. God mode 4v4 shows a stalled ball (stall 16-21%,
  inside an enemy egg 7%) with the egg and without it: a god-mode artefact (nobody dies).
- **Friendly fire** stays small: 1.3-1.9% of real bullet hits in 4v4 play with the egg (0.8-0.9%
  off; 2.0-2.3% with the old rule), 4.9% in god mode (4.5% off).
- The largest turn sets the pace monotonically: 30 deg 1.56 / 7.79 kills per minute (1v1 / 4v4),
  45 deg 1.45 / 6.57, 60 deg 1.23 / 5.87; hit chance per bullet 0.37 / 0.25 / 0.19.

## What does not matter (in this agent's play)

- **The spread.** From 1.5 to 20 mrad, kills per minute move by at most 4% (1v1 egg 1.45 ->
  1.39, 4v4 6.57 -> 6.35; off 3.25 -> 3.23 and 11.1 -> 11.2), the hit chance per bullet not at
  all (0.25-0.30). The agent fires at about 200 m (median nearest-enemy distance at a shot)
  and holds fire more at a wide spread (59 -> 50 bullets per kill in 1v1); its misses are aim
  (the trigger's gate), not dispersion. The spread decides the workbench's TTK at range (the
  author's band), not this agent's kill rate.
- **Hit damage** 7 / 10.5 / 14: 1v1 kills per minute 1.23 / 1.45 / 1.48, 4v4 5.95 / 6.57 / 7.12:
  the kill rate is set by getting into a firing position, not by the hits needed.
- **Egg front** 24 / 53.3 / 80 m: 1v1 kills per minute 1.53 / 1.45 / 1.39; the shorter egg lets
  more head-on hits through (18% of hits head-on at 24 m, 9% at 53 m, 7% at 80 m).
- **The agent's lead error** (0 / 0.25 / 1): no measurable change.

## Open questions

1. The time-to-kill band: the workbench puts hit damage about 8.5 (spread up to about 5 mrad)
   or 7 (today's spread) in the author's band; today's 10.5 is under the 0.4 s floor (0.33 s).
   In this agent's play TTK is 7-8 s with any of them (aim and positioning dominate). Which
   should the band be judged on?
2. The spread: no play number depends on it with this agent; choose it from the workbench's
   TTK at range (5 mrad keeps the 500 m burst under 1 s at hit damage 8.5)?
3. The tick cost of the egg: +0.39 ms (+40%) on the GPU tick (0.98 -> 1.37 ms) after one cut;
   further cuts not tried (the turned bullets' per-bullet gathers and the bullet block's copy
   are the next candidates).
4. `store_crowd`: the situational store weight wins 1v1 (0.60 against 0.25) and is even in
   4v4; adopt (store_weight 0.5, store_crowd 1) as the yardstick's rows?
5. `fire_tolerance`: spraying wins (free shots); a shot cost or ammunition would change this
   (not in scope).

The tables above ran before the hit test's cut (a ship whose egg a bullet enters has no
straight contact that tick, so the entry and contact reductions run as one pass; it changes
only a second egg entered in the same tick). Re-run after the cut (egg_runs/recheck_after_cut):
1v1 egg 1.39 kills per minute (1.45 before), 4v4 6.67 (6.57), god-mode 4v4 friendly fire 4.4%
(4.9%): within the seed-to-seed spread.

## Damage per pass, per engagement, and always fire (2026-11-27)

Script `egg_pass.py`, results `egg_runs/pass_2026_11_27/*.json` (raw logs in
data/egg-pass-2026-11-27). Look-ahead agent on both sides, today's settings (hit damage 10.5,
spread 1.5 mrad), same game counts as `egg_play.py` (1v1: 512 environments, 4v4: 256, 1800
ticks = 60 s each, seed 0, games re-dealt). Damage is in full shields (`StepOut.hit_damage`:
bullets, plus collisions credited to the other ship; collisions are rare in these cells).
A pass is a local minimum of the 1v1 distance under the gun range, 1 s apart, events assigned to
the nearest minimum of the same game; a game that ends on a kill inside the gun range ends its
last pass at the kill tick (without that, the 1v1 off cell loses a third of its kills: the game
ends at the merge, before the distance turns up; the first run of the script did this wrongly
and was replaced). An engagement is the passes of one game in a row, ended by a kill or by the
distance rising above the gun range between two passes (or the game's end). "Always fire": after
the agent decides, every ship's fire action is 1 (the gun still waits its cooldown, and a blocked
shot at stall speed with an empty store stays blocked), and `GameConfig.fire_thrust_fraction =
1.0`: `core._fly` withholds `1 - fire_thrust_fraction` of the thrust on a tick that fires, so 1.0
withholds none (the default 0.0 withholds all of it on those ticks).

### Per pass (1v1, both shooters summed; the first row of 1v1 egg matches egg_play's 2.72 and 43%)

| 1v1 | hits mean | median [p25, p75] p90 | damage mean (shields) | damage median [p25, p75] p90 | no hit | kill | partial (>0, no kill) | damage >= 1 shield |
|---|---|---|---|---|---|---|---|---|
| egg, normal | 2.72 | 1 [0, 4] 9 | 0.24 | 0.09 [0, 0.38] 0.77 | 43.2% | 17.6% | 39.3% | 4.8% |
| off, normal | 9.17 | 8 [3, 14] 19 | 0.76 | 0.72 [0.23, 1.16] 1.61 | 15.7% | 50.7% | 33.6% | 37.8% |
| egg, always fire | 2.96 | 1 [0, 5] 9 | 0.26 | 0.11 [0, 0.39] 0.80 | 33.6% | 19.5% | 47.0% | 4.4% |
| off, always fire | 9.46 | 9 [4, 14] 20 | 0.77 | 0.72 [0.28, 1.17] 1.61 | 13.9% | 50.7% | 35.4% | 37.9% |

Passes: 4116 / 3216 / 4157 / 3256. Per shooter (one sample per pass and shooter, the attacker's
own damage): egg normal 71% of attacks do no hit, 8.8% kill, 20.9% partial, 2.3% deliver at
least a shield; the median attack that hits at all does 0.30 of a shield (mean 0.40); off normal
44.7% no hit, 25.4% kill, 30.1% partial, 16.8% at least a shield, median hitting attack 0.67.
A pass that kills (summed): egg normal 0.67 shields mean (the kill's pass also holds the
damage of earlier passes' remainder only through the shield's carry: the victim was already
hurt), off normal 1.06.

Overkill per pass: refuted for the egg in normal play. The god-mode mean (15.3 hits per pass)
came from a ship that cannot die; with real shields the egg pass delivers a median of 0.09 of a
shield, and only 4.8% of passes deliver a whole shield or more (the egg's attacker alone: 2.3%).
Off (no egg) is different: 37.8% of passes deliver a shield or more and 50.7% kill, median 0.72.

### Per engagement (1v1)

| 1v1 | engagements | passes mean (median) | damage mean (median) | share that kill | damage in a killing engagement (median) | passes to kill mean (median) | kills in one pass |
|---|---|---|---|---|---|---|---|
| egg, normal | 1617 | 2.55 (2) | 0.60 (0.60) | 44.8% | 1.15 (1.14) | 3.64 (4) | 2.9% |
| off, normal | 2253 | 1.43 (1) | 1.09 (1.15) | 72.4% | 1.34 (1.19) | 1.54 (1) | 55.2% |
| egg, always fire | 1722 | 2.41 (2) | 0.62 (0.69) | 47.1% | 1.16 (1.15) | 3.47 (3) | 2.8% |
| off, always fire | 2262 | 1.44 (1) | 1.12 (1.16) | 73.0% | 1.33 (1.20) | 1.56 (1) | 53.8% |

With the egg a kill takes 3 to 4 passes (3.64 mean, 1 pass 2.9%); without it one pass (55%).
Engagements that do not kill (55% with the egg) end by separation or the 60 s clock with 0.6
shields dealt on average.

### Damage lost to shield recharge

The shield did not refill in any cell: shield recharge is 0.0 shields in 1v1 egg (both),
0.12 in 1v1 off (of 2460 dealt), 0.62 in 4v4 egg (of 2434), 0 with always fire. The agent
essentially never coasts straight with no trigger for the wait, so no damage is lost to
recharge: every ship that dies took, over its life, 1.139 shields (1v1 egg; 1.165 off, 1.156 4v4)
= its 1.0 shield counted + 0.139 overkill (the last hit's excess; 12-14% of the damage on killed
ships) + 0.00 refilled. Damage on ships that did not die (games ended by the clock, ships alive
at the end) is 14.9% (1v1 egg), 22.2% (1v1 off), 21.2% (4v4 egg) of all damage dealt: that,
not recharge, is what does not count toward a kill.

### Firing and hits

| cell | fire action share | shots per pass | shots per kill (real bullets ended) | hit chance per real bullet |
|---|---|---|---|---|
| 1v1 egg normal | 2.5% | 11.2 mean (6 median) | 59.5 | 26.0% |
| 1v1 off normal | 3.6% | 20.1 (18) | 34.8 | 51.7% |
| 4v4 egg normal | 3.7% | - | 63.7 | 23.6% |
| 1v1 egg always fire | 100% (99.9% fired) | 422 (302) | 2236 | 0.68% |
| 1v1 off always fire | 100% | 514 (380) | 1075 | 1.71% |
| 4v4 egg always fire | 100% | - | 1107 | 1.39% |

### Always fire against normal play

| cell | kills / min | median time to kill [p25, p75] (s) | hits / pass (1v1) | damage / pass (shields) | passes with no hit |
|---|---|---|---|---|---|
| 1v1 egg normal | 1.41 | 7.87 [4.03, 12.48] | 2.72 | 0.24 | 43.2% |
| 1v1 egg always | 1.59 | 7.83 [4.23, 11.78] | 2.96 | 0.26 | 33.6% |
| 1v1 off normal | 3.21 | 0.80 [0.40, 4.61] | 9.17 | 0.76 | 15.7% |
| 1v1 off always | 3.28 | 0.73 [0.37, 4.60] | 9.46 | 0.77 | 13.9% |
| 4v4 egg normal | 6.48 | 7.27 [2.41, 14.53] | - | - | - |
| 4v4 egg always | 9.86 | 6.80 [2.70, 12.20] | - | - | - |

Always fire raises damage per pass by 8% (egg) and 1% (off) and kills per minute by 12%, 2% and
52% (1v1 egg, 1v1 off, 4v4 egg), while the shots rise 40-fold in the passes: the agent's fire
gate (2-4% of ticks) loses little of what a spray would deliver in 1v1, because the bullets
that hit are the aimed ones; the extra shots are 99% misses (hit chance 0.7-1.7%). The 4v4 gain
is the largest (a spray hits whoever crosses the line, and the thrust a shot withholds is the
cost the normal agent pays: with no firing penalty a firing ship also keeps its speed). The
cell mixes the two changes (fire every tick; no withheld thrust); they were not separated. The
theoretical maximum for this agent's flight is therefore about 1.1x the damage per pass in 1v1
and 1.5x the kills in 4v4; passes still end in a kill only 20% (egg) of the time.
