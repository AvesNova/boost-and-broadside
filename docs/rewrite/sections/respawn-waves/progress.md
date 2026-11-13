# Respawn waves — progress

Branch `tune/respawn-waves` (from `tune/frontline-design` at 2540d718), 2026-11-12, box
gpubox7 (RTX 5060 Ti). Bench-only: nothing here changes the game.

## Method

Hell Let Loose style respawns on top of the decided hard capture lock, in
`bench/frontline_design.py` (`Design.waves`, per-environment tensors like `lock`, so every
variant of a fleet size shares one compile):

- `respawn_minimum` m (s): a dead ship waits at least m s;
- `wave_interval` I (s): it then respawns at its team's next wave; each team's waves come
  every I s from a random phase per team at the episode start (I = 0: no waves, respawn at
  m);
- `wave_effect`, on the team that lost the zone at a capture: none, `reset` (its next wave
  a full I away), `delay X` (its wave clock pushed back X s), `skip` (pushed back I).

A ship's wait is counted from its death tick to its respawn tick (the record's
`respawn_wait`: every respawn of each environment's first game, 0.25 s bins and the exact
sum). Checked by `tests/sections/test_respawn_waves.py`: a synthetic tick loop with deaths
and captures at chosen ticks, the respawn ticks exactly as the wave schedule gives them.

Settings as the frontline design search: real_scale_frontline, no fields, fog off, 2 Yongma
\+ 3 Jujak a team, scripted mirror T = 1, the tree's cards, the hard lock L = 45 / 55 s at
5v5 / 10v10 (radius 7800 / 11031 m), clock 1200 s and threshold off, replayed at threshold
2 / 3 and clock 900 / 1200 by `bench/respawn_waves_report.py`. The game's wait is 16 s
fixed (frontline) and 10 s fixed (deathmatch). Deathmatch: real_scale_deathmatch, no
tickets, 600 s. Run with `run.sh`; records in `records/`.

## Results

## 2026-11-13 06:10 KST: deathmatch 5v5 (main session; the agent's queue stalled)

The capture-effect and 10v10 runs never started: the run script's waiting loop tested
`tmux has-session -t waves`, which matched its own session `waves2` by prefix, so it spun
until it was killed at about 06:00. The records of the runs that finished (w5, dm5) are in
`records/`. Deathmatch 5v5, 512 games each:

| variant | kills / min | life median (s) | trade share (1 s) | between deaths median (s) | first hit to first kill (s) | wait mean / p90 (s) |
|---|---|---|---|---|---|---|
| fixed 10 s | 5.11 | 86.5 | 0.079 | 9.2 | 12.1 | 10.0 / 10.2 |
| min 10, waves 15 s | 4.73 | 87.5 | 0.072 | 10.0 | 11.6 | 17.4 / 23.5 |
| min 10, waves 20 s | 4.59 | 88.5 | 0.070 | 10.5 | 12.1 | 20.0 / 28.2 |
| min 10, waves 30 s | 4.38 | 87.5 | 0.064 | 10.9 | 11.4 | 24.9 / 37.0 |

Waves cost deathmatch pace (kills a minute down 7-14%) at the same life; they don't change
trades or engagement. Still to run: the capture effects at 5v5 and the 10v10 frontline set.

## 2026-11-13 08:40 KST: frontline 5v5 minimum × interval, deathmatch fixed timers (gpubox4)

Box gpubox4 (RTX 5060 Ti), own clone and venv under `/workspace/waves2`. Frontline 5v5,
lock 45 s, 512 games each; `w5` (2026-11-12) and `w5b` (today) together. Threshold 2 /
clock 1200 s, threshold 3 / clock 900 s in brackets:

| variant | P(same team next) | by the front | median length (s) | first capture (s) | wait mean / p90 (s) |
|---|---|---|---|---|---|
| fixed 16 | 0.494 (0.491) | 0.85 (0.33) | 570 | 182 | 16.0 / 16.2 |
| min 10, no waves | 0.519 (0.531) | 0.82 (0.38) | 561 | 214 | 10.0 / 10.2 |
| min 5, waves 15 | 0.498 (0.492) | 0.88 (0.35) | 546 | 200 | 12.5 / 18.8 |
| min 5, waves 20 | 0.490 (0.493) | 0.86 (0.40) | 541 | 184 | 15.0 / 23.0 |
| min 5, waves 25 | 0.478 (0.479) | 0.85 (0.38) | 563 | 176 | 17.5 / 27.5 |
| min 5, waves 30 | 0.467 (0.471) | 0.89 (0.42) | 484 | 153 | 19.9 / 32.2 |
| min 8, waves 15 | 0.505 (0.497) | 0.86 (0.35) | 570 | 182 | 15.5 / 21.5 |
| min 8, waves 20 | 0.481 (0.475) | 0.88 (0.37) | 506 | 174 | 17.8 / 26.0 |
| min 8, waves 25 | 0.486 (0.490) | 0.87 (0.39) | 478 | 163 | 20.6 / 30.8 |
| min 8, waves 30 | 0.481 (0.472) | 0.89 (0.37) | 491 | 149 | 23.1 / 35.2 |
| min 10, waves 15 | 0.480 (0.481) | 0.85 (0.36) | 534 | 165 | 17.5 / 23.5 |
| min 10, waves 20 | 0.482 (0.478) | 0.88 (0.38) | 499 | 164 | 19.8 / 28.0 |
| min 10, waves 25 | 0.474 (0.474) | 0.86 (0.43) | 508 | 160 | 22.5 / 32.5 |
| min 10, waves 30 | 0.506 (0.492) | 0.91 (0.46) | 446 | 144 | 25.1 / 37.2 |

P(same team next) stays at 0.47-0.53 (± 0.013-0.017) over every minimum and interval: the
waves add no momentum and remove none. What the interval moves is the pace: the first
capture comes earlier (200 → 150 s from waves 15 to 30 at minimum 5) and the median game
shortens, because a wave lands its ships together. The minimum matters less than the
interval at the same wait (min 5 waves 30 and min 10 waves 20 both wait about 20 s; the
longer interval is the faster game). The wait is about m + I/2 on average and m + 0.9 I at
the 90th percentile, as the schedule gives.

Deathmatch 5v5 (`dm5b`), fixed timers, no waves, 512 games each:

| variant | kills / min | life median (s) | trade share (1 s) | between deaths median (s) | first hit to first kill (s) | wait mean / p90 (s) |
|---|---|---|---|---|---|---|
| fixed 5 s | 5.41 | 84.5 | 0.084 | 8.7 | 12.1 | 5.0 / 5.2 |
| fixed 8 s | 5.24 | 86.5 | 0.084 | 9.1 | 11.6 | 8.0 / 8.2 |
| fixed 10 s (the game) | 5.09 | 86.5 | 0.079 | 9.3 | 12.1 | 10.0 / 10.2 |
| fixed 16 s | 4.80 | 87.5 | 0.073 | 10.1 | 11.4 | 16.0 / 16.2 |

The timer moves the kill rate (+6% at 5 s, +3% at 8 s, -6% at 16 s against 10 s) and
little else: life, engagement and trades barely move. A dead ship is out for 5.6 / 8.5 /
10.4 / 15.5% of its time at 5 / 8 / 10 / 16 s.

## 2026-11-13 08:50 KST: capture effects at 5v5 (e5b; lock 45, 576 games each)

On a capture, the team that lost the zone has its wave clock reset (next wave a full
interval away), delayed 10 s, or pushed back one interval (a wave skipped). Threshold 2 /
clock 1200 s, threshold 3 / clock 900 s in brackets:

| variant | P(same team next) | by the front | median length (s) | first capture (s) | wait mean / p90 (s) |
|---|---|---|---|---|---|
| min 8, waves 20, none | 0.483 (0.471) | 0.86 (0.35) | 549 | 177 | 17.8 / 26.0 |
| min 8, waves 20, reset | 0.539 (0.538) | 0.90 (0.42) | 512 | 175 | 18.7 / 27.2 |
| min 8, waves 20, delay 10 | 0.516 (0.520) | 0.90 (0.42) | 527 | 164 | 18.6 / 27.0 |
| min 8, waves 20, skip | 0.619 (0.608) | 0.94 (0.54) | 430 | 170 | 19.5 / 27.8 |
| min 5, waves 30, none | 0.468 (0.468) | 0.89 (0.42) | 499 | 153 | 19.8 / 32.2 |
| min 5, waves 30, reset | 0.586 (0.589) | 0.95 (0.58) | 429 | 164 | 22.3 / 35.0 |
| min 5, waves 30, delay 10 | 0.524 (0.521) | 0.92 (0.45) | 474 | 162 | 21.0 / 33.5 |
| min 5, waves 30, skip | 0.682 (0.690) | 0.97 (0.70) | 344 | 162 | 24.3 / 46.5 |

(± 0.011-0.015 on P.) Every capture effect is momentum: the team that just captured is
likelier to capture next, as much as the effect costs the losing team in time (delay 10 <
reset ≈ I/2 < skip = I). The effects end more games by the front and shorten them, but
they undo the lock's calibration (the lock was set where P = 0.5). Without an effect the
waves keep P at 0.47-0.49. Mean wait rises by the effect (skip at 30 s: p90 46.5 s, over
the Hell Let Loose bound).

## 2026-11-13 08:56 KST: frontline 10v10 (w10b; lock 55, radius 11031 m, 384 games each)

| variant | P(same team next) | by the front | median length (s) | first capture (s) | wait mean / p90 (s) |
|---|---|---|---|---|---|
| fixed 16 | 0.448 (0.452) | 0.95 (0.60) | 419 | 168 | 16.0 / 16.2 |
| min 5, waves 20 | 0.435 (0.440) | 0.95 (0.59) | 384 | 170 | 14.9 / 23.2 |
| min 8, waves 20 | 0.424 (0.444) | 0.95 (0.61) | 406 | 164 | 18.0 / 26.2 |
| min 5, waves 30 | 0.441 (0.444) | 0.96 (0.60) | 379 | 152 | 20.0 / 32.2 |
| min 10, waves 30 | 0.451 (0.463) | 0.96 (0.63) | 389 | 138 | 25.1 / 37.2 |

(± 0.012-0.019 on P.) As at 5v5 the waves leave P where the fixed wait has it, and the
30 s waves bring the first capture forward. But lock 55 gives P ≈ 0.44-0.45 here, against 0.489 (0.450) in the frontline design
calibration (s10, 192 games, ± about 0.025; the game source is unchanged since). With
twice the games, 55 s sits below 0.5 at 10v10: a longer lock raises P (the capturer keeps
the zone), so 10v10 wants a little more than 55 s. Locks 60 / 65 / 70 checked below.

## 2026-11-13 09:05 KST: the lock at 10v10 (w10c; 384 games each)

| variant | lock (s) | P(same team next) | by the front | median length (s) | first capture (s) |
|---|---|---|---|---|---|
| fixed 16 | 55 | 0.448 (0.452) | 0.95 (0.60) | 419 | 168 |
| fixed 16 | 60 | 0.543 (0.552) | 0.96 (0.66) | 359 | 168 |
| fixed 16 | 65 | 0.645 (0.634) | 0.97 (0.69) | 346 | 156 |
| fixed 16 | 70 | 0.706 (0.725) | 0.99 (0.79) | 327 | 165 |
| min 8, waves 20 | 55 | 0.424 (0.444) | 0.95 (0.61) | 406 | 164 |
| min 8, waves 20 | 65 | 0.640 (0.640) | 0.99 (0.74) | 320 | 149 |
| min 5, waves 30 | 55 | 0.441 (0.444) | 0.96 (0.60) | 379 | 152 |
| min 5, waves 30 | 65 | 0.589 (0.608) | 0.98 (0.72) | 354 | 149 |

P rises about 0.02 a second of lock here, so P = 0.5 sits at about 57-58 s at 10v10
(interpolated between 55 and 60; the 57 / 58 s run was stopped when the calibration chain
took the box again). The waves do not move it at 55 s; at 65 s the 30 s waves sit lower
(0.59 against 0.64, about 2 SE). At 5v5 lock 45 holds: 0.47-0.50 over every wave variant
without a capture effect.

## Recommendation

- **Frontline: minimum 8 s, waves every 20 s, no capture effect.** Wait mean 17.8 s, p90
  26 s (Hell Let Loose: about 10 s plus 30 s waves, mean about 25 s, worst 40 s), so a
  death costs 8-28 s, a little more on average than today's 16 s, and the waves land the
  team together: the game is shorter (median 506 against 570 s at 5v5) and the first
  capture earlier (174 against 182 s). P(same team next) stays at the lock's 0.5. A faster
  alternative is minimum 5 s, waves 30 s (mean 20 s, p90 32 s, median 484 s, first capture
  153 s). Capture effects all add momentum (delay 10 the least, +0.03-0.05; skip the most,
  +0.14-0.21) and would need the lock recalibrated; leave them out.
- **Deathmatch: fixed 8 s, no waves** (5.24 kills / min against 5.09 at 10 s; out 8.5% of
  the time). 5 s makes a death matter little (5.6% of the time), 16 s costs 6% of the kills.
- **Lock at 10v10: 55 s gives P ≈ 0.45; about 58 s gives 0.5.** 5v5 lock 45 holds.

These are scripted mirror games: the scripted agent does not read the wave clock, so a
human or a trained agent timing pushes to the waves may make them matter more than here.
