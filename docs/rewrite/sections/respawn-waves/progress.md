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
