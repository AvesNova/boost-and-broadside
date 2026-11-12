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
