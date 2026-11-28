# Item 13: shrinking zone (branch zone-13)

Started 17:53 (power performance, load 1.3). ETA 20:00 (worst 20:50).

## Log

- 17:53 read the brief, queue, notes and code.
- 18:05 zone, no clock, derived bound (3,934 ticks), zone globals, scripted boundary on the
  zone, tests (test_zone.py) committed; quick tier scripted/model/eval/tool 163 passed.
- 18:06 behaviour unit (yardstick 1v1 2,048 / 4v4 1,024 games) started; replays on the CPU.
  1v1 done: every game ends by a wipe (99.8% one team, 0.2% both), median 59.8 s, p90 111.3 s,
  max 121.6 s; 15.0% of deaths by the boundary.
- 18:20 queued bnb-13-gpu: real trainer 2 BC + 2 RL static, then the cold compile check.
  ETA all done 19:15 (worst 19:45).
- 18:11 behaviour done (tables: zone-13/tables.md). 4v4: every game a wipe (0.10% both),
  median 52.3 s, p90 76.2 s, max 118.4 s; 3.6% of deaths by the boundary.
- 18:15 replays: data/replays-2026-11-28-zone13/ (1v1 seeds 0-2: 66.1 / 104.8 / 74.7 s; 4v4:
  84.3 / 70.8 / 47.7 s; all wipes), the zone drawn (yellow edge, red ramp end, map faint).
- 18:22 real trainer (lean.train, static, 2,048 environments): 2 BC + 2 RL updates, no
  recompile (fail_on_recompile); second-update rates 8,698 (BC) / 8,603 (RL) steps/s with the
  quick tier running on the CPU beside it (not a timing).
- 18:26 compile check: within the baseline; 6 graphs, 11,477 traced operations (item 12:
  11,439), 0 graph breaks, cold 172.9 s at load 3.9 (budget 175 s: little headroom).
- Quick tier (game, scripted, model, eval, tool, learn) 410 passed, 3 skipped; ruff clean.
  One learn test changed: it relied on the clock's tie-break for wins at a 6-tick test clock;
  a test-clock end is now a draw (wins are covered by test_rewards).
- 18:32 slow learn: the fullgraph league-tick test failed: observe's math.isfinite on the
  stage's end breaks the graph under dynamic shapes. Fixed (a check of the shrink speed).
- 18:38 recheck on the fixed tree (919209e1), quiet (load 1.4): compile check within the
  baseline, 11,477 traced operations, 0 graph breaks, cold 161.3 s (budget 175 s); slow learn
  6 passed. Done 18:47.
