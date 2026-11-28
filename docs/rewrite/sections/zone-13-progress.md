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
