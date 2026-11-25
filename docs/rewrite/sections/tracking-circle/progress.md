# Tracking circle: progress

- 06:45 tried: GameConfig.fog_tracking (default on), core.tracked_by_enemy (ticks since seen 0
  and known: no new state), sight_pairs(state, tracked) with the circle reach p / (1 - e) for
  tracked targets (torch.where, square-root free), perceive(state, revealed, tracking).
  worked: 5 new fog tests (16 in test_fog.py), each fails with tracking stubbed out; quick tier
  lean/game + lean/scripted 143 passed; ruff clean. next: measure.py tracking option and the
  16 conditions (CPU), compile check (unit bnb-tracking), replays, NOTES + core-decisions.
