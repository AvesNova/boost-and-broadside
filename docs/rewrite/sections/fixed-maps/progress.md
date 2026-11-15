# Fixed maps: progress

The author's rule (2026-11-15): every map is fixed-size, each game on its preset's own
arena radius at every team size, until an agent has trained successfully on variable-size
maps (the map-size arm, A1). Variable maps are an opt-in setting for that arm only.

- **Item 1, training (done):** `TrainConfig.arena_scale` defaults to None (each pool's own
  arena); `ArenaScale` is the map-size arm's opt-in setting. Tests: test_map_sampling,
  test_keyed_draws, test_training_lineup, test_mixed (the override for fixed radii there
  is no longer needed and is gone).
- **Item 2, evaluation (done):** `EvaluationConfig.map_rule` ("preset" by default,
  "density" the map-size arm's opt-in); `game_of` plays every condition on its preset's own
  radius at every size, break-even included (`crossover.sized_game`,
  `crossover.break_even_game`). A break-even condition's game is now the larger of the
  opponent's count and the maximum count a side (`Condition.ships_per_team` is None for
  break-even). Also fixed on the way: `read_settings` returned the opponent as its
  recorded identity ("scripted"), so a read-back config differed from the written one
  after the Imugi rename (a failing test on the integration branch).
