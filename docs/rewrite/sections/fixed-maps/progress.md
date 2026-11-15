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
- **Item 3, break-even (done):** `bnb crossover --map-rule` (preset by default) and the
  maximum count raised to 30 (`Crossover.maximum_count`, `CrossoverConfig.maximum_count`,
  `EvaluationConfig.maximum_count`). Unequal teams with ours above the opponent's needed no
  engine change: `CrossoverConfig.ships_per_team` was already the larger count and the
  benching takes either team's count; a 30-a-side game of each real-scale preset resets
  and steps on the CPU with (1, 20), (30, 20) and (20, 30) active, on both map rules. The
  cost: a benched slot costs about an active one, so a 30-a-side break-even game is about
  1.5 times a 20-a-side one by slot count (not measured; the GPU is busy).
  The evaluation's slow CPU tests (`test_a_small_evaluation_decides_and_decide_reads_it`,
  `test_networks_play_both_kinds_of_condition`) pass with the tiny break-even at 1 to 6
  ships against 3 (a 6-a-side game); at 1 to 3 on the tiny preset's own 400 m map neither
  scripted player crossed within 20 batches.
- **Item 4, frontline break-even left out (done):** `EvaluationConfig.unscored` (default
  `frontline_break_even`); `scored()` drops it, so it is neither played nor waited on, and
  frontline's third is shared by 5v5 and 10v10. `--play` of an unscored condition is
  refused.
