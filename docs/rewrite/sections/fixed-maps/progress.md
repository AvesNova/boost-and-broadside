# Fixed maps: progress

The author's rule (2026-11-15): every map is fixed-size, each game on its preset's own
arena radius at every team size, until an agent has trained successfully on variable-size
maps (the map-size arm, A1). Variable maps are an opt-in setting for that arm only.

- **Item 1, training (done):** `TrainConfig.arena_scale` defaults to None (each pool's own
  arena); `ArenaScale` is the map-size arm's opt-in setting. Tests: test_map_sampling,
  test_keyed_draws, test_training_lineup, test_mixed (the override for fixed radii there
  is no longer needed and is gone).
