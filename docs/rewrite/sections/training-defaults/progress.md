# Training defaults: progress

Branch `feat/training-defaults` from `integrate/next-runs` (e0840b71). The six decided
defaults of [arms-todo.md](../../arms-todo.md) ("Decided, to implement"). CPU only.

## 2026-11-15 07:03 to 07:15: deathmatch outcome `kill_lead`

- **Tried:** two settings chose deathmatch scoring: the learner's outcome
  (`RewardConfig.deathmatch_outcome`, set on `ModeRules.deathmatch_outcome`) and the
  ratings' fit (`deathmatch_score` of `bnb ratings`, `bnb head-to-head`, `bnb pool create`,
  `bnb.league.tournament`). Both defaulted to kill share.
- **Worked:** both default to the kill lead now: the learner's outcome `"kill_lead"`, the
  ratings' `"win_loss"` (the same rule: the sign of the final kill difference). Kill share
  stays as the setting for an ablation. `bnb evaluate` already scored by win, draw or loss.
- **Kept:** the ratings' value keeps its name `win_loss` rather than `kill_lead`: it is the
  name recorded with every batch (`scoring`) and the rule the other modes are rated by.
- **Fixed on the way:** a record batch that names no scoring was read under "the default";
  with the default changed it would have been misread. Such batches were all written while
  kill share was the default, so `tournament.UNNAMED_SCORE = "kill_share"` names that now
  (`unnamed_scoring`), apart from the default.
- **Next:** mirrored coin-flip lineups.

## 07:15 to 07:40: mirrored coin-flip lineups

- **Tried:** the rollout's lineup was fixed for a run (`Rules.team_types`, or
  `TrainConfig.lineup`), and the game kept a ship's type through every reset.
- **Worked:** `TrainConfig.lineup_draw` (on by default): each episode draws its lineup
  from a keyed stream (`purpose_key("lineup")` on the episode's key, so the map and seat
  draws are unchanged), every slot one of the set's drawn types with equal chance
  (`api.DRAWN_LINEUPS`: Yongma or Jujak in the real-scale set), both teams alike. The game's
  `resolve` takes `next_ship_type` and applies it to an environment whose episode ends,
  before its ships spawn (each spawns with its new type's shield and energy); without it the
  game computes as before (the fingerprint is unchanged). The rating lanes draw too. A
  fixed `lineup` replaces the draw.
- **Ambiguous (my reading):** the decision names Yongma and Jujak, which only the
  real-scale set has. The standard set (the tests' and toy runs' set) draws Fighter or
  Heavy, its two types, so the default is on for every set with drawn types; a set without
  any (design-seed) refuses `lineup_draw` at the config.
- **Not changed:** BC's teacher-distribution sample (`cloning.teacher_distributions`, a
  calibration rollout of the scripted agent alone) still plays the game's own lineup.
- **Tests:** test_training_lineup (9 passed); the Trainer-heavy files (test_mixed,
  test_map_sampling, test_absent_fields, test_training_checkpoints, test_league_slots,
  test_lanes, test_bc, test_game_section: 145 passed) still pass with the draw on.
- **Next:** mode mix 6/5/5.

## 07:40 to 07:50: mode mix 6/5/5

- **Worked:** `TRAINING_MIXES["real_scale_all_modes"]` is frontline 0.375, deathmatch
  0.3125, elimination 0.3125: of 2048 environments 768 / 640 / 640, each mode's self-play
  and league pools 384 / 320 / 320. `environment_multiple` stays 32.
- **Tests:** test_mixed, test_game_version (the game fingerprint is unchanged by the
  lineup draw's `next_ship_type`), test_references: 622 passed.
- **Next:** league sampling (1 - p)^2 with no floor.

## 07:50 to 08:15: league sampling (1 - p)^2, no floor

- **Worked:** `sampling_weights(p, sampling, floor)`: `LeagueConfig.sampling = "hard"`,
  (1 - p)^2 over every entry (snapshots, both averages, the scripted agent, the cheese
  agents alike), and `LeagueConfig.floor = 0.0`. The old rule stays as an ablation:
  `sampling="variance"` (p (1 - p)) and a positive floor. Self-play is unchanged (the
  lineup's fixed 50%). No rating lanes built.
- **Didn't, then fixed:** two slot tests assumed the initial draws put no scripted agent in
  a slot; under (1 - p)^2 an unrated live policy draws the scripted agent (rated far above
  it) often. The tests now set the slots they need.
- **Tests:** test_league_pool, test_league_slots, test_league_metrics,
  test_league_ratings, test_league_cheese, test_league_graph, test_lanes,
  test_dashboards: 72 passed.
- **Next:** the snapshot schedules.
