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

## 08:15 to 08:45: snapshot schedules

- **Worked:** `bnb.runtime.snapshots.SnapshotSchedule(steps, every)`: a list of steps,
  then a period (None: the list's last gap). `due(before, after)` says whether an update
  holds a point; a period alone is the old cadence exactly.
  - RL policy files: `TrainConfig.policy_snapshots = RL_SNAPSHOTS` (1, 2, 3, 4, 6, 8, 10,
    12, 16 and 20M, then every 5M), replacing `policy_snapshot_every` (10M). Overridable on
    a resume as before (`policy_snapshots.steps`, `policy_snapshots.every`).
  - League members: `LeagueConfig.snapshots = LEAGUE_SNAPSHOTS` (5 ... 160M, then every
    30M), replacing `snapshot_every`; None keeps the measured 30-minute period. The
    league's checkpoint state records `snapshot_schedule`.
- **Open (not settled by the decisions):**
  - PPO runs write every update's policy files by default (`update_policies`), so the
    RL schedule applies only to runs without them (BC by default, or
    `--update-policies False`). I left that default alone: turning it off would drop
    files an author may want (the averaging sweep reads them).
  - League retention (`pool.retained`: the newest 8, then one per 2^k) still applies on
    top of the schedule, so at 160M the members at 20, 25, 30 and 40M are dropped.
  - BC uses the same policy schedule (denser than its old 10M early on).
- **Tests:** test_snapshot_schedules (new), test_league_slots, test_league_cheese,
  test_league_graph, test_league_metrics, test_league_ratings, test_update_policies,
  test_training_checkpoints, test_references, test_formatting: 679 passed.
- **Next:** BC fixed at 100M, the plateau logging only.

## 08:45 to 09:15: BC fixed at 100M, the plateau logging only

- **Worked:** `StopConfig.budget_steps = 100_000_000` and `StopConfig.plateau_stops =
  False`: the plateau rule is still computed and logged (`bc/stop/plateau/reached`, new,
  beside `fraction` and `held_steps`; "(reached)" on the update line), and stops BC only
  with `--set stop.plateau_stops=true`. `bnb train --bc-budget-steps` defaults to None,
  StopConfig's 100M.
- **Also changed (the decision implies it):** a chain's BC stage took the combined budget
  as its own and relied on the plateau to stop first. With the plateau logging only it
  would have used the whole combined budget, so its budget is now
  `cli.chain_bc_budget(combined) = min(combined, 100M)`. The chain tests set
  `stop.plateau_stops=true`, as they test the hand-over at a fired rule.
- **Tests:** test_bc, test_chain, test_cli, test_bc_toy, test_references,
  test_formatting: 676 passed.
