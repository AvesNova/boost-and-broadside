# League arm options: progress

## Resume note (2026-11-15, paused)

- Done: all three options built, committed (c2ecf594) and pushed; the new tests (11) pass;
  the docs tests pass; `--print-config --device cpu` resolves A5, A6 and R3.
- Half-done: the quick tier over the existing league, mix and runs tests (test_league_pool,
  test_league_slots, test_league_cheese, test_mixed, test_snapshot_schedules, test_lanes,
  test_league_graph, tests/runs) was stopped before it finished; rerun it.
- Next: rerun those tests, merge origin/feat/imugi-boundary (2 new commits), push, report.

Branch `feat/league-arm-options` (from `feat/imugi-boundary`). Three arms launch from
settings alone; every default is today's behaviour.

## 2026-11-15: all three built

- **A5, mode mix** (`TrainConfig.mode_mix`, preset names to whole weights). `bnb train`
  builds the pools after `--set` (`bnb.runtime.train.arranged`), each game with its own
  map, clocks, respawn and spawn groups; a trainer refuses pools that are not the mix's
  (`check_pools`). 8/4/4 at 2048: 512/512/256/256/256/256.
- **A6, one seat** (`TrainConfig.self_play_seats="one"` + `LeagueConfig.latest_share`).
  No self-play pool: every game is learner against the league seat
  (`LEAGUE_ONLY`). "latest" holds the first `latest_share × slots` slots, reloaded with
  the live weights after every update; never drawn, cut or rated. Learner rows = one
  per environment.
- **R3, exploiter** (`LeagueConfig.fixed_opponent`, a policy file). The pool is that entry
  alone (kind "fixed") in every slot; no Imugi, cheese, averages or snapshots as
  opponents; no self-play; the build must match the live policy's.
- Tests: `tests/sections/test_league_arm_options.py` (11, CPU). Docs:
  `docs/explanation/league.md` (One-sided self-play and the exploiter),
  `docs/guides/training.md` (mode_mix).

Open for the author: A6's environment count (3,072 keeps today's learner rows per update;
4,096 doubles the opponent rows); A6's slot count (8 keeps 4 drawn slots beside 4 latest).
