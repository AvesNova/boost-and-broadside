# A2 team sizes: progress

Branch `feat/a2-team-sizes` from `origin/feat/imugi-boundary` (83404bbf). CPU only.

## Design (2026-11-16)

- **Setting:** `TrainConfig.team_sizes` (`bnb.contracts.TeamSizes`: `low`, `high`, `weights`),
  copied onto every training pool (`Pool.team_sizes`) by `training_pools`; None (the
  default) plays every slot, so every other run is unchanged. A2 launches with
  `--set 'team_sizes={"low": 1}'` (with `active_fields`, see the open questions).
- **Default distribution for A2 (Claude's choice):** uniform over the allowed sizes up to the
  preset's 5: deathmatch and elimination 1v1 to 5v5 (1/5 each), Frontline 3v3 to 5v5 (1/3
  each); the mode's floor is `api.MINIMUM_SHIPS_PER_TEAM` (Frontline 3). `weights` gives
  other shares, renormalised after the floor.
- **One size per game, both teams** (N against N). A team's slots past N are benched (the
  game's existing benching: dead, out of play, the benched flag shown), so the batch keeps
  its static 5v5 shapes: no recompiles per size.
- **The draw is the runtime's**, not the game's: `Rollout._team_sizes`, keyed by (seed,
  environment, episode) under its own purpose key, so it moves no other draw. The game gains
  only `resolve(..., next_active=)`: an environment whose episode ends spawns and benches by
  its next episode's size; the carry's `active` follows from the next tick. No state leaf
  is added, so `tests/sections/game_reference.pt` is untouched.
- **Per-size rules** come from `active` as before: deathmatch's tickets 4 N^(4/3) to the
  nearest 5 (5, 10, 15, 25, 35 for 1-5), elimination's wipe on the enemy's active ships;
  the clocks and the elimination shrink are the same at every size (the freeze).
- **Learning:** `valued` excludes benched slots; rewards give a benched slot nothing and
  split the enemy's unforced events over the present slots only (`enemy_team_share`'s
  `present`); the Yemong masks benched ships (`mask_benched`, set by the Trainer, kept by
  the league's and lanes' copies). The policy, enemy-action, next-state (delta: alive at t
  and t+1) and geometry masks already exclude dead slots.
- **BC and RL** both use the Trainer's rollout over the training pools, so both follow it.
  The rating lanes, evaluation and play keep their games' own sizes.

## Log

### 04:20 code and tests
- Worked: contract, game `next_active`, rollout draw, reward and valued masks, Trainer
  mask, 22 tests in `tests/sections/test_team_sizes.py` (draw per mode, uniform, keyed,
  isolation from other draws, all-slots = plain rollout, resume, rewards, settings, a BC
  and an RL update on a tiny Yemong).
- Next: related suites, docs, merge.

### 05:10 suites and merge
- Worked: quick tier on the touched and related files: 227 + 172 tests pass (game reference,
  benched, absent fields, crossover, map sampling, mixed, rewards and their v1 fixtures,
  rollout records, training checkpoints, resume, BC, split critic, lanes, learner shots,
  modes, contracts, docstrings, staleness). Fixed on the way: the rollout-records test's
  reward recorder (new `present` argument); `layout.py` was unformatted on the base.
- Didn't: a compiled CPU rollout fails in Inductor (KeyError in a buffer) with or without
  team sizes, so it says nothing about this branch; the GPU compile and graph-capture check
  is queued behind the balance and GRU jobs on the shared lock.
- Merged into `feat/imugi-boundary` as 238db2e4.
- Open (for the author): field counts in A2; the per-game vs per-ship-step distribution;
  small deathmatch games mostly running the clock (arms-todo item 9).

### 05:02 GPU check (laptop 4070, performance profile, shared lock; unit a2-check, 3 min)
- Worked: a compiled, CUDA-graph-captured rollout of the three real-scale modes (96
  environments, 20 s episodes, `TeamSizes(1)`) compiles and captures once (133 s cold) and
  runs 20 more segments with `error_on_recompile` on: no recompile over 211 episode resets.
  Sizes seen: deathmatch and elimination 1-5, Frontline 3-5. Script:
  the session scratchpad's `gpu_check.py` (not kept).

## Joint sizes and fields (2026-11-16, the author's answers)

The author: ship and field counts vary together; 5v5 at least half the games; uneven sizes
included; the equations are discussed first. Proposal for the author:
[../../a2-sampling-proposal.md](../../a2-sampling-proposal.md).

### 06:00 mechanism, tests, proposal
- Worked: `TeamSizes` is now a table of (team 0's ships, team 1's ships, fields,
  probability) from a ship rule (`uniform_pairs` default, `independent_teams`, `near_full`,
  `table`; `full_share` 0.5, `uneven`) and a field rule (`per_ship` default: one field per
  ship to the nearest even count, ties up, optional `field_spread`; `game`: the game's own).
  The rollout draws one row per episode by the same keyed uniform (purpose "team size"),
  so nothing else moves; the game takes `field_count` (reset) and `next_field_count`
  (resolve), -1 for the game's own draw; `api.active_field_count` reads a state's. Tickets
  unchanged (each team's winning score from the enemy's ships, as break-even plays it).
  Series per training mode, on by default (`series`): `team_size/larger`, `/smaller`,
  `/fields`, `/share/{a}v{b}`. `samples_field_counts` counts the coupled fields, so the
  Yemong masks absent fields too. Tests: 32 in `test_team_sizes.py` (rules, table, field
  counts, refusals, uneven tickets, the draw against the table, field counts at reset,
  the series, isolation, all-slots = plain, resume, settings, BC and RL updates).
- Provisional launch setting: `--set 'team_sizes={}'`.

### 07:30 suites and integration
- Worked (all runs with `PYTHONPATH=$PWD/src`, checked to import this worktree's `bnb`; the
  worktree's `.venv` links the main venv, whose editable install points at the main tree):
  the related quick tier, 846 tests (game version and reference, benched, absent fields,
  crossover, map sampling, mixed, diagnostics games, rewards, lanes, training lineup,
  rollout records, keyed draws, contracts, docstrings, staleness, references, team sizes):
  one failure, a docs reference (`api.MINIMUM_SHIPS_PER_TEAM` needs its full name), fixed.
  Merged `origin/feat/imugi-boundary` (split-critic fixes 7 and 8); on the merged tree team
  sizes, rewards, split critic, rollout records, references and staleness: 678 pass.
- The first round's pre-merge runs may have imported the main tree's code; the same test
  files pass now on this tree, which covers them.
- Merged tree, game side: game version and reference, benched, absent fields, crossover,
  mixed, keyed draws, contracts, docstrings: 135 pass.

### 07:50 GPU check (laptop 4070, performance profile, shared lock; unit a2-joint-check, 2 min)
- Worked (imports this worktree's `bnb`, tree before the imugi-boundary merge): a compiled,
  CUDA-graph-captured rollout of the three real-scale modes (96 environments, 20 s
  episodes, `TeamSizes()`) compiles and captures once (97 s cold) and runs 20 more segments
  with `error_on_recompile` on: no recompile over 211 episode resets. Seen: deathmatch 24
  of the 25 (team 0, team 1, fields) rows, Frontline all 9 (3 to 5 a side), elimination 19;
  fields as the table gives (2 at 1v1 to 10 at 5v5); the size sums counted 211 of 211
  ended episodes. Script: the session scratchpad's `a2_joint_gpu_check.py` (not kept).
