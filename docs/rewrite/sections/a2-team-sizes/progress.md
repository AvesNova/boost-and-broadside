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

### 11:30 code and tests
- Worked: contract, game `next_active`, rollout draw, reward and valued masks, Trainer
  mask, 22 tests in `tests/sections/test_team_sizes.py` (draw per mode, uniform, keyed,
  isolation from other draws, all-slots = plain rollout, resume, rewards, settings, a BC
  and an RL update on a tiny Yemong).
- Next: related suites, docs, merge.
