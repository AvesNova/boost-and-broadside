# Zero-sum rewards: the two failing reward tests (2026-11-12)

**Verdict: the rewards are zero-sum; the tests' coverage expectations were stale.** No
reward or game bug, no training run affected.

## What failed

On integrate/next-runs (e80c08eb), quick tier of `tests/sections/test_rewards.py`:

- `test_levels_are_zero_sum_and_the_ledger_holds_every_deficit[grouped]`: the per-tick
  zero-sum and ledger checks all passed; the final *coverage* assertion failed:
  `seen = {friendly 3, boundary 15, respawns 16, resets 16, recharge 0, kills 3}` (no
  shield recharge in 16 environments x 300 ticks of random play, seed 5).
- `test_rollout_stores_every_level_zero_sum_per_environment[team_deathmatch]`: the
  per-environment zero-sum check passed; `trajectory.reward.abs().sum() > 0` failed: 8
  environments x 64 ticks of random play with the real types had no reward at all.

Neither is a zero-sum violation (the test names say "zero_sum", the failing lines do not).

## Since when

Checked commit by commit: both pass at 1a796aac, 27af0004, 69ac926b, 09c6d642, f20e1dfc;
both fail from 63d36cb7 (game version 4: a random spawn heading leaves room to turn away
from the edge). That change reshuffles random play in the deathmatch modes (fewer
boundary deaths, different encounters), so seed-fixed coverage flipped.

## Why the tests, not the rewards

- Zero-sum, over 8 seeds x 16 environments x 300 ticks in all five modes: the worst
  per-tick level sum relative to the tick's largest reward is 3.5e-7 to 4.8e-7 (float32
  rounding); the ledger holds every deficit within 1e-3 points throughout.
- Recharge in the deathmatch modes is luck, before and after version 4. Seeds 0-15
  (batch 16): recharge is 0 in 10/16 (team deathmatch), 12/16 (grouped), 10/16
  (elimination) seeds; at f20e1dfc (version 3) it was 0 in about half the seeds too
  (team deathmatch seeds 0-7: 0, 46, 88, 0, 0, 33, 0, 76 recharge ticks). A calm ship
  has to be hurt and then spared for the delay. Batch 32 still misses in 4-8/16. Sandbox
  and Frontline recharge in every seed. `reward_step` does not depend on the mode.
- The rollout's first reward with random play and the real types: seeds 0-7 at 64 ticks
  give 0.0 / 2.9 / 13.0 (HEAD) where version 3 gave 8.9 / 4.2 / 3.5; at 256 ticks every
  seed 0-7 has damage reward (11.6-39.5 summed |reward|).

## Fix (tests only)

- The real-play coverage test requires recharge only in sandbox and Frontline, with the
  reason in a comment; deaths of every kind, respawns and resets stay required in every
  mode.
- The rollout zero-sum test runs 256 ticks, not 64 (+1.2 s).

Still flaky by nature, measured: enemy kills with random play in team deathmatch /
elimination are 0 in 2-3 of 16 seeds (seed 5, used, has them).

Quick tier of test_rewards: 21 passed. ruff clean.
