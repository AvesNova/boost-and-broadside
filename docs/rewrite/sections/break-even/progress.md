# Break-even ship count: `bnb crossover` rewritten (2026-11-14)

Branch `feat/break-even`, from `origin/feat/seed-isolation` (a13173da). CPU only (the GPU
was busy until about 01:10). The decided metric (the author, 2026-11-14): "N of ours hold
20 Imugi", the scripted agent fixed at 20 ships, the policy at a varying count; N\* where
the policy's score (a draw half a win) crosses 0.5, with a standard error, and 20 / N\*,
per real-scale mode. Reader-facing description: `docs/explanation/league.md`, "The
break-even ship count".

## Tried and worked

- **Fit.** Logistic regression of the summed score on log(count) (binomial likelihood,
  Newton with step halving, x centred), N\* = exp(-a / s), standard error by the delta
  method (the ratio's relative error is N\*'s). Separated outcomes (no maximum) give a
  bound (`Fit.lower`, `Fit.upper`); a falling curve gives no N\*.
- **Allocation.** First batch: the bracket 6, 10, 14, 20, dealt over the blocks. Then, with
  a crossing, a greedy c-optimal design: each block in turn to the count that most reduces
  the predicted variance of log N\* (current information plus the block's expected
  information), averaged over N\* at the estimate and one standard error either side.
  Separated: inside the bound (up to three counts nearest its geometric middle), or halve
  the lowest count (all won) / double the highest (all lost).
- **Stop.** Standard error at or under 0.3 ships, or 8192 games per mode (the batch that
  reaches it is the last), or a final bound (all won at count 1, all lost at count 20).
- **Synthetic check** (8 blocks x 128 games per batch, 20 repeats per case): batches to
  the 0.3-ship target were 1 to 2 for N\* = 3.5, 8, 13 (slope 2 or 5 in log count), 2 to 6
  for N\* = 19 at slope 2; the errors over their standard errors had a standard deviation
  of 0.7 to 1.3 (20 repeats). The 0.3-ship target is cheap at this layout: a tighter one
  (0.1 to 0.15) is worth considering.
- **Lineups.** The engine had no random mirrored lineup: `Arena(game_lineups=True)` now takes
  a (games, ships per team) lineup per batch, both teams flying it slot for slot.
  `crossover.lineup_draw` draws each slot from (Yongma, Jujak) by a keyed hash of the seed,
  the batch and the game.
- **Records.** One JSONL line per batch, `artifacts/crossover/<preset>.batches.jsonl`,
  with each block's count, wins, draws and losses and the fit after it; a rerun filters
  the lines by the measurement's key (identities, game, seed, opponent count, lineup types),
  reuses them and numbers new batches past them (the batch number keys the games' draws).
- **Arena.** The game is the preset at 20 ships per team with the arena by the density
  rule (`arena_radius_for`): 15600 m for frontline and elimination, 8400 m for
  deathmatch. This is my choice, not a decided one: the alternative is the preset's own
  (5v5) arena with 20 ships crowding it.

## Benched-slot cost (CPU, eager, performance profile, a training run on the GPU)

Scripted against scripted, real-scale deathmatch at 20 per side (2 blocks x 64 games),
ms per tick: all 20 active 132 and 151 (two runs), the first team at 4 active 122 and 149,
at 10 active 132; a 12v12 game 92. A benched slot costs what an active one does (the game,
the scripted agent and the network run at fixed shapes over every slot); noise about 10%.
The scripted pass also runs over the network's rows (the match seat's `step`, as in
head-to-head), so a policy-vs-scripted batch pays the scripted agent on every row. The
network runs one stacked forward over its lanes (one per block, `default_lanes`), on its
own rows only. One engine, so one compile, per mode.

## Didn't / open

- No real-scale CPU end-to-end: a 600 s real-scale game at 20 per side is about 18000 ticks
  at about 1 ms per game-tick on the CPU. The tests run a 3v3 game.
- The historical session scripts `docs/rewrite/sections/gpu-box/crossover_adaptive.py` and
  `fixed_odds.py` import the old grid's `summary` and `wilson_interval`, which are gone;
  they are left as the record of those sessions.

## Next: the first GPU measurement (after the GPU frees, about 01:10)

rl-v20's final exponential average against the scripted agent at 20, all three modes:

```bash
uv run --no-sync bnb crossover \
  /home/vizia/avesnova/box-5090-backup/runs/rl-v20/policies/step_170917888/exponential_average \
  --out /home/vizia/avesnova/runs-v2/break-even/rl-v20
```

Defaults: `--opponent scripted --opponent-count 20 --minimum-count 1 --maximum-count 20
--presets real_scale_frontline real_scale_deathmatch real_scale_elimination --bracket 6 10
14 20 --target-stderr 0.3 --budget-games 8192 --lineup-types Yongma Jujak --blocks 8
--block-environments 128 --seed 0`. Check the first batch's time per mode before
trusting the ETA line (the first batch includes the compile); if a 20v20 batch of 1024
games does not fit the 8 GB laptop GPU, lower `--block-environments` to 64.
