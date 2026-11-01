---
covers: [src/bnb/cli.py, src/bnb/runtime/train.py, src/bnb/contracts/pools.py, src/bnb/contracts/config.py, src/bnb/game/modes/rules.py]
---

# Training

Training today runs the walking skeleton's loop: the MLP in `bnb.models.mlp`, trained by
the plain PPO in `bnb.learn.ppo`. Section 6 replaces the learner and trains Yemong (see
[The learner](../explanation/in-progress/learner.md)). How the loop works is in
[The training loop](../explanation/training-loop.md); this page is about running and
configuring it.

## Choose the game

A run plays one game or several. Three ways to choose:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb --preset frontline                 # one game preset
uv run --no-sync bnb --preset mixed                     # the default training mix
uv run --no-sync bnb --mode team_deathmatch --ships-per-team 2 --num-fields 4 \
    --arena-radius 1200                                 # any mode at any size
```

The game presets, from `bnb.game.modes.rules.PRESETS`:

<!-- generated-table: game-presets -->

Without a preset, the command builds a `bnb.contracts.GameConfig` from `--mode`,
`--ships-per-team`, `--num-fields` and `--arena-radius`, with the mode's clocks:

<!-- generated-table: mode-clocks -->

`GameConfig` validates itself on construction: two teams, at most
`bnb.contracts.config.MAX_SHIPS` ships, 1 to `bnb.contracts.config.MAX_FIELDS` fields, a
positive arena, a known mode, and spawn groups only in team deathmatch, summing to the
ships per team. Every field:

<!-- generated-table: game-config -->

`episode_s` and `respawn_s` must be whole multiples of the tick, or building the rules
fails (`bnb.contracts.config.ticks_of`).

## Pools: who plays whom

A batch of environments is cut into **pools** (`bnb.contracts.Pool`): contiguous slices,
each with a fixed lineup. `seats` names the controller for each role. Role `r` plays team
`(r + seat) mod 2`, and the seat is redrawn for every environment at every reset, so every
controller plays both sides. The skeleton lineup, `bnb.runtime.train.LINEUP`:

<!-- generated-table: training-lineup -->

- `learner` is the model being trained.
- `league_0` is a copy of it, refreshed every `league_every` updates.
- `chaser` turns toward and shoots at the nearest visible enemy
  (`bnb.controllers.scripted.Chaser`).
- `random` takes sticky uniform random actions (`bnb.controllers.scripted.Random`).

Each pool's `share` times the batch size must be a whole number of environments, and the
shares must sum to 1. `fog_share` is the fraction of the pool's environments with fog of
war; it too must come to a whole number. Every field of a pool:

<!-- generated-table: pool-fields -->

## The mixed batch

A pool may also set its own game: `mode`, `arena_radius`, `spawn_groups`, `episode_s` and
`respawn_s`. `None` takes the `GameConfig`'s value. The sizes (ships per team, fields) are
the `GameConfig`'s and are shared by the whole batch. One batch then holds several games
in the same tensors and the same compiled graph; [Mixed batches](../explanation/game/mixed-batches.md)
explains how.

`--preset mixed` builds such a batch with `bnb.runtime.train.mixed`: every game of the
training mix crossed with every pool of the lineup. The training mix,
`bnb.runtime.train.TRAINING_MIX`:

<!-- generated-table: training-mix -->

The crossed pools, game by game so that each game is one contiguous slice:

<!-- generated-table: mixed-pools -->

`bnb.contracts.pools.environment_multiple` gives the smallest batch the pools split into;
every valid `--num-envs` is a multiple of it.

<!-- runnable -->
```python
from bnb.contracts.pools import environment_multiple
from bnb.runtime.train import LINEUP, mixed

game, pools = mixed()
print(game.mode, len(pools), environment_multiple(pools), environment_multiple(LINEUP))
```

## Configure a run in Python

`bnb.runtime.train.TrainConfig` holds the loop's settings, and `bnb.runtime.train.train`
runs it. The command builds the same objects.

<!-- generated-table: train-config -->

`pools` defaults to `LINEUP`, and `ppo` to `bnb.learn.ppo.PPOConfig`:

<!-- generated-table: ppo-config -->

A custom lineup, here learner against the chaser on half the batch and self-play on the
other half, with no fog in the chaser pool:

<!-- not-runnable: needs CUDA -->
```python
import torch

from bnb.contracts import GameConfig, Pool
from bnb.runtime.train import LEARNER, TrainConfig, train

pools = (Pool("self_play", 0.5, (LEARNER, LEARNER)),
         Pool("chaser", 0.5, (LEARNER, "chaser"), fog_share=0.0))
config = TrainConfig(game=GameConfig(5, 10, 2600.0), pools=pools, num_envs=1024, updates=200)
train(config, torch.device("cuda"))
```

The trainer knows four controller names: `learner`, `league_0`, `chaser` and `random`. A
pool that names any other controller fails when the rollout is built.

## Seeds and process settings

`train` seeds torch's default generator from `TrainConfig.seed` once, at the start. The
`bnb` command sets `TRITON_DEFAULT_FP_FUSION` before torch is imported. Library code sets
no process-wide options; only entry points do (see
[Architecture](../explanation/architecture.md)).
