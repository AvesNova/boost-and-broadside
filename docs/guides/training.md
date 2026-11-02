---
covers: [src/bnb/cli.py, src/bnb/runtime/train.py, src/bnb/contracts/pools.py, src/bnb/contracts/config.py, src/bnb/game/modes/rules.py]
reviewed: 2026-11-02
---

# Training

Training trains Yemong (`bnb.models.yemong`), or the toy relational model, with the
learner in `bnb.learn.ppo` (see [The learner](../explanation/learner.md)). How the loop
works is in [The training loop](../explanation/training-loop.md); this page is about running
and configuring it.

## Choose the game

A run plays one game or several. Three ways to choose:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset frontline                 # one game preset
uv run --no-sync bnb train --preset mixed                     # the default training mix
uv run --no-sync bnb train --mode team_deathmatch --ships-per-team 2 --num-fields 4 \
    --arena-radius 1200                                 # any mode at any size
```

The game presets, from `bnb.game.modes.rules.PRESETS`:

<!-- generated: game-presets -->

Without a preset, the command builds a `bnb.contracts.GameConfig` from `--mode`,
`--ships-per-team`, `--num-fields` and `--arena-radius`, with the mode's clocks:

<!-- generated: mode-clocks -->

`GameConfig` validates itself on construction: two teams, at most
`bnb.contracts.config.MAX_SHIPS` ships, 1 to `bnb.contracts.config.MAX_FIELDS` fields, a
positive arena, a known mode, and spawn groups only in team deathmatch, summing to the
ships per team. Every field:

<!-- generated: game-config -->

`episode_seconds` and `respawn_seconds` must be whole multiples of the tick, or building the rules
fails (`bnb.contracts.config.ticks_of`).

## Pools: who plays whom

A batch of environments is cut into **pools** (`bnb.contracts.Pool`): contiguous slices,
each with a fixed lineup. `seats` names the controller for each role. Role `r` plays team
`(r + seat) mod 2`, and the seat is redrawn for every environment at every reset, so every
controller plays both sides. The skeleton lineup, `bnb.runtime.train.LINEUP`:

<!-- generated: training-lineup -->

The seats name the trainer's controllers, which
[The training loop](../explanation/training-loop.md#controllers) lists. Behaviour cloning
has its own lineup and objective; see [Behaviour cloning](behaviour-cloning.md).

Each pool's `share` times the batch size must be a whole number of environments, and the
shares must sum to 1. `fog_share` is the fraction of the pool's environments with fog of
war; it too must come to a whole number. Every field of a pool:

<!-- generated: pool-fields -->

## The mixed batch

A pool may also set its own game: `mode`, `arena_radius`, `spawn_groups`, `episode_seconds` and
`respawn_seconds`. `None` takes the `GameConfig`'s value. The sizes (ships per team, fields) are
the `GameConfig`'s and are shared by the whole batch. One batch then holds several games
in the same tensors and the same compiled graph; [Mixed batches](../explanation/game/mixed-batches.md)
explains how.

`--preset mixed` builds such a batch with `bnb.runtime.train.mixed`: every game of the
training mix crossed with every pool of the lineup. The training mix,
`bnb.runtime.train.TRAINING_MIX`:

<!-- generated: training-mix -->

The crossed pools, game by game so that each game is one contiguous slice:

<!-- generated: mixed-pools -->

`bnb.contracts.pools.environment_multiple` gives the smallest batch the pools split into;
every valid `--environments` is a multiple of it.

<!-- runnable -->
```python
from bnb.contracts.pools import environment_multiple
from bnb.runtime.train import LINEUP, mixed

game, pools = mixed()
print(game.mode, len(pools), environment_multiple(pools), environment_multiple(LINEUP))
```

## Choose the model and the learner's settings

The command's other options set the model and the learner:

| Option | Values | Sets |
|---|---|---|
| `--model` | `yemong`, `relational` | `TrainConfig.model`: Yemong or the toy relational model |
| `--precision` | `bfloat16` (default), `float32`, `tf32` | `TrainConfig.precision`: the matmuls' precision |
| `--shards` | a count | `TrainConfig.shards`: segments collected per update |
| `--old-log-probabilities` | `rollout`, `recomputed` | `LearnerConfig.old_log_probabilities`: the rollout's, or recomputed by `unroll` before the first epoch |
| `--micro-batch-rows` | a count | `LearnerConfig.micro_batch_rows`: ship rows per micro-batch (memory only; the result is the same) |

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --precision tf32 --shards 2 --micro-batch-rows 320
```

## Configure a run in Python

`bnb.runtime.train.TrainConfig` holds the loop's settings, and `bnb.runtime.train.train`
runs it. The command builds the same objects.

<!-- generated: train-config -->

`pools` defaults to `bnb.runtime.train.LINEUP`, and `learner` to `bnb.learn.ppo.LearnerConfig`:

<!-- generated: learner-config -->

A custom lineup, here learner against the chaser on half the batch and self-play on the
other half, with no fog in the chaser pool:

<!-- not-runnable: needs CUDA -->
```python
import torch

from bnb.contracts import GameConfig, Pool
from bnb.runtime.train import LEARNER, TrainConfig, train

pools = (Pool("self_play", 0.5, (LEARNER, LEARNER)),
         Pool("chaser", 0.5, (LEARNER, "chaser"), fog_share=0.0))
config = TrainConfig(game=GameConfig(5, 10, 2600.0), pools=pools, environments=1024, updates=200)
train(config, torch.device("cuda"))
```

A pool may seat only the trainer's controllers
([The training loop](../explanation/training-loop.md#controllers)); a pool that names any
other controller fails when the rollout is built.

## The toy learner

The toy learner checks the whole loop on a small game: the relational model
(`bnb.models.relational.Relational`) trained from scratch at 2v2 team deathmatch against the
chaser. `bnb.runtime.train.win_rate` plays fresh games against a scripted opponent and
returns the shares won, drawn and lost.

<!-- not-runnable: needs CUDA and runs for minutes -->
```python
import torch

from bnb.contracts import GameConfig, Pool
from bnb.runtime.train import LEARNER, TrainConfig, Trainer, win_rate

torch.manual_seed(0)
config = TrainConfig(game=GameConfig(2, 4, 1200.0), environments=256, model="relational",
                     pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),))
trainer = Trainer(config, torch.device("cuda"))
for _ in range(400):
    trainer.update()
print(win_rate(trainer, games=256))
```

`tests/loop/test_learning.py` runs this and requires a win rate of at least 0.6.

## Seeds and process settings

The `bnb` command seeds torch's default generator from `--seed` (the run's
`TrainConfig.seed`, which also seeds the learner's streams; `--set seed=N` is the same)
and sets the matmul precision (`bnb.cli.set_precision`) once, before training,
and sets `TRITON_DEFAULT_FP_FUSION` before torch is imported. A script that calls
`bnb.runtime.train.train` does the same itself. Library code sets no process-wide options;
only entry points do (see [Architecture](../explanation/architecture.md)).
