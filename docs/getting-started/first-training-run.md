---
covers: [src/bnb/cli.py, src/bnb/runtime/train.py]
reviewed: 2026-11-02
---

# A first training run

## The command

`bnb` is the only entry point (`src/bnb/cli.py`). `bnb train` trains Yemong with the learner
(see [The learner](../explanation/learner.md)). On a GPU:

<!-- not-runnable: needs CUDA and runs for minutes -->
```bash
uv run --no-sync bnb train --preset mixed --environments 1024 --updates 100
```

`--preset mixed` trains on the default training mix: Frontline and team deathmatch on two
maps, in one batch. The [training guide](../guides/training.md) explains the presets, the
pools and the mix. Every option:

<!-- generated: cli-options -->

`--environments` must be a multiple of what the pools split into; the command stops with a
message naming the multiple when it is not.

<!-- runnable -->
```bash
uv run --no-sync bnb train --help
```

## What it prints

One line per update:

- `update`: the update count;
- `rollout` and `learn`: the time of the rollout and of the update, in milliseconds;
- `entropy`, `kl`{.plain} and `clip`{.plain}: the policy entropy, and the approximate KL and clip fraction of
  the update's first epoch;
- `reward/s`: the learner's reward per ship per second, the reward levels summed, for each
  pool.

The first update includes compilation and the CUDA-graph capture, so it takes longer than
the rest. It is also the memory probe: the run prints its allocator peaks after it, and
stops if they exceed the device's memory less a margin.

## Where it goes

Each `bnb train` is a run in its own directory under `runs/`, named for the date, the
preset and two words. It holds the run's record, its configuration and its training
checkpoints. Ctrl-C stops the run after the current update, with a checkpoint, and
`bnb train --resume RUN` continues it exactly; `bnb runs` lists the runs. See
[Resuming, forking and sharing runs](../guides/resuming-forking-sharing.md).

## The same loop in Python

`bnb.runtime.train.Trainer` is the loop the command runs. Eagerly on the CPU, with a tiny
game and the toy relational model, one update takes a few seconds:

<!-- runnable -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.learn.ppo import LearnerConfig
from bnb.runtime.train import TrainConfig, Trainer

config = TrainConfig(game=GameConfig(ships_per_team=1, num_fields=2, arena_radius=1200.0),
                     environments=8, updates=1, length=8, model="relational",
                     relational_hidden=32, compiled=False,
                     learner=LearnerConfig(minibatches=2, micro_batch_rows=8))
trainer = Trainer(config, torch.device("cpu"))
row = trainer.update()
print(row["update"], sorted(row["pools"]), round(row["loss/value"], 2))
```

Next: [a benchmark](benchmark.md).
