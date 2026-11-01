---
covers: [src/bnb/cli.py, src/bnb/runtime/train.py]
---

# A first training run

## The command

`bnb` is the only entry point (`src/bnb/cli.py`). It trains the skeleton model, a small
MLP, with plain PPO. On a GPU:

<!-- not-runnable: needs CUDA and runs for minutes -->
```bash
uv run --no-sync bnb --preset mixed --num-envs 1024 --updates 100
```

`--preset mixed` trains on the default training mix: Frontline and team deathmatch on two
maps, in one batch. The [training guide](../guides/training.md) explains the presets, the
pools and the mix. Every option:

<!-- generated-table: cli-options -->

`--num-envs` must be a multiple of what the pools split into; the command stops with a
message naming the multiple when it is not.

<!-- runnable -->
```bash
uv run --no-sync bnb --help
```

## What it prints

One line per update:

- `update`: the update count;
- `rollout` and `learn`: the time of the rollout and of the PPO update, in milliseconds;
- `entropy` and `kl`: the policy entropy and the approximate KL of the update;
- `reward/s`: the learner's reward per ship per second, the reward levels summed, for each
  pool.

The first update includes compilation and the CUDA-graph capture, so it takes longer than
the rest.

## The same loop in Python

`bnb.runtime.train.Trainer` is the loop the command runs. Eagerly on the CPU, with a tiny
game, one update takes a few seconds:

<!-- runnable -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.runtime.train import TrainConfig, Trainer

config = TrainConfig(game=GameConfig(ships_per_team=1, num_fields=2, arena_radius=1200.0),
                     num_envs=8, updates=1, length=8, hidden=32, compile=False)
trainer = Trainer(config, torch.device("cpu"))
row = trainer.update()
print(row["update"], sorted(row["reward_per_s"]))
```

Next: [a benchmark](benchmark.md).
