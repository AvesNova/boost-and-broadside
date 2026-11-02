---
covers: [src/bnb/cli.py, src/bnb/runtime/cloning.py, src/bnb/learn/bc.py]
reviewed: 2026-11-02
---

# Behaviour cloning

Behaviour cloning trains a model to imitate the scripted agent before reinforcement
learning. How it works is in [Behaviour cloning](../explanation/behaviour-cloning.md);
this page is about running it.

## Clone the scripted agent

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb --preset mixed --objective bc
uv run --no-sync bnb --preset mixed --objective bc --teacher-temperature 1.5 \
    --bc-budget-steps 200000000
```

`--objective bc` plays the BC lineup (half self-play, half the scripted agent) in the
chosen game and runs until a stop rule fires: the live score against the scripted agent,
a KL plateau, or the budget. `--updates` is not used. Each update prints the BC KL, the
labels' and the student's entropy and the live score per opponent; the last line names
the rule that stopped it. The other options (`--model`, `--num-envs`, `--shards`,
`--precision`, `--micro-batch-rows`, `--seed`) are training's
([Training](training.md)).

The command does not yet save the clone: it ends when the stop rule fires, and the clone
leaves with the process. Starting reinforcement learning from a clone goes through Python
([below](#from-python)) until policy files exist.

## Choose the labels' entropy

The labels' entropy is the entropy the clone starts reinforcement learning with. To find
the temperature that gives a target mean entropy:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb --preset mixed --calibrate-entropy 0.5
uv run --no-sync bnb --preset mixed --calibrate-entropy 1.2 --teacher-floor 0.1
```

It plays the scripted agent against itself in the preset's games, prints its natural
entropy (temperature 1, no floor) and the temperature whose labels reach the target at the given
floor, and exits. A target outside what the temperature reaches at that floor fails with
the reachable range; raise `--teacher-floor` for a higher one. `--calibration-segments`
(4 by default) sets how many segments of 128 ticks it records. The natural entropy is low
(0.40 nats on the training mix,
[measured](../explanation/behaviour-cloning.md#measurements)), and at floor 0 the
temperature alone reaches about 0.75. `--teacher-floor` defaults to 0.01, so no label
gives an action probability 0. Pass the temperature to the BC run with
`--teacher-temperature`.

## From Python

<!-- not-runnable: needs CUDA and runs for minutes -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.learn import bc
from bnb.runtime import cloning
from bnb.runtime.train import TrainConfig, Trainer

torch.manual_seed(0)
game = GameConfig(2, 4, 1200.0)
config = TrainConfig(game=game, pools=cloning.LINEUP, num_envs=256, model="relational",
                     learner=bc.config(), objective="bc",
                     teacher=bc.TeacherConfig(temperature=1.0))
student, reason = cloning.clone(config, torch.device("cuda"),
                                bc.StopConfig(budget_steps=20_000_000))
print("stopped by", reason)

# Reinforcement learning from the clone: its weights and return scale, a fresh optimizer.
rl = Trainer(TrainConfig(game=game, num_envs=256, model="relational"), torch.device("cuda"))
cloning.warm_start(rl, student)
for _ in range(100):
    rl.update()
```

`tests/sections/test_bc_toy.py` runs the toy case and checks the clone's score against its
teacher and the chaser.
