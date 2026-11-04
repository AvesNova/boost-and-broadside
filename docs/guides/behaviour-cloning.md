---
covers: [src/bnb/cli.py, src/bnb/runtime/cloning.py, src/bnb/learn/bc.py, src/bnb/runs/session.py]
reviewed: 2026-11-04
---

# Behaviour cloning

Behaviour cloning trains a model to imitate the scripted agent before reinforcement
learning. How it works is in [Behaviour cloning](../explanation/behaviour-cloning.md);
this page is about running it.

## Clone the scripted agent

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --preset mixed --objective bc --name clone
uv run --no-sync bnb train --preset mixed --objective bc --teacher-temperature 1.5 \
    --bc-budget-steps 200000000
```

`--objective bc` plays the BC lineup (half self-play, half the scripted agent) in the
chosen game and runs until a stop rule fires: the live score against the scripted agent
(over 1000 games, less two standard errors), a KL plateau, or the budget
([When it stops](../explanation/behaviour-cloning.md#when-it-stops)). The scripted
opponent plays the labels' distribution, at `--teacher-floor` and `--teacher-temperature`.
`--updates` is not used. Each update prints the BC KL, the labels' and the student's
entropy, the live score per opponent and, once 1000 games have finished, the stop rule's
window score and its bound; the last line names the rule that stopped it. The other
options (`--model`, `--environments`, `--shards`, `--precision`, `--micro-batch-rows`,
`--seed`, `--set`, `--name`, `--runs-directory`) are training's ([Training](training.md)).

The run lives in a run directory like a PPO run's
([Runs and checkpoints](../explanation/runs.md)) and writes training checkpoints on the
same cadence. When a stop rule fires, or after the update in progress at Ctrl-C, it writes
a checkpoint and saves the clone as the run's policy file, `runs/<run>/policy/`{.plain}:
the weights, the return scale and the step counts. `run.json`{.plain} records the rule as
`stopped_by` (`interrupted` for Ctrl-C).

## Resume a BC run

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb train --resume clone
uv run --no-sync bnb train --resume clone --bc-budget-steps 800000000
```

A resume continues a stopped or crashed BC run from its newest checkpoint, as for PPO
([Resuming, forking and sharing runs](resuming-forking-sharing.md)): the stop rule's score
window, the KL's average and its history come back with the rest, so the rules decide as
they would have in one process. `--bc-budget-steps` is the run's total budget and may be
raised on a resume; the other stop settings come from the checkpoint. A finished run
resumed trains until a rule fires again: a raised budget continues one that stopped on its
budget, while the score and plateau rules usually fire on the first update. The new clone
replaces the old policy file. A BC run itself starts fresh or resumes: it does not start
from another run (`--from`) or a policy file (`--init-from`).

## Start reinforcement learning from the clone

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --preset mixed --init-from runs/clone/policy
```

`--init-from` starts a new PPO run from the policy file's weights and return scale, with a
fresh optimizer and no cloning term
([Resuming, forking and sharing runs](resuming-forking-sharing.md)). Its entropy
coefficients are `bnb.learn.ppo.WARM_START_ENTROPY` (0) rather than v1's −0.0081, on each
action factor that `--set learner.entropy_power=...` (or `entropy_stick`, `entropy_shoot`)
does not set: the clone already samples every action through its
labels' floor, and at its advantage scale v1's bonus outweighs the policy gradient on rare
actions such as shooting, which took the first clone's score against the scripted agent
from 0.23 to 0.03 in 24 updates.

## Choose the labels' entropy

The labels' entropy is the entropy the clone starts reinforcement learning with. To find
the temperature that gives a target mean entropy:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --calibrate-entropy 0.5
uv run --no-sync bnb train --preset mixed --calibrate-entropy 1.2 --teacher-floor 0.1
```

It plays the scripted agent against itself in the preset's games, prints its natural
entropy (temperature 1, no floor) and the temperature whose labels reach the target at the given
floor, and exits without writing a run. A target outside what the temperature reaches at that floor fails with
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
config = TrainConfig(game=game, pools=cloning.LINEUP, environments=256, model="relational",
                     learner=bc.config(), objective="bc",
                     teacher=bc.TeacherConfig(temperature=1.0),
                     stop=bc.StopConfig(budget_steps=20_000_000))
student = Trainer(config, torch.device("cuda"))
reason = cloning.clone(student)  # its stop rule: config.stop
print("stopped by", reason)

# Reinforcement learning from the clone in memory: its weights and return scale, a fresh
# optimizer (bnb.runs.checkpoint.save_policy writes it to a policy file instead).
rl = Trainer(TrainConfig(game=game, environments=256, model="relational"), torch.device("cuda"))
cloning.warm_start(rl, student)
for _ in range(100):
    rl.update()
```

`tests/sections/test_bc_toy.py` runs the toy case and checks the clone's score against its
teacher and the chaser.
