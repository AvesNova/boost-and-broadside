---
covers: [src/bnb/cli.py, src/bnb/runtime/cloning.py, src/bnb/learn/bc.py, src/bnb/runs/session.py]
reviewed: 2026-11-11 15:15
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
chosen game and runs until a stop rule fires: the plateau of the BC KL, or the budget, a
cap; a score target (`--set stop.score_target=0.5`, off by default) adds parity against
the scripted agent over 1000 lane games, read on the better of the two averages of the
clone's weights ([When it stops](../explanation/behaviour-cloning.md#when-it-stops)). The
scripted opponent plays the labels' distribution, at `--teacher-floor` and
`--teacher-temperature`. `--updates` is not used. Each update prints the BC KL, the
labels' and the student's entropy, the live score per opponent, once an average's 1000
games have finished the average the score window reads with its window score and standard
error, and once the plateau reads one its fraction (the KL's recent improvement over its
improvement since 5M steps; the rule fires once it has stayed below 0.05 for 10M steps)
and how long it has been below; the last line names the rule that stopped it and the
average the run's policy holds. To clone and then reinforce in one command, chain them
([Training](training.md#chain-bc-and-rl)). The other
options (`--model`, `--environments`, `--shards`, `--precision`, `--micro-batch-rows`,
`--seed`, `--set`, `--name`, `--runs-directory`) are training's ([Training](training.md)).

The run lives in a run directory like a PPO run's
([Runs and checkpoints](../explanation/runs.md)) and writes training checkpoints on the
same cadence. When a stop rule fires, or after the update in progress at a stop request (Ctrl-C,
SIGTERM or a `STOP` file: [Resuming](resuming-forking-sharing.md#start-and-stop-a-run)), it writes
a checkpoint and saves its policy files, each with its weights, the return scale, the step
counts and its own lane ratings
([The hand-off](../explanation/behaviour-cloning.md#the-hand-off)):

| Directory | Holds |
| --- | --- |
| `runs/<run>/policy/`{.plain} | the run's policy and the hand-off to RL: always the exponential average |
| `runs/<run>/policy-uniform/`{.plain} | the uniform average (after the warm-up) |
| `runs/<run>/policy-exponential/`{.plain} | the exponential moving average |
| `runs/<run>/policy-live/`{.plain} | the live clone |

Every 10M environment steps (`--set policy_snapshot_every=...`) and at each end, the run
also keeps the live clone and both averages under
`runs/<run>/policies/step_<steps>/{live,exponential_average,uniform_average}/`{.plain},
never replaced, so a run resumed past its budget keeps its earlier averages; `bnb ratings
--with <run>/policies/step_<steps>/uniform_average` rates one
([Runs and checkpoints](../explanation/runs.md#every-updates-policy-files)).

`run.json`{.plain} records the rule as `stopped_by` (`interrupted` for a stop request) and the
average in `policy/` as `policy_player` (always `exponential_average`).
The averages take the league's settings: the uniform average starts after the run's
learning-rate warm-up (`LearnerConfig.warmup_steps`), `--set league.uniform_from=20000000` starts
it elsewhere, `--set league.exponential_decay=0.98` slows the exponential
one.

## Run a BC experiment to a fixed budget

A comparison between builds (an "arm") trains every build for the same number of
environment steps, so that no stop rule ends one early:

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --preset mixed --objective bc --bc-budget-steps 30000000 \
    --set stop.plateau_fraction=null --name 2026-11-09-bc-arm-example
```

No plateau fraction (`StopConfig.plateau_fraction`) turns the plateau off, and the score
stop is off without a target, so the budget ends the run (the plateau's reading is still
logged). The build's own changes go on as more `--set` lines.
The model and learner settings the arms build on (the trunk order, BC's discounts and
trace decays) are the defaults: a run copied from an older arm's command line may carry
them as `--set` lines too, which changes nothing. The exact command of any run is in its
`run.json`{.plain} (`segments[].command`): copy it from there rather than rebuilding it.

## Resume a BC run

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb train --resume clone
uv run --no-sync bnb train --resume clone --bc-budget-steps 800000000
```

A resume continues a stopped or crashed BC run from its newest checkpoint, as for PPO
([Resuming, forking and sharing runs](resuming-forking-sharing.md)): the stop rule's score
windows and the plateau's state come back with the rest, so the rules decide as
they would have in one process, and so do both averages. `--bc-budget-steps` is the run's total budget and may be
raised on a resume; the other stop settings come from the checkpoint. A finished run
resumed trains until a rule fires again: a raised budget continues one that stopped on its
budget, while the plateau (and a score target) usually fire on the first update. The new policy
files replace the old ones. A BC run itself starts fresh or resumes: it does not start
from another run (`--from`) or a policy file (`--init-from`).

## Start reinforcement learning from the clone

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --preset mixed --init-from runs/clone/policy
```

`--init-from` starts a new PPO run from the policy file's weights and return scale, with a
fresh optimizer and step count (so the learning-rate warm-up runs again) and no cloning
term, and its live rating starts from the file's lane
ratings ([Resuming, forking and sharing runs](resuming-forking-sharing.md)). Any of the
four files works: `--init-from runs/clone/policy-live` starts from the live clone. Its entropy
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
from bnb.league.lanes import LaneHooks
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
hooks = LaneHooks(student)  # the lanes' ratings and the averages the stop rule reads
reason = cloning.clone(student)  # its stop rule: config.stop
average = student.stop_rule.read or config.stop.reads[0]
print("stopped by", reason, "reading", average)

# Reinforcement learning from that average in memory: its weights and the return scale, a
# fresh optimizer (bnb.runs.session.save_policies writes the policy files instead).
student.model.load_state_dict(hooks.averages.weights[average])
rl = Trainer(TrainConfig(game=game, environments=256, model="relational"), torch.device("cuda"))
cloning.warm_start(rl, student)
for _ in range(100):
    rl.update()
```

`tests/sections/test_bc_toy.py` runs the toy case and checks the clone's score against its
teacher and the chaser.
