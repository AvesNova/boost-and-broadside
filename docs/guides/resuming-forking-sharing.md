---
covers: [src/bnb/cli.py, src/bnb/runs/]
reviewed: 2026-11-03 16:00
---

# Resuming, forking and sharing runs

Every `bnb train` is a run with a directory under `runs/` at the repository root. This page is about continuing,
branching and sharing runs; [Runs and checkpoints](../explanation/runs.md) explains what is
saved and why.

## Start and stop a run

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --updates 2000
```

The first line names the run, such as `2026-11-02-mixed-purple-pumpkin`; `--name` sets
it. Ctrl-C finishes the update in progress, writes a checkpoint and marks the run
`stopped` (a second Ctrl-C stops at once, without a checkpoint).

## See your runs

<!-- runnable -->
```bash
uv run --no-sync bnb runs
```

lists every run: its status, newest checkpoint step, start time and last commit. A run
whose process died without stopping shows as `running (stale)`. `bnb runs show RUN`
prints a run's record: its segments (one per process that ran it, with the commit and
steps), the overrides and the kept checkpoints.

## Resume

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb train --resume 2026-11-02-mixed-purple-pumpkin
uv run --no-sync bnb train --resume 2026-11-02-mixed-purple-pumpkin --updates 4000 \
    --set learner.learning_rate=0.0001
uv run --no-sync bnb train --resume 2026-11-02-mixed-purple-pumpkin --at 268435456
```

- A resume continues from the newest checkpoint with every setting the checkpoint holds,
  exactly where the run stopped.
- `--updates` is the run's total budget. `--precision`, `--micro-batch-rows` and
  `--checkpoint-every` may change, and `--set` the fields
  `bnb.runs.checkpoint.OVERRIDABLE` lists (the learning rate, the warm-up, the loss
  coefficients, the reward weights). Each change is recorded with the step it took effect
  at.
- A run whose process is still alive on this machine does not resume: stop it first.
- Any other option fails with a message pointing to `--from`.
- `--at STEP` resumes from an older kept checkpoint; the newer ones move to
  `checkpoints/rewound_<segment>/` and stay there.
- On another commit, the resume warns and records the commit in the new segment. A model
  whose version changed does not load: resume from a worktree at the run's commit.

## Fork

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb train --from 2026-11-02-mixed-purple-pumpkin --preset mixed \
    --set learner.entropy=-0.005 --updates 500
```

A fork is a new run with the command line's settings that starts from another run's
checkpoint (the newest, or `--at STEP`): the model, optimizer, return scale and
environment-step count; its league starts afresh from the model. `--updates` is the number of updates the fork trains (500
here), whatever the parent had done. Its record names the parent. It takes the parent's
model settings unless `--set` gives one (`--set yemong.width=256`); then the parent's
build record must match them, and a model whose build record differs (another width,
another ship card) is refused: that is a new run.

## Warm start from a policy file

<!-- not-runnable: needs CUDA and a policy file -->
```bash
uv run --no-sync bnb train --init-from runs/2026-11-02-mixed-quiet-otter/checkpoints/step_536870912 \
    --preset mixed
```

The new run's model starts from the policy file's weights and return scale, with a fresh
optimizer: behaviour cloning's hand-off to RL. A BC run saves its clone at
`runs/<run>/policy/`{.plain} ([Behaviour cloning](behaviour-cloning.md)). Any policy file or
training checkpoint works, if its build record matches the run's model; as for a fork,
the model's settings come from the file unless `--set` gives one. In Python, the same files are
`bnb.runs.checkpoint.save_policy` and `bnb.runs.checkpoint.load_policy`:

<!-- runnable -->
```python
import tempfile

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.runs.checkpoint import load_policy, save_policy

model = Relational(RelationalBuild(RelationalConfig(hidden=32), 2, 4))
with tempfile.TemporaryDirectory() as directory:
    save_policy(f"{directory}/policy", model, game=GameConfig(2, 2, 1500.0),
                types=api.ship_types())
    loaded = load_policy(f"{directory}/policy", torch.device("cpu"))
print(type(loaded).__name__, loaded.build.config.hidden)
```

## Check the memory first

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --environments 1536 --probe
```

runs the setup and one update, prints the allocator's peaks against the budget, and exits
without writing a run. Every run does the same on its first update and stops there if it
is over the budget.

## Share a checkpoint

<!-- not-runnable: needs a W&B login -->
```bash
uv run --no-sync bnb runs upload 2026-11-02-mixed-purple-pumpkin --at 268435456
uv run --no-sync bnb runs upload 2026-11-02-mixed-purple-pumpkin --league
```

uploads the chosen checkpoint (the newest without `--at`), or the league's snapshots, as a
W&B artifact named for the run and the step. Nothing uploads by itself.
