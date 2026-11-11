---
covers: [src/bnb/cli.py, src/bnb/runs/chain.py, src/bnb/runtime/train.py, src/bnb/contracts/pools.py, src/bnb/contracts/config.py, src/bnb/game/modes/rules.py]
reviewed: 2026-11-11 15:15
---

# Training

Training trains Yemong (`bnb.models.yemong`), the toy relational model or the
nearest-neighbour MLP, with the learner in `bnb.learn.ppo` (see
[The learner](../explanation/learner.md)). How the loop
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
| `--model` | `yemong`, `relational`, `nearest` | `TrainConfig.model`: Yemong, the toy relational model, or the nearest-neighbour MLP |
| `--precision` | `bfloat16` (default), `float32`, `tf32` | `TrainConfig.precision`: the matmuls' precision |
| `--environments` | a count | `TrainConfig.environments`: training environments (the rating lanes, `--set lanes.environments=N`, are beyond them) |
| `--shards` | a count | `TrainConfig.shards`: consecutive segments collected per update; with `--environments` the update's layout (`--set lookahead=N` sets the ticks played after the last shard) |
| `--old-log-probabilities` | `rollout`, `recomputed` | `LearnerConfig.old_log_probabilities`: the rollout's, or recomputed by `unroll` before the first epoch |
| `--micro-batch-rows` | a count | `LearnerConfig.micro_batch_rows`: ship rows per micro-batch (memory only; the result is the same) |

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --precision tf32 --shards 2 --micro-batch-rows 320
```

The model's settings are `TrainConfig.yemong` (`bnb.models.yemong.model.YemongConfig`) and
`TrainConfig.relational` (`bnb.models.relational.RelationalConfig`) and `TrainConfig.nearest`
(`bnb.models.nearest.NearestConfig`: `hidden`, `allies`, `enemies`, `fields`); the run builds
the one `--model` names. The ship card the run plays is `TrainConfig.ship_card`, its values by
field, one per ship type (the default is `bnb.game.api.default_card`). `--set` changes
either by its dotted path; the run's `config.json`{.plain} and its checkpoints record them:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --set yemong.width=256
uv run --no-sync bnb train --preset mixed --set ship_card.hit_damage=[14,18]
```

A tuple is a JSON list and None is `null`. The single value arm (one per-ship value over
the summed reward levels, no outcome head;
[the learner](../explanation/learner.md#the-single-value-arm)) sets the critic's levels and
one discount, trace decay and team spirit for its one level:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --set 'yemong.levels=["summed_reward"]' \
  --set yemong.outcome_level=null --set 'learner.discount=[0.999]' \
  --set 'learner.trace_decay=[0.97]' --set 'learner.team_spirit=[0.0]'
```

`bfloat16_matmuls` is not set there: `--precision` sets it.

The ablations are settings of the same code. Yemong without temporal layers keeps its
spatial sublayers and drops each block's temporal one (`YemongConfig.order`
`"spatial_only"`), so the fed-back beliefs are its only memory. The nearest-neighbour MLP is
`--model nearest`:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --set 'yemong.order="spatial_only"'
uv run --no-sync bnb train --preset mixed --model nearest
```

## Chain BC and RL

A model is trained in two stages: behaviour cloning of the scripted agent until its
imitation plateaus, then PPO from the clone. `--objective chain` runs both in one command,
to a combined budget of environment steps:

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --objective chain --preset mixed --chain-budget-steps 200000000 \
    --bc-set learner.micro_batch_rows=2560 --rl-set learner.micro_batch_rows=320 \
    --name 2026-11-12-chain-example
uv run --no-sync bnb train --resume 2026-11-12-chain-example
uv run --no-sync bnb train --resume 2026-11-12-chain-example --chain-budget-steps 300000000
```

The BC stage is `bnb train --objective bc` with the combined budget as its own, a cap: in a
chain that goes as planned its plateau rule stops it first
([When it stops](../explanation/behaviour-cloning.md#when-it-stops)). The RL stage is a warm
start (`--init-from`) from the BC stage's exponential average, `policy-exponential/`{.plain},
for every model: a fresh optimizer and step count, so the learning-rate warm-up runs again,
the warm start's entropy coefficients (`bnb.learn.ppo.WARM_START_ENTROPY`) and no term
pulling it towards the clone. It trains the whole updates that fit in what the combined
budget leaves (`bnb.runs.chain.reinforcement_updates`), none when BC used it all.

Every other option (`--preset`, `--model`, `--environments`, `--set`, `--wandb`, ...)
applies to both stages; `--bc-set` and `--rl-set` are `--set` lines for one stage, applied
after the shared ones. `--updates`, `--bc-budget-steps`, `--from` and `--init-from` are the
chain's to set and are refused. `--print-config` prints both stages' configurations, as
`{"bc": ..., "rl": ...}`.

A chain is two ordinary runs and a record (`bnb.runs.chain`):

<!-- not-runnable: a directory layout -->
```text
runs/2026-11-12-chain-example/chain.json   the chain: its command, budget, stages, handover
runs/2026-11-12-chain-example-bc/          the BC stage, a BC run
runs/2026-11-12-chain-example-rl/          the RL stage, a PPO run
```

The handover is in `chain.json`{.plain}: the BC stage's last environment step, the rule that
stopped it and the policy file the RL stage started from. Each stage's
`run.json`{.plain} names the chain, its stage and its `start_step`, the chain's steps
before it, and the RL stage's `parent` is the policy file. Each stage's records count
their own environment steps from 0 and carry `counters/chain_environment_steps`, the step
on the chain's axis, so the two `metrics.jsonl`{.plain} files plot as one curve.

`--resume NAME` continues the chain where it is: the BC stage from its newest checkpoint,
the handover (a finished BC stage with no RL stage yet), or the RL stage. Each stage
writes a checkpoint at its step 0, so an RL stage that failed before its first cadence
checkpoint resumes from its start. A resume takes its settings from the chain's record;
it may change `--chain-budget-steps` (the RL stage's updates follow; a BC stage not yet
finished gets it as its budget), `--device`, `--runs-directory`, `--checkpoint-every` and
`--wandb`. A stop request (Ctrl-C, SIGTERM, or a `STOP`{.plain} file in the running stage's
directory) ends the process after the stage in progress, with its checkpoint; so does a
BC stage that did not finish. Each stage's run can also be resumed alone as a plain run,
which does not go on to the next stage.

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
and sets `TRITON_DEFAULT_FP_FUSION` before torch is imported and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first CUDA allocation (less
reserved memory, a slightly faster update; an explicit setting wins). A script that calls
`bnb.runtime.train.train` does the same itself. Library code sets no process-wide options;
only entry points do (see [Architecture](../explanation/architecture.md)).
