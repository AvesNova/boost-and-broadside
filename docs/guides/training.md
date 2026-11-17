---
covers: [src/bnb/cli.py, src/bnb/runs/chain.py, src/bnb/runtime/train.py, src/bnb/contracts/pools.py, src/bnb/contracts/config.py, src/bnb/game/modes/rules.py, src/bnb/runs/mirror.py]
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
war where its game has fog (none in team deathmatch: `GameConfig.fogged_share`); it too
must come to a whole number. Every field of a pool:

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

`TrainConfig.mode_mix` sets the mix as a setting rather than a preset: game presets and
whole weights, each game's share its weight over their sum. `bnb train` builds the pools
from it (`bnb.runtime.train.arranged`), every game with its own map, clocks, respawn and
spawn groups. The 8/4/4 arm (A5), on top of `--preset real_scale_all_modes`:

<!-- not-runnable: options added to a bnb train command -->
```bash
--set 'mode_mix={"real_scale_frontline": 8, "real_scale_deathmatch": 4, "real_scale_elimination": 4}'
```

A training preset may carry settings besides its games
(`bnb.runtime.train.PRESET_SETTINGS`, applied before the command's own `--set`), so a run
of it traces to the committed settings alone. `--preset release_jujak_elimination`, the
release run's (the author, 2026-11-17), is real-scale elimination alone
(`real_scale_elimination_hemisphere`: no fog, the hemisphere spawn of
[Modes](../explanation/game/modes.md)) with:

- Jujak in every slot of both teams (`lineup`, `lineup_draw` false), the rating lanes too;
- the belief off: no next-state head, so no forecast fed back and no error head
  (`YemongConfig.next_state_head`, `forecast_feedback`, `delta_split` false,
  `physics_hidden_decay` null), and no enemy-action head; the density, shot and geometry
  heads stay;
- the lineups (`team_sizes`, a `bnb.contracts.TeamSizes` table with `pools` "opponents"
  and `learner_side` "smaller"): self-play 5v5; every pool with an opponent (the league's,
  BC's Imugi pool) 5v5 in half its games, the learner 4 against 5 in a quarter and 3
  against 5 in a quarter, the learner always the smaller side, every field in play. The
  rating lanes play 5v5, so the training Elo curve stays one game.

`box/release_run.sh` launches it on the 5090 box.

Setting `pools` by hand gives every pool the `GameConfig`'s game unless each pool names its
own; `mode_mix` does not have that trap. A trainer refuses pools whose games and shares
are not the mix's (`bnb.runtime.train.check_pools`).

Every map is fixed-size by default: each pool plays its own arena radius every episode
(`TrainConfig.arena_scale=None`). This holds until an agent has trained successfully on
variable-size maps (the author's rule, 2026-11-15); map-size sampling is an opt-in setting
for the map-size arm, `--set 'arena_scale={"sigma": 0.35, "low": 0.5, "high": 2.0}'`
(every training environment draws its arena scale at each reset, ln scale ~ Normal(0,
0.35) truncated to 0.5 to 2, then the mode's minimum radius). The rating lanes always play
their games' own sizes. See
[Mixed batches](../explanation/game/mixed-batches.md#map-size-sampling).

The number of fields can vary too: `TrainConfig.active_fields=ActiveFields(low, high)`
plays a drawn count of the game's field slots each episode, the rest absent, and the
Yemong learner masks the absent ones. It is off by default (every field). See
[Mixed batches](../explanation/game/mixed-batches.md#field-count-sampling).

Team sizes can vary too (arm A2): `TrainConfig.team_sizes=TeamSizes(...)` draws each
episode's two team sizes, equal or not, and its field count with them, within each mode's
smallest team (Frontline 3 a side), the rest of each team's slots benched and the other
fields absent, both masked. It is off by default (every slot). `--set 'team_sizes={}'`
plays 5v5 in half the games and spreads the other half evenly over the other pairs of
sizes, one field per ship; the training log records the drawn sizes per mode
(`team_size/...`). See
[Mixed batches](../explanation/game/mixed-batches.md#team-size-sampling).

Each episode draws its lineup by default (`TrainConfig.lineup_draw`): every slot is one of
the ship set's drawn types with equal chance (`bnb.game.api.DRAWN_LINEUPS`: Yongma or
Jujak in the real-scale set), and both teams fly the lineup slot for slot, as the
evaluation suite's games do. The draw is keyed by the run's seed, the environment and
the episode, and covers the rating lanes too. Set `lineup_draw=false` to play the game's
own lineup, or `lineup` to fly one fixed lineup, which replaces the draw.

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
(`bnb.models.nearest.NearestConfig`: `hidden`, `allies`, `enemies`, `fields`,
`full_mode_inputs`); the run builds
the one `--model` names. The ship card the run plays is `TrainConfig.ship_card`, its values by
field, one per ship type (the default is `bnb.game.api.default_card`). `--set` changes
either by its dotted path; the run's `config.json`{.plain} and its checkpoints record them:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --set yemong.width=256
uv run --no-sync bnb train --preset mixed --set ship_card.hit_damage=[14,18]
```

A tuple is a JSON list and None is `null`. The offence premium
([the learner](../explanation/learner.md#the-offence-premium)) is a keypoint schedule, a
JSON object; 2:1 annealed to 1:1 between 20M and 60M steps:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed \
  --set 'learner.offence_premium={"steps": [0, 20000000, 60000000], "values": [2.0, 2.0, 1.0]}'
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

A model is trained in two stages: behaviour cloning of Imugi, the scripted agent, for a fixed
budget, then PPO from the clone. `--objective chain` runs both in one command,
to a combined budget of environment steps:

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --objective chain --preset mixed --chain-budget-steps 200000000 \
    --bc-set learner.micro_batch_rows=2560 --rl-set learner.micro_batch_rows=320 \
    --name 2026-11-12-chain-example
uv run --no-sync bnb train --resume 2026-11-12-chain-example
uv run --no-sync bnb train --resume 2026-11-12-chain-example --chain-budget-steps 300000000
```

The BC stage is `bnb train --objective bc` with its own budget, 100M environment steps for
every arm, at most the combined one (`bnb.cli.chain_bc_budget`; `--bc-set
stop.budget_steps=N` or the chain's `--bc-budget-steps N` sets another); its plateau rule only logs
([When it stops](../explanation/behaviour-cloning.md#when-it-stops)). The RL stage is a warm
start (`--init-from`) from the BC stage's exponential average, `policy-exponential/`{.plain},
for every model: a fresh optimizer and step count, so the learning-rate warm-up runs again,
the warm start's entropy coefficients (`bnb.learn.ppo.WARM_START_ENTROPY`) and no term
pulling it towards the clone. It trains the whole updates that fit in what the combined
budget leaves (`bnb.runs.chain.reinforcement_updates`), none when BC used it all.

Every other option (`--preset`, `--model`, `--environments`, `--set`, `--wandb`, ...)
applies to both stages; `--bc-set` and `--rl-set` are `--set` lines for one stage, applied
after the shared ones. `--updates`, `--from` and `--init-from` are the
chain's to set and are refused. `--set update_policies=rl` writes every update's policy
files in the RL stage only (`TrainConfig.update_policies`; `all` adds BC's, about 1 GB more).
`--print-config` prints both stages' configurations, as `{"bc": ..., "rl": ...}`.

`--bc-budget-steps 0` is a chain with no BC stage (arm A24, RL from scratch): the handover
is recorded at step 0 with no policy (`stopped_by` `no_bc`), no `-bc` run is made, and the
RL stage is a fresh run over the whole combined budget with the RL stage's settings, the warm
start's entropy coefficients (`WARM_START_ENTROPY`, 0) included, so it differs from BC→RL only
by the clone. A plain fresh PPO run (`--objective ppo`) keeps `LearnerConfig`'s −0.0081
instead. `--print-config` then prints `{"bc": null, "rl": ...}`:

<!-- not-runnable: needs CUDA and runs for hours -->
```bash
uv run --no-sync bnb train --objective chain --preset mixed \
    --chain-budget-steps 300000000 --bc-budget-steps 0
```

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

## Mirror a run into W&B

A run trained without `--wandb` (on a box with no W&B login) can be mirrored into W&B from
a machine that is logged in. `bnb runs mirror` (`bnb.runs.mirror`) reads the run's
`metrics.jsonl`{.plain}, `config.json`{.plain} and `run.json`{.plain}, locally or fetched with
rsync over ssh into `~/.cache/bnb/mirror/HOST`{.plain}, and logs each record as `--wandb` would
have: the same project, name, config and `counters/environment_steps` x-axis, a chain's
stages as two W&B runs in the chain's group. A chain's name mirrors its stages that exist.

<!-- not-runnable: needs W&B and a box -->
```bash
uv run --no-sync bnb runs mirror 2026-11-12-chain-example
uv run --no-sync bnb runs mirror dry-v24-s1 --follow --host box5090 --remote-runs /workspace/runs
```

It can be run again at any time: the W&B run's id is derived from the run's name (or is the
one its `run.json`{.plain} keeps, for a run that streamed live) and resumed, and the newest
step mirrored, kept in the run's `wandb_mirror.json`{.plain}, is not logged twice. `--follow`
polls every `--every` seconds (300) until the chain's `<name>.exit`{.plain} file
(box/dry_run.sh writes one) appears or no stage is left running, then syncs once more.

## Stop a run that stops fighting

Each update line ends with the trigger rules that are up (`alerts: ...`). Two of them watch
the training games' combat: `combat_collapse/team_deathmatch` and
`combat_collapse/elimination` are up while the mean over the last three updates of that
mode's combat deaths or hits per living ship-minute (`combat/deaths_per_ship_minute/{mode}`,
`combat/hits_per_ship_minute/{mode}`) is below its floor: the lower of 0.075 deaths or 6.5
hits (a quarter of a BC policy's team deathmatch on the real-scale set) and a quarter of the
run's own first five updates. `TrainConfig.combat_collapse_stop` ends a PPO run after that
many consecutive updates with either up, and `run.json`{.plain} records the rule as
`stopped_by`; 0, the default, never stops:

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync bnb train --preset mixed --set combat_collapse_stop=5
```

The `skipped_step` rule is up for an update with non-finite minibatches (their optimizer
steps are skipped). It writes a checkpoint and the update's batch under
`dumps/step_<steps>/` in the run, for replaying the step, at most
`TrainConfig.skipped_step_dumps` times per run (default 1: a full-scale dump is about 10 GB);
later skipped updates are alerted and logged only.

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

`--seed` (the run's `TrainConfig.seed`; `--set seed=N` is the same) keys every random
draw of the run ([Random draws](../explanation/architecture.md#random-draws)): two runs
of one seed that differ in a setting draw alike wherever they are alike. The `bnb`
command sets the matmul precision (`bnb.cli.set_precision`) once, before training,
and sets `TRITON_DEFAULT_FP_FUSION` before torch is imported and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first CUDA allocation (less
reserved memory, a slightly faster update; an explicit setting wins). A script that calls
`bnb.runtime.train.train` does the same itself. Library code sets no process-wide options;
only entry points do (see [Architecture](../explanation/architecture.md)).
