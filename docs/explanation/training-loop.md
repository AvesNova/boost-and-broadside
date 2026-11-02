---
covers: [src/bnb/runtime/, src/bnb/controllers/, src/bnb/models/relational.py]
reviewed: 2026-11-02
---

# The training loop

The loop lives in `bnb.runtime.train`. It trains Yemong, or the toy relational model, with
the learner (see [The learner](learner.md)), or clones the scripted agent with
`TrainConfig.objective` set to `bc` (see [Behaviour cloning](behaviour-cloning.md), whose
loop with its stop rule is `bnb.runtime.cloning.clone`). How to run and configure it is in
the [training guide](../guides/training.md).

## One update

`bnb.runtime.train.Trainer.update` does, in order:

1. **Collect** `shards` segments of `length` ticks over the whole batch (`Rollout.run`),
   each followed by the learner's bootstrap evaluation (`Rollout.bootstrap`); the segments
   go into the rollout's segment store.
2. **Learn:** one update of the learner over the stored segments
   (`bnb.learn.ppo.Learner.update`; with BC, `bnb.learn.bc.BehaviourCloning`, the same
   update with the cloning loss).
3. **Refresh the league:** every `league_every` updates, copy the learner's weights into the
   `league_0` controller's model, in place.

It returns a row of measurements: the times of the rollout and the update, the learner's
statistics (each loss term, the value loss and explained variance per level, the clip
fraction and approximate KL of the first epoch, or with BC the BC KL and the entropies,
the step/unroll ratio drift, the return scale and the learning rate), and per pool the learner's reward per level per ship per
second and its episode results.

## Controllers

The trainer builds the learner, its league copy, and a controller for every other name its
pools seat:

- `learner`: the model being trained, through `bnb.controllers.yemong.YemongController` or
  `bnb.controllers.relational.RelationalController`;
- `league_0`: a frozen copy of it;
- `chaser`: `bnb.controllers.scripted.Chaser`, which turns toward and shoots at the nearest
  visible living enemy;
- `random`: `bnb.controllers.scripted.Random`, sticky uniform random actions;
- `scripted`: `bnb.controllers.scripted.Scripted`, [the scripted agent](scripted-agent.md),
  with its characteristics measured once for the trainer. With BC it is wrapped in
  `bnb.learn.bc.TeacherOpponent`, which plays the labels' distribution.

The keyboard pilot, `bnb.controllers.human.human_action`, is for the
[play shell](in-progress/play-shell.md) only; no pool seats it.

`TrainConfig.model` chooses the model: `yemong` (`bnb.models.yemong`) or `relational`, the
toy learner's fixed-size model (`bnb.models.relational.Relational`). The skeleton MLP
(`bnb.models.mlp.MLP`) and its controller (`bnb.controllers.model.ModelController`) remain
for the contract tests; the loop does not train them.

## The rollout

`bnb.runtime.rollout.Rollout` owns the [carry](glossary.md#carry): the game state, the ship card, every
controller's carry, the current observation, the seat, the tick's random draws, the
reward ledger, the learner's staging ring and its segment store, all allocated when it is
built. A tick
is a sequence of compiled stages over that carry (see [Architecture](architecture.md)).

- **The lineup** (`bnb.runtime.lineup.Lineup`) lists, for each controller, the environments
  and roles its views come from, pool by pool. A row's team is `(role + seat) mod 2`.
- **The seat** is redrawn for every environment at every reset, in the resolve stage, so
  every controller plays both sides equally.
- **Staggered clocks.** At the start each environment's episode clock is set to a random
  tick within its own episode length, so resets are spread over time rather than
  simultaneous.
- **Act.** Each controller receives its views of the current observation, the parts it
  `reads` plus the globals, and returns actions for its ships. The runtime scatters them
  into the batch's action tensor by environment and team.
- **Record.** For the learner's views, the act stage writes the observation parts it reads
  (and any `extra_reads`, such as the parts BC's teacher reads),
  every row's action, the log-probability and value, anything else its `Outputs` give (the
  values per level, the outcome probabilities, the replay record) and the label inputs into
  a staging ring of two graph blocks. The advance stage writes the shot capture, the resolve
  stage the learner's rewards per level, the `done` flags and each view's match result. After each block the rollout
  copies the ring's block into its segment store (`bnb.runtime.store.SegmentStore`), in
  pinned host memory or on the device.
- **Rewards** are computed every tick from the tick's events and the ledger (see
  [Rewards](game/rewards.md)), for every ship, and routed to the learner's views.

`Rollout.run` writes one shard of the store and returns it as a `bnb.contracts.Trajectory`
that views the store; it is valid until the shard is written again.

## Realtime

`bnb.runtime.realtime.Realtime` runs one match on the CPU at batch size 1 for interactive
play: the same game functions, with the whole tick compiled as one function. It takes
actions in sim order and returns every team's view.

<!-- runnable -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.runtime.realtime import Realtime

match = Realtime(GameConfig(ships_per_team=1, num_fields=2, arena_radius=1200.0),
                 compile=False)
obs = match.step(torch.zeros((2, 3), dtype=torch.int32))
print(obs.ships.shape)
```
