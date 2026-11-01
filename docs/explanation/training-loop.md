---
covers: [src/bnb/runtime/, src/bnb/learn/ppo.py, src/bnb/models/mlp.py, src/bnb/models/heads.py, src/bnb/controllers/]
---

# The training loop

The loop today is the walking skeleton's: everything runs end to end, and every learning
part is a stub that a later section replaces. It lives in `bnb.runtime.train`. How to run
and configure it is in the [training guide](../guides/training.md).

## One update

`bnb.runtime.train.Trainer.update` does, in order:

1. **Collect** a rollout of `length` ticks over the whole batch (`Rollout.run`), and the
   value of the learner's views after it, for the bootstrap.
2. **Learn:** one PPO update on the learner's views (`bnb.learn.ppo.PPO.update`).
3. **Refresh the league:** every `league_every` updates, copy the learner's weights into the
   `league_0` controller's model, in place.

It returns a row of measurements: the times of the rollout and the update, the PPO
statistics, and the learner's reward per ship per second for each pool.

## Controllers

The trainer builds one controller per name a pool can seat:

- `learner`: the model being trained, through `bnb.controllers.model.ModelController`;
- `league_0`: a frozen copy of it;
- `chaser`: `bnb.controllers.scripted.Chaser`, which turns toward and shoots at the nearest
  visible living enemy;
- `random`: `bnb.controllers.scripted.Random`, sticky uniform random actions.

The model is the skeleton MLP (`bnb.models.mlp.MLP`). Each own ship reads its own row and a
summary of the whole view, with fixed scales, through a factored policy over power, stick
and shoot (`bnb.models.heads`). It exists to exercise the loop. Because its summary is a
flattened view, it reads token position, and it relies on rows being ordered by
`team_relative`.

## The rollout

`bnb.runtime.rollout.Rollout` owns the carry: the game state, the ship card, every
controller's carry, the current observation, the seat, the tick's random draws, the
reward ledger and the learner's trajectory buffer, all allocated when it is built. A tick
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
- **Record.** For the learner's views, the act stage writes the observation parts it reads,
  the action, log-probability and value, and anything else its `Outputs` give (the values
  per level, the outcome probabilities, the replay record) into the trajectory buffer. The
  resolve stage writes the learner's rewards per level and the `done` flags.
- **Rewards** are computed every tick from the tick's events and the ledger (see
  [Rewards](game/rewards.md)), for every ship, and routed to the learner's views.

`Rollout.run` returns a `bnb.contracts.Trajectory` that views the buffers; it is valid
until the next run.

## The learner

`bnb.learn.ppo.PPO` is plain PPO, the skeleton's stub:

- GAE over the levels summed: the stub's critic is one value per ship;
- a clipped policy loss, a squared-error value loss and an entropy bonus;
- several epochs of shuffled minibatches, with gradient clipping, through a fused Adam on
  the GPU.

The loss is compiled. The optimizer updates the parameters in place, which keeps the
rollout's captured CUDA graph pointing at the current weights. The defaults are in the
[training guide](../guides/training.md).

Section 6 replaces this learner: per-level returns with the categorical critic, the return
scale, the auxiliary losses and their labels, over Yemong (see
[The learner](in-progress/learner.md)).

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
