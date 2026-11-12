---
covers: [src/bnb/runtime/, src/bnb/controllers/, src/bnb/models/relational.py]
reviewed: 2026-11-06 12:00
---

# The training loop

The loop lives in `bnb.runtime.train`. It trains Yemong, or the toy relational model, with
the learner (see [The learner](learner.md)), or clones the scripted agent with
`TrainConfig.objective` set to `bc` (see [Behaviour cloning](behaviour-cloning.md), whose
loop with its stop rule is `bnb.runtime.cloning.clone`). How to run and configure it is in
the [training guide](../guides/training.md).

## One update

`bnb.runtime.train.Trainer.update` does, in order:

1. **Collect** `shards` consecutive segments of `length` ticks over the whole batch
   (`Rollout.run`), then the lookahead: `TrainConfig.lookahead` more ticks of the same
   games (`Rollout.run_lookahead`), and the learner's bootstrap evaluation after it
   (`Rollout.bootstrap`). The segments go into the rollout's segment store; of the
   lookahead the store keeps only what the learner's targets read there (the returns'
   records, and a bullet's flight of the shot capture). The lookahead is never trained on,
   and the next update continues from its end, so nothing is played twice.
2. **Learn:** one update of the learner over the stored segments
   (`bnb.learn.ppo.Learner.update`; with BC, `bnb.learn.bc.BehaviourCloning`, the same
   update with the cloning loss).

The default batch is v1's twice over, as v2 trains several modes: 2,048 training environments
x 4 shards x 128 ticks, 1,048,576 trained ticks per update, with a lookahead of 128 ticks
(`TrainConfig.environments`, `shards`, `length`, `lookahead`). For the same trained ticks,
more environments are more independent games, where more shards continue the same ones;
2,048 x 4 is the widest layout that fits an 8 GB GPU with room to spare. Measured with the lanes on (rollout per update,
device memory): 1,024 x 8, 22.7 s and 5.74 GB; 2,048 x 4, 21.8 s and 6.82 GB; 4,096 x 2,
24.0 s and 7.55 GB of 7.65 at 8 graph ticks; 8,192 x 1 does not fit. The learner's time
does not change with the layout. The rollout's
records stream to pinned host memory in chunks of `TrainConfig.graph_ticks` (8), two on the
device at a time, so the trajectory does not bind the device's memory. Every trained tick has at least
128 played ticks after it in the update, so the advantages are not cut short at a shard's end.

When a league is attached (`bnb.league.league.League`, which `bnb train` attaches to a PPO
run), its hooks run after each segment and after the lookahead, after the collection and
after the learning; see [The league](league.md). A BC run attaches the rating lanes'
hooks alone (`bnb.league.lanes.LaneHooks`), which also keep the averages of its weights.

It returns a row of measurements: the times of the rollout and the update, the learner's
statistics (each loss term, the value loss and explained variance per level, the clip
fraction and approximate KL of the first epoch, or with BC the BC KL and the entropies,
the step/unroll ratio drift, the return scale and the learning rate), and per pool the learner's reward per level per ship per
second and its episode results.

## Controllers

The trainer builds the learner and a controller for every other name its pools seat. The
default lineup (`bnb.runtime.train.LINEUP`) is self-play and the league's seat, half each:

- `learner`: the model being trained, through `bnb.controllers.yemong.YemongController` or
  `bnb.controllers.relational.RelationalController` (the relational and nearest models);
- `opponents`: the league's seat, `bnb.runtime.opponents.OpponentSlots`: network slots
  and the stationary ladder in one controller (see [The league](league.md));
- `chaser`: `bnb.controllers.scripted.Chaser`, which turns toward and shoots at the nearest
  visible living enemy, boosting when far and reversing when close; its distances are in
  the card's lengths (the race's 600, 120 and 250 m on the standard card), from the same
  characteristics as the scripted agent's;
- `random`: `bnb.controllers.scripted.Random`, sticky uniform random actions;
- `scripted`: `bnb.controllers.scripted.Scripted`, [the scripted agent](scripted-agent.md),
  with its characteristics measured once for the trainer. With BC it is wrapped in
  `bnb.learn.bc.TeacherOpponent`, which plays the labels' distribution.

- `lanes`: the rating lanes' seat, `bnb.runtime.lanes.LaneSeat` (see below).

The keyboard pilot, `bnb.controllers.human.human_action`, is for the
[play shell](in-progress/play-shell.md) only; no pool seats it.

`TrainConfig.model` chooses the model: `yemong` (`bnb.models.yemong`), `relational`, the
toy learner's fixed-size model (`bnb.models.relational.Relational`), or `nearest`, the
nearest-neighbour MLP of any fleet size (`bnb.models.nearest.Nearest`; see
[The learner](learner.md#the-nearest-neighbour-mlp)). The skeleton MLP
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
that views the store; it is valid until the shard is written again. `Rollout.run_lookahead`
plays the lookahead into the store's lookahead leaves (`SegmentStore.lookahead`).

## The rating lanes

The rating lanes (`bnb.runtime.lanes`) are extra environments of the same rollout, in the
same tensors as the training games, that play rated matchups: they are never stored for the
learner and count in none of its losses. `TrainConfig.lanes` (`bnb.runtime.lanes.LaneConfig`)
sets how many (`environments`, beyond the training ones) and what they play (`matchups`).

- **Players:** `live`, the weights the learner trains (copied into the lanes before each
  collection); a weight set loaded by name (`exponential_average` and `uniform_average`, the
  averages of the live weights that the league and BC keep, and `league_<k>`, the entry
  the league's network slot k holds); and the ladder's rungs,
  `random`, `rung_T<t>_p<p>`, `rung_T<t>` and `scripted` (the scripted agent at T = 1, with
  its mode's parameters, which is also BC's default teacher).
  A matchup is any pair of them.
- **Layout:** every mode of the training mix plays every matchup. The lanes are split evenly
  over the modes and, within a mode, by the matchups' weights; each (mode, matchup) is a
  pool of its own, `<mode>/lane/<first>_vs_<second>`, whose two seats are both the one
  controller `lanes`, so the seat is redrawn at every reset as in any pool. The rollout's
  shapes are fixed at capture: which row plays which player is static, only a network
  player's weights change, copied in place between segments (`LaneSeat.load`).
- **Defaults:** a reinforcement-learning run's lanes (`bnb.runtime.lanes.RL_MATCHUPS`) play,
  per mode: the live policy and the averaged agent against each of the league's four slot
  entries and each other; the slot entries and both against the scripted agent, a small
  share, the anchor; and the ladder chained from the scripted agent down (each rung against
  the next one up), which calibrates the ladder. A game tells the rating fit `p (1 − p)`
  about its players' gap: 0.25 at even odds, 0.004 at the 950 points v1's policy reached
  above the scripted agent. The league draws its slots' entries by `p (1 − p)` against the
  live policy, so they are near-even opponents, and older entries, which played lower ones
  before they cycled out, chain the newer to the scripted agent. BC's
  (`bnb.runtime.cloning.LANES`) play the live clone, the uniform average and the
  exponential average against the scripted agent alone, a third of each mode's lanes each.
- **A game counts once, by the weights that played it:** each lane row stamps its player's
  weight generation when its game begins, and loading another entry into a slot's lane
  player (`LaneSeat.load` with `fresh`) starts a new generation, so a game in progress then
  is left out rather than credited to the new entry. Each row's first game is left out
  too: the rollout's clocks are staggered, so it is cut short, and in the first updates
  those short games, mostly ties, read as a rating near the scripted agent's.
- **Results:** the rollout tallies each ended lane game once, from the first player's side,
  and the trainer takes the tally after each collection (`Trainer.lane_record`), logged as
  `lanes/games/<mode>/<matchup>` and `lanes/score/<mode>/<matchup>`. They are rated in
  one fit with the league's training games ([The league](league.md#the-ratings)).
- **Size:** 1,024 lanes by default. They cost rollout time in proportion to their network
  rows, about 5.5 s per 1,024 rows per update on the development laptop; the measurements
  are in `docs/rewrite/sections/rollout-lanes-notes.md`.

## Realtime

`bnb.runtime.realtime.Realtime` runs one match on the CPU at batch size 1 for interactive
play: the same game functions, with the whole tick compiled as one function. It takes
actions in sim order and returns every team's view; it keeps the tick's `Events`. The
match draws from its own `torch.Generator`, so a seed, a keyframe (`keyframe`, `restore`)
and the actions decide every state, and `background=True` compiles the tick on a thread
while the match runs it uncompiled.

<!-- runnable -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.runtime.realtime import Realtime

match = Realtime(GameConfig(ships_per_team=1, num_fields=2, arena_radius=1200.0),
                 compiled=False)
obs = match.step(torch.zeros((2, 3), dtype=torch.int32))
print(obs.ships.shape)
```
