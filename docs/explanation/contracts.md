---
covers: [src/bnb/contracts/]
reviewed: 2026-11-03 09:45
---

# Contracts

`bnb.contracts` holds the types that cross a seam between sections: their shapes, dtypes
and meanings, and no game logic. The code is the source of truth; the tables on this page
are generated from it.

- The data contracts (`Obs`, `Privileged`, `Action`, `Events`, `Outputs`, `Trajectory`)
  are frozen dataclasses of tensors, registered as [pytrees](glossary.md#pytree) by
  `bnb.contracts.tree.contract`.
- The configuration contracts (`GameConfig`, `Pool`) are plain frozen dataclasses of Python
  values. `Controller` is a protocol.
- `Obs`, `Action` and `Events` each carry a schema version (`bnb.contracts.Obs.SCHEMA_VERSION`
  and the like). A real break of their layout bumps it. Adding an observation column does
  not: models read columns by name, a column's meaning never changes under the same name
  (a new meaning is a new name), and a policy file records the columns it trained with
  ([runs](runs.md)).
- `tests/contract` checks every implementation of every contract: shapes, dtypes, the
  import direction and ONNX export of each model's step.

Shapes use `B` environments, `N` ships (two teams of `S`), `M` fields, `Z` zone slots,
`R` [views](glossary.md#view) and `L` steps.

## The observation

[`Obs`][bnb.contracts.obs.Obs] is what one team sees. `bnb.game.api.perceive` builds one view per
team, with a leading view axis: view `v` is what team `v` sees.

- **Raw and in world coordinates.** No rotation, no mirroring, no model features. Each
  model composes its own inputs from it.
- **Team-relative.** Every ship carries `team_relative = (team - viewer) mod 2`, and rows
  are ordered by it: the viewer's own `S` ships first, then the enemies, each by slot.
  Nothing in the observation reveals which team index, or which side of the map, the
  viewer has. The real team index exists only in the game state and in `Privileged`.
- **Hidden ships.** For a ship the team cannot see, the first
  `bnb.contracts.obs.SHIP_HIDDEN_ZERO` columns are zero; the rest are always shown.
- **Accelerations.** `acceleration_x`, `acceleration_y` (m/s^2) and `nose_acceleration`
  (rad/s^2) are the change of the velocity and of the nose rate over the last tick, per
  second, in world coordinates like the velocity. `acceleration_valid` is 1 only for a real
  tick of motion the team saw: the ship alive at both ticks, not spawned or respawned at
  this one, past the game's first tick, and seen by the team at both ticks. So a ship just
  spotted shows none, and nothing about the tick before it was seen leaks. Otherwise the
  three are 0. The game state keeps each ship's velocity, nose rate, life and each team's
  sight from the last perception for this (`bnb.game.state.Match`).
- **Times are in seconds.** That covers the shield recharge wait, the gun cooldown, the
  respawn wait and the time since seen.
- **No bullets.** Models do not read them, a ship that fires is visible that tick, and the
  learner rebuilds shot labels from stored truth.

The parts and their columns, from `bnb.contracts.obs`:

**Ships**, `(..., N, len(SHIP_COLUMNS))`:

<!-- generated: ship-columns -->

**Fields**, `(..., M, len(FIELD_COLUMNS))`, the same in every view:

<!-- generated: field-columns -->

**Zones**, `(..., Z, len(ZONE_COLUMNS))`. Every mode has the same zone slots; a mode
without zones shows them absent, with `present` 0 and every other column 0. Roles are the
viewer's: 0 own spawn, 1 own defense, 2 neutral, 3 enemy defense, 4 enemy spawn. Rows are
in each viewer's own ring order. `lock_own` and `lock_enemy` are the seconds left of the
capture lock on the viewer's team and on the enemy's: the team that lost the zone gains no
capture progress on it until 0. Both are public.

<!-- generated: zone-columns -->

**Globals**, `(..., len(GLOBAL_COLUMNS))`. The first columns are the game's; the rest are
the mode's block: a mode one-hot (all zero in sandbox), the scores by team and the score
that wins (team deathmatch; in elimination the enemy's ships), the front for the viewer
and the front that wins (Frontline), and the seconds to the viewer's own team's next
respawn wave (`next_wave_own`, Frontline; the enemy's clock is not shown).
An absent quantity is 0.

Every time column is in seconds. In Frontline a ship's `respawn_wait` is its minimum
wait left; the ship then respawns at its team's next wave. An own ship's time to respawn
is therefore the first of `next_wave_own` + k x the wave interval at or after its
`respawn_wait`. An enemy ship's `respawn_wait` shows only its minimum, so it does not
reveal the enemy's wave clock.

<!-- generated: global-columns -->

`bnb.contracts.obs.SHIP`, `FIELD`, `ZONE` and `GLOBAL` map each column name to its index.
A stored batch may leave out (`None`) a part its model does not read.

## Privileged truth

`bnb.contracts.Privileged` is the truth only learners may read: each ship's sim team, the
sim index of each observation row, and every ship's row as if it were seen (its true
columns, the dead included). The
perceive stage produces it. A controller's `step` has no parameter for it, so no
controller can receive it.

## The action

[`Action`][bnb.contracts.action.Action] is integer components per ship, each shaped `(..., ships)`:

<!-- generated: action-layout -->

- `power`: 0 coast, 1 boost, 2 reverse.
- `stick`: 0 straight, then turn levels alternating right and left (the stick levels are
  tabled in [Physics](game/physics.md#flight)).
- `shoot`: 0 hold, 1 fire.

The joint index is `power * (NUM_STICK * NUM_SHOOT) + stick * NUM_SHOOT + shoot`, the
index of the action in the product of the components. `bnb.contracts.action.to_joint` and
`bnb.contracts.action.from_joint` convert. A model may use a joint head internally, as
Yemong does, and convert at the boundary.

<!-- runnable -->
```python
import torch

from bnb.contracts import Action
from bnb.contracts.action import NUM_JOINT, from_joint, to_joint

joint = torch.arange(NUM_JOINT)
assert torch.equal(to_joint(from_joint(joint)), joint)
action = Action(power=torch.tensor([1]), stick=torch.tensor([6]), shoot=torch.tensor([1]))
print(to_joint(action))
```

## Events

`bnb.contracts.Events` is what happened in one tick, in sim order (team-major), not in
observation order. Rewards and diagnostics read it; controllers never do.

- `damage (B, N, N)`: hit damage from shooter to target. It is a per-tick transient: the
  reward step consumes it and trajectories never store it.
- `boundary_damage`, `died`, `fired`, `spawned`, `killer` per ship; `done (B,)`.
- `shield_lost` and `shield_restored`: what each shield actually lost and regained.
- `objective_progress` and `objective`: the mode's two objective amounts, each summing to
  zero over the environment.
- `result (B,)`: the match result for team 0, nonzero only where `done`.
- `enemy_damage_by_aspect (B, N, 4)` and `kill_aspect`: for the statistics, the hit damage
  each ship took from the enemy team by the hit's aspect class, and the class of the hit
  that killed it (-1 if none). The classes (`bnb.contracts.ASPECTS`) split the angle
  between the bullet's reversed direction and the target's nose: a mutual head-on under
  10 degrees (both ships can fire), an offset head-on from 10 to 45 (only the shooter
  can), the flank to 135, the tail beyond. The damage itself follows the smooth aspect curve
  ([Physics](game/physics.md)); the classes only count it.

## Controllers

`bnb.contracts.Controller` is the interface every controller implements: models,
scripted agents, random, and later humans.

- A controller plays **one team**: it receives that team's views and returns actions for
  that team's `S` ships.
- `init_carry(views, device)` gives its opaque [carry](glossary.md#carry), a pytree the runtime stores and
  never inspects (`None` if stateless). `step(carry, obs, reset)` returns the action, the
  next carry and `bnb.contracts.Outputs`; `reset` flags the views whose environment reset
  before this observation.
- `reads` lists the observation parts it uses, so the rollout stores only those.
- A model's parameters live in the controller's own module, following PyTorch idiom. The
  league's seat (`bnb.runtime.opponents.OpponentSlots`) holds its slots' weights as stacked
  parameters and copies new ones in.

`bnb.contracts.Outputs` holds what a learner needs per controlled ship: the log-probability
of the chosen action, the entropy and the value. A controller that gives them adds the
critic per value level (`values`), the outcome probabilities (a critic with an outcome
level) and a `replay` record: what
the decision read from its carry, so the learner can recompute a stored segment without
replaying it. Under DPPO's exact trust region the learner's controller also gives its whole
joint action distribution (`probabilities`, float16), which the trajectory keeps
([The learner](learner.md#the-trust-region)).

## Pools and trajectories

- `bnb.contracts.Pool` is a contiguous slice of the batch with a fixed lineup and one game.
  The [training guide](../guides/training.md) lists its fields.
- `bnb.contracts.Trajectory` is a rollout as the learner receives it: `L` steps of the
  learner's `R` views, `S` ships each. Its reward is per level, in the order of
  `bnb.contracts.REWARD_LEVELS` (see [Rewards](game/rewards.md)). Its action holds every
  row of the view, and it carries the learner's label inputs: the enemy rows of the
  privileged view, the `alive` and `valued` flags, and the shot capture
  (see [The learner](learner.md)). Its `result` is each view's match result where an
  episode ended, from its own team's side.

## Configuration

`bnb.contracts.GameConfig` holds the sizes and the rules of a match: ships per team,
fields, arena radius, mode, spawn groups, the winning score and the two time-native rules.
It validates itself on construction. The game turns it into static rules with
`bnb.game.api.rules_of`.
