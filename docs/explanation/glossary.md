---
covers: [src/bnb/contracts/, src/bnb/runtime/lineup.py, src/bnb/runtime/rollout.py]
---

# Glossary

The terms these pages use, in the meaning the code gives them.

## Adapter

The one file a section writes for a set of reference fixtures: it translates the fixtures'
interface to `bnb`. See [Reference fixtures](../guides/reference-fixtures.md).

## Belief

What a model believes about every ship row of its view: the ship's state where it is
visible, the model's own forecast where it is hidden. Yemong keeps it in its carry. See
[The model (Yemong)](yemong.md#the-belief).

## Carry

A controller's or the rollout's state from one decision step to the next, held between
calls: for Yemong, its recurrent state and its belief; for the rollout, the game state, every
controller's carry, the buffers and the random draws. A carry is a pytree of tensors, so
the runtime can store, copy and capture it without knowing what is inside. A stateless
controller's carry is `None`.

## Controller

Whatever chooses the actions of one team: a model, a scripted agent, random play, later a
human. It implements `bnb.contracts.Controller`. See [Contracts](contracts.md#controllers).

## Decision step, tick

One step of the game, `Rules.tick_length` seconds. The physics tick and the decision step
are the same: every controller acts on every tick.

## Environment

One match in the batch. A batch of `B` environments steps together, one tensor row each.

## Golden

A file of inputs and outputs computed by an independent reference, which a section's code
must reproduce. Goldens are never edited. See [Reference fixtures](../guides/reference-fixtures.md).

## Lineup, pool, seat

- A **pool** (`bnb.contracts.Pool`) is a contiguous slice of the batch's environments with
  a fixed lineup and one game.
- A **lineup** is which controller plays each role in a pool: `Pool.seats`.
- The **seat** of an environment decides which team each role plays: role `r` plays team
  `(r + seat) mod 2`. It is redrawn at every reset, so every controller plays both sides.

## Pytree

A nested structure of tensors that torch can walk: here, a frozen dataclass whose fields
are tensors or other such dataclasses, registered with `torch.utils._pytree`
(`bnb.contracts.tree.contract`). The runtime maps a function, such as a copy or a move to
the GPU, over every tensor of a pytree at once.

## Rules, ship card

- **Rules** (`bnb.game.rules.Rules`) are the sizes and shared constants of a batch, as
  Python numbers. Compiled code specializes on them.
- The **ship card** (`bnb.game.ship_types.ShipTypes`) is the per-type parameters, as
  tensors. Changing them recompiles nothing.

## Sim index, team-relative

- The **sim index** of a ship is its position in the game state: team-major, ship
  `n = team * ships_per_team + slot`. Events use it.
- **Team-relative** order is the observation's: the viewer's own ships first, then the
  enemies. A ship's `team_relative` column is 0 for the viewer's team and 1 for the other.

## Stage

A pure function of the rollout's carry that the runtime compiles on its own and whose
outputs it writes back into the carry. A tick is a sequence of stages. See
[Architecture](architecture.md#compiled-stages).

## View

One team's observation of one environment. An `Obs` with a leading `(B, 2)` holds both
teams' views of every environment; a controller receives `R` views, the rows of the batch
it plays.
