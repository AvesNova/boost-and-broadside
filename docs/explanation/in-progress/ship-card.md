---
covers: []
---

# The ship card (in progress)

<!-- generated: section-status -->

Today the
card is the per-type parameter table in `bnb.game.ship_types`
(see [Physics](../game/physics.md)).

It will cover **handles**: the quantities a designer sets, such as stall speed, g limit and
maximum level speed. An offline solver finds the physics parameters that make the simulated ship meet
them, by flying standard manoeuvres in the game's own code. The handles are independent
coordinates: changing one re-solves with every other held. The section also moves to the
card every parameter a handle sets, such as the vision range.
