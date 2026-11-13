---
covers: [src/bnb/game/perception/, src/bnb/contracts/obs.py, src/bnb/runtime/lineup.py]
reviewed: 2026-11-03 09:45
---

# Perception and fog

Perception turns the game state into each team's view: who it can see, and the
observation contract built from that (`bnb.contracts.Obs`, described in
[Contracts](../contracts.md)). It runs after the tick, on the final state, in
`bnb.game.api.perceive`, which returns the state, the observation and the privileged truth.

## Visibility

`bnb.game.perception.sight.visibility` decides which ships each team sees. A ship is
visible to a team when:

- it is on that team; or
- it is alive and it fired this tick; or
- it is alive and it spawned this tick; or
- it is alive and some living ship of that team is within that observer's vision range of
  it with a clear sight line.

A **sight line** is blocked by a field's opaque core, a disc of radius
`max(radius - band_width / 2, 0)`, when the segment passes strictly inside that disc,
unless both ends are strictly inside the same core. Range tests compare squared
distances. Bullets are not perceived.

The vision range is the observer's: its type's scale (`vision_range_scale` on the
[ship card](ship-card.md)) times the shared `Rules.vision_range`. Perception takes no card,
so every reset and every tick write the ranges into `Match.vision_range`, which
`bnb.game.perception.sight.visibility` reads.

## Fog of war

Fog is per environment. `Match.fogged` is set at reset and kept through the episode resets
that follow (`bnb.game.state.Match`).

- **Fogged:** a team sees only what the visibility rule gives it. For a ship it cannot see,
  the hidden columns are zero.
- **Clear:** every living ship is visible to both teams, and `time_since_seen` is 0.

The observation's `fogged` global column says which. In training, `Pool.fog_share` sets the
fogged fraction of each pool. The split is fixed and contiguous: the first `fog_share` of
each pool's environments are fogged for the whole run
(`bnb.runtime.lineup.Lineup`). The seat is redrawn every episode, so neither side gains
from the split.

## Time since seen

`Match.ticks_since_seen (B, 2, N)` counts, per team and ship, the ticks since that team
last saw the ship. It is 0 while the ship is visible, and in a clear environment. It is
the only state perception changes, which is why `perceive` returns the state. The
observation shows it in seconds, in `time_since_seen`.

## Building the observation

- Each view lists rows by `team_relative`: view `v`'s row block `j` holds team
  `(j + v) mod 2`. With team-major storage this is a static gather, fused into the copy the
  stage writes anyway (`bnb.game.perception.observe.view_order`).
- The ship rows are the hidden columns where the viewer sees the ship and zero where it
  does not, then the always-shown columns.
- The fields are the same in every view. The zones and the mode's global columns come from
  the mode, on its own slice of the batch (see [Modes](modes.md)).
  Frontline's capture locks are public zone columns. Each view's `next_wave_own` is its own
  team's wave clock only, and an enemy ship's `respawn_wait` shows only its minimum wait,
  so nothing in a view gives the enemy's wave clock directly.
- **Privileged** (`bnb.contracts.Privileged`) is built from the same rows with every ship
  visible (the dead included, with `time_since_seen` 0), plus each ship's sim team and the
  sim index of every row. No controller receives it. The rollout stores its enemy rows for
  the learner's labels (`Trajectory.truth`).

## The seat

Nothing in the observation may reveal which team index or which side of the map a view
belongs to. Rows are team-relative, maps are symmetric, spawn formations are shuffled, and
side-specific roles are drawn per episode. The tests compare the distribution of
observations from either seat:
`tests/sections/test_perception.py::test_spawn_layout_does_not_reveal_the_seat`, its
per-mode form in `tests/sections/test_modes.py`, and the mixed-batch form
`tests/sections/test_mixed.py::test_a_mixed_batch_does_not_reveal_the_seat`.
