---
covers: [src/bnb/game/modes/, src/bnb/contracts/config.py]
---

# Modes

A mode is the game's rules on top of the physics: a clock, a win rule, an objective, where
ships spawn, which ships are protected, and the mode's part of the observation. Modes live
in `bnb.game.modes` and are pure like the rest of the game. The spec is
`docs/rewrite/sections/modes-spec.md`; Frontline's capture rules are checked against v1's
code by the fixtures in `tests/reference/rewards`.

The modes are `bnb.contracts.config.MODES`:

- **team deathmatch**: score by deaths;
- **Frontline**: capture zones around a ring and push the front;
- **sandbox**: the physics game alone, with no score and no win, only the clock. The race's
  reference tests run in it.

## Clocks and presets

A match ends when its clock runs out or when the win rule ends it. The clocks by mode
(`bnb.game.modes.rules.CLOCKS`), which a `GameConfig` or a pool may override:

<!-- generated: mode-clocks -->

The training presets (`bnb.game.modes.rules.PRESETS`):

<!-- generated: game-presets -->

## The mode phase

The mode phase (`bnb.game.modes.phase.mode_phase`) runs inside `resolve`, after damage,
death and shields and before the clocks. It updates the mode state, emits the two
objective events and decides whether the match is won.

- `done` is the clock or the win.
- `result` is for team 0: +1 a win, -1 a loss, 0 a draw. It is the win rule's when the win
  ended the match, otherwise the clock rule's (`bnb.game.modes.phase.clock_result`), and 0
  where the match goes on.
- `objective_progress` and `objective` are signed amounts per ship, each summing to zero
  over the environment. Team deathmatch and sandbox emit zero.

When a match ends, the environment gets a new map and a fresh mode state
(`bnb.game.modes.phase.new_match`), and every ship spawns by the mode's placement.

## Team deathmatch

- **Score:** every death scores one point for the other team, whatever its cause: an
  enemy's hit, friendly fire or the boundary.
- **Win:** the first team to `kills_per_ship * ships_per_team` points. Both reaching it on
  the same tick is a draw. At the clock the higher score wins, and equal scores draw.
- **Spawns, uniform:** each spawning ship draws candidate points uniformly over the arena
  and takes the one farthest from the nearest living enemy. At a reset, when every ship
  spawns at once, each ship's first candidate stands in for its position, and each ship
  takes the candidate farthest from the nearest enemy's. One pass serves both, and nothing
  in it depends on the team or sim index.
- **Spawns, grouped** (`spawn_groups`, for example `(2, 2, 1)`): at a reset the groups'
  centres are spread over the arena, and each team's ships are assigned to groups by
  formation rank. A respawn lands at a distance from an anchor, a random living ship of
  either team, choosing the candidate farthest from the nearest living enemy.

## Frontline

- **The ring:** `bnb.game.modes.rules.FRONTLINE_ZONES` zones on a circle around the
  centre, at a random rotation and a random handedness (clockwise or counter-clockwise). At front 0 their roles are, in ring order:
  neutral, team 0's spawn, team 0's defense, team 1's defense, team 1's spawn.
- **Meters:** only the two defense zones have meters, from 0 to 1. A meter moves only on a
  strict majority of the living ships inside its zone: it rises while the attackers
  outnumber the defenders there, and falls while the defenders outnumber the attackers. Its
  rate is the harmonic number of the lead (1, 1.5, 1.83, ... for a lead of 1, 2, 3, ...)
  over `capture_seconds`, so a lead of one fills an empty meter in `capture_seconds`.
- **Captures:** a full meter captures. The front moves one step in the capturing team's
  favour, every meter empties, and the roles rotate around the ring.
- **Win:** a front of `front_threshold` captures ahead. At the clock the sign of the front
  decides, and 0 draws.
- **Spawns:** a ship spawns at a random point near its team's current spawn zone, facing
  the zone its team attacks.
- **Spawn protection:** a hit on a ship inside its own team's spawn zone deals no damage.
  The bullet is still spent. Boundary damage is unaffected.
- **Objective events,** two amounts per ship, each summing to zero over the environment:
  - `objective_progress` follows the meters. When a meter moves, the side the move favours
    shares the size of the move among its ships inside the zone, and the other side loses
    the same amount, shared among its ships outside the zone (all its ships, if none is
    outside). The tick that completes a capture pays nothing here.
  - `objective` follows captures. A capture is worth +1 to the attacking side and -1 to the
    defending side. Within each side, half is shared among the ships in the zone on the
    completing tick, and half by an attempt ledger kept while the meter rose: attackers are
    credited for being present, defenders charged for being absent.
- **The map** is mirror-symmetric about the axis through the centre and the neutral zone.

## The mode's rules

`bnb.game.modes.rules.ModeRules` holds each mode's constants, as Python numbers that
compiled graphs specialize on:

<!-- generated: mode-rules -->

## The mode's observation

- **Zones:** every mode has the same zone slots (`bnb.game.modes.rules.ZONE_SLOTS`), so a
  model sees the same tokens in any batch. A mode without zones shows them absent. Each
  view lists the zones in its own ring order, with roles, meter direction and the
  captures still needed to win or lose all relative to the viewer, so the order reveals no
  seat (`bnb.game.modes.observation.zone_rows`).
- **Globals:** the mode one-hot, the scores by team and the winning score, the front for
  the viewer and the winning front (`bnb.game.modes.observation.mode_globals`).

The columns are listed in [Contracts](../contracts.md).

## How a mode runs in a batch

Each hook takes its mode's `ModeRules` and dispatches on the mode's name in Python, so a
compiled graph holds only the rules of the modes in its batch. In a batch of several games
each hook runs on its mode's slice only; see [Mixed batches](mixed-batches.md).
[Adding a mode](../../guides/adding-a-mode.md) lists every hook.
