---
covers: [src/bnb/game/physics/, src/bnb/game/rules.py, src/bnb/game/ship_types.py, src/bnb/game/state.py, src/bnb/game/api.py]
reviewed: 2026-11-03 09:45
---

# Physics

The game is the package `bnb.game`: pure functions over a batch of environments. Its
public functions are in `bnb.game.api`. This page covers the physics: the tick, flight,
fields, bullets, combat, spawns and maps. The full mathematical statement is the game
spec, `docs/rewrite/sections/game-spec.md`; the code follows it and the reference tests
check it against a float64 oracle.

## The API

<!-- runnable -->
```python
import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api

rules = api.rules_of(GameConfig(ships_per_team=2, num_fields=4, arena_radius=1200.0))
types = api.ship_types()
batch = 3
state = api.reset(rules, types, api.uniforms(rules, batch))
idle = torch.zeros((batch, rules.num_ships), dtype=torch.int32)
action = Action(idle, idle, idle)
moved, sweep = api.move(rules, types, state, action, api.spread(rules, batch))
state, events = api.resolve(rules, types, moved, sweep, action, api.uniforms(rules, batch))
state, obs, privileged = api.perceive(rules, state)
print(obs.ships.shape, events.damage.shape)
```

- `rules_of` turns a `GameConfig` into `bnb.game.rules.Rules`: the sizes and every shared
  constant, as Python numbers.
- `ship_types` gives the ship card, `bnb.game.ship_types.ShipTypes`: the per-type
  parameters, as tensors. Every game function takes it.
- The state, `bnb.game.state.State`, is opaque outside `bnb.game`. The runtime copies it as
  a pytree and never reads a field.
- Randomness is an explicit input. `spread` draws the bullet spread (two standard normals
  per ship per tick) and `uniforms` the tick's uniforms (the map, the spawn formation and
  the mode's spawn draws). Tests pass their own draws in.
- `step` is `move` then `resolve`. The runtime calls the two separately, so it can cut a
  compiled stage between them; `bnb.game.state.Sweep` carries the tick's motions from one
  to the other.
- For display, with no rules of their own: `log_index` (the medium at points),
  `core_radius` (the cores the sight test uses), `zone_sides` (Frontline's zone owners and
  attackers) and `visible` (each team's observation in sim order). The
  [play shell](../in-progress/play-shell.md) draws the game through them.

## The tick

One tick is one decision step, `Rules.tick_length` seconds. The phases, numbered as in
`src/bnb/game/physics/step.py`:

1. **Fire**, from the tick-start state.
2. **Ships fly.** Only ships alive at tick start move.
3. **Bullets fly**, including any fired this tick.
4. **Hits**, between the ship and bullet motions of this tick.
5. **Damage, death and shields.**

   **The mode phase** (unnumbered) runs here: the mode's state, objective and win rule
   (see [Modes](modes.md)).
6. **Clocks:** bullet lifetimes and gun cooldowns.
7. **Death and respawn.**
8. **Episode end:** an environment whose clock ran out, or whose match was won, resets.

`move` is phases 1 to 3; `resolve` is phases 4 to 8 with the mode phase. Perception is
separate (`perceive`, see [Perception and fog](perception.md)).

## Ships and their card

Every ship has a type, 0 fighter or 1 heavy (`FIGHTER` and `HEAVY` in `bnb.game.ship_types`).
Within each team, slots alternate starting with a fighter. The per-type parameters are
the ship card, from `bnb.game.ship_types.DEFAULTS`:

<!-- generated: ship-card -->

A parameter is a tensor of shape `(types,)`, or `(B, types)` to give every environment its
own card. `bnb.game.ship_types.per_ship` reads a ship's value as a select over the types,
which fuses into any kernel and needs no host copy. Changing a card value recompiles
nothing. The fields ending in `_scale`{.plain} scale a shared value per type
(`bnb.game.ship_types.SCALED`); [The ship card](ship-card.md) explains them and how a card
is designed from handles.

## Flight

A ship's state is its position, its nose direction, its **proper speed**, its angle of
attack, its nose rotation rate and an **energy store**.

- **The store is altitude.** Boost (power index 1) trades store for speed, like a dive;
  reverse (index 2) trades speed for store, like a climb; coast exchanges slowly. The
  engine charges the store, with less power as the store fills past a knee.
- **Lift turns the ship.** The stick commands lateral lift. A lift budget, capped by the
  g limit, is shared between the turn and the lift that holds level flight against
  gravity, so a hard turn at low speed sinks.
- **The nose is an oscillator.** It follows the commanded angle of attack through a
  damped second-order system, solved exactly over each half tick.
- **Integration:** a kick, a drift along the exact arc of the nose rotation, and another
  kick, with the fields evaluated at the new position in between.

The stick levels and power modes by index (`bnb.game.rules.STICK_LEVELS`,
`bnb.game.rules.POWER_MODES`):

<!-- generated: stick-levels -->

<!-- generated: power-modes -->

The power mode is the sign of the energy exchange: negative (boost) trades store for speed,
positive (reverse) trades speed for store, and zero (coast) leaves the slow exchange that
fades as the store fills.

The shared flight constants (`bnb.game.rules.FlightConstants`). The induced-drag factor,
the stall angle and the three exchanges are scaled per type by the card:

<!-- generated: flight-constants -->

## Fields

A map has `M` circular **fields** that change the refractive index `n` of space. Each has a
centre, a radius, a band width and a level in `bnb.game.rules.FIELD_LEVELS`; a level
`l`{.plain} sets `ln n = l ln √2`, so `n` spans one half to two. The profile is a smooth
step across the band, and overlapping fields compose as a union.

Inside a field a ship or a bullet keeps its proper speed while its world speed is that
divided by `n`, and its path bends toward higher `n`. Each field also has an opaque core,
which blocks sight (see [Perception and fog](perception.md)).

## Bullets and combat

- **Firing.** A ship fires when it is alive, its gun is cooled down and its store holds the
  shot's energy cost. The bullet starts at the ship's position with the ship's velocity plus
  the muzzle speed along the nose, plus the shooter's random spread. It lives its shooter's
  type's bullet life (the card's `bullet_life`, a whole number of moves at the rules'
  tick; 0.4 s for the standard types, 0.6 s for the real-scale ones).
- **Drag.** Bullets slow quadratically on their proper speed, with the game's bullet drag
  times the shooter's drag scale, so a shot has a range. A type's gun range sets both its
  bullet life and its drag scale ([The ship card](ship-card.md)).
- **Hits.** A swept test between each bullet's and each ship's straight motion this tick
  finds contacts within the target's hit radius. A bullet hits only its first contact.
  **Friendly fire is on:** a bullet can hit any ship but its own shooter.
- **Damage.** A hit deals the shooter's damage times the aspect
  `1 - (1 - f_0) exp(-ln 2 (θ / θ_h)²)`, with `θ` the angle between the bullet's reversed
  direction and the target's nose: the target's head-on fraction `f_0` from dead ahead,
  half way to full damage at the target's head-on angle `θ_h` (42.27 degrees by default,
  the old fixed curve's `exp(-4θ²/π)`), and nearly all of it from the side or rear.
- **Shields.** Damage beyond the current shield breaks it, and the overkill is ignored.
  Any damage to a ship whose shield was already broken at tick start kills it. After a
  delay without damage, the shield recharges at a fixed rate; both are per type. The delay
  counts down and the shield recharges only on a **quiet** tick: the ship coasts, its stick
  straight or at a slight turn (the lightest level, either side), and holds its fire
  (`bnb.game.rules.QUIET_STICKS`, `bnb.game.physics.combat.quiet`). Any other input (boost,
  reverse, a normal or sharp turn, a shot) resets the delay, as a hit does, and the shield
  gains nothing that tick: the recharge needs the whole delay of unbroken quiet flight, a
  real disengagement (game version 5; in version 3 and 4 such an input only paused the
  delay, so quiet stretches between passes added up and the shield refilled between them).
- **The arena** is a disc with no wall. A ship outside it takes boundary damage that grows
  with its depth.
- **The killer** of a ship is the owner of its earliest hit this tick, ties to the lower
  bullet index.
- **Aspect statistics.** For the statistics only, each hit is classed by `θ`: a mutual
  head-on under 10 degrees (the shooter in front of the target's guns, both can fire), an
  offset head-on from 10 to 45 (only the shooter can), the flank to 135, the tail beyond
  (`bnb.contracts.ASPECTS`). The tick's
  events carry the enemy damage each ship took by class and the class of its killing hit;
  the training series (`combat/damage_share/{aspect}`, `combat/kill_share/{aspect}`, per
  pool too), the match statistics (`damage_{aspect}`, `kills_{aspect}`) and the
  gun-lethality bench report the shares.

The shared constants, from `bnb.game.rules.Rules`. The card scales the spread, the
bullet drag, the head-on fraction and angle, the shield recharge rate and delay, and the vision range
per type:

<!-- generated: game-constants -->

## Spawns and maps

- **Death and respawn.** A ship that dies stays where it is until its respawn wait ends;
  then it spawns. The mode decides where and facing which way (see [Modes](modes.md)).
  A spawning ship starts with its type's spawn energy and a full shield.
- **The formation.** Sandbox spawns each team in a disc around its spawn centre, at
  opposite ends of the spawn axis. Ships spawning together take evenly spread places,
  shuffled and rotated at random, so nothing about the formation depends on the team
  index.
- **Fair maps.** A reset draws a new map. Fields come in pairs, each the other rotated by
  180 degrees about the centre, with one at the centre when the count is odd; Frontline
  mirrors them about its neutral axis instead. Either way the map looks the same from
  either side.
- **Map scale.** The arena radius sets the map scale, `arena_radius / 2600`
  (`bnb.game.rules.MAP_SCALE_RADIUS`), which scales spawn discs, field radii and zone radii.
