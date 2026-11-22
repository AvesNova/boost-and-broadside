"""The fixed interfaces between the lean packages (see lean/CONTRACT.md).

Sizes: E environments, N ship slots (both teams together), K bullet slots per ship.
Layout: batch first, features last. Units: metres, radians, seconds, ticks where stated.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

STICKS, POWERS, FIRES = 7, 3, 2
JOINT = STICKS * POWERS * FIRES
BULLET_SLOTS = 11
SHIP_TYPES = 5  # len(lean.game.card.TYPE_NAMES): v2's three, then p51d and vizia_plain

# Per-ship observation row: 20 scaled scalars, 4 flags, then the ship-type one-hot.
SHIP_SCALARS = 20
SHIP_FLAGS = 4
SHIP_FEATURES = SHIP_SCALARS + SHIP_FLAGS + SHIP_TYPES
GLOBAL_FEATURES = 3
# Per-field observation row (v2's field inputs, src/bnb/models/yemong/inputs.py): radius over
# 1000 m, band width over 40 m, level (half-octaves of the refractive index) over 2.
FIELD_FEATURES = 3


def split_joint(joint: Tensor) -> Tensor:
    """Split a joint action index (...,) long in [0, 42) into (..., 3) long [stick, power, fire]."""
    stick = torch.div(joint, POWERS * FIRES, rounding_mode="floor")
    power = torch.div(joint, FIRES, rounding_mode="floor") % POWERS
    fire = joint % FIRES
    return torch.stack([stick, power, fire], dim=-1)


def join_action(action: Tensor) -> Tensor:
    """Join (..., 3) long [stick, power, fire] into the joint index stick * 6 + power * 2 + fire."""
    return action[..., 0] * (POWERS * FIRES) + action[..., 1] * FIRES + action[..., 2]


@dataclass
class State:
    """The whole game state. Every field is a tensor of shape (E, N) unless noted.

    Stick index: 0 straight, then (1, 2) fine, (3, 4) sustained, (5, 6) full, odd indices
    one direction and even the other. Power index: 0 coast, 1 boost, 2 reverse.
    """

    x: Tensor  # float32 metres, arena centre at 0
    y: Tensor
    heading_cos: Tensor  # nose direction
    heading_sin: Tensor
    speed: Tensor  # float32 metres per second (proper speed)
    angle: Tensor  # angle of attack, radians
    angle_cos: Tensor
    angle_sin: Tensor
    nose_rate: Tensor  # radians per second
    energy: Tensor  # health and fuel in one store
    shield: Tensor
    shield_wait: Tensor  # seconds until the shield may recharge
    cooldown: Tensor  # long, ticks until the gun may fire
    next_slot: Tensor  # long, bullet slot the next shot is written to
    alive: Tensor  # bool
    exists: Tensor  # bool, fixed for an episode
    team: Tensor  # long, 0 or 1
    type: Tensor  # long, index into the ship card
    constants: dict[str, Tensor]  # per-ship constants gathered at reset
    bullet_x: Tensor  # (E, N, K) float32
    bullet_y: Tensor
    bullet_dir_x: Tensor  # unit direction of travel
    bullet_dir_y: Tensor
    bullet_speed: Tensor  # metres per second
    bullet_moves_left: Tensor  # (E, N, K) long; 0 is inactive
    tick: Tensor  # (E,) long
    fired: Tensor  # (E, N) bool: the ship fired a real bullet on the last tick (stored, not inferred)
    bullet_real: Tensor  # (E, N, K) bool: a real bullet (False: a ghost, which does no damage)
    bullet_fired_tick: Tensor  # (E, N, K) long: the episode tick the bullet left the gun
    # The medium (ln n, d/dx ln n, d/dy ln n) at each ship and bullet, cached from the end of the
    # last tick (fields are fixed within an episode), as v2 caches it.
    medium: Tensor  # (E, N, 3) float32
    bullet_medium: Tensor  # (E, N, K, 3) float32
    # Fog (None with fog off). Each ship as its enemy team last saw it: (E, SEEN_COLUMNS, N)
    # float32 raw values in the order of lean.game.core.SEEN_COLUMNS (the last two are "alive"
    # at that sighting, 1 until seen dead, and "known", 1 once ever seen); and the ticks since
    # the enemy team last saw it, (E, N) long, 0 while visible, capped (v2's ticks_since_seen).
    seen: Tensor | None = None
    ticks_since_seen: Tensor | None = None


@dataclass
class StepOut:
    """What happened in one tick."""

    damage_dealt: Tensor  # (E, N) float32, damage this ship's bullets did to enemies
    damage_taken: Tensor  # (E, N) float32, all energy and shield lost to bullets and the boundary
    friendly_damage: Tensor  # (E, N) float32, damage this ship's bullets did to team-mates
    died: Tensor  # (E, N) bool
    fired: Tensor  # (E, N) bool
    killer: Tensor  # (E, N) long, slot of the killer or -1
    done: Tensor  # (E,) bool
    result: Tensor  # (E,) long: 0 or 1 = winning team, 2 = draw, -1 = not done
    # Bullets whose flight ended this tick (real and ghost alike), per slot (E, N, K):
    # shot_outcome -1 none ended, 0 missed (ran out of moves), 1 hit an enemy, 2 hit a team-mate;
    # shot_tick the episode tick that bullet left the gun (the tick the hit head labels).
    shot_outcome: Tensor
    shot_tick: Tensor
    # Damage detail for the rewards (v2's ledger), each before the shield's clamp:
    hit_damage: Tensor  # (E, N, N) float32 [attacker, target]: real bullets' damage this tick
    boundary_damage: Tensor  # (E, N) float32: damage from the arena boundary this tick
    shield_restored: Tensor  # (E, N) float32: shield recharged this tick


@dataclass
class Obs:
    """One team's view of the game: full information with fog off; with fog on, enemy rows are
    their last-seen values and nothing depends on what the team cannot see."""

    ship: Tensor  # (E, N, SHIP_FEATURES) float32
    pos: Tensor  # (E, N, 2) float32 metres
    heading: Tensor  # (E, N) float32 radians
    key_mask: Tensor  # (E, N) bool: row may be attended (exists and alive)
    own: Tensor  # (E, N) bool: rows this view controls (own team, alive)
    globals: Tensor  # (E, GLOBAL_FEATURES) float32
    field: Tensor  # (E, F, FIELD_FEATURES) float32; F may be 0
    field_pos: Tensor  # (E, F, 2) float32 metres, the field's centre
    field_mask: Tensor  # (E, F) bool: the field is present
    visible: Tensor  # (E, N) bool: the row is seen this tick (own rows always; all with fog off)
    since_seen: Tensor  # (E, N) float32: symlog of the seconds since the row was seen (0 visible)
