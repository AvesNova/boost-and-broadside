"""The fixed interfaces between the lean packages (see lean/CONTRACT.md).

Sizes: E environments, N ship slots (both teams together), K bullet slots per ship.
Layout: batch first, features last. Units: metres, radians, seconds, ticks where stated.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

STICKS, POWERS, FIRES = 7, 3, 2
JOINT = STICKS * POWERS * FIRES
BULLET_SLOTS = 11
SHIP_TYPES = 3

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
class Constants:
    """Per-ship constants of one run, read by name: `constants["lift"]` or `constants.lift`.

    A constant equal for every ship of the run (every type in the lineup, or fixed by the
    rules) is a Python number in `uniform`, compiled into the graph. One that differs by ship
    type is a row of `table` (E, C, N) float32, gathered once per slot at reset; `columns`
    gives its first row and width (width 1 reads as (E, N), a wider one, an action-indexed
    table, as (E, width, N)). Action-indexed tables are always rows of `table`.
    """

    uniform: dict[str, float | int]
    columns: dict[str, tuple[int, int]]
    table: Tensor

    def __getitem__(self, name: str) -> Tensor | float | int:
        if name in self.uniform:
            return self.uniform[name]
        start, width = self.columns[name]
        return self.table[:, start] if width == 1 else self.table[:, start : start + width]

    def __getattr__(self, name: str) -> Tensor | float | int:
        if name.startswith("__") or name in ("uniform", "columns", "table"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self.uniform or name in self.columns


# The packed state's rows, by name. Each name reads as an (E, N) view (bullets (E, N, K)).
# Per-ship quantities that change in an episode, float32 (counts held as whole numbers):
SHIP_COLUMNS = (
    "x",  # metres, arena centre at 0
    "y",
    "heading_cos",  # nose direction
    "heading_sin",
    "speed",  # metres per second (proper speed)
    "angle",  # angle of attack, radians
    "angle_cos",
    "angle_sin",
    "nose_rate",  # radians per second
    "energy",  # health and fuel in one store
    "shield",
    "shield_wait",  # seconds until the shield may recharge
    "cooldown",  # whole ticks until the gun may fire
    # The medium (ln n, d/dx ln n, d/dy ln n) at the ship, cached from the end of the last tick
    # (fields are fixed within an episode), as v2 caches it.
    "log_index",
    "gradient_x",
    "gradient_y",
    # The velocity at proper speed along the flight path (the nose turned back by the angle of
    # attack), stored by the step once (the world velocity is this over the refractive index).
    "velocity_x",
    "velocity_y",
    "next_slot",  # the bullet slot the next shot is written to, a whole number
)
FLAG_COLUMNS = (
    "alive",
    "exists",  # fixed for an episode
    "fired",  # the ship fired a real bullet on the last tick (stored, not inferred)
)
IDENTITY_COLUMNS = ("team", "type")  # long, fixed for an episode: 0 or 1; index into the ship card
BULLET_COLUMNS = (
    "bullet_x",
    "bullet_y",
    "bullet_dir_x",  # unit direction of travel
    "bullet_dir_y",
    "bullet_speed",  # metres per second (proper speed)
    "bullet_log_index",  # the medium at the bullet, cached from the end of its last move
    "bullet_gradient_x",
    "bullet_gradient_y",
)
BULLET_COUNT_COLUMNS = (
    "bullet_moves_left",  # 0 is inactive
    "bullet_fired_tick",  # the episode tick the bullet left the gun
)
BLOCKS = {
    "ships": SHIP_COLUMNS,
    "flags": FLAG_COLUMNS,
    "identity": IDENTITY_COLUMNS,
    "bullets": BULLET_COLUMNS,
    "bullet_counts": BULLET_COUNT_COLUMNS,
}
COLUMN = {name: (block, index) for block, names in BLOCKS.items() for index, name in enumerate(names)}


@dataclass
class State:
    """The whole game state, packed so that per-field work (a reset's select, holding a dead
    ship) is one operation per block. Every quantity in the *_COLUMNS tuples reads by name as a
    view: `state.x` is `state.ships[:, 0]`, (E, N); `state.bullet_x` is (E, N, K).

    Stick index: 0 straight, then (1, 2) fine, (3, 4) sustained, (5, 6) full, odd indices
    one direction and even the other. Power index: 0 coast, 1 boost, 2 reverse.
    """

    ships: Tensor  # (E, SHIP_COLUMNS, N) float32
    flags: Tensor  # (E, FLAG_COLUMNS, N) bool
    identity: Tensor  # (E, IDENTITY_COLUMNS, N) long
    constants: Constants  # per-ship constants: Python numbers and one table gathered at reset
    bullets: Tensor  # (E, BULLET_COLUMNS, N, K) float32
    bullet_counts: Tensor  # (E, BULLET_COUNT_COLUMNS, N, K) long
    bullet_real: Tensor  # (E, N, K) bool: a real bullet (False: a ghost, which does no damage)
    tick: Tensor  # (E,) long
    # The environment's refractive fields, (E, 5, F) float32 rows [x, y, radius, band width,
    # target ln n]; F may be 0 (no medium).
    fields: Tensor
    # Fog (None with fog off). Each ship as its enemy team last saw it: (E, SEEN_COLUMNS, N)
    # float32 raw values in the order of lean.game.core.SEEN_COLUMNS (the last two are "alive"
    # at that sighting, 1 until seen dead, and "known", 1 once ever seen); and the ticks since
    # the enemy team last saw it, (E, N) long, 0 while visible, capped (v2's ticks_since_seen).
    seen: Tensor | None = None
    ticks_since_seen: Tensor | None = None

    @classmethod
    def pack(cls, *, constants: Constants, tick: Tensor, fields: Tensor, bullet_real: Tensor,
             seen: Tensor | None = None, ticks_since_seen: Tensor | None = None, **named: Tensor) -> State:
        """A state from every named quantity of the *_COLUMNS tuples, each (E, N) or (E, N, K)."""
        missing = set(COLUMN) - set(named)
        if missing or len(named) != len(COLUMN):
            raise TypeError(f"State.pack needs every column once; missing {sorted(missing)}")
        dtypes = {"ships": torch.float32, "flags": torch.bool, "identity": torch.long,
                  "bullets": torch.float32, "bullet_counts": torch.long}
        blocks = {
            block: torch.stack([named[name].to(dtypes[block]) for name in names], 1)
            for block, names in BLOCKS.items()
        }
        return cls(**blocks, constants=constants, tick=tick, fields=fields, bullet_real=bullet_real,
                   seen=seen, ticks_since_seen=ticks_since_seen)

    def replace(self, **changes) -> State:
        """A copy with named quantities (and any dataclass fields) replaced; a changed block is
        copied, the others shared."""
        blocks: dict[str, Tensor] = {}
        fields = {}
        for name, value in changes.items():
            if name in COLUMN:
                block, index = COLUMN[name]
                if block not in blocks:
                    blocks[block] = getattr(self, block).clone()
                blocks[block][:, index] = value
            else:
                fields[name] = value
        return dataclasses.replace(self, **fields, **blocks)


def _column(block: str, index: int) -> property:
    return property(lambda state: getattr(state, block)[:, index])


for _name, (_block, _index) in COLUMN.items():
    setattr(State, _name, _column(_block, _index))


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
