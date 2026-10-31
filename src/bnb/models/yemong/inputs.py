"""What the model composes from one team view and what it read from its carry (spec 4).

Per view the tokens are the ship rows (own ships first), the global token and the fields.
A ship row's physical state is the belief's, so it is truth where visible. Nothing here
reads a hidden ship's hidden-zero columns.

Every position the model reads (ships, fields, the map centre) is shifted by the view's
random offset, drawn per match. The categorical position codes then cannot memorise
absolute map positions; rotary attention and local presence read only differences, where
the offset cancels.
"""

import dataclasses
import math

import torch

from bnb.contracts import Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import FIELD, GLOBAL, SHIP
from bnb.contracts.tree import contract
from bnb.game.rules import TEAMS

from . import codes
from .features import local_presence, respawn_features, symlog
from .state_code import POSITION_LEVELS, POSITION_PERIOD, ShipStateCode

NULL_ACTION = 0  # the joint index of coast, straight, hold fire
# The log-index gradient's scale: a full-level band edge (1.875 x 2 ln 2 over 40 m).
GRADIENT_SCALE = 1.875 * 2.0 * math.log(2.0) / 40.0
ARENA_SCALE = 2600.0  # m
FIELD_RADIUS_SCALE = 1000.0  # m
FIELD_BAND_SCALE = 40.0  # m
FIELD_LEVEL_SCALE = 2.0
# Stored rounded: the forecast probabilities and spreads (the step reads the same values).
STORED_DTYPE = torch.bfloat16

SHIP_FLAGS = ("alive", "visible", "fired")
POSITION_CODE = 9 * POSITION_LEVELS  # the width of a position code
GLOBAL_INPUTS = POSITION_CODE + 3
FIELD_INPUTS = POSITION_CODE + 3


def ship_inputs(code_width: int, ship_types: int) -> int:
    """The ship token's width: the code, team and type one-hots, the flags, time since seen,
    respawn features (9), the pending action, the gradient and presence (2 each)."""
    return code_width + TEAMS + ship_types + len(SHIP_FLAGS) + 1 + 9 + NUM_JOINT + 2 + 2


@contract
class Replay:
    """What one decision read from the carry, after assimilating sight: valid (R, N);
    age (R, N) int32 ticks since seen; means (R, N, 12) in the shifted frame; spreads
    (R, N, U) bf16; enemy_action (R, N - S, 42) bf16, the action belief of the enemy rows (the
    only rows it is read for); previous_action (R, S); offset (R, 2) m, the view's position
    offset."""

    valid: torch.Tensor
    age: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor
    enemy_action: torch.Tensor
    previous_action: torch.Tensor
    offset: torch.Tensor


def truth(ships: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """(..., N, 12) the observed ship state in state_code.MEANS order, positions shifted by
    offset (..., 2) (only read where visible)."""
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    heading = torch.atan2(column["heading_sin"], column["heading_cos"])
    return torch.stack([column["x"] + offset[..., 0:1], column["y"] + offset[..., 1:2],
                        column["velocity_x"], column["velocity_y"], heading, column["nose_rate"],
                        column["angle_of_attack"], column["shield_recharge_wait"],
                        column["shield"], column["energy"], column["gun_cooldown"],
                        column["log_refractive_index"]], -1)


@dataclasses.dataclass(frozen=True)
class Tokens:
    """The encoder's inputs and the geometry for the rotary tables, (...) leading axes.

    ship (..., N, ship_inputs), global_token (..., 1, 84), field (..., M, 84); code (..., N,
    490) the belief's state code (the next-state head's baseline); ship_type (..., N) int64;
    query_position (..., N + 1, 2) and query_heading (..., N + 1): the ships' belief and the
    map centre (no heading); field_position (..., M, 2). Positions are shifted.
    """

    ship: torch.Tensor
    global_token: torch.Tensor
    field: torch.Tensor
    code: torch.Tensor
    ship_type: torch.Tensor
    query_position: torch.Tensor
    query_heading: torch.Tensor
    field_position: torch.Tensor


def _position_code(position: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(position[..., 0])
    return codes.position_encode(position, zero, (POSITION_PERIOD,) * 2,
                                 POSITION_LEVELS).flatten(-2)


def compose(obs: Obs, replay: Replay, state_code: ShipStateCode, ship_types: int,
            presence_radius: float, tick: float) -> Tokens:
    ships, dtype = obs.ships, obs.ships.dtype
    own = replay.previous_action.shape[-1]
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    visible = column["visible"] > 0.5
    ship_type = column["ship_type"].long()
    team = column["team_relative"].long()
    spreads = replay.spreads.to(dtype)
    centre = replay.offset  # the map centre, the origin, in the shifted frame
    # A ship with no belief sits at the map centre, not at the frame's origin (which would
    # reveal the offset to rotary attention).
    position = torch.where(replay.valid[..., None], replay.means[..., 0:2], centre[..., None, :])
    means = torch.cat([position, replay.means[..., 2:]], -1)
    code = state_code.encode(means, spreads, ship_type)

    joint = torch.arange(NUM_JOINT, device=ships.device)
    own_action = (replay.previous_action[..., None] == joint).to(dtype)
    spawned = column["spawned"][..., own:, None] > 0.5
    null = (joint == NULL_ACTION).to(dtype)
    pending = torch.cat([own_action, torch.where(spawned, null, replay.enemy_action.to(dtype))],
                        -2)

    gradient = torch.stack([column["log_index_gradient_x"], column["log_index_gradient_y"]], -1)
    gradient = torch.where(visible[..., None], gradient, 0.0) / GRADIENT_SCALE
    presence = local_presence(position, team, replay.valid, presence_radius)
    since_seen = replay.age.to(dtype) * tick
    ship = torch.cat([code, (team[..., None] == torch.arange(TEAMS, device=ships.device)).to(dtype),
                      (ship_type[..., None] == torch.arange(ship_types, device=ships.device))
                      .to(dtype),
                      torch.stack([column[name] for name in SHIP_FLAGS], -1),
                      symlog(since_seen)[..., None], respawn_features(column["respawn_wait"]),
                      pending, gradient, presence], -1)

    globals_ = obs.globals
    global_token = torch.cat([_position_code(centre),
                              torch.stack([globals_[..., GLOBAL["episode_progress"]],
                                           globals_[..., GLOBAL["arena_radius"]] / ARENA_SCALE,
                                           globals_[..., GLOBAL["fogged"]]], -1)], -1)[..., None, :]

    fields = obs.fields
    field_position = torch.stack([fields[..., FIELD["x"]], fields[..., FIELD["y"]]], -1) \
        + centre[..., None, :]
    field = torch.cat([_position_code(field_position),
                       torch.stack([fields[..., FIELD["radius"]] / FIELD_RADIUS_SCALE,
                                    fields[..., FIELD["band_width"]] / FIELD_BAND_SCALE,
                                    fields[..., FIELD["level"]] / FIELD_LEVEL_SCALE], -1)], -1)
    query_position = torch.cat([position, centre[..., None, :]], -2)
    query_heading = torch.cat([replay.means[..., 4], torch.zeros_like(centre[..., :1])], -1)
    return Tokens(ship=ship, global_token=global_token, field=field, code=code,
                  ship_type=ship_type, query_position=query_position, query_heading=query_heading,
                  field_position=field_position)
