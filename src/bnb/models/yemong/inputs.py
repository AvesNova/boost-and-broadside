"""What the model composes from one team view and its belief (spec 4).

Per view the tokens are the ship rows (own ships first), the global token and the fields.
A ship row's physical state is the belief's, so it is truth where visible. Nothing here
reads a hidden ship's hidden-zero columns.
"""

import dataclasses
import math

import torch

from bnb.contracts import Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import FIELD, GLOBAL, SHIP

from . import codes
from .features import local_presence, respawn_features, symlog
from .state_code import POSITION_LEVELS, POSITION_PERIOD, ShipStateCode

TEAMS = 2
NULL_ACTION = 0  # the joint index of coast, straight, hold fire
# The log-index gradient's scale: a full-level band edge (1.875 x 2 ln 2 over 40 m).
GRADIENT_SCALE = 1.875 * 2.0 * math.log(2.0) / 40.0
ARENA_SCALE = 2600.0  # m
FIELD_RADIUS_SCALE = 1000.0  # m
FIELD_BAND_SCALE = 40.0  # m
FIELD_LEVEL_SCALE = 2.0

SHIP_FLAGS = ("alive", "visible", "fired")
GLOBAL_INPUTS = 3
FIELD_INPUTS = 9 * POSITION_LEVELS + 3


def ship_inputs(code_width: int, ship_types: int) -> int:
    """The ship token's width: the code, team and type one-hots, the flags, time since seen,
    respawn features (9), the pending action, the gradient and presence (2 each)."""
    return code_width + TEAMS + ship_types + len(SHIP_FLAGS) + 1 + 9 + NUM_JOINT + 2 + 2


def truth(ships: torch.Tensor) -> torch.Tensor:
    """(..., N, 12) the observed ship state in state_code.MEANS order (zero when hidden)."""
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    heading = torch.atan2(column["heading_sin"], column["heading_cos"])
    return torch.stack([column["x"], column["y"], column["velocity_x"], column["velocity_y"],
                        heading, column["nose_rate"], column["angle_of_attack"],
                        column["shield_recharge_wait"], column["shield"], column["energy"],
                        column["gun_cooldown"], column["log_refractive_index"]], -1)


@dataclasses.dataclass(frozen=True)
class Tokens:
    """The encoder's inputs and the geometry for the rotary tables, (...) leading axes.

    ship (..., N, ship_inputs), global_token (..., 1, 3), field (..., M, 84); code (..., N,
    490) the belief's state code (the next-state head's baseline); ship_type (..., N) int64;
    ship_position (..., N, 2) and ship_heading (..., N) the belief's; field_position (..., M, 2).
    """

    ship: torch.Tensor
    global_token: torch.Tensor
    field: torch.Tensor
    code: torch.Tensor
    ship_type: torch.Tensor
    ship_position: torch.Tensor
    ship_heading: torch.Tensor
    field_position: torch.Tensor


def compose(obs: Obs, valid: torch.Tensor, means: torch.Tensor, spreads: torch.Tensor,
            enemy_action: torch.Tensor, previous_action: torch.Tensor, state_code: ShipStateCode,
            ship_types: int, presence_radius: float) -> Tokens:
    """valid (..., N), means (..., N, 12), spreads (..., N, 15): the belief after this
    observation; enemy_action (..., N - S, 42): its action forecasts for the enemy rows (the
    own rows' pending actions are previous_action (..., S), their last joint actions)."""
    ships = obs.ships
    own = previous_action.shape[-1]
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    visible = column["visible"] > 0.5
    ship_type = column["ship_type"].long()
    team = column["team_relative"].long()
    code = state_code.encode(means, spreads, ship_type)

    joint = torch.arange(NUM_JOINT, device=ships.device)
    own_action = (previous_action[..., None] == joint).to(ships.dtype)
    spawned = column["spawned"][..., own:, None] > 0.5
    null = (joint == NULL_ACTION).to(ships.dtype)
    pending = torch.cat([own_action, torch.where(spawned, null, enemy_action)], -2)

    gradient = torch.stack([column["log_index_gradient_x"], column["log_index_gradient_y"]], -1)
    gradient = torch.where(visible[..., None], gradient, 0.0) / GRADIENT_SCALE
    position = means[..., 0:2]
    presence = local_presence(position, team, valid, presence_radius)
    dtype = ships.dtype
    ship = torch.cat([code, (team[..., None] == torch.arange(TEAMS, device=ships.device)).to(dtype),
                      (ship_type[..., None] == torch.arange(ship_types, device=ships.device))
                      .to(dtype),
                      torch.stack([column[name] for name in SHIP_FLAGS], -1),
                      symlog(column["time_since_seen"])[..., None],
                      respawn_features(column["respawn_wait"]), pending, gradient, presence], -1)

    globals_ = obs.globals
    global_token = torch.stack([globals_[..., GLOBAL["episode_progress"]],
                                globals_[..., GLOBAL["arena_radius"]] / ARENA_SCALE,
                                globals_[..., GLOBAL["fogged"]]], -1)[..., None, :]

    fields = obs.fields
    field_position = torch.stack([fields[..., FIELD["x"]], fields[..., FIELD["y"]]], -1)
    field_code = codes.position_encode(field_position, torch.zeros_like(fields[..., 0]),
                                       (POSITION_PERIOD,) * 2, POSITION_LEVELS)
    field = torch.cat([field_code.flatten(-2),
                       torch.stack([fields[..., FIELD["radius"]] / FIELD_RADIUS_SCALE,
                                    fields[..., FIELD["band_width"]] / FIELD_BAND_SCALE,
                                    fields[..., FIELD["level"]] / FIELD_LEVEL_SCALE], -1)], -1)
    return Tokens(ship=ship, global_token=global_token, field=field, code=code,
                  ship_type=ship_type, ship_position=position, ship_heading=means[..., 4],
                  field_position=field_position)
