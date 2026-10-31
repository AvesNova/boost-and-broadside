"""What the model composes from one team view and what it read from its carry (spec 4).

Per view the tokens are the ship rows (own ships first), the global token, the fields and
the mode's zones (modes spec 7; none outside Frontline). A ship row's physical state is the
belief's, so it is truth where visible. Nothing here reads a hidden ship's hidden-zero
columns.

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
from bnb.contracts.obs import FIELD, GLOBAL, SHIP, ZONE, ZONE_COLUMNS
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
ZONE_ROLES = 5  # own spawn, own defense, neutral, enemy defense, enemy spawn
# The enemy-action forecasts are stored as int16 fixed point (2 bytes, resolution 3e-5), and
# the step composes from the stored integers, so unroll reads exactly what the step read.
# (Not bf16: inside a compiled kernel inductor drops a fused fp32 -> bf16 -> fp32 round trip
# by default, so the step would read unrounded values while the replay held rounded ones.)
# The spreads stay float32: rounding them moved the state code by up to 5e-3.
ACTION_SCALE = 32767.0


def store_action(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.round(probabilities * ACTION_SCALE).to(torch.int16)

SHIP_FLAGS = ("alive", "visible", "fired")
POSITION_CODE = 9 * POSITION_LEVELS  # the width of a position code
# The centre's code, then episode progress, arena radius, fog, the mode one-hot (2), both
# scores over the target, the target, the front and its threshold.
GLOBAL_INPUTS = POSITION_CODE + 10
FIELD_INPUTS = POSITION_CODE + 3
# The centre's code, radius, the role one-hot, progress, direction and the two distances.
ZONE_INPUTS = POSITION_CODE + 1 + ZONE_ROLES + 4


def ship_inputs(code_width: int, ship_types: int) -> int:
    """The ship token's width: the code, team and type one-hots, the flags, time since seen,
    respawn features (9), the pending action, the gradient and presence (2 each)."""
    return code_width + TEAMS + ship_types + len(SHIP_FLAGS) + 1 + 9 + NUM_JOINT + 2 + 2


@contract
class Replay:
    """What one decision read from the carry, after assimilating sight: valid (R, N);
    age (R, N) int32 ticks since seen; means (R, N, 12) in the shifted frame; spreads
    (R, N, U); enemy_action (R, N - S, 42) int16 (store_action), the action belief of the
    enemy rows (the only rows it is read for); previous_action (R, S); offset (R, 2) m, the
    view's position offset."""

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

    ship (..., N, ship_inputs), global_token (..., 1, GLOBAL_INPUTS), field (..., M,
    FIELD_INPUTS), zone (..., Z, ZONE_INPUTS); code (..., N, 490) the belief's state code (the
    next-state head's baseline); ship_type (..., N) int64; query_position (..., N + 1, 2) and
    query_heading (..., N + 1): the ships' belief and the map centre (no heading);
    memory_position (..., M + Z, 2), the fields' and zones' centres. Positions are shifted.
    """

    ship: torch.Tensor
    global_token: torch.Tensor
    field: torch.Tensor
    zone: torch.Tensor
    code: torch.Tensor
    ship_type: torch.Tensor
    query_position: torch.Tensor
    query_heading: torch.Tensor
    memory_position: torch.Tensor


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
    spreads = replay.spreads
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
    forecast = replay.enemy_action.to(dtype) / ACTION_SCALE
    pending = torch.cat([own_action, torch.where(spawned, null, forecast)], -2)

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
    target = globals_[..., GLOBAL["score_target"]]
    per_target = torch.where(target > 0.0, 1.0 / torch.where(target > 0.0, target, 1.0), 0.0)
    global_token = torch.cat([_position_code(centre), torch.stack([
        globals_[..., GLOBAL["episode_progress"]],
        globals_[..., GLOBAL["arena_radius"]] / ARENA_SCALE, globals_[..., GLOBAL["fogged"]],
        globals_[..., GLOBAL["mode_team_deathmatch"]], globals_[..., GLOBAL["mode_frontline"]],
        globals_[..., GLOBAL["score_own"]] * per_target,
        globals_[..., GLOBAL["score_enemy"]] * per_target, symlog(target),
        globals_[..., GLOBAL["front"]], globals_[..., GLOBAL["front_threshold"]]], -1)],
        -1)[..., None, :]

    fields = obs.fields
    field_position = torch.stack([fields[..., FIELD["x"]], fields[..., FIELD["y"]]], -1) \
        + centre[..., None, :]
    field = torch.cat([_position_code(field_position),
                       torch.stack([fields[..., FIELD["radius"]] / FIELD_RADIUS_SCALE,
                                    fields[..., FIELD["band_width"]] / FIELD_BAND_SCALE,
                                    fields[..., FIELD["level"]] / FIELD_LEVEL_SCALE], -1)], -1)
    zones = obs.zones if obs.zones is not None \
        else fields.new_zeros(fields.shape[:-2] + (0, len(ZONE_COLUMNS)))
    zone_position = torch.stack([zones[..., ZONE["x"]], zones[..., ZONE["y"]]], -1) \
        + centre[..., None, :]
    role = zones[..., ZONE["role"]].long()
    zone = torch.cat([_position_code(zone_position), torch.cat([
        (zones[..., ZONE["radius"]] / FIELD_RADIUS_SCALE)[..., None],
        (role[..., None] == torch.arange(ZONE_ROLES, device=role.device)).to(dtype),
        torch.stack([zones[..., ZONE["progress"]], zones[..., ZONE["direction"]],
                     symlog(zones[..., ZONE["offensive_distance"]]),
                     symlog(zones[..., ZONE["defensive_distance"]])], -1)], -1)], -1)
    query_position = torch.cat([position, centre[..., None, :]], -2)
    query_heading = torch.cat([replay.means[..., 4], torch.zeros_like(centre[..., :1])], -1)
    return Tokens(ship=ship, global_token=global_token, field=field, zone=zone, code=code,
                  ship_type=ship_type, query_position=query_position, query_heading=query_heading,
                  memory_position=torch.cat([field_position, zone_position], -2))
