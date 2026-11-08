"""What the model composes from one team view and what it read from its carry (spec 4).

Per view the tokens are the ship rows (own ships first), the global token, the fields and
the mode's zones (modes spec 7; none outside Frontline). A ship row's physical state is the
belief's, so it is truth where visible. Nothing here reads a hidden ship's hidden-zero
columns.

Every position the model reads (ships, fields, zones, the map centre) is shifted by the
view's random offset, drawn per match. The categorical position codes then cannot memorise
absolute map positions; rotary attention and local presence read only differences, where
the offset cancels.
"""

import dataclasses
import math

import torch

from bnb.contracts import Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import FIELD, GLOBAL, SHIP, ZONE
from bnb.contracts.tree import contract
from bnb.game.rules import TEAMS

from . import codes
from .delta import DeltaComponent
from .features import RESPAWN_HARMONICS, local_presence, respawn_features, symlog
from .state_code import POSITION_LEVELS, POSITION_PERIOD, ShipStateCode

NULL_ACTION = 0  # the joint index of coast, straight, hold fire
# The model's input scales: constants of the model (its VERSION), not the game's rules. They
# equal the game's defaults (`MAP_SCALE_RADIUS`, `Rules.field_band_width`) by choice, and do
# not follow them if those change.
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
    """Probabilities as int16 fixed point, round(p ACTION_SCALE): the replay's storage of the enemy-
    action forecasts."""
    return torch.round(probabilities * ACTION_SCALE).to(torch.int16)


# The acceleration inputs (`YemongConfig.acceleration_inputs`): per row the one-tick change
# of the velocity, along and across the velocity at t (left positive), and of the nose rate,
# per second, each a symlog HL-Gauss code (`codes.symlog_encode`), and a validity flag.
# Knees: the median |value| of the living truth's one-tick pairs on the recorded segments,
# respawns excluded, pooled at equal weight over both players and three presets (3.81 m/s^2,
# 14.1 m/s^2, 1.55 rad/s^2; bench/acceleration_inputs_measure.py). Bounds (the last real
# bin's centre): the hard one-tick bounds of the change under persistence (state-deltas
# 1.2: 258 m/s, 24.7 rad/s) over a tick of 1/30 s. Constants of the model, like the scales.
ACCELERATION_BINS = 31
ACCELERATIONS = (
    DeltaComponent("along", 3.8, 258.0 * 30.0, ACCELERATION_BINS),  # m/s^2
    DeltaComponent("across", 14.0, 258.0 * 30.0, ACCELERATION_BINS),  # m/s^2
    DeltaComponent("angular", 1.55, 24.7 * 30.0, ACCELERATION_BINS),  # rad/s^2
)
ACCELERATION_CODE = sum(component.width for component in ACCELERATIONS)
ACCELERATION_INPUTS = ACCELERATION_CODE + 1  # the codes, then the validity flag
SLOW_SPEED = 1.0  # m/s: below it the frame is the heading, not the velocity

SHIP_FLAGS = ("alive", "visible", "fired")
POSITION_CODE = 9 * POSITION_LEVELS  # the width of a position code
# The centre's code, then episode progress, arena radius, fog, the mode one-hot (2), both
# scores over the target, the target, the front and its threshold.
GLOBAL_INPUTS = POSITION_CODE + 10
FIELD_INPUTS = POSITION_CODE + 3
# The centre's code, radius, the role one-hot, progress, direction and the two distances.
ZONE_INPUTS = POSITION_CODE + 1 + ZONE_ROLES + 4


def ship_inputs(
    code_width: int, ship_types: int, accelerations: bool = False, ellipse_width: int = 0
) -> int:
    """The ship token's width: the code, team and type one-hots, the flags, time since seen,
    respawn features (1 + 2 RESPAWN_HARMONICS), the pending action, the gradient and presence
    (2 each), then with accelerations the acceleration inputs (ACCELERATION_INPUTS), then
    the ellipse inputs (`ShipStateCode.ellipse_code`, ellipse_width)."""
    respawn = 1 + 2 * RESPAWN_HARMONICS
    width = code_width + TEAMS + ship_types + len(SHIP_FLAGS) + 1 + respawn + NUM_JOINT + 2 + 2
    return width + (ACCELERATION_INPUTS if accelerations else 0) + ellipse_width


@contract
class Replay:
    """What one decision read from the carry, after assimilating sight, for the enemy rows
    (the only rows a belief can hide; an own ship is always visible, so its row is the
    observation's truth with no spread, and `compose` rebuilds it): valid (R, N - S);
    age (R, N - S) int32 ticks since seen; means (R, N - S, 12) in the shifted frame;
    spreads (R, N - S, U); enemy_action (R, N - S, 42) int16 (store_action), the action
    belief; previous_action (R, S); offset (R, 2) m, the view's position offset."""

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
    return torch.stack(
        [
            column["x"] + offset[..., 0:1],
            column["y"] + offset[..., 1:2],
            column["velocity_x"],
            column["velocity_y"],
            heading,
            column["nose_rate"],
            column["angle_of_attack"],
            column["shield_recharge_wait"],
            column["shield"],
            column["energy"],
            column["gun_cooldown"],
            column["log_refractive_index"],
        ],
        -1,
    )


@contract
class Motion:
    """What a tick's accelerations of hidden rows difference against (the acceleration
    inputs, `YemongConfig.acceleration_inputs`, or the extrapolation baseline's rates), per
    enemy row (..., N - S), the only rows a belief can hide: velocity (..., N - S, 2) m/s
    and nose_rate rad/s, the row as the model read it (`believed`: truth where visible, the
    belief's forecast where hidden); present, alive with a belief."""

    velocity: torch.Tensor
    nose_rate: torch.Tensor
    present: torch.Tensor


def motion(ships: torch.Tensor, valid: torch.Tensor, means: torch.Tensor, own: int) -> Motion:
    """The Motion of the enemy rows the model reads: ships (..., N, columns), the
    observation; valid (..., N) and means (..., N, 12) as `believed` gives them; own, S."""
    return Motion(
        velocity=means[..., own:, 2:4],
        nose_rate=means[..., own:, 5],
        present=(ships[..., own:, SHIP["alive"]] > 0.5) & valid[..., own:],
    )


def preceding(now: Motion, start: Motion, reset: torch.Tensor) -> Motion:
    """The Motion before each tick of a segment (L, ...): start (the carry's, (...)) before
    the first, the tick before after it; nothing is present before a reset (L, ...)."""
    shifted = {
        name: torch.cat([getattr(start, name)[None], getattr(now, name)[:-1]], 0)
        for name in ("velocity", "nose_rate", "present")
    }
    shape = reset.shape + (1,) * (shifted["present"].dim() - reset.dim())
    shifted["present"] = shifted["present"] & ~reset.view(shape)
    return Motion(**shifted)


@dataclasses.dataclass(frozen=True)
class Accelerations:
    """The acceleration inputs of rows (..., N): values (..., N, 3), along and across the
    velocity at t (the heading below SLOW_SPEED; across to the left) in m/s^2 and the nose
    rate's in rad/s^2, zero where not valid; valid, a real tick of motion; slow, the frame
    was the heading; world (..., N, 3), the same change in the world frame (x, y, the nose
    rate's), zero where not valid: what an extrapolation baseline reads
    (`delta.Rates`)."""

    values: torch.Tensor
    valid: torch.Tensor
    slow: torch.Tensor
    world: torch.Tensor | None = None


def accelerations(
    ships: torch.Tensor,
    means: torch.Tensor,
    now: Motion,
    before: Motion,
    tick_length: float,
) -> Accelerations:
    """The change over the last tick per second of every row, in the velocity frame at t:
    ships (..., N, columns) the observation, means (..., N, 12) the rows as the model reads
    them (`believed`), now and before the enemy rows' Motion at t and t - 1.

    A visible row (every own row) reads the observation's world-frame change and its flag
    (`acceleration_valid`: alive and seen at both ticks, not spawned, not a game's first
    tick; so just spotted is invalid). A hidden row differences the belief: valid when
    present at both ticks and not spawned at t; just hidden is the forecast minus the last
    truth."""
    own = ships.shape[-2] - now.present.shape[-1]
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    visible = column["visible"] > 0.5
    observed = torch.stack(
        [column["acceleration_x"], column["acceleration_y"], column["nose_acceleration"]], -1
    )
    spawned = column["spawned"][..., own:] > 0.5
    believed_valid = now.present & before.present & ~spawned
    believed_change = torch.cat(
        [
            (now.velocity - before.velocity) / tick_length,
            ((now.nose_rate - before.nose_rate) / tick_length)[..., None],
        ],
        -1,
    )
    believed_change = torch.cat([torch.zeros_like(observed[..., :own, :]), believed_change], -2)
    believed_valid = torch.cat([torch.zeros_like(visible[..., :own]), believed_valid], -1)
    valid = torch.where(visible, column["acceleration_valid"] > 0.5, believed_valid)
    change = torch.where(visible[..., None], observed, believed_change)
    velocity, heading = means[..., 2:4], means[..., 4]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    slow = speed < SLOW_SPEED
    safe = torch.where(slow, 1.0, speed)
    unit_x = torch.where(slow, torch.cos(heading), velocity[..., 0] / safe)
    unit_y = torch.where(slow, torch.sin(heading), velocity[..., 1] / safe)
    along = change[..., 0] * unit_x + change[..., 1] * unit_y
    across = change[..., 1] * unit_x - change[..., 0] * unit_y
    values = torch.where(valid[..., None], torch.stack([along, across, change[..., 2]], -1), 0.0)
    world = torch.where(valid[..., None], change, 0.0)
    return Accelerations(values=values, valid=valid, slow=slow & valid, world=world)


def acceleration_code(inputs: Accelerations) -> torch.Tensor:
    """(..., N, ACCELERATION_INPUTS): each component's symlog HL-Gauss code (an invalid row
    codes its zero), then the validity flag."""
    parts = [
        codes.symlog_encode(inputs.values[..., i], c.knee, c.bound, c.bins)[0]
        for i, c in enumerate(ACCELERATIONS)
    ]
    return torch.cat([*parts, inputs.valid.to(inputs.values.dtype)[..., None]], -1)


@dataclasses.dataclass(frozen=True)
class Tokens:
    """The encoder's inputs and the geometry for the rotary tables, (...) leading axes.

    ship (..., N, ship_inputs), global_token (..., 1, GLOBAL_INPUTS), field (..., M,
    FIELD_INPUTS), zone (..., Z, ZONE_INPUTS); zone_present (..., Z) bool, the zone is
    the mode's (an absent one becomes the learned absent token); code (..., N, state_code.width) the
    belief's state code (the next-state head's baseline); ship_type (..., N) int64;
    query_position (..., N + 1, 2) and query_heading (..., N + 1): the ships' belief and the
    map centre (no heading); memory_position (..., M + Z, 2), the fields' and zones' centres,
    an absent zone at the map centre (the global token's rotation). Positions are shifted.
    Given the Motion the tick before (the acceleration inputs or the extrapolation
    baseline): motion, the enemy rows' Motion (the next tick's previous), and accelerations,
    whether or not the ship token reads them (None without).
    """

    ship: torch.Tensor
    global_token: torch.Tensor
    field: torch.Tensor
    zone: torch.Tensor
    zone_present: torch.Tensor
    code: torch.Tensor
    ship_type: torch.Tensor
    query_position: torch.Tensor
    query_heading: torch.Tensor
    memory_position: torch.Tensor
    motion: Motion | None = None
    accelerations: Accelerations | None = None


def _position_code(position: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(position[..., 0])
    return codes.position_encode(position, zero, (POSITION_PERIOD,) * 2, POSITION_LEVELS).flatten(
        -2
    )


def believed(ships: torch.Tensor, replay: Replay) -> tuple[torch.Tensor, ...]:
    """Every row's (valid, age, means, spreads) from the replay's enemy rows and the own
    rows' observed truth (valid, seen now, no spread), exactly what sight gave the step."""
    own = replay.previous_action.shape[-1]
    own_means = truth(ships[..., :own, :], replay.offset)
    rows = own_means.shape[:-1]
    valid = torch.cat([replay.valid.new_ones(rows), replay.valid], -1)
    age = torch.cat([replay.age.new_zeros(rows), replay.age], -1)
    spreads = torch.cat(
        [replay.spreads.new_zeros(rows + replay.spreads.shape[-1:]), replay.spreads], -2
    )
    return valid, age, torch.cat([own_means, replay.means], -2), spreads


def compose(
    obs: Obs,
    replay: Replay,
    state_code: ShipStateCode,
    ship_types: int,
    presence_radius: float,
    tick_length: float,
    previous: Motion | None = None,
    read_accelerations: bool = True,
) -> Tokens:
    """The Tokens of one team's views (...) from the observation and the replay: the ships' belief
    codes and features, the global token, the fields and the zones, every position shifted by the
    replay's offset. presence_radius in m, tick_length in s. previous: the enemy rows' Motion
    at the tick before (`preceding` in a segment), given with the acceleration inputs or the
    extrapolation baseline (its rates, `Tokens.accelerations`); with read_accelerations the
    acceleration inputs end the ship token, before the ellipse inputs
    (`state_code.spread_inputs` "ellipse"), and without it the token does not carry them."""
    ships, dtype = obs.ships, obs.ships.dtype
    own = replay.previous_action.shape[-1]
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    visible = column["visible"] > 0.5
    ship_type = column["ship_type"].long()
    team = column["team_relative"].long()
    valid, age, believed_means, spreads = believed(ships, replay)
    centre = replay.offset  # the map centre, the origin, in the shifted frame
    # A ship with no belief sits at the map centre, not at the frame's origin (which would
    # reveal the offset to rotary attention).
    position = torch.where(valid[..., None], believed_means[..., 0:2], centre[..., None, :])
    means = torch.cat([position, believed_means[..., 2:]], -1)
    code = state_code.encode(means, spreads, ship_type)

    joint = torch.arange(NUM_JOINT, device=ships.device)
    own_action = (replay.previous_action[..., None] == joint).to(dtype)
    spawned = column["spawned"][..., own:, None] > 0.5
    null = (joint == NULL_ACTION).to(dtype)
    forecast = replay.enemy_action.to(dtype) / ACTION_SCALE
    pending = torch.cat([own_action, torch.where(spawned, null, forecast)], -2)

    gradient = torch.stack([column["log_index_gradient_x"], column["log_index_gradient_y"]], -1)
    gradient = torch.where(visible[..., None], gradient, 0.0) / GRADIENT_SCALE
    presence = local_presence(position, team, valid, presence_radius)
    since_seen = age.to(dtype) * tick_length
    now, change = None, None
    if previous is not None:
        now = motion(ships, valid, believed_means, own)
        change = accelerations(ships, believed_means, now, previous, tick_length)
    ship = torch.cat(
        [
            code,
            (team[..., None] == torch.arange(TEAMS, device=ships.device)).to(dtype),
            (ship_type[..., None] == torch.arange(ship_types, device=ships.device)).to(dtype),
            torch.stack([column[name] for name in SHIP_FLAGS], -1),
            symlog(since_seen)[..., None],
            respawn_features(column["respawn_wait"]),
            pending,
            gradient,
            presence,
            *((acceleration_code(change),) if change is not None and read_accelerations else ()),
            *((state_code.ellipse_code(spreads),) if state_code.ellipse_width else ()),
        ],
        -1,
    )

    globals_ = obs.globals
    target = globals_[..., GLOBAL["score_target"]]
    per_target = torch.where(target > 0.0, 1.0 / torch.where(target > 0.0, target, 1.0), 0.0)
    global_token = torch.cat(
        [
            _position_code(centre),
            torch.stack(
                [
                    globals_[..., GLOBAL["episode_progress"]],
                    globals_[..., GLOBAL["arena_radius"]] / ARENA_SCALE,
                    globals_[..., GLOBAL["fogged"]],
                    globals_[..., GLOBAL["mode_team_deathmatch"]],
                    globals_[..., GLOBAL["mode_frontline"]],
                    globals_[..., GLOBAL["score_own"]] * per_target,
                    globals_[..., GLOBAL["score_enemy"]] * per_target,
                    symlog(target),
                    globals_[..., GLOBAL["front"]],
                    globals_[..., GLOBAL["front_threshold"]],
                ],
                -1,
            ),
        ],
        -1,
    )[..., None, :]

    fields = obs.fields
    field_position = (
        torch.stack([fields[..., FIELD["x"]], fields[..., FIELD["y"]]], -1) + centre[..., None, :]
    )
    field = torch.cat(
        [
            _position_code(field_position),
            torch.stack(
                [
                    fields[..., FIELD["radius"]] / FIELD_RADIUS_SCALE,
                    fields[..., FIELD["band_width"]] / FIELD_BAND_SCALE,
                    fields[..., FIELD["level"]] / FIELD_LEVEL_SCALE,
                ],
                -1,
            ),
        ],
        -1,
    )
    zones = obs.zones
    # An absent zone takes the global token's rotation (the map centre; heading 0, like
    # every memory token): at the frame's origin it would reveal the offset to attention.
    present = zones[..., ZONE["present"]] > 0.5
    zone_position = torch.where(
        present[..., None],
        torch.stack([zones[..., ZONE["x"]], zones[..., ZONE["y"]]], -1) + centre[..., None, :],
        centre[..., None, :],
    )
    role = zones[..., ZONE["role"]].long()
    zone = torch.cat(
        [
            _position_code(zone_position),
            torch.cat(
                [
                    (zones[..., ZONE["radius"]] / FIELD_RADIUS_SCALE)[..., None],
                    (role[..., None] == torch.arange(ZONE_ROLES, device=role.device)).to(dtype),
                    torch.stack(
                        [
                            zones[..., ZONE["progress"]],
                            zones[..., ZONE["direction"]],
                            symlog(zones[..., ZONE["offensive_distance"]]),
                            symlog(zones[..., ZONE["defensive_distance"]]),
                        ],
                        -1,
                    ),
                ],
                -1,
            ),
        ],
        -1,
    )
    query_position = torch.cat([position, centre[..., None, :]], -2)
    query_heading = torch.cat([believed_means[..., 4], torch.zeros_like(centre[..., :1])], -1)
    return Tokens(
        ship=ship,
        global_token=global_token,
        field=field,
        zone=zone,
        zone_present=present,
        code=code,
        ship_type=ship_type,
        query_position=query_position,
        query_heading=query_heading,
        memory_position=torch.cat([field_position, zone_position], -2),
        motion=now,
        accelerations=change,
    )
