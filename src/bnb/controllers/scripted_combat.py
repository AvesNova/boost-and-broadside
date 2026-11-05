"""The scripted agent's combat layer (scripted spec A4) and the geometry both layers share.

Everything is per own ship, (R, S), from one team's observation: the own rows are the first
S, the enemy rows the rest. An enemy row is read only where it is visible and alive, so a
hidden row's zeroed columns never matter.
"""

import math
from typing import NamedTuple

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.handling.characteristics import Characteristics, time_of_flight
from bnb.game.ship_types import per_ship

from .scripted_parameters import ScriptedParameters, floored, logistic, ramp

TARGET_SIZE_RANGE = (math.radians(1.0), math.radians(45.0))  # the clamp on its angular size
COAST, BOOST, REVERSE = 0, 1, 2  # power indices (bnb.game.rules.POWER_MODES)
TINY = 1e-6  # m: keeps a direction's length away from zero


class Marginals(NamedTuple):
    """Per own ship: power (R, S, 3), stick (R, S, 7) and shoot (R, S, 2) probabilities."""

    power: torch.Tensor
    stick: torch.Tensor
    shoot: torch.Tensor


class Context(NamedTuple):
    """What both layers read, per own ship (R, S) unless shown; enemy rows (R, E)."""

    own: torch.Tensor  # (R, S, columns)
    enemy: torch.Tensor  # (R, E, columns)
    own_type: torch.Tensor
    enemy_type: torch.Tensor
    enemy_seen: torch.Tensor  # visible and alive
    x: torch.Tensor
    y: torch.Tensor
    velocity_x: torch.Tensor
    velocity_y: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    speed: torch.Tensor  # proper
    energy_fraction: torch.Tensor
    gun_range: torch.Tensor
    stall_angle: torch.Tensor
    corner_speed: torch.Tensor
    arena_radius: torch.Tensor  # (R, 1)
    distance_from_centre: torch.Tensor
    boundary_weight: torch.Tensor  # the steer toward the centre, in [0, 1]
    temperature: torch.Tensor | float  # the agent's: a float, or (R, 1) per view


def column(rows: torch.Tensor, name: str) -> torch.Tensor:
    """The named SHIP column of ship rows (..., columns), shaped (...)."""
    return rows[..., SHIP[name]]


def context(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    temperature: torch.Tensor | float = 1.0,
) -> Context:
    """The Context both layers read, from one team's views (R, ...), own rows first, at the
    agent's temperature (a float, or (R, 1) per view)."""
    ships = obs.ships
    S = ships.shape[-2] // 2  # two teams, own rows first
    own, enemy = ships[:, :S], ships[:, S:]
    own_type = column(own, "ship_type").to(torch.int32)
    enemy_type = column(enemy, "ship_type").to(torch.int32)
    enemy_seen = (column(enemy, "visible") > 0.5) & (column(enemy, "alive") > 0.5)
    x, y = column(own, "x"), column(own, "y")
    velocity_x, velocity_y = column(own, "velocity_x"), column(own, "velocity_y")
    speed = column(own, "proper_speed")
    arena_radius = obs.globals[:, None, GLOBAL["arena_radius"]]
    # The boundary: within a margin of turn radii of the edge and flying outward.
    distance_from_centre = torch.sqrt(x * x + y * y).clamp_min(TINY)
    turn_radius = speed * speed / per_ship(characteristics.max_acceleration, own_type)
    margin = (parameters.boundary_margin_in_turn_radius * turn_radius).clamp_min(1.0)
    depth = (distance_from_centre - (arena_radius - margin)) / margin
    outward = x * velocity_x + y * velocity_y > 0.0
    boundary_weight = torch.where(
        outward, ramp(depth, parameters.boundary_steer_ramp_in_margin), 0.0
    )
    return Context(
        own=own,
        enemy=enemy,
        own_type=own_type,
        enemy_type=enemy_type,
        enemy_seen=enemy_seen,
        x=x,
        y=y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        cos=column(own, "heading_cos"),
        sin=column(own, "heading_sin"),
        speed=speed,
        energy_fraction=column(own, "energy") / per_ship(characteristics.max_energy, own_type),
        gun_range=per_ship(characteristics.gun_range, own_type),
        stall_angle=per_ship(characteristics.stall_angle, own_type),
        corner_speed=per_ship(characteristics.corner_speed, own_type),
        arena_radius=arena_radius,
        distance_from_centre=distance_from_centre,
        boundary_weight=boundary_weight,
        temperature=temperature,
    )


def signed_angle(
    cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """The angle from the nose (cos, sin) to the direction (x, y), positive to the left."""
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


def steer(
    fleet: Context, direction_x: torch.Tensor, direction_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A wanted direction with the boundary's steer toward the centre blended in."""
    length = torch.sqrt(direction_x * direction_x + direction_y * direction_y).clamp_min(TINY)
    weight = fleet.boundary_weight
    centre_x = -fleet.x / fleet.distance_from_centre
    centre_y = -fleet.y / fleet.distance_from_centre
    return (
        (1.0 - weight) * direction_x / length + weight * centre_x,
        (1.0 - weight) * direction_y / length + weight * centre_y,
    )


def stick_marginal(
    fleet: Context, error: torch.Tensor, parameters: ScriptedParameters
) -> torch.Tensor:
    """(R, S, 7) the stick for an angle error (positive left): P(turn) and the level by
    two further logistics, all in stall angles, each floored; the side follows the error's
    sign, floored (an error of exactly 0 picks either side evenly). Indices: 0 straight,
    then by level, odd right and even left."""
    size = error.abs() / fleet.stall_angle
    temperature = fleet.temperature
    turn = floored(logistic(size, parameters.turn_in_stall_angle, temperature), "turn")
    above_one = floored(
        logistic(size, parameters.level_two_in_stall_angle, temperature), "level_two"
    )
    above_two = floored(
        logistic(size, parameters.level_three_in_stall_angle, temperature), "level_three"
    )
    levels = torch.stack(
        [
            turn * (1.0 - above_one),
            turn * above_one * (1.0 - above_two),
            turn * above_one * above_two,
        ],
        -1,
    )  # (R, S, 3)
    left = floored((error > 0.0).float(), "side")
    left = torch.where(error == 0.0, 0.5, left)[..., None]
    sides = torch.stack([(1.0 - left) * levels, left * levels], -1)
    return torch.cat([(1.0 - turn)[..., None], sides.flatten(-2)], -1)


def power_marginal(boost: torch.Tensor, reverse: torch.Tensor) -> torch.Tensor:
    """(R, S, 3) from P(reverse) and P(boost | not reverse), each floored, coast otherwise."""
    reverse = floored(reverse, "reverse")
    boost = (1.0 - reverse) * floored(boost, "boost")
    return torch.stack([1.0 - boost - reverse, boost, reverse], -1)


def slow(fleet: Context, parameters: ScriptedParameters) -> torch.Tensor:
    """P(boost) for speed, unfloored: high when slow, in corner speeds."""
    return logistic(
        fleet.speed / fleet.corner_speed, parameters.boost_slow_in_corner_speed, fleet.temperature
    )


def far(fleet: Context, distance: torch.Tensor, parameters: ScriptedParameters) -> torch.Tensor:
    """P(boost) for distance, unfloored: the store fraction times the distance in gun
    ranges."""
    return logistic(
        fleet.energy_fraction * distance / fleet.gun_range,
        parameters.boost_far_in_store_gun_ranges,
        fleet.temperature,
    )


def either(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """P(first or second) for independent events: 1 - (1 - first)(1 - second)."""
    return 1.0 - (1.0 - first) * (1.0 - second)


def lead(
    fleet: Context,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_velocity_x: torch.Tensor,
    target_velocity_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The aim point (first-order interception), relative to the shooter: the target's
    displacement when a bullet fired now arrives, `t = time_of_flight(distance)` with the
    distance capped at lead_cap_in_gun_range. The bullet inherits the shooter's velocity, so
    the relative velocity carries the target. No field bending."""
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    cap = parameters.lead_cap_in_gun_range * fleet.gun_range
    flight_time = time_of_flight(characteristics, fleet.own_type, torch.minimum(distance, cap))
    return (
        target_x + (target_velocity_x - fleet.velocity_x) * flight_time,
        target_y + (target_velocity_y - fleet.velocity_y) * flight_time,
    )


def combat(
    fleet: Context, characteristics: Characteristics, parameters: ScriptedParameters
) -> tuple[Marginals, torch.Tensor, torch.Tensor]:
    """The combat layer: (marginals, has_target (R, S), the target's distance (R, S), 0
    without one). The target is the nearest visible, living enemy (ties: lowest row)."""
    enemy = fleet.enemy
    offset_x = column(enemy, "x")[:, None, :] - fleet.x[..., None]  # (R, S, E)
    offset_y = column(enemy, "y")[:, None, :] - fleet.y[..., None]
    seen = fleet.enemy_seen[:, None, :]
    squared = torch.where(seen, offset_x * offset_x + offset_y * offset_y, math.inf)
    pick = torch.nn.functional.one_hot(squared.argmin(-1), squared.shape[-1]).bool() & seen
    has_target = seen.any(-1).expand_as(fleet.x)

    def target(values: torch.Tensor) -> torch.Tensor:
        """The target's value from per-pair (R, S, E) or per-enemy (R, E) values."""
        values = values if values.dim() == 3 else values[:, None, :]
        return torch.where(pick, values, 0.0).sum(-1)

    target_x, target_y = target(offset_x), target(offset_y)
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    safe_distance = distance.clamp_min(TINY)
    aim_x, aim_y = lead(
        fleet,
        characteristics,
        parameters,
        target_x,
        target_y,
        target(column(enemy, "velocity_x")),
        target(column(enemy, "velocity_y")),
    )
    aim_error = signed_angle(fleet.cos, fleet.sin, aim_x, aim_y)

    # Without a target: head for the centre unless already near it.
    toward_centre = fleet.distance_from_centre > (
        parameters.centre_hold_in_arena_radius * fleet.arena_radius
    )
    wanted_x, wanted_y = steer(
        fleet, torch.where(has_target, aim_x, -fleet.x), torch.where(has_target, aim_y, -fleet.y)
    )
    steered = stick_marginal(
        fleet, signed_angle(fleet.cos, fleet.sin, wanted_x, wanted_y), parameters
    )
    hold = (~has_target & ~toward_centre)[..., None]
    stick = torch.where(hold, stick_marginal(fleet, torch.zeros_like(fleet.x), parameters), steered)

    boost = either(
        slow(fleet, parameters), torch.where(has_target, far(fleet, distance, parameters), 0.0)
    )
    reverse = torch.where(
        has_target,
        logistic(distance / fleet.gun_range, parameters.reverse_in_gun_range, fleet.temperature),
        0.0,
    )
    power = power_marginal(boost, reverse)

    hit_radius = target(per_ship(characteristics.hit_radius, fleet.enemy_type))
    size = (2.0 * torch.atan(hit_radius / safe_distance)).clamp(*TARGET_SIZE_RANGE)
    aligned = logistic(aim_error.abs() / size, parameters.aim_in_target_size, fleet.temperature)
    in_range = logistic(
        distance / fleet.gun_range, parameters.shoot_in_gun_range, fleet.temperature
    )
    cost = per_ship(characteristics.bullet_energy_cost, fleet.own_type)
    ready = (column(fleet.own, "gun_cooldown") <= 0.0) & (column(fleet.own, "energy") >= cost)
    shoot = floored(torch.where(has_target & ready, aligned * in_range, 0.0), "shoot")
    shoot = torch.stack([1.0 - shoot, shoot], -1)
    return Marginals(power, stick, shoot), has_target, distance


def blend(first: torch.Tensor, second: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The log-probability blend first^w second^(1 - w), renormalized; where the two have
    no common support, their mixture. weight (R, S) in [0, 1]."""
    weight = weight[..., None]
    geometric = first.pow(weight) * second.pow(1.0 - weight)
    total = geometric.sum(-1, keepdim=True)
    return torch.where(
        total > 0.0, geometric / total.clamp_min(1e-30), weight * first + (1.0 - weight) * second
    )
