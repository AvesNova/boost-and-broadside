"""The scripted agent's parameters (scripted spec A3): dimensionless numbers, each named for
the characteristic quantity it scales (`bnb.game.handling.Characteristics`).

A value is a float, or a tensor (R, 1) with one value per view (`unpack` of a parameter
table), so one compiled controller can play several parameter sets side by side (the
parameter search). The defaults are the search's choice; the section notes record the search.
"""

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class Ramp:
    """A linear ramp: `at_low` at or below `low`, `at_high` at or above `high`."""

    low: float
    high: float
    at_low: float
    at_high: float


def ramp(x: torch.Tensor, shape: Ramp) -> torch.Tensor:
    """The ramp at x, broadcasting its values against x."""
    fraction = ((x - shape.low) / (shape.high - shape.low)).clamp(0.0, 1.0)
    return shape.at_low + (shape.at_high - shape.at_low) * fraction


@dataclasses.dataclass(frozen=True)
class ScriptedParameters:
    """Ramps map a quantity (in the unit the name gives) to a probability or a weight.

    Combat (spec A4): turn_ramp_in_stall_angle, P(turn) by the angle error;
    level_two_ramp_in_stall_angle and level_three_ramp_in_stall_angle, P(level >= 2 | turn)
    and P(level 3 | level >= 2) by the same error; boost_slow_ramp_in_corner_speed,
    P(boost) by speed; boost_far_ramp_in_store_gun_ranges, P(boost) by the store fraction
    times the target's distance in gun ranges; reverse_ramp_in_gun_range, P(reverse) by
    distance; aim_ramp_in_target_size, P(aligned) by the nose's angle to the aim point over
    the target's angular size; shoot_ramp_in_gun_range, P(in range) by distance;
    lead_cap_in_gun_range, the distance beyond which the lead stops growing;
    boundary_margin_in_turn_radius, the boundary margin in g-limited turn radii;
    boundary_steer_ramp_in_margin, the weight of the steer toward the centre by the depth
    into the margin; centre_hold_in_arena_radius, how near the centre a ship without a
    target holds its course.

    Frontline (spec A5): combat_radius_in_gun_range, the strength kernel's width and where
    the strategy takes over from combat; support_radius_in_zone_radius, the zone strength
    kernel's width; defensive_floor and defensive_margin, the strength the defense zone
    wants at least and beyond the enemy's near it (a ship at full shield is 1);
    aggression, the attackers attack while their strength times exp(aggression) exceeds
    the enemy's near the attack zone; defense_urgency, added to the defense zone's want
    while its meter moves against the viewer; separation_radius_in_gun_range and
    separation_weight; recovery_shield_fraction and recovery_threat, below and above which a
    ship falls back to its spawn.
    """

    turn_ramp_in_stall_angle: Ramp = Ramp(0.0185, 0.16, 0.0, 1.0)
    level_two_ramp_in_stall_angle: Ramp = Ramp(0.153, 0.4, 0.0, 1.0)
    level_three_ramp_in_stall_angle: Ramp = Ramp(1.04, 2.09, 0.0, 1.0)
    boost_slow_ramp_in_corner_speed: Ramp = Ramp(0.471, 0.532, 1.0, 0.0)
    boost_far_ramp_in_store_gun_ranges: Ramp = Ramp(2.91, 4.91, 0.0, 1.0)
    reverse_ramp_in_gun_range: Ramp = Ramp(0.368, 0.512, 1.0, 0.0)
    aim_ramp_in_target_size: Ramp = Ramp(0.335, 1.09, 1.0, 0.0)
    shoot_ramp_in_gun_range: Ramp = Ramp(0.773, 0.876, 1.0, 0.0)
    lead_cap_in_gun_range: float = 1.16
    boundary_margin_in_turn_radius: float = 2.3
    boundary_steer_ramp_in_margin: Ramp = Ramp(0.0133, 0.479, 0.0, 1.0)
    centre_hold_in_arena_radius: float = 0.48
    combat_radius_in_gun_range: float = 2.33
    support_radius_in_zone_radius: float = 1.0
    defensive_floor: float = 0.81
    defensive_margin: float = 0.496
    aggression: float = 0.628
    defense_urgency: float = 1.82
    separation_radius_in_gun_range: float = 0.853
    separation_weight: float = 0.444
    recovery_shield_fraction: float = 0.185
    recovery_threat: float = 0.945


NAMES = tuple(field.name for field in dataclasses.fields(ScriptedParameters))


def _width(value) -> int:
    return 4 if isinstance(value, Ramp) else 1


def pack(parameters: ScriptedParameters) -> list[float]:
    """The parameters as a flat list (a ramp is low, high, at_low, at_high)."""
    out: list[float] = []
    for name in NAMES:
        value = getattr(parameters, name)
        out.extend(dataclasses.astuple(value) if isinstance(value, Ramp) else (value,))
    return out


def unpack(table: torch.Tensor) -> ScriptedParameters:
    """Per-view parameters from a table (R, len(pack(...))): each value a view (R, 1) of
    its column, so an in-place change to the table changes them."""
    default, values, column = ScriptedParameters(), {}, 0
    for name in NAMES:
        width = _width(getattr(default, name))
        part = [table[:, column + i:column + i + 1] for i in range(width)]
        values[name] = Ramp(*part) if width == 4 else part[0]
        column += width
    return ScriptedParameters(**values)


def table(parameters: list[ScriptedParameters], device=None) -> torch.Tensor:
    """A parameter table (R, P) with one row per view."""
    return torch.tensor([pack(p) for p in parameters], dtype=torch.float32, device=device)
