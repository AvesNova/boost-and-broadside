"""The scripted agent's parameters (scripted spec A3): dimensionless numbers, each named for
the characteristic quantity it scales (`bnb.game.handling.characteristics.Characteristics`).

A value is a float, or a tensor (R, 1) with one value per view (`unpack` of a parameter
table), so one compiled controller can play several parameter sets side by side (the
parameter search). The defaults are the search's choice; the section notes record the search.

Every probability the agent reads from a quantity is a `Logistic` in the quantity's
logarithm, flattened by the agent's temperature; every binary decision is then floored at
`EPSILON` (`floored`), so no action of a living ship has probability exactly 0 or 1 at a
finite temperature. Weights that are not probabilities (the boundary's steer toward the
centre, the Frontline layer's hand-over to combat) are linear `Ramp`s.
"""

import dataclasses
import math

import torch

EPSILON = 0.04  # the floor of every binary decision (v1's): p in [EPSILON, 1 - EPSILON]
TINY_RATIO = 1e-6  # a logistic's input below this reads as this (its logarithm stays finite)


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
class Logistic:
    """sigma(+-(ln x - ln centre) / (scale T)) of a positive quantity x: 1/2 at `centre` (in
    x's unit), rising with x when `rising`, falling otherwise; `scale` in natural-log units
    (at T = 1 the 10-90% transition spans a factor exp(4.39 scale) in x). It is the Hill
    function x^k / (x^k + centre^k), k = 1 / (scale T). The direction is the field's and
    never searched (`pack` leaves it out)."""

    centre: float
    scale: float
    rising: bool


def logistic(x: torch.Tensor, shape: Logistic, temperature=1.0) -> torch.Tensor:
    """The logistic at x, without the floor, at a temperature (a float, or (R, 1) per
    view); at T = +inf it is 1/2 everywhere."""
    sign = 1.0 if shape.rising else -1.0
    centre = shape.centre
    log_centre = torch.log(centre) if isinstance(centre, torch.Tensor) else math.log(centre)
    argument = sign * (torch.log(x.clamp_min(TINY_RATIO)) - log_centre)
    return torch.sigmoid(argument / (shape.scale * temperature))


def floored(probability: torch.Tensor) -> torch.Tensor:
    """A binary decision's probability squeezed into [EPSILON, 1 - EPSILON]."""
    return EPSILON + (1.0 - 2.0 * EPSILON) * probability


@dataclasses.dataclass(frozen=True)
class ScriptedParameters:
    """Logistics map a quantity (in the unit the name gives) to a probability; ramps to a
    weight.

    Combat (spec A4): turn_in_stall_angle, P(turn) by the angle error;
    level_two_in_stall_angle and level_three_in_stall_angle, P(level >= 2 | turn) and
    P(level 3 | level >= 2) by the same error; boost_slow_in_corner_speed, P(boost) by
    speed (falling); boost_far_in_store_gun_ranges, P(boost) by the store fraction times
    the target's distance in gun ranges; reverse_in_gun_range, P(reverse) by distance
    (falling); aim_in_target_size, P(aligned) by the nose's angle to the aim point over the
    target's angular size (falling); shoot_in_gun_range, P(in range) by distance (falling);
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
    while its meter moves against the viewer, times 1 + the meter's progress (the enemy's
    share of the capture); separation_radius_in_gun_range and
    separation_weight; recovery_shield_fraction and recovery_threat, below and above which a
    ship falls back to its spawn.
    """

    turn_in_stall_angle: Logistic = Logistic(0.0544, 0.49, True)
    level_two_in_stall_angle: Logistic = Logistic(0.247, 0.22, True)
    level_three_in_stall_angle: Logistic = Logistic(1.47, 0.16, True)
    boost_slow_in_corner_speed: Logistic = Logistic(0.5, 0.031, False)
    boost_far_in_store_gun_ranges: Logistic = Logistic(3.78, 0.143, True)
    reverse_in_gun_range: Logistic = Logistic(0.434, 0.127, False)
    aim_in_target_size: Logistic = Logistic(0.604, 0.27, False)
    shoot_in_gun_range: Logistic = Logistic(0.823, 0.127, False)
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


# Each logistic's smallest scale: 0.1 x the standard deviation of the logarithm of its
# input over the earlier (linear-ramp) agent's own games, the smallest over the three
# training presets (the section notes give the measurement). Narrower, it would act as a
# step.
SCALE_FLOORS = {
    "turn_in_stall_angle": 0.158,
    "level_two_in_stall_angle": 0.158,
    "level_three_in_stall_angle": 0.158,
    "boost_slow_in_corner_speed": 0.031,
    "boost_far_in_store_gun_ranges": 0.143,
    "reverse_in_gun_range": 0.127,
    "aim_in_target_size": 0.205,
    "shoot_in_gun_range": 0.127,
}

NAMES = tuple(field.name for field in dataclasses.fields(ScriptedParameters))


def _columns(value) -> tuple:
    """A value's packed columns: a ramp's four, a logistic's centre and scale, or itself."""
    if isinstance(value, Ramp):
        return dataclasses.astuple(value)
    if isinstance(value, Logistic):
        return (value.centre, value.scale)
    return (value,)


def pack(parameters: ScriptedParameters) -> list[float]:
    """The parameters as a flat list (a ramp is low, high, at_low, at_high; a logistic is
    centre, scale)."""
    return [float(number) for name in NAMES for number in _columns(getattr(parameters, name))]


def unpack(table: torch.Tensor) -> ScriptedParameters:
    """Per-view parameters from a table (R, len(pack(...))): each value a view (R, 1) of
    its column, so an in-place change to the table changes them. A logistic's direction is
    the default's."""
    default, values, column = ScriptedParameters(), {}, 0
    for name in NAMES:
        kind = getattr(default, name)
        width = len(_columns(kind))
        part = [table[:, column + i : column + i + 1] for i in range(width)]
        if isinstance(kind, Ramp):
            values[name] = Ramp(*part)
        elif isinstance(kind, Logistic):
            values[name] = Logistic(*part, kind.rising)
        else:
            values[name] = part[0]
        column += width
    return ScriptedParameters(**values)


def from_row(row: list[float]) -> ScriptedParameters:
    """One parameter set of floats from a packed row: `pack`'s inverse."""
    values = unpack(torch.tensor([row], dtype=torch.float64))
    out = {}
    for name in NAMES:
        value = getattr(values, name)
        if isinstance(value, Ramp):
            out[name] = Ramp(*(float(end) for end in dataclasses.astuple(value)))
        elif isinstance(value, Logistic):
            out[name] = Logistic(float(value.centre), float(value.scale), value.rising)
        else:
            out[name] = float(value)
    return ScriptedParameters(**out)


def table(parameters: list[ScriptedParameters], device=None) -> torch.Tensor:
    """A parameter table (R, P) with one row per view."""
    return torch.tensor([pack(p) for p in parameters], dtype=torch.float32, device=device)
