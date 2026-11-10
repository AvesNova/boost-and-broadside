"""The scripted agent's parameters (scripted spec A3): dimensionless numbers, each named for
the characteristic quantity it scales (`bnb.game.handling.characteristics.Characteristics`).

A value is a float, or a tensor (R, 1) with one value per view (`unpack` of a parameter
table), so one compiled controller can play several parameter sets side by side (the
parameter search). The defaults are the search's choice; the section notes record the search.

Every probability the agent reads from a quantity is a `Logistic` in the quantity's
logarithm, flattened by the agent's temperature and held off exactly 0 and 1
(`LOGIT_LIMIT`); every binary decision but the shot is then floored at its own `EPSILONS`
(`floored`). The shot's only exact values are its gates: no target, or a gun that cannot
fire, and P(shoot) is 0. Weights that are not probabilities (the boundary's steer toward the
centre, the Frontline layer's hand-over to combat) are linear `Ramp`s.

The plan choices the agent remembers (its target, its turn side, its Frontline goal and its
recovery) change at a `Switch`'s rate: with probability 1 - exp(-rate dt) per decision, the
rate rising smoothly with how much better the best alternative is than the committed option.
"""

import dataclasses
import math
from typing import NamedTuple

import torch

# Each binary decision's floor: p in [epsilon, 1 - epsilon]. Not searched; each is the
# largest that costs little strength against the earlier agent (the section notes give the
# ablation). The shot has none: a per-tick floor fires energy-costly shots at 30 decisions
# a second, and even 0.001 cost about 0.05 of the score; its softness is its logistics'.
EPSILONS = {
    "turn": 0.02,
    "level_two": 0.02,
    "level_three": 0.02,
    "side": 0.02,
    "reverse": 0.04,
    "boost": 0.04,
    "shoot": 0.0,
}
# A logistic's argument is held within +-LOGIT_LIMIT, so it never rounds to exactly 0 or 1
# (sigma(9.2) = 1 - 1.0e-4).
LOGIT_LIMIT = 9.2
TINY_RATIO = 1e-6  # a logistic's input below this reads as this (its logarithm stays finite)


class Ramp(NamedTuple):
    """A linear ramp: `at_low` at or below `low`, `at_high` at or above `high`."""

    low: float
    high: float
    at_low: float
    at_high: float


def ramp(x: torch.Tensor, shape: Ramp) -> torch.Tensor:
    """The ramp at x, broadcasting its values against x."""
    fraction = ((x - shape.low) / (shape.high - shape.low)).clamp(0.0, 1.0)
    return shape.at_low + (shape.at_high - shape.at_low) * fraction


class Logistic(NamedTuple):
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
    view), within sigma(+-LOGIT_LIMIT); at T = +inf it is 1/2 everywhere."""
    sign = 1.0 if shape.rising else -1.0
    centre = shape.centre
    log_centre = torch.log(centre) if isinstance(centre, torch.Tensor) else math.log(centre)
    argument = sign * (torch.log(x.clamp_min(TINY_RATIO)) - log_centre)
    argument = argument / (shape.scale * temperature)
    return torch.sigmoid(argument.clamp(-LOGIT_LIMIT, LOGIT_LIMIT))


class Switch(NamedTuple):
    """A plan choice's switching rate (per second): rate sigma((advantage - centre) / (scale
    T)), advantage how much better the best alternative is than the committed option, in the
    unit the field's name gives. Near-equal options switch at rate sigma(-centre / scale),
    a clearly better one at up to `rate`."""

    rate: float
    centre: float
    scale: float


def switch_probability(
    advantage: torch.Tensor, shape: Switch, temperature, seconds: float
) -> torch.Tensor:
    """The probability of switching within a decision of `seconds`: 1 - exp(-rate seconds)
    for the switch's rate at `advantage`, at a temperature (a float, or (R, 1) per view)."""
    argument = (advantage - shape.centre) / (shape.scale * temperature)
    rate = shape.rate * torch.sigmoid(argument.clamp(-LOGIT_LIMIT, LOGIT_LIMIT))
    return -torch.expm1(-rate * seconds)


def floored(probability: torch.Tensor, decision: str) -> torch.Tensor:
    """A binary decision's probability squeezed into [epsilon, 1 - epsilon], epsilon the
    decision's `EPSILONS`."""
    epsilon = EPSILONS[decision]
    return epsilon + (1.0 - 2.0 * epsilon) * probability


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

    Memory: memory_fade_in_seconds, a hidden enemy's weight as a target by the time since it
    was seen (falling; its dead-reckoned distance is divided by the weight);
    target_switch_in_log_distance, the target's switch by the log ratio of the committed
    target's (weighted) distance to the best alternative's; side_commit_in_stall_angle, how
    much the committed turn side outweighs the error's sign, by the error;
    side_switch_in_radians, the side's switch by how much shorter the other way round is;
    goal_switch_in_strength, the Frontline goal's (defend or attack) by the margin of the
    other goal's rule, in ship strengths; recovery_switch_in_log_ratio, recovery's by the
    log margin of the other choice's rule.
    """

    turn_in_stall_angle: Logistic = Logistic(0.0398, 0.306, True)
    level_two_in_stall_angle: Logistic = Logistic(0.171, 0.17, True)
    level_three_in_stall_angle: Logistic = Logistic(0.734, 0.171, True)
    boost_slow_in_corner_speed: Logistic = Logistic(0.536, 0.0345, False)
    boost_far_in_store_gun_ranges: Logistic = Logistic(10.1, 0.161, True)
    reverse_in_gun_range: Logistic = Logistic(0.362, 0.135, False)
    aim_in_target_size: Logistic = Logistic(0.41, 0.237, False)
    shoot_in_gun_range: Logistic = Logistic(0.513, 0.127, False)
    lead_cap_in_gun_range: float = 1.33
    boundary_margin_in_turn_radius: float = 3.41
    boundary_steer_ramp_in_margin: Ramp = Ramp(0.0169, 0.292, 0.0, 1.0)
    centre_hold_in_arena_radius: float = 0.469
    combat_radius_in_gun_range: float = 3.4
    support_radius_in_zone_radius: float = 1.29
    defensive_floor: float = 0.298
    defensive_margin: float = 0.378
    aggression: float = 1.81
    defense_urgency: float = 3.87
    separation_radius_in_gun_range: float = 0.625
    separation_weight: float = 0.553
    recovery_shield_fraction: float = 0.133
    recovery_threat: float = 1.56
    memory_fade_in_seconds: Logistic = Logistic(1.0, 0.25, False)
    target_switch_in_log_distance: Switch = Switch(8.0, 0.3, 0.06)
    side_commit_in_stall_angle: Logistic = Logistic(4.0, 0.2, True)
    side_switch_in_radians: Switch = Switch(8.0, 0.6, 0.12)
    goal_switch_in_strength: Switch = Switch(8.0, 0.3, 0.06)
    recovery_switch_in_log_ratio: Switch = Switch(8.0, 0.3, 0.06)


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
    # the memory's: a tenth of a factor e in time, the turn logistics' in stall angles
    "memory_fade_in_seconds": 0.1,
    "side_commit_in_stall_angle": 0.158,
}
# Each switch's smallest scale, in its advantage's unit: about a tenth of the range its
# advantage spans in play (a log distance ratio, radians of turn, ship strengths, a log
# margin).
SWITCH_SCALE_FLOORS = {
    "target_switch_in_log_distance": 0.05,
    "side_switch_in_radians": 0.1,
    "goal_switch_in_strength": 0.05,
    "recovery_switch_in_log_ratio": 0.05,
}

NAMES = tuple(field.name for field in dataclasses.fields(ScriptedParameters))
_DEFAULT = ScriptedParameters()  # the logistics' directions (`unpack`)


def _columns(value) -> tuple:
    """A value's packed columns: a ramp's four, a switch's rate, centre and scale, a
    logistic's centre and scale, or itself."""
    if isinstance(value, (Ramp, Switch)):
        return tuple(value)
    if isinstance(value, Logistic):
        return (value.centre, value.scale)
    return (value,)


def pack(parameters: ScriptedParameters) -> list[float]:
    """The parameters as a flat list (a ramp is low, high, at_low, at_high; a switch is rate,
    centre, scale; a logistic is centre, scale)."""
    return [float(number) for name in NAMES for number in _columns(getattr(parameters, name))]


def unpack(table: torch.Tensor) -> ScriptedParameters:
    """Per-view parameters from a table (R, len(pack(...))): each value a view (R, 1) of
    its column, so an in-place change to the table changes them. A logistic's direction is
    the default's."""
    default, values, column = _DEFAULT, {}, 0
    for name in NAMES:
        kind = getattr(default, name)
        width = len(_columns(kind))
        part = [table[:, column + i : column + i + 1] for i in range(width)]
        if isinstance(kind, Ramp):
            values[name] = Ramp(*part)
        elif isinstance(kind, Switch):
            values[name] = Switch(*part)
        elif isinstance(kind, Logistic):
            values[name] = Logistic(*part, kind.rising)
        else:
            values[name] = part[0]
        column += width
    return _assembled(values)


def _assembled(values: dict) -> ScriptedParameters:
    """ScriptedParameters from every field's value, without its generated __init__: a
    compiled graph cannot bind that signature's defaults (the logistic constants), so a
    per-view set can be made inside one."""
    parameters = object.__new__(ScriptedParameters)
    for name, value in values.items():
        object.__setattr__(parameters, name, value)
    return parameters


def from_row(row: list[float]) -> ScriptedParameters:
    """One parameter set of floats from a packed row: `pack`'s inverse."""
    values = unpack(torch.tensor([row], dtype=torch.float64))
    out = {}
    for name in NAMES:
        value = getattr(values, name)
        if isinstance(value, (Ramp, Switch)):
            out[name] = type(value)(*(float(end) for end in value))
        elif isinstance(value, Logistic):
            out[name] = Logistic(float(value.centre), float(value.scale), value.rising)
        else:
            out[name] = float(value)
    return ScriptedParameters(**out)


def table(parameters: list[ScriptedParameters], device=None) -> torch.Tensor:
    """A parameter table (R, P) with one row per view."""
    return torch.tensor([pack(p) for p in parameters], dtype=torch.float32, device=device)


# The game modes the agent is tuned for, one parameter set each, in `mode_index` order: the
# standard training presets and the real-scale ones (`bnb.game.modes.rules.PRESETS`), each
# a preset's name. `mode_index` tells them apart by the map in gun ranges, the mode's flag
# and the card's gravity.
STANDARD_MODES = ("deathmatch", "deathmatch_wide", "frontline")
REAL_SCALE_MODES = ("real_scale_deathmatch", "real_scale_frontline")
MODES = STANDARD_MODES + REAL_SCALE_MODES


class ModeParameters:
    """One parameter set per game mode (`MODES`): each view plays with its mode's set,
    chosen from its observation (`mode_index`), so one batch may mix the modes."""

    def __init__(self, sets: dict[str, ScriptedParameters]) -> None:
        if set(sets) != set(MODES):
            raise ValueError(f"need one parameter set per mode {MODES}, got {tuple(sets)}")
        self.sets = {mode: sets[mode] for mode in MODES}
        self._tables: dict[torch.device, torch.Tensor] = {}

    def __eq__(self, other) -> bool:
        return isinstance(other, ModeParameters) and self.sets == other.sets

    def __repr__(self) -> str:
        return f"ModeParameters({self.sets!r})"

    def table(self, device) -> torch.Tensor:
        """(len(MODES), P) the sets' packed rows on `device` (made once per device, so a
        graph capture after the first use reads the same tensor)."""
        device = torch.device(device)
        if device not in self._tables:
            self._tables[device] = table(list(self.sets.values()), device)
        return self._tables[device]

    def per_view(self, mode: torch.Tensor) -> ScriptedParameters:
        """Per-view parameters for views of modes `mode` (R,) int64 (`mode_index`)."""
        return unpack(self.table(mode.device)[mode])
