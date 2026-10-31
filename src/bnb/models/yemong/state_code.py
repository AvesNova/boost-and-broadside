"""The ship-state code (spec 3): the belief's means and spreads as one vector of softmax
groups, and back.

The belief stores physical values (m, m/s, rad, s, J). Shield and energy are coded as
fractions of the ship type's maximum, read from the ship card the model was built with.
"""

import math

import torch
import torch.nn as nn

from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, per_ship

from . import codes

MEANS = ("x", "y", "velocity_x", "velocity_y", "heading", "nose_rate", "angle_of_attack",
         "shield_recharge_wait", "shield", "energy", "gun_cooldown", "log_refractive_index")
SPREADS = ("position", "velocity_xx", "velocity_xy", "velocity_yy", "heading", "nose_rate",
           "angle_of_attack", "shield_recharge_wait", "shield", "energy",
           "log_refractive_index", "cooldown_0", "cooldown_1", "cooldown_2", "cooldown_3")
MEAN = {name: i for i, name in enumerate(MEANS)}

POSITION_PERIOD = 4.0 * 3.0**8  # m per axis: 26 244 m, finest level 4 m
POSITION_LEVELS = 9
VELOCITY_KNEE = 50.0  # m/s
VELOCITY_BINS = 81
VELOCITY_SPACING = 2.5
HEADING_COLOURS = 4
HEADING_LEVELS = 4
NOSE_RATE_LIMIT = 8.0  # rad/s
NOSE_RATE_BINS = 41
SCALAR_BINS = 21
LOG_INDEX_SPAN = math.log(2.0)  # field levels span n from 1/2 to 2


def _centred(value: torch.Tensor, period: float) -> torch.Tensor:
    """value mod period, in [-period / 2, period / 2)."""
    return torch.remainder(value + 0.5 * period, period) - 0.5 * period


def velocity_reach() -> float:
    """The largest world speed the velocity code resolves (m/s): 50 (e^2 - 1)."""
    return VELOCITY_KNEE * math.expm1(0.5 * (VELOCITY_BINS - 1) * VELOCITY_SPACING
                                      / VELOCITY_KNEE)


class ShipStateCode(nn.Module):
    """Encodes (means (..., 12), spreads (..., 15), ship_type (...)) to (..., 490) and back.

    Runs in order: position 9x9, velocity 3x81, heading 4x4, nose rate 41, angle of attack,
    shield wait, shield, energy and log index 21 each, gun cooldown 4.
    """

    def __init__(self, rules: Rules, types: ShipTypes) -> None:
        super().__init__()
        self.angle_of_attack_limit = 2.0 * rules.flight.stall_angle
        self.shield_delay = rules.shield_recharge_delay
        self.tick = rules.tick_length
        self.cooldown_bins = 1 + int(types.ticks_between_shots.max())
        self.register_buffer("max_shield", types.max_shield.detach().clone().float())
        self.register_buffer("max_energy", types.max_energy.detach().clone().float())
        scalar = (1, SCALAR_BINS)
        self.runs: codes.Runs = ((POSITION_LEVELS, 9), (3, VELOCITY_BINS),
                                 (HEADING_LEVELS, HEADING_COLOURS), (1, NOSE_RATE_BINS),
                                 scalar, scalar, scalar, scalar, scalar, (1, self.cooldown_bins))

    @property
    def width(self) -> int:
        return sum(groups * size for groups, size in self.runs)

    @property
    def groups(self) -> int:
        return sum(groups for groups, _ in self.runs)

    def _scalar_ranges(self) -> tuple[tuple[float, float, int], ...]:
        """(low, high, bins) of the five 21-bin scalars, in code order."""
        return ((-self.angle_of_attack_limit, self.angle_of_attack_limit, SCALAR_BINS),
                (0.0, self.shield_delay, SCALAR_BINS), (0.0, 1.0, SCALAR_BINS),
                (0.0, 1.0, SCALAR_BINS), (-LOG_INDEX_SPAN, LOG_INDEX_SPAN, SCALAR_BINS))

    def _maxima(self, ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return per_ship(self.max_shield, ship_type), per_ship(self.max_energy, ship_type)

    def _cooldown_sharp(self, cooldown: torch.Tensor) -> torch.Tensor:
        return codes.scalar_sharp(cooldown, 0.0, (self.cooldown_bins - 1) * self.tick,
                                  self.cooldown_bins)

    def encode(self, means: torch.Tensor, spreads: torch.Tensor,
               ship_type: torch.Tensor) -> torch.Tensor:
        max_shield, max_energy = self._maxima(ship_type)
        m = means.unbind(-1)
        s = spreads.unbind(-1)
        position = codes.position_encode(means[..., 0:2], s[0], (POSITION_PERIOD,) * 2,
                                         POSITION_LEVELS)
        velocity = codes.velocity_encode(means[..., 2:4], spreads[..., 1:4], VELOCITY_BINS,
                                         VELOCITY_SPACING, VELOCITY_KNEE)
        heading = codes.circular_encode(m[4], s[4], 2.0 * math.pi, HEADING_COLOURS,
                                        HEADING_LEVELS)
        nose_rate = codes.scalar_encode(m[5], s[5], -NOSE_RATE_LIMIT, NOSE_RATE_LIMIT,
                                        NOSE_RATE_BINS)
        values = (m[6], m[7], m[8] / max_shield, m[9] / max_energy, m[11])
        sigmas = (s[6], s[7], s[8] / max_shield, s[9] / max_energy, s[10])
        scalars = [codes.scalar_encode(value, sigma, *limits)
                   for value, sigma, limits in zip(values, sigmas, self._scalar_ranges())]
        cooldown = torch.clamp_min(self._cooldown_sharp(m[10]) + spreads[..., 11:], 0.0)
        return torch.cat([position.flatten(-2), velocity.flatten(-2), heading.flatten(-2),
                          nose_rate, *scalars, cooldown], -1)

    def sharp(self, means: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
        """The code of a state known exactly."""
        return self.encode(means, torch.zeros(means.shape[:-1] + (len(SPREADS),),
                                              dtype=means.dtype, device=means.device), ship_type)

    def decode(self, probabilities: torch.Tensor,
               ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., 490) probabilities to (means (..., 12), spreads (..., 15))."""
        max_shield, max_energy = self._maxima(ship_type)
        parts = probabilities.split([groups * size for groups, size in self.runs], -1)
        position, position_spread = codes.position_decode(
            parts[0].unflatten(-1, (POSITION_LEVELS, 9)), (POSITION_PERIOD,) * 2, POSITION_LEVELS)
        # The arena is centred on the origin, so the period is centred too (a deviation:
        # the spec decodes to [0, P)).
        position = _centred(position, POSITION_PERIOD)
        velocity, covariance = codes.velocity_decode(parts[1].unflatten(-1, (3, VELOCITY_BINS)),
                                                     VELOCITY_BINS, VELOCITY_SPACING,
                                                     VELOCITY_KNEE)
        heading, heading_spread = codes.circular_decode(
            parts[2].unflatten(-1, (HEADING_LEVELS, HEADING_COLOURS)), 2.0 * math.pi,
            HEADING_COLOURS, HEADING_LEVELS)
        heading = _centred(heading, 2.0 * math.pi)
        nose_rate, nose_spread = codes.scalar_decode(parts[3], -NOSE_RATE_LIMIT, NOSE_RATE_LIMIT,
                                                     NOSE_RATE_BINS)
        scalars = [codes.scalar_decode(part, *limits)
                   for part, limits in zip(parts[4:9], self._scalar_ranges())]
        (attack, attack_spread), (wait, wait_spread), (shield, shield_spread), \
            (energy, energy_spread), (log_index, log_index_spread) = scalars
        cooldown_code = parts[9]
        centres = self.tick * torch.arange(self.cooldown_bins, dtype=probabilities.dtype,
                                           device=probabilities.device)
        cooldown = (cooldown_code * centres).sum(-1)
        residual = cooldown_code - self._cooldown_sharp(cooldown)
        means = torch.stack([position[..., 0], position[..., 1], velocity[..., 0],
                             velocity[..., 1], heading, nose_rate, attack, wait,
                             shield * max_shield, energy * max_energy, cooldown, log_index], -1)
        spreads = torch.cat([torch.stack([position_spread], -1), covariance,
                             torch.stack([heading_spread, nose_spread, attack_spread, wait_spread,
                                          shield_spread * max_shield, energy_spread * max_energy,
                                          log_index_spread], -1), residual], -1)
        return means, spreads

    def decode_logits(self, logits: torch.Tensor,
                      ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-state head logits to (means, spreads): floor removal per group, then decode."""
        return self.decode(codes.remove_baseline_floor(logits, self.runs), ship_type)
