"""The ship-state code (spec 3): the belief's means and spreads as one vector of softmax
groups, and back.

The belief stores physical values (m, m/s, rad, s, J). Shield and energy are coded as
fractions of the ship type's maximum, read from the ship card the model was built with.
The gun cooldown's bins (and so its residual spreads) are one more than the card's largest
ticks between shots.
"""

import math

import torch
import torch.nn as nn

from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, per_ship

from . import codes

MEANS = ("x", "y", "velocity_x", "velocity_y", "heading", "nose_rate", "angle_of_attack",
         "shield_recharge_wait", "shield", "energy", "gun_cooldown", "log_refractive_index")
# The spreads before the cooldown residuals (one per cooldown bin, which follow).
SPREADS = ("position", "velocity_xx", "velocity_xy", "velocity_yy", "heading", "nose_rate",
           "angle_of_attack", "shield_recharge_wait", "shield", "energy",
           "log_refractive_index")
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
# The five 21-bin scalars in code order; shield and energy are coded as fractions.
SCALARS = ("angle_of_attack", "shield_recharge_wait", "shield", "energy",
           "log_refractive_index")


def velocity_reach() -> float:
    """The largest world speed the velocity code resolves (m/s): 50 (e^2 - 1)."""
    return VELOCITY_KNEE * math.expm1(0.5 * (VELOCITY_BINS - 1) * VELOCITY_SPACING
                                      / VELOCITY_KNEE)


class ShipStateCode(nn.Module):
    """Encodes (means (..., 12), spreads (..., spread_count), ship_type (...)) to
    (..., width) and back. With the default card the spreads are 15 wide and the code 490.

    Runs in order: position 9x9, velocity 3x81, heading 4x4, nose rate 41, the five SCALARS
    of 21 bins each, gun cooldown.
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

    @property
    def spread_count(self) -> int:
        """SPREADS, then one cooldown residual per cooldown bin."""
        return len(SPREADS) + self.cooldown_bins

    def _scalar_ranges(self) -> dict[str, tuple[float, float, int]]:
        """(low, high, bins) of each of SCALARS."""
        unit = (0.0, 1.0, SCALAR_BINS)
        return dict(angle_of_attack=(-self.angle_of_attack_limit, self.angle_of_attack_limit,
                                     SCALAR_BINS),
                    shield_recharge_wait=(0.0, self.shield_delay, SCALAR_BINS), shield=unit,
                    energy=unit, log_refractive_index=(-LOG_INDEX_SPAN, LOG_INDEX_SPAN,
                                                       SCALAR_BINS))

    def _maxima(self, ship_type: torch.Tensor) -> dict[str, torch.Tensor]:
        """The divisor that makes each scalar a fraction (1 for the others)."""
        return dict(shield=per_ship(self.max_shield, ship_type),
                    energy=per_ship(self.max_energy, ship_type))

    def _cooldown_sharp(self, cooldown: torch.Tensor) -> torch.Tensor:
        return codes.scalar_sharp(cooldown, 0.0, (self.cooldown_bins - 1) * self.tick,
                                  self.cooldown_bins)

    def encode(self, means: torch.Tensor, spreads: torch.Tensor,
               ship_type: torch.Tensor) -> torch.Tensor:
        maxima = self._maxima(ship_type)
        mean = dict(zip(MEANS, means.unbind(-1), strict=True))
        spread = dict(zip(SPREADS, spreads[..., :len(SPREADS)].unbind(-1), strict=True))
        residual = spreads[..., len(SPREADS):]
        position = codes.position_encode(means[..., 0:2], spread["position"],
                                         (POSITION_PERIOD,) * 2, POSITION_LEVELS)
        velocity = codes.velocity_encode(means[..., 2:4], spreads[..., 1:4], VELOCITY_BINS,
                                         VELOCITY_SPACING, VELOCITY_KNEE)
        heading = codes.circular_encode(mean["heading"], spread["heading"], 2.0 * math.pi,
                                        HEADING_COLOURS, HEADING_LEVELS)
        nose_rate = codes.scalar_encode(mean["nose_rate"], spread["nose_rate"],
                                        -NOSE_RATE_LIMIT, NOSE_RATE_LIMIT, NOSE_RATE_BINS)
        ranges = self._scalar_ranges()
        scalars = [codes.scalar_encode(mean[name] / maxima.get(name, 1.0),
                                       spread[name] / maxima.get(name, 1.0), *ranges[name])
                   for name in SCALARS]
        cooldown = torch.clamp_min(self._cooldown_sharp(mean["gun_cooldown"]) + residual, 0.0)
        return torch.cat([position.flatten(-2), velocity.flatten(-2), heading.flatten(-2),
                          nose_rate, *scalars, cooldown], -1)

    def sharp(self, means: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
        """The code of a state known exactly."""
        return self.encode(means, means.new_zeros(means.shape[:-1] + (self.spread_count,)),
                           ship_type)

    def decode(self, probabilities: torch.Tensor,
               ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., width) probabilities to (means (..., 12), spreads (..., spread_count)).
        Positions come back in [0, P) per axis (the code's period), the heading in
        [-pi, pi)."""
        maxima = self._maxima(ship_type)
        parts = probabilities.split([groups * size for groups, size in self.runs], -1)
        position, position_spread = codes.position_decode(
            parts[0].unflatten(-1, (POSITION_LEVELS, 9)), (POSITION_PERIOD,) * 2, POSITION_LEVELS)
        velocity, covariance = codes.velocity_decode(parts[1].unflatten(-1, (3, VELOCITY_BINS)),
                                                     VELOCITY_BINS, VELOCITY_SPACING,
                                                     VELOCITY_KNEE)
        heading, heading_spread = codes.circular_decode(
            parts[2].unflatten(-1, (HEADING_LEVELS, HEADING_COLOURS)), 2.0 * math.pi,
            HEADING_COLOURS, HEADING_LEVELS)
        heading = torch.remainder(heading + math.pi, 2.0 * math.pi) - math.pi
        nose_rate, nose_spread = codes.scalar_decode(parts[3], -NOSE_RATE_LIMIT, NOSE_RATE_LIMIT,
                                                     NOSE_RATE_BINS)
        ranges = self._scalar_ranges()
        scalar, scalar_spread = {}, {}
        for name, part in zip(SCALARS, parts[4:9], strict=True):
            value, sigma = codes.scalar_decode(part, *ranges[name])
            scale = maxima.get(name, 1.0)
            scalar[name], scalar_spread[name] = value * scale, sigma * scale
        cooldown_code = parts[9]
        centres = self.tick * torch.arange(self.cooldown_bins, dtype=probabilities.dtype,
                                           device=probabilities.device)
        cooldown = (cooldown_code * centres).sum(-1)
        residual = cooldown_code - self._cooldown_sharp(cooldown)
        means = torch.stack([position[..., 0], position[..., 1], velocity[..., 0],
                             velocity[..., 1], heading, nose_rate, scalar["angle_of_attack"],
                             scalar["shield_recharge_wait"], scalar["shield"], scalar["energy"],
                             cooldown, scalar["log_refractive_index"]], -1)
        spreads = torch.cat([position_spread[..., None], covariance,
                             torch.stack([heading_spread, nose_spread]
                                         + [scalar_spread[name] for name in SCALARS], -1),
                             residual], -1)
        return means, spreads

    def decode_logits(self, logits: torch.Tensor,
                      ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-state head logits to (means, spreads): floor removal per group, then decode."""
        return self.decode(codes.remove_baseline_floor(logits, self.runs), ship_type)
