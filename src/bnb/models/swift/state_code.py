"""The ship-state code (spec 3) and the game constants the model is built with.

Belief channels (C = 12): x, y (m); velocity x, y (m/s); heading (rad, [-pi, pi)); nose rate
(rad/s); angle of attack (rad); shield wait (s); shield; energy (J); gun cooldown (s);
log refractive index.
Spreads (U = 15): position sigma; velocity covariance xx, xy, yy; sigmas of heading, nose
rate, angle of attack, shield wait, shield, energy and log index; 4 cooldown residuals.
"""

import dataclasses
import math

import torch

from . import codes

MEAN_CHANNELS = 12
SPREAD_CHANNELS = 15
PREDICTION_WIDTH = MEAN_CHANNELS + SPREAD_CHANNELS

POSITION_PERIOD = 26244.0  # m, 4 * 3^8: the finest of 9 levels is 4 m
POSITION_LEVELS = 9
VELOCITY_KNEE = 50.0  # m/s
VELOCITY_BINS = 81
VELOCITY_SPACING = 2.5
HEADING_COLOURS = 4
HEADING_LEVELS = 4
NOSE_RATE_LIMIT = 8.0  # rad/s
SHIELD_DELAY = 6.0  # s
LOG_INDEX_LIMIT = math.log(2.0)
SCALAR_BINS = 21
NOSE_RATE_BINS = 41


@dataclasses.dataclass(frozen=True)
class ShipCard:
    """Game constants the model reads. The defaults are stand-ins: the integration builds the
    card from the game's ship types. Nothing in the weights depends on them except the
    number of types (the ship-type one-hot)."""

    tick_seconds: float = 1.0 / 30.0
    shield_maximum: tuple[float, ...] = (100.0, 150.0)  # per type
    energy_maximum: tuple[float, ...] = (1000.0, 1500.0)  # J, per type
    lift_clip: float = 0.35  # rad, the largest alpha_s over types
    ticks_between_shots: int = 3  # the largest over types
    bullet_life_moves: int = 12
    muzzle_speed: float = 600.0  # m/s, the largest over types
    hit_radius: float = 9.0  # m, the smallest over types

    @property
    def ship_types(self) -> int:
        return len(self.shield_maximum)

    @property
    def cooldown_bins(self) -> int:
        return 1 + self.ticks_between_shots

    def shot_parameters(self) -> dict:
        """The shot codes' constants (spec 9)."""
        reach = VELOCITY_KNEE * (math.e**2 - 1)
        top = (self.muzzle_speed + reach) * 2
        return {
            "displacement_period": 2.5 * top * self.bullet_life_moves * self.tick_seconds,
            "displacement_levels": 8, "direction_levels": 5, "top_speed": top,
            "speed_bins": 41, "index_span": LOG_INDEX_LIMIT, "index_bins": 21,
            "collision_radius": self.hit_radius, "near_bins": 8, "near_span_radii": 6.0,
            "near_sigma_bins": 0.75,
        }


class ShipStateCodec:
    """encode / sharp / decode / decode_logits for the 490-wide, 23-group code."""

    def __init__(self, card: ShipCard):
        self.card = card
        self.angle_limit = 2 * card.lift_clip
        self.cooldown_high = card.ticks_between_shots * card.tick_seconds
        # (name, groups, size) in vector order
        self.layout = [
            ("position", POSITION_LEVELS, 9),
            ("velocity", 3, VELOCITY_BINS),
            ("heading", HEADING_LEVELS, HEADING_COLOURS),
            ("nose_rate", 1, NOSE_RATE_BINS),
            ("angle_of_attack", 1, SCALAR_BINS),
            ("shield_recharge_wait", 1, SCALAR_BINS),
            ("shield", 1, SCALAR_BINS),
            ("energy", 1, SCALAR_BINS),
            ("log_refractive_index", 1, SCALAR_BINS),
            ("gun_cooldown", 1, card.cooldown_bins),
        ]
        self.runs = [(groups, size) for _, groups, size in self.layout]
        self.width = sum(groups * size for groups, size in self.runs)
        self.groups = sum(groups for groups, _ in self.runs)
        self.scalars = {  # name: (low, high, bins, mean channel, spread channel)
            "nose_rate": (-NOSE_RATE_LIMIT, NOSE_RATE_LIMIT, NOSE_RATE_BINS, 5, 5),
            "angle_of_attack": (-self.angle_limit, self.angle_limit, SCALAR_BINS, 6, 6),
            "shield_recharge_wait": (0.0, SHIELD_DELAY, SCALAR_BINS, 7, 7),
            "shield": (0.0, 1.0, SCALAR_BINS, 8, 8),
            "energy": (0.0, 1.0, SCALAR_BINS, 9, 9),
            "log_refractive_index": (-LOG_INDEX_LIMIT, LOG_INDEX_LIMIT, SCALAR_BINS, 11, 10),
        }

    def maxima(self, ship_type, like):
        """(shield maximum, energy maximum) per ship, each shaped like ship_type."""
        shield = torch.tensor(self.card.shield_maximum, dtype=like.dtype, device=like.device)
        energy = torch.tensor(self.card.energy_maximum, dtype=like.dtype, device=like.device)
        return shield[ship_type], energy[ship_type]

    def cooldown_sharp(self, cooldown):
        return codes.scalar_sharp(0.0, self.cooldown_high, self.card.cooldown_bins, cooldown)

    def encode(self, means, spreads, ship_type):
        """means (..., 12), spreads (..., 15), ship_type (...) int -> code (..., 490)."""
        shield_maximum, energy_maximum = self.maxima(ship_type, means)
        scale = {"shield": shield_maximum, "energy": energy_maximum}
        pieces = [
            codes.position_encode((POSITION_PERIOD, POSITION_PERIOD), POSITION_LEVELS,
                                  means[..., 0:2], spreads[..., 0]).flatten(-2),
            codes.velocity_encode(VELOCITY_BINS, VELOCITY_SPACING, VELOCITY_KNEE,
                                  means[..., 2:4], spreads[..., 1:4]).flatten(-2),
            codes.circular_encode(2 * math.pi, HEADING_COLOURS, HEADING_LEVELS, means[..., 4],
                                  spreads[..., 4]).flatten(-2),
        ]
        for name, (low, high, bins, mean_channel, spread_channel) in self.scalars.items():
            mean, sigma = means[..., mean_channel], spreads[..., spread_channel]
            if name in scale:
                mean, sigma = mean / scale[name], sigma / scale[name]
            pieces.append(codes.scalar_encode(low, high, bins, mean, sigma))
        pieces.append(torch.clamp_min(self.cooldown_sharp(means[..., 10]) + spreads[..., 11:15],
                                      0))
        return torch.cat(pieces, -1)

    def sharp(self, means, ship_type):
        spreads = torch.zeros(*means.shape[:-1], SPREAD_CHANNELS, dtype=means.dtype,
                              device=means.device)
        return self.encode(means, spreads, ship_type)

    def split(self, code):
        out, start = {}, 0
        for name, groups, size in self.layout:
            out[name] = code[..., start:start + groups * size]
            start += groups * size
        return out

    def decode(self, probabilities, ship_type):
        """code (..., 490) probabilities -> (means (..., 12), spreads (..., 15))."""
        parts = self.split(probabilities)
        position, position_spread = codes.position_decode(
            (POSITION_PERIOD, POSITION_PERIOD), POSITION_LEVELS,
            parts["position"].unflatten(-1, (POSITION_LEVELS, 9)))
        position = torch.remainder(position + POSITION_PERIOD / 2, POSITION_PERIOD) \
            - POSITION_PERIOD / 2
        velocity, covariance = codes.velocity_decode(
            VELOCITY_BINS, VELOCITY_SPACING, VELOCITY_KNEE,
            parts["velocity"].unflatten(-1, (3, VELOCITY_BINS)))
        heading, heading_spread = codes.circular_decode(
            2 * math.pi, HEADING_COLOURS, HEADING_LEVELS,
            parts["heading"].unflatten(-1, (HEADING_LEVELS, HEADING_COLOURS)))
        heading = torch.remainder(heading + math.pi, 2 * math.pi) - math.pi
        shield_maximum, energy_maximum = self.maxima(ship_type, probabilities)
        scale = {"shield": shield_maximum, "energy": energy_maximum}
        means = [None] * MEAN_CHANNELS
        spreads = [None] * SPREAD_CHANNELS
        means[0], means[1] = position[..., 0], position[..., 1]
        means[2], means[3] = velocity[..., 0], velocity[..., 1]
        means[4] = heading
        spreads[0] = position_spread
        spreads[1], spreads[2], spreads[3] = covariance.unbind(-1)
        spreads[4] = heading_spread
        for name, (low, high, bins, mean_channel, spread_channel) in self.scalars.items():
            mean, spread = codes.scalar_decode(low, high, bins, parts[name])
            if name in scale:
                mean, spread = mean * scale[name], spread * scale[name]
            means[mean_channel], spreads[spread_channel] = mean, spread
        cooldown_code = parts["gun_cooldown"]
        centres = torch.arange(self.card.cooldown_bins, dtype=probabilities.dtype,
                               device=probabilities.device) * self.card.tick_seconds
        cooldown = (cooldown_code * centres).sum(-1)
        residual = cooldown_code - self.cooldown_sharp(cooldown)
        means[10] = cooldown
        return (torch.stack(means, -1),
                torch.cat([torch.stack(spreads[:11], -1), residual], -1))

    def decode_logits(self, logits, ship_type, floor: float = 1e-3):
        return self.decode(codes.remove_baseline_floor(logits, self.runs, floor), ship_type)
