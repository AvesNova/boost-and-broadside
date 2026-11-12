"""The ship-state code (spec 3): the belief's means and spreads as one vector of softmax
groups, and back.

The belief stores physical values (m, m/s, rad, s, J). Shield and energy are coded as
fractions of the ship type's maximum, read from the ship card the model was built with.
The other ranges are the card's largest over its types: the angle of attack spans twice the
widest stall angle, the recharge wait the longest recharge delay, and the gun cooldown's
bins (and so its residual spreads) are one more than the most ticks between shots.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, per_ship, ticks_between_shots

from . import codes

MEANS = (
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "heading",
    "nose_rate",
    "angle_of_attack",
    "shield_recharge_wait",
    "shield",
    "energy",
    "gun_cooldown",
    "log_refractive_index",
)
# The spreads before the cooldown residuals (one per cooldown bin, which follow).
SPREADS = (
    "position",
    "velocity_xx",
    "velocity_xy",
    "velocity_yy",
    "heading",
    "nose_rate",
    "angle_of_attack",
    "shield_recharge_wait",
    "shield",
    "energy",
    "log_refractive_index",
)

# The kinematic state whose full covariance a delta build's belief carries
# (`YemongConfig.next_state = "delta"`): appended to the spreads as its upper triangle,
# row-major (COVARIANCE_ENTRIES).
COVARIANCE_STATE = ("x", "y", "velocity_x", "velocity_y", "heading", "nose_rate")
COVARIANCE_ENTRIES = len(COVARIANCE_STATE) * (len(COVARIANCE_STATE) + 1) // 2
SPREAD_INPUTS = ("scalar", "ellipse")  # `YemongConfig.spread_inputs`
POSITION_INPUTS = ("colour", "fourier")  # `YemongConfig.position_inputs`

# Sizing rule for the position codes (this colour code, `YemongConfig.position_period`, and
# the rotary bands, `YemongConfig.rotary_*`); revisit both whenever ship speeds, the tick or
# the largest map change (yemong spec, "Sizing the position codes"):
# - colour code, finest period > 2 x the largest distance a ship moves in one tick
#   (Nyquist; real scale: about 320 m/s at 30 Hz is 10.7 m a tick, so above 21 m);
#   coarsest > the largest absolute span (a map's diameter plus the overshoot);
# - rotary, finest > 2 x the largest relative move in a tick (about 640 m/s closing, 21 m
#   a tick, so above 43 m) and fine enough to aim (about 1 degree at 550 m); coarsest >
#   2 x the largest pairwise distance, since a relative phase is unique only within +-P/2
#   (the largest evaluation maps are about 62 km across).
# POSITION_PERIOD is the record default (every build before 2026-11-12: 4 m to 26 244 m,
# breaking both); a new run's is 236 196 m (36 m finest; `model.COLOUR_PERIOD`) and its
# rotary bands 48 m to 254 771 m at ratio 3.405, each turned 35 degrees
# (`model.ROTARY_*`).
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
SCALARS = ("angle_of_attack", "shield_recharge_wait", "shield", "energy", "log_refractive_index")


# The ellipse inputs (`spread_inputs = "ellipse"`): per ellipse (position, velocity) log
# sigma_major, log sigma_minor, and the doubled angle (cos 2 theta, sin 2 theta) times the
# anisotropy (sigma_major - sigma_minor) / (sigma_major + sigma_minor), world frame, each a
# two-hot code of SCALAR_BINS like the state code's scalars. A sigma is coded as
# ln(1 + sigma / knee) from 0 up to its bound's (a log above the knee, 0 for a sharp row).
ELLIPSE_FEATURES = 4
# The sigmas' knees; each is coded up to its bound: the position code's period, the
# velocity code's reach (`ShipStateCode`).
ELLIPSE_KNEE = dict(position=1.0, velocity=0.1)


def pack_covariance(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 6, 6) to its upper triangle (..., COVARIANCE_ENTRIES), row-major."""
    size = len(COVARIANCE_STATE)
    rows, columns = torch.triu_indices(size, size, device=matrix.device)  # no host copy
    return matrix[..., rows, columns]


def unpack_covariance(packed: torch.Tensor) -> torch.Tensor:
    """The inverse of `pack_covariance`: (..., COVARIANCE_ENTRIES) to symmetric (..., 6, 6)."""
    size = len(COVARIANCE_STATE)
    # made on the packed tensor's device: no host-to-device copy (which a CUDA graph's
    # capture, the uncompiled rollout's, refuses)
    device = packed.device
    rows, columns = torch.triu_indices(size, size, device=device)
    entries = torch.arange(COVARIANCE_ENTRIES, device=device)
    index = torch.zeros((size, size), dtype=torch.long, device=device)
    index[rows, columns] = entries
    index[columns, rows] = entries
    return packed[..., index]


def ellipse(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A packed 2 x 2 covariance (..., 3) (xx, xy, yy) to (sigma_major, sigma_minor,
    anisotropy cos 2 theta, anisotropy sin 2 theta), theta the major axis's angle from x.
    A round (or zero) covariance has both angle features 0."""
    xx, xy, yy = packed.unbind(-1)
    centre = 0.5 * (xx + yy)
    half = 0.5 * (xx - yy)
    radius = torch.sqrt(half * half + xy * xy)
    major = torch.sqrt(torch.clamp_min(centre + radius, 0.0))
    minor = torch.sqrt(torch.clamp_min(centre - radius, 0.0))
    total = major + minor
    anisotropy = torch.where(total > 0, (major - minor) / torch.where(total > 0, total, 1.0), 0.0)
    safe = torch.where(radius > 0, radius, 1.0)
    cos = torch.where(radius > 0, half / safe, 0.0)
    sin = torch.where(radius > 0, xy / safe, 0.0)
    return major, minor, anisotropy * cos, anisotropy * sin


def velocity_reach() -> float:
    """The largest world speed the velocity code resolves (m/s): 50 (e^2 - 1)."""
    return VELOCITY_KNEE * math.expm1(0.5 * (VELOCITY_BINS - 1) * VELOCITY_SPACING / VELOCITY_KNEE)


@dataclasses.dataclass(frozen=True)
class StateCodeRanges:
    """What the code reads from the rules and the ship card (`state_code_ranges`), so a
    built model never reads the current game. tick_length (s); angle_of_attack_limit (rad):
    twice the widest stall angle; shield_recharge_delay (s): the longest; cooldown_bins:
    one more than the most ticks between shots; max_shield and max_energy: per type."""

    tick_length: float
    angle_of_attack_limit: float
    shield_recharge_delay: float
    cooldown_bins: int
    max_shield: tuple[float, ...]
    max_energy: tuple[float, ...]


def state_code_ranges(rules: Rules, types: ShipTypes) -> StateCodeRanges:
    """The ranges of the rules and a (types,) card: the shared values times the card's
    largest scale (exact at the default card's 1)."""
    return StateCodeRanges(
        tick_length=rules.tick_length,
        angle_of_attack_limit=2.0 * rules.flight.stall_angle * float(types.stall_angle_scale.max()),
        shield_recharge_delay=rules.shield_recharge_delay
        * float(types.shield_recharge_delay_scale.max()),
        cooldown_bins=1 + int(ticks_between_shots(types.fire_rate, rules.tick_length).max()),
        max_shield=tuple(types.max_shield.tolist()),
        max_energy=tuple(types.max_energy.tolist()),
    )


class ShipStateCode(nn.Module):
    """Encodes (means (..., 12), spreads (..., spread_count), ship_type (...)) to
    (..., width) and back. With the default card the spreads are 15 wide and the code 490.

    Runs in order: position 9x9, velocity 3x81, heading 4x4, nose rate 41, the five SCALARS
    of 21 bins each, gun cooldown. spread_fit: how `decode` reads the position's and the
    heading's spreads (`codes.circular_decode`). position_inputs (POSITION_INPUTS): what the
    ship token reads for the position and the heading (`inputs`): the code's own runs
    ("colour"), or Fourier features at the same levels ("fourier"); the code is the same
    either way. position_period (m per axis): the position code's coarsest period
    (`YemongConfig.position_period`), which the position ellipse's sigma code also spans.
    """

    def __init__(
        self,
        ranges: StateCodeRanges,
        spread_fit: str = "coarsest",
        covariance: bool = False,
        spread_inputs: str = "scalar",
        position_inputs: str = "colour",
        position_period: float = POSITION_PERIOD,
    ) -> None:
        super().__init__()
        self.position_period = position_period
        if spread_fit not in codes.SPREAD_FITS:
            raise ValueError(f"spread_fit {spread_fit!r} is not one of {codes.SPREAD_FITS}")
        if spread_inputs not in SPREAD_INPUTS:
            raise ValueError(f"spread_inputs {spread_inputs!r} is not one of {SPREAD_INPUTS}")
        if position_inputs not in POSITION_INPUTS:
            raise ValueError(f"position_inputs {position_inputs!r} is not one of {POSITION_INPUTS}")
        self.spread_fit = spread_fit
        self.position_inputs = position_inputs
        self.covariance = covariance
        self.spread_inputs = spread_inputs
        self.angle_of_attack_limit = ranges.angle_of_attack_limit
        self.shield_delay = ranges.shield_recharge_delay
        self.tick_length = ranges.tick_length
        self.cooldown_bins = ranges.cooldown_bins
        self.register_buffer("max_shield", torch.tensor(ranges.max_shield, dtype=torch.float32))
        self.register_buffer("max_energy", torch.tensor(ranges.max_energy, dtype=torch.float32))
        # SCALARS' lows and highs (`_scalar_axis`): made once, on the module's device (a
        # tensor built from host numbers inside a step is a copy a CUDA graph cannot take)
        bounds = self._scalar_ranges()
        for end, name in enumerate(("scalar_low", "scalar_high")):
            values = [bounds[scalar][end] for scalar in SCALARS]
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32), persistent=False)
        # the ellipse features' lows and highs (`ellipse_code`), likewise
        ellipse_low, ellipse_high = [], []
        for name in ("position", "velocity"):
            bound = {"position": position_period, "velocity": velocity_reach()}[name]
            top = math.log1p(bound / ELLIPSE_KNEE[name])
            ellipse_low += [0.0, 0.0, -1.0, -1.0]
            ellipse_high += [top, top, 1.0, 1.0]
        for name, values in (("ellipse_low", ellipse_low), ("ellipse_high", ellipse_high)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32), persistent=False)
        scalar = (1, SCALAR_BINS)
        self.runs: codes.Runs = (
            (POSITION_LEVELS, 9),
            (3, VELOCITY_BINS),
            (HEADING_LEVELS, HEADING_COLOURS),
            (1, NOSE_RATE_BINS),
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            (1, self.cooldown_bins),
        )

    @property
    def width(self) -> int:
        """The code's width: groups times size, summed over runs."""
        return sum(groups * size for groups, size in self.runs)

    @property
    def groups(self) -> int:
        """The softmax groups of the code, summed over runs."""
        return sum(groups for groups, _ in self.runs)

    @property
    def input_width(self) -> int:
        """The width the ship token reads (`inputs`): the code's, or with Fourier inputs
        (cos, sin) per level of each position axis and of the heading in place of their
        runs (36 for 81, 8 for 16)."""
        if self.position_inputs == "colour":
            return self.width
        position, heading = self.runs[0], self.runs[2]
        replaced = position[0] * position[1] + heading[0] * heading[1]
        return self.width - replaced + 2 * (2 * POSITION_LEVELS + HEADING_LEVELS)

    @property
    def input_groups(self) -> int:
        """The groups of `inputs` (a Fourier (cos, sin) pair counts as one): what the
        encoder's first projection scales its code columns by."""
        if self.position_inputs == "colour":
            return self.groups
        return self.groups + POSITION_LEVELS  # a pair per axis and level, not one joint group

    @property
    def spread_count(self) -> int:
        """SPREADS, then one cooldown residual per cooldown bin, then with `covariance` the
        kinematic covariance (COVARIANCE_ENTRIES)."""
        return len(SPREADS) + self.cooldown_bins + (COVARIANCE_ENTRIES if self.covariance else 0)

    @property
    def residual_slice(self) -> slice:
        """The cooldown residuals' columns of the spreads."""
        return slice(len(SPREADS), len(SPREADS) + self.cooldown_bins)

    @property
    def covariance_slice(self) -> slice:
        """The covariance's columns of the spreads (with `covariance`)."""
        start = len(SPREADS) + self.cooldown_bins
        return slice(start, start + COVARIANCE_ENTRIES)

    @property
    def ellipse_width(self) -> int:
        """The ellipse inputs' width (0 with scalar spread inputs)."""
        return 2 * ELLIPSE_FEATURES * SCALAR_BINS if self.spread_inputs == "ellipse" else 0

    def blocks(self, spreads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(position, velocity) packed 2 x 2 covariances (..., 3) of spreads: the
        covariance's blocks with `covariance`, else the scalar sigma's round one and the
        velocity slots."""
        if self.covariance:
            matrix = unpack_covariance(spreads[..., self.covariance_slice])
            position = torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]], -1)
            velocity = torch.stack([matrix[..., 2, 2], matrix[..., 2, 3], matrix[..., 3, 3]], -1)
            return position, velocity
        variance = spreads[..., 0] ** 2
        position = torch.stack([variance, torch.zeros_like(variance), variance], -1)
        return position, spreads[..., 1:4]

    def ellipse_code(self, spreads: torch.Tensor) -> torch.Tensor:
        """(..., ellipse_width): per ellipse (position, velocity; `blocks`) the four
        ELLIPSE_FEATURES, each a two-hot code of SCALAR_BINS (`ELLIPSE_KNEE`)."""
        features = []
        for name, packed in zip(("position", "velocity"), self.blocks(spreads), strict=True):
            knee = ELLIPSE_KNEE[name]
            major, minor, cos, sin = ellipse(packed)
            features += [torch.log1p(major / knee), torch.log1p(minor / knee), cos, sin]
        # one code call over the eight features, their bounds per feature (module init)
        low = self.ellipse_low.to(spreads.dtype)
        high = self.ellipse_high.to(spreads.dtype)
        return codes.scalar_sharp(torch.stack(features, -1), low, high, SCALAR_BINS).flatten(-2)

    def _scalar_ranges(self) -> dict[str, tuple[float, float, int]]:
        """(low, high, bins) of each of SCALARS."""
        unit = (0.0, 1.0, SCALAR_BINS)
        return dict(
            angle_of_attack=(-self.angle_of_attack_limit, self.angle_of_attack_limit, SCALAR_BINS),
            shield_recharge_wait=(0.0, self.shield_delay, SCALAR_BINS),
            shield=unit,
            energy=unit,
            log_refractive_index=(-LOG_INDEX_SPAN, LOG_INDEX_SPAN, SCALAR_BINS),
        )

    def _scalar_axis(
        self, maxima: dict[str, torch.Tensor], like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SCALARS' lows and highs (len(SCALARS),) and divisors (..., len(SCALARS)), for one
        code call over the scalars."""
        low, high = self.scalar_low.to(like.dtype), self.scalar_high.to(like.dtype)
        shape = next(iter(maxima.values())).shape if maxima else like.shape[:-1]
        ones = torch.ones(shape, dtype=like.dtype, device=like.device)
        divisor = torch.stack([maxima.get(name, ones) for name in SCALARS], -1)
        return low, high, divisor

    def _maxima(self, ship_type: torch.Tensor) -> dict[str, torch.Tensor]:
        """The divisor that makes each scalar a fraction (1 for the others)."""
        return dict(
            shield=per_ship(self.max_shield, ship_type), energy=per_ship(self.max_energy, ship_type)
        )

    def _cooldown_sharp(self, cooldown: torch.Tensor) -> torch.Tensor:
        return codes.scalar_sharp(
            cooldown, 0.0, (self.cooldown_bins - 1) * self.tick_length, self.cooldown_bins
        )

    def encode(
        self, means: torch.Tensor, spreads: torch.Tensor, ship_type: torch.Tensor
    ) -> torch.Tensor:
        """The state code (..., width) of means (..., len(MEANS)), spreads (..., spread_count) and
        ship_type (...), one run per quantity in runs order."""
        maxima = self._maxima(ship_type)
        mean = dict(zip(MEANS, means.unbind(-1), strict=True))
        spread = dict(zip(SPREADS, spreads[..., : len(SPREADS)].unbind(-1), strict=True))
        residual = spreads[..., self.residual_slice]
        position = codes.position_encode(
            means[..., 0:2],
            self._position_sigma(spreads),
            (self.position_period,) * 2,
            POSITION_LEVELS,
        )
        velocity = codes.velocity_encode(
            means[..., 2:4], spreads[..., 1:4], VELOCITY_BINS, VELOCITY_SPACING, VELOCITY_KNEE
        )
        heading = codes.circular_encode(
            mean["heading"], spread["heading"], 2.0 * math.pi, HEADING_COLOURS, HEADING_LEVELS
        )
        nose_rate = codes.scalar_encode(
            mean["nose_rate"],
            spread["nose_rate"],
            -NOSE_RATE_LIMIT,
            NOSE_RATE_LIMIT,
            NOSE_RATE_BINS,
        )
        # every scalar shares SCALAR_BINS: one code call over a scalar axis, the bounds per
        # scalar as tensors
        low, high, divisor = self._scalar_axis(maxima, means)
        scalars = codes.scalar_encode(
            torch.stack([mean[name] for name in SCALARS], -1) / divisor,
            torch.stack([spread[name] for name in SCALARS], -1) / divisor,
            low,
            high,
            SCALAR_BINS,
        )
        cooldown = torch.clamp_min(self._cooldown_sharp(mean["gun_cooldown"]) + residual, 0.0)
        return torch.cat(
            [
                position.flatten(-2),
                velocity.flatten(-2),
                heading.flatten(-2),
                nose_rate,
                scalars.flatten(-2),
                cooldown,
            ],
            -1,
        )

    def _position_sigma(self, spreads: torch.Tensor) -> torch.Tensor:
        """The position's sigma: (...) for both axes, or with ellipse spread inputs (..., 2)
        per world axis, the covariance's marginals."""
        if self.spread_inputs == "ellipse":
            block, _ = self.blocks(spreads)
            return torch.sqrt(torch.clamp_min(block[..., 0::2], 0.0))
        return spreads[..., 0]

    def inputs(
        self, code: torch.Tensor, means: torch.Tensor, spreads: torch.Tensor
    ) -> torch.Tensor:
        """What the ship token reads of the state (..., input_width): with colour inputs the
        code (..., width) itself; with Fourier inputs the code with its position and heading
        runs replaced by `codes.fourier_encode` of the same means and spreads at the same
        periods (position per world axis, x first, POSITION_LEVELS levels from
        position_period; heading HEADING_LEVELS from 2 pi), each level attenuated by its
        sigma as the code's is."""
        if self.position_inputs == "colour":
            return code
        sigma = self._position_sigma(spreads)
        if sigma.dim() < means.dim():
            sigma = torch.stack([sigma, sigma], -1)
        position = torch.stack(
            [
                codes.fourier_encode(
                    means[..., axis], sigma[..., axis], self.position_period, 3, POSITION_LEVELS
                )
                for axis in range(2)
            ],
            -3,
        )
        heading = codes.fourier_encode(
            means[..., MEANS.index("heading")],
            spreads[..., SPREADS.index("heading")],
            2.0 * math.pi,
            HEADING_COLOURS,
            HEADING_LEVELS,
        )
        parts = code.split([groups * size for groups, size in self.runs], -1)
        return torch.cat([position.flatten(-3), parts[1], heading.flatten(-2), *parts[3:]], -1)

    def sharp(self, means: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
        """The code of a state known exactly."""
        return self.encode(
            means, means.new_zeros(means.shape[:-1] + (self.spread_count,)), ship_type
        )

    def decode(
        self, probabilities: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., width) probabilities to (means (..., 12), spreads (..., spread_count)).
        Positions come back in [0, P) per axis (the code's period), the heading in
        [-pi, pi)."""
        maxima = self._maxima(ship_type)
        parts = probabilities.split([groups * size for groups, size in self.runs], -1)
        position, position_spread = codes.position_decode(
            parts[0].unflatten(-1, (POSITION_LEVELS, 9)),
            (self.position_period,) * 2,
            POSITION_LEVELS,
            self.spread_fit,
        )
        velocity, covariance = codes.velocity_decode(
            parts[1].unflatten(-1, (3, VELOCITY_BINS)),
            VELOCITY_BINS,
            VELOCITY_SPACING,
            VELOCITY_KNEE,
        )
        heading, heading_spread = codes.circular_decode(
            parts[2].unflatten(-1, (HEADING_LEVELS, HEADING_COLOURS)),
            2.0 * math.pi,
            HEADING_COLOURS,
            HEADING_LEVELS,
            self.spread_fit,
        )
        heading = torch.remainder(heading + math.pi, 2.0 * math.pi) - math.pi
        nose_rate, nose_spread = codes.scalar_decode(
            parts[3], -NOSE_RATE_LIMIT, NOSE_RATE_LIMIT, NOSE_RATE_BINS
        )
        low, high, divisor = self._scalar_axis(maxima, probabilities)
        values, sigmas = codes.scalar_decode(
            torch.cat(parts[4:9], -1).unflatten(-1, (len(SCALARS), SCALAR_BINS)),
            low,
            high,
            SCALAR_BINS,
        )
        values, sigmas = values * divisor, sigmas * divisor
        scalar = dict(zip(SCALARS, values.unbind(-1), strict=True))
        scalar_spread = dict(zip(SCALARS, sigmas.unbind(-1), strict=True))
        cooldown_code = parts[9]
        centres = self.tick_length * torch.arange(
            self.cooldown_bins, dtype=probabilities.dtype, device=probabilities.device
        )
        cooldown = (cooldown_code * centres).sum(-1)
        residual = cooldown_code - self._cooldown_sharp(cooldown)
        means = torch.stack(
            [
                position[..., 0],
                position[..., 1],
                velocity[..., 0],
                velocity[..., 1],
                heading,
                nose_rate,
                scalar["angle_of_attack"],
                scalar["shield_recharge_wait"],
                scalar["shield"],
                scalar["energy"],
                cooldown,
                scalar["log_refractive_index"],
            ],
            -1,
        )
        spreads = torch.cat(
            [
                position_spread[..., None],
                covariance,
                torch.stack(
                    [heading_spread, nose_spread] + [scalar_spread[name] for name in SCALARS], -1
                ),
                residual,
            ],
            -1,
        )
        return means, spreads

    def decode_logits(
        self, logits: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-state head logits to (means, spreads): floor removal per group, then decode."""
        return self.decode(codes.remove_baseline_floor(logits, self.runs), ship_type)
