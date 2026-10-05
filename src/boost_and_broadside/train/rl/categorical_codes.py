"""Categorical position/attitude codes and physical velocity compression.

Circular codes interpolate between adjacent colours on each nested lattice.
Uncertainty attenuates each level towards uniform. Decoding inverts the chord
interpolation before unwrapping: the argument of an interpolated phasor is not
the linearly interpolated physical angle.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

POSITION_LEVELS = 9
POSITION_COLOURS = 3
ATTITUDE_LEVELS = 4
ATTITUDE_COLOURS = 4
VELOCITY_KNEE = 100.0
# Levels whose first harmonic is below this cannot locate a mean reliably in fp32.
MIN_SHARPNESS = 1e-4
SHARP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CircularCode:
    """Nested two-hot circular code with a shared physical spread.

    Args:
        period: Physical circumference of the coarsest level.
        colours: Number of distinct colours per level (at least three).
        levels: Number of nested levels.
    """

    period: float
    colours: int
    levels: int

    def __post_init__(self) -> None:
        if self.period <= 0 or self.colours < 3 or self.levels < 1:
            raise ValueError("circular code needs a positive period, >=3 colours and >=1 levels")

    @property
    def width(self) -> int:
        return self.colours * self.levels

    def _periods(self, like: torch.Tensor) -> torch.Tensor:
        level = torch.arange(self.levels, device=like.device, dtype=like.dtype)  # (L,)
        return self.period / self.colours**level

    def encode(self, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Encode physical moments to probabilities shaped ``(..., L, C)``."""
        periods = self._periods(mean)  # (L,)
        coordinate = mean.remainder(self.period).unsqueeze(-1) / periods * self.colours
        left = coordinate.floor()  # (..., L)
        fraction = coordinate - left  # (..., L)
        colour = torch.arange(self.colours, device=mean.device)  # (C,)
        sharp = (colour == left.remainder(self.colours).unsqueeze(-1)) * (1.0 - fraction).unsqueeze(
            -1
        ) + (colour == (left + 1).remainder(self.colours).unsqueeze(-1)) * fraction.unsqueeze(
            -1
        )  # (..., L, C)
        attenuation = torch.exp(-0.5 * (2.0 * math.pi * sigma.unsqueeze(-1) / periods).square())
        return sharp * attenuation.unsqueeze(-1) + (1.0 - attenuation.unsqueeze(-1)) / self.colours

    def decode(self, probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover mean and spread by chord inversion and closed-form least squares.

        Finer phases unwrap against the last informative level. Uniform levels
        have no position information and do not overwrite that estimate.
        """
        angle_step = 2.0 * math.pi / self.colours
        colours = torch.arange(self.colours, device=probabilities.device, dtype=probabilities.dtype)
        real = (probabilities * (colours * angle_step).cos()).sum(-1)  # (..., L)
        imag = (probabilities * (colours * angle_step).sin()).sum(-1)  # (..., L)
        phase = torch.atan2(imag, real).remainder(2.0 * math.pi)  # (..., L)
        sector = (phase / angle_step).floor().clamp(max=self.colours - 1)  # (..., L)
        local_angle = phase - sector * angle_step  # (..., L)
        # A ray intersects the polygon edge at (1-t) + t*exp(i*angle_step).
        fraction = local_angle.sin() / (
            math.sin(angle_step) * local_angle.cos()
            + (1.0 - math.cos(angle_step)) * local_angle.sin()
        ).clamp_min(torch.finfo(probabilities.dtype).eps)  # (..., L)
        fraction = fraction.clamp(0.0, 1.0)
        sharp_radius = (
            (1.0 - fraction + fraction * math.cos(angle_step)).square()
            + (fraction * math.sin(angle_step)).square()
        ).sqrt()  # (..., L)
        attenuation = ((real.square() + imag.square()).sqrt() / sharp_radius).clamp(max=1.0)
        periods = self._periods(probabilities)  # (L,)
        local_mean = (sector + fraction) * (periods / self.colours)  # (..., L)
        precision = attenuation.square() * (self.period / periods).square()  # (..., L)
        precision = precision * (attenuation > MIN_SHARPNESS)
        mean = local_mean[..., 0]
        total_precision = precision[..., 0]
        for level in range(1, self.levels):
            period = periods[level]
            unwrapped = (
                local_mean[..., level] + ((mean - local_mean[..., level]) / period).round() * period
            )
            total_precision = total_precision + precision[..., level]
            fraction = precision[..., level] / total_precision.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            )
            mean = mean + fraction * (unwrapped - mean)
        # Fit log sharpness = -sigma^2 * (2*pi/period)^2 / 2. Use only
        # resolved levels, and suppress roundoff on genuinely sharp codes.
        attenuation = torch.where(attenuation > 1.0 - SHARP_TOLERANCE, 1.0, attenuation)
        informative = attenuation > MIN_SHARPNESS  # (..., L)
        slope = 0.5 * (2.0 * math.pi / periods).square()  # (L,)
        weight = informative * attenuation.square()  # (..., L)
        variance = (
            -(weight * slope * attenuation.clamp_min(MIN_SHARPNESS).log()).sum(-1)
            / (weight * slope.square()).sum(-1).clamp_min(torch.finfo(probabilities.dtype).tiny)
        ).clamp_min(0.0)
        sharp = ((probabilities > 0).sum(-1) <= 2).all(-1)
        variance = torch.where(sharp, 0.0, variance)
        # Complete attenuation has no identifiable mean or finite spread. Use
        # one circumference as the saturated spread; it re-encodes as uniform
        # to fp32 precision, instead of falsely claiming certainty.
        spread = torch.where(informative.any(-1), variance.sqrt(), self.period)
        return mean.remainder(self.period), spread


@dataclass(frozen=True)
class PositionCode:
    """Bilinear nine-colour levels on a rectangular physical torus."""

    world_size: tuple[float, float]
    levels: int = POSITION_LEVELS

    @property
    def width(self) -> int:
        return self.levels * POSITION_COLOURS**2

    def encode(self, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Return ``(..., L, 9)`` probabilities from a 2D mean and one sigma."""
        axes = [
            CircularCode(period, POSITION_COLOURS, self.levels).encode(mean[..., axis], sigma)
            for axis, period in enumerate(self.world_size)
        ]  # each (..., L, 3)
        return (axes[0].unsqueeze(-1) * axes[1].unsqueeze(-2)).flatten(-2)

    def decode(self, probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode axis marginals, combining the two fitted variances equally."""
        joint = probabilities.unflatten(-1, (POSITION_COLOURS, POSITION_COLOURS))
        marginals = (joint.sum(-1), joint.sum(-2))  # each (..., L, 3)
        decoded = [
            CircularCode(period, POSITION_COLOURS, self.levels).decode(marginals[axis])
            for axis, period in enumerate(self.world_size)
        ]
        mean = torch.stack([value[0] for value in decoded], dim=-1)  # (..., 2)
        variance = (decoded[0][1].square() + decoded[1][1].square()) / 2.0
        return mean, variance.sqrt()


def compress_velocity(velocity: torch.Tensor) -> torch.Tensor:
    """Radial log compression with an identity derivative at zero."""
    radius = velocity.norm(dim=-1, keepdim=True)  # (..., 1)
    safe_radius = radius.clamp_min(torch.finfo(velocity.dtype).eps)
    scale = VELOCITY_KNEE * torch.log1p(radius / VELOCITY_KNEE) / safe_radius
    return velocity * torch.where(radius > 0, scale, 1.0)


def expand_velocity(compressed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`compress_velocity`, including zero velocity."""
    radius = compressed.norm(dim=-1, keepdim=True)  # (..., 1)
    safe_radius = radius.clamp_min(torch.finfo(compressed.dtype).eps)
    scale = VELOCITY_KNEE * torch.expm1(radius / VELOCITY_KNEE) / safe_radius
    return compressed * torch.where(radius > 0, scale, 1.0)


def velocity_jacobian(velocity: torch.Tensor) -> torch.Tensor:
    """Jacobian mapping physical covariance into compressed coordinates."""
    radius = velocity.norm(dim=-1, keepdim=True)  # (..., 1)
    safe_radius = radius.clamp_min(torch.finfo(velocity.dtype).eps)
    direction = velocity / safe_radius  # (..., 2)
    radial = 1.0 / (1.0 + radius / VELOCITY_KNEE)  # (..., 1)
    tangential = VELOCITY_KNEE * torch.log1p(radius / VELOCITY_KNEE) / safe_radius
    tangential = torch.where(radius > 0, tangential, 1.0)
    projector = direction.unsqueeze(-1) * direction.unsqueeze(-2)  # (..., 2, 2)
    identity = torch.eye(2, device=velocity.device, dtype=velocity.dtype)  # (2, 2)
    return tangential.unsqueeze(-1) * identity + (radial - tangential).unsqueeze(-1) * projector


#: Projection axes of the velocity code, at 0, 120 and 240 degrees.
VELOCITY_AXES = 3
VELOCITY_BINS = 81
VELOCITY_SPACING = 5.0
#: Multiples of machine epsilon below which a moment correction or a fitted
#: variance is rounding noise, not information (about 1e-6 in float32).
SCALAR_TOLERANCE_EPS = 8.0


@dataclass(frozen=True)
class ScalarCode:
    """Two-hot histogram over evenly spaced bins, with a Gaussian spread.

    Encoding convolves the exact two-hot of the mean with a discrete Gaussian
    kernel of variance ``sigma**2``. Variances add under convolution and the
    kernel is symmetric, so the histogram keeps the mean and its variance is
    ``sigma**2`` plus the two-hot's own ``h**2 t (1 - t)``, where ``t`` is the
    mean's fraction between its two bins. Decoding subtracts that term. On a
    lattice the two-hot is the least-variance distribution with a given mean,
    so the subtraction is never negative.

    Mass the kernel spills past either end is folded into the end bin, which
    pulls the mean inward and narrows the spread. One closed-form step on the
    interior bins restores both: each sends a fraction one bin toward the
    deficit (the mean) and a fraction to both neighbours (the variance). End
    bins never send, so nothing spills again. Where the end bins hold too
    much of the mass for that step to reach the target the moments are
    projected instead of matched; the round trip is then approximate.

    Args:
        low: Centre of the first bin.
        high: Centre of the last bin.
        bins: Number of bins, at least three.
    """

    low: float
    high: float
    bins: int

    def __post_init__(self) -> None:
        if self.bins < 3 or not self.high > self.low:
            raise ValueError("scalar code needs at least three bins and high > low")

    @property
    def spacing(self) -> float:
        return (self.high - self.low) / (self.bins - 1)

    def centres(self, like: torch.Tensor) -> torch.Tensor:
        """``(bins,)`` bin centres on ``like``'s device and dtype."""
        index = torch.arange(self.bins, device=like.device, dtype=like.dtype)
        return self.low + index * self.spacing

    def _coordinate(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clamped bin coordinate, its lower bin, and the fraction above it."""
        coordinate = ((mean - self.low) / self.spacing).clamp(0.0, self.bins - 1.0)
        lower = coordinate.floor().clamp(max=self.bins - 2.0)
        return coordinate, lower, coordinate - lower

    def sharp(self, value: torch.Tensor) -> torch.Tensor:
        """The exact two-hot of ``value``, ``(..., bins)``."""
        _, lower, fraction = self._coordinate(value)
        index = torch.arange(self.bins, device=value.device, dtype=value.dtype)
        offset = index - lower.unsqueeze(-1)  # (..., bins)
        return (offset == 0) * (1.0 - fraction).unsqueeze(-1) + (offset == 1) * fraction.unsqueeze(
            -1
        )

    def encode(self, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Encode a mean and a spread to probabilities, ``(..., bins)``."""
        coordinate, lower, fraction = self._coordinate(mean)
        last = self.bins - 1.0
        # A histogram on [0, last] with mean m has variance at most m (last - m);
        # the physical share of that is what is left after the two-hot's own.
        admissible = (coordinate * (last - coordinate) - fraction * (1.0 - fraction)).clamp_min(0)
        variance = (sigma / self.spacing).square().minimum(admissible)  # (...,) square bins
        index = torch.arange(self.bins, device=mean.device, dtype=mean.dtype)
        # Bin offsets from the lower two-hot bin: moments are taken about it, so
        # they stay O(1) in float32 wherever the mean sits.
        offset = index - lower.unsqueeze(-1)  # (..., bins)
        # Discrete Gaussian kernel with variance exactly ``variance``: a sampled
        # Gaussian when it is at least one square bin (its variance error is
        # below 1e-8 there), otherwise the spike mixed with the unit sampled
        # Gaussian in proportion ``variance``. Both are symmetric and continuous
        # at one square bin.
        width = variance.clamp_min(1.0)  # (...,)
        weight = variance.clamp_max(1.0).unsqueeze(-1)  # (..., 1)

        def gaussian(distance: torch.Tensor) -> torch.Tensor:
            return torch.exp(-0.5 * distance.square() / width.unsqueeze(-1)) / torch.sqrt(
                2.0 * math.pi * width.unsqueeze(-1)
            )

        two_hot = (offset == 0) * (1.0 - fraction).unsqueeze(-1) + (
            offset == 1
        ) * fraction.unsqueeze(-1)
        smooth = (1.0 - fraction).unsqueeze(-1) * gaussian(offset) + fraction.unsqueeze(
            -1
        ) * gaussian(offset - 1.0)  # (..., bins)
        # Fold the Gaussian tails beyond either end into the end bin.
        scale = torch.sqrt(2.0 * width)
        below = 0.5 * (
            (1.0 - fraction) * torch.erfc((lower + 0.5) / scale)
            + fraction * torch.erfc((lower + 1.5) / scale)
        )
        above = 0.5 * (
            (1.0 - fraction) * torch.erfc((last - lower + 0.5) / scale)
            + fraction * torch.erfc((last - lower - 0.5) / scale)
        )
        smooth = smooth + (index == 0) * below.unsqueeze(-1) + (index == last) * above.unsqueeze(-1)
        probabilities = (1.0 - weight) * two_hot + weight * smooth
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        return self._match_moments(
            probabilities, offset, fraction, variance + fraction * (1.0 - fraction), index
        )

    def _match_moments(
        self,
        probabilities: torch.Tensor,
        offset: torch.Tensor,
        target_mean: torch.Tensor,
        target_variance: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """One drift-and-diffusion step on interior bins toward the target moments.

        ``offset`` is each bin's position relative to the origin the target
        moments are stated about.
        """
        tolerance = SCALAR_TOLERANCE_EPS * torch.finfo(probabilities.dtype).eps
        mean = (probabilities * offset).sum(-1)
        second = (probabilities * offset.square()).sum(-1)
        interior = (index > 0) & (index < self.bins - 1)
        source = probabilities * interior  # (..., bins)
        mass = source.sum(-1).clamp_min(torch.finfo(probabilities.dtype).tiny)
        drift = (target_mean - mean) / mass  # signed fraction moved one bin up
        drift = torch.where(drift.abs() > tolerance, drift, 0.0)
        direction = torch.sign(drift).unsqueeze(-1)
        drift_second = drift.abs() * (source * (2.0 * direction * offset + 1.0)).sum(-1)
        target_second = target_variance + target_mean.square()
        diffusion = ((target_second - second - drift_second) / (2.0 * mass)).clamp_min(0.0)
        diffusion = torch.where(diffusion > tolerance, diffusion, 0.0)
        total = 2.0 * diffusion + drift.abs()
        excess = total.clamp_min(1.0)
        diffusion, drift = diffusion / excess, drift / excess
        up = drift.clamp_min(0.0).unsqueeze(-1) * source
        down = (-drift).clamp_min(0.0).unsqueeze(-1) * source
        spread = diffusion.unsqueeze(-1) * source
        moved = up + down + 2.0 * spread
        received = F.pad(up + spread, (1, -1)) + F.pad(down + spread, (-1, 1))
        return probabilities - moved + received

    def decode(self, probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Histogram mean, and its variance less the two-hot's, as a spread."""
        index = torch.arange(self.bins, device=probabilities.device, dtype=probabilities.dtype)
        coordinate = (probabilities * index).sum(-1).clamp(0.0, self.bins - 1.0)
        variance = (probabilities * (index - coordinate.unsqueeze(-1)).square()).sum(-1)
        fraction = coordinate - coordinate.floor()
        physical = variance - fraction * (1.0 - fraction)
        tolerance = SCALAR_TOLERANCE_EPS * torch.finfo(probabilities.dtype).eps
        physical = torch.where(physical > tolerance, physical, 0.0)
        return self.low + coordinate * self.spacing, physical.sqrt() * self.spacing


@dataclass(frozen=True)
class VelocityCode:
    """Three projections of log-compressed velocity, each a :class:`ScalarCode`.

    The moments are the mean and covariance of raw world velocity, the
    covariance packed as ``(xx, xy, yy)``. They meet the compressed code through
    the compression's Jacobian at the mean, used in both directions, so the
    round trip is exact even where the linearisation is not.

    Everything is elementwise: under bf16 autocast a matrix product would be
    cast down, and the round trip needs float32.
    """

    bins: int = VELOCITY_BINS
    spacing: float = VELOCITY_SPACING

    @property
    def axis_code(self) -> ScalarCode:
        reach = 0.5 * (self.bins - 1) * self.spacing
        return ScalarCode(-reach, reach, self.bins)

    @property
    def width(self) -> int:
        return VELOCITY_AXES * self.bins

    @staticmethod
    def _project(vector: torch.Tensor) -> torch.Tensor:
        """``(..., 3)`` projections of a ``(..., 2)`` vector on the three axes."""
        x, y = vector[..., 0], vector[..., 1]
        half_root3 = 0.5 * math.sqrt(3.0)
        return torch.stack([x, -0.5 * x + half_root3 * y, -0.5 * x - half_root3 * y], dim=-1)

    def sharp(self, velocity: torch.Tensor) -> torch.Tensor:
        """``(..., 3, bins)`` exact two-hots of the three projections."""
        return self.axis_code.sharp(self._project(compress_velocity(velocity)))

    def encode(self, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        """``(..., 3, bins)`` from a ``(..., 2)`` mean and a packed ``(..., 3)`` covariance."""
        xx, xy, yy = sandwich(packed_velocity_jacobian(mean), covariance).unbind(-1)
        quarter = 0.25
        three_quarters = 0.75
        cross = 0.5 * math.sqrt(3.0) * xy
        variance = torch.stack(
            [
                xx,
                quarter * xx - cross + three_quarters * yy,
                quarter * xx + cross + three_quarters * yy,
            ],
            dim=-1,
        )
        projection = self._project(compress_velocity(mean))
        return self.axis_code.encode(projection, variance.clamp_min(0.0).sqrt())

    def decode(self, probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Least-squares mean from the three axes, covariance solved exactly.

        Returns the ``(..., 2)`` mean and the packed ``(..., 3)`` covariance.
        """
        projection, sigma = self.axis_code.decode(probabilities)  # each (..., 3)
        half_root3 = 0.5 * math.sqrt(3.0)
        compressed = torch.stack(
            [
                projection[..., 0] - 0.5 * (projection[..., 1] + projection[..., 2]),
                half_root3 * (projection[..., 1] - projection[..., 2]),
            ],
            dim=-1,
        ) * (2.0 / VELOCITY_AXES)
        variance = sigma.square()
        xx = variance[..., 0]
        xy = (variance[..., 2] - variance[..., 1]) / math.sqrt(3.0)
        yy = (2.0 * (variance[..., 1] + variance[..., 2]) - xx) / 3.0
        # Three marginals from a model need not come from one covariance.
        xx, yy = xx.clamp_min(0.0), yy.clamp_min(0.0)
        bound = (xx * yy).sqrt()
        xy = torch.maximum(torch.minimum(xy, bound), -bound)
        mean = expand_velocity(compressed)
        inverse = packed_inverse_velocity_jacobian(mean)
        return mean, sandwich(inverse, torch.stack([xx, xy, yy], dim=-1))


def sandwich(matrix: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """``A S A`` for symmetric 2x2 ``A`` and ``S``, both packed as ``(xx, xy, yy)``."""
    a, b, c = matrix.unbind(-1)
    s1, s2, s3 = covariance.unbind(-1)
    left_x, left_y = a * s1 + b * s2, a * s2 + b * s3
    right_x, right_y = b * s1 + c * s2, b * s2 + c * s3
    return torch.stack(
        [left_x * a + left_y * b, left_x * b + left_y * c, right_x * b + right_y * c], dim=-1
    )


def _packed_radial_matrix(
    velocity: torch.Tensor, radial: torch.Tensor, tangential: torch.Tensor
) -> torch.Tensor:
    """``tangential I + (radial - tangential) v̂v̂ᵀ``, packed."""
    radius = velocity.norm(dim=-1)
    direction = velocity / radius.clamp_min(torch.finfo(velocity.dtype).eps).unsqueeze(-1)
    dx, dy = direction[..., 0], direction[..., 1]
    excess = radial - tangential
    return torch.stack(
        [tangential + excess * dx * dx, excess * dx * dy, tangential + excess * dy * dy], dim=-1
    )


def packed_velocity_jacobian(velocity: torch.Tensor) -> torch.Tensor:
    """:func:`velocity_jacobian`, packed as ``(xx, xy, yy)``; it is symmetric."""
    radius = velocity.norm(dim=-1)
    safe_radius = radius.clamp_min(torch.finfo(velocity.dtype).eps)
    radial = 1.0 / (1.0 + radius / VELOCITY_KNEE)
    tangential = torch.where(
        radius > 0, VELOCITY_KNEE * torch.log1p(radius / VELOCITY_KNEE) / safe_radius, 1.0
    )
    return _packed_radial_matrix(velocity, radial, tangential)


def packed_inverse_velocity_jacobian(velocity: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of :func:`packed_velocity_jacobian`."""
    radius = velocity.norm(dim=-1)
    radial = 1.0 + radius / VELOCITY_KNEE
    compressed = (VELOCITY_KNEE * torch.log1p(radius / VELOCITY_KNEE)).clamp_min(
        torch.finfo(velocity.dtype).tiny
    )
    tangential = torch.where(radius > 0, radius / compressed, 1.0)
    return _packed_radial_matrix(velocity, radial, tangential)
