"""Categorical codes of a continuous state (spec 3.3): circular, position, scalar, velocity,
and the grouped-softmax operations over a code vector.

Every function is elementwise over leading axes, has no data-dependent control flow, and
works in float32 or float64, eagerly or compiled.
"""

import math

import torch
import torch.nn.functional as F

BASELINE_FLOOR = 1e-3  # the next-state baseline log(code + floor) and its removal
INFORMATIVE = 1e-4  # attenuation below which a circular level carries no information


def _tolerance(dtype: torch.dtype) -> float:
    return 8.0 * torch.finfo(dtype).eps


def _one_hot(index: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot by comparison (no scatter), so it fuses and exports."""
    return (index[..., None] == torch.arange(size, device=index.device)).to(dtype)


# --------------------------------------------------------------------------- circular


def circular_encode(mean: torch.Tensor, sigma: torch.Tensor, period: float, colours: int,
                    levels: int) -> torch.Tensor:
    """(...) mean and sigma to (..., levels, colours) probabilities.

    The spec's coordinate is (m mod P) / p_l * c. Only its fraction and its floor mod c are
    used, and m / p_l * c differs from it by a multiple of c^(l + 1), so the mod is left out:
    in float32 it would cost the finest levels their precision (x mod 26 km rounds to 2 mm,
    0.15% of a 4 m level's 1.3 m colour).
    """
    level_periods = period / colours ** torch.arange(levels, dtype=mean.dtype, device=mean.device)
    position = mean[..., None] * (colours / level_periods)
    lower = torch.floor(position)
    fraction = position - lower
    colour = torch.remainder(lower, colours).long()
    sharp = (1.0 - fraction)[..., None] * _one_hot(colour, colours, mean.dtype) \
        + fraction[..., None] * _one_hot(torch.remainder(colour + 1, colours), colours, mean.dtype)
    attenuation = torch.exp(-0.5 * (2.0 * math.pi * sigma[..., None] / level_periods) ** 2)
    return sharp * attenuation[..., None] + ((1.0 - attenuation) / colours)[..., None]


def circular_decode(probabilities: torch.Tensor, period: float, colours: int,
                    levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., levels, colours) probabilities to (mean, spread), each (...)."""
    dtype, device = probabilities.dtype, probabilities.device
    eps, tiny = torch.finfo(dtype).eps, torch.finfo(dtype).tiny
    step = 2.0 * math.pi / colours
    angles = step * torch.arange(colours, dtype=dtype, device=device)
    real = (probabilities * torch.cos(angles)).sum(-1)
    imaginary = (probabilities * torch.sin(angles)).sum(-1)
    phase = torch.remainder(torch.atan2(imaginary, real), 2.0 * math.pi)
    sector = torch.clamp_max(torch.floor(phase / step), float(colours - 1))
    offset = phase - sector * step
    denominator = math.sin(step) * torch.cos(offset) + (1.0 - math.cos(step)) * torch.sin(offset)
    fraction = torch.clamp(torch.sin(offset) / torch.clamp_min(denominator, eps), 0.0, 1.0)
    chord = torch.sqrt((1.0 - fraction + fraction * math.cos(step)) ** 2
                       + (fraction * math.sin(step)) ** 2)
    attenuation = torch.clamp_max(torch.sqrt(real * real + imaginary * imaginary) / chord, 1.0)
    level_periods = [period / colours**level for level in range(levels)]
    # Per-level constants built on the device (no host copy, so eager graph capture works).
    refinement = colours ** torch.arange(levels, dtype=dtype, device=device)  # period / p_l
    local_mean = (sector + fraction) * (period / refinement) / colours
    precision = torch.where(attenuation > INFORMATIVE, attenuation**2, 0.0) * refinement**2

    mean, total = local_mean[..., 0], precision[..., 0]
    for level in range(1, levels):
        level_period = level_periods[level]
        unwrapped = local_mean[..., level] \
            + torch.round((mean - local_mean[..., level]) / level_period) * level_period
        total = total + precision[..., level]
        mean = mean + precision[..., level] / torch.clamp_min(total, tiny) * (unwrapped - mean)
    mean = torch.remainder(mean, period)

    attenuation = torch.where(attenuation > 1.0 - 1e-6, 1.0, attenuation)
    informative = attenuation > INFORMATIVE
    curvature = 0.5 * (2.0 * math.pi * refinement / period) ** 2
    weight = informative.to(dtype) * attenuation**2
    variance = -(weight * curvature * torch.log(torch.clamp_min(attenuation, INFORMATIVE))).sum(-1)
    variance = torch.clamp_min(variance / torch.clamp_min((weight * curvature**2).sum(-1), tiny),
                               0.0)
    two_hot = ((probabilities > 0).sum(-1) <= 2).all(-1)
    variance = torch.where(two_hot, 0.0, variance)
    spread = torch.where(informative.any(-1), torch.sqrt(variance), period)
    return mean, spread


# --------------------------------------------------------------------------- position


def position_encode(mean: torch.Tensor, sigma: torch.Tensor, periods: tuple[float, float],
                    levels: int) -> torch.Tensor:
    """(..., 2) mean and (...) sigma to (..., levels, 9): the outer product of a 3-colour
    circular code per axis, x-major."""
    x = circular_encode(mean[..., 0], sigma, periods[0], 3, levels)
    y = circular_encode(mean[..., 1], sigma, periods[1], 3, levels)
    return (x[..., :, None] * y[..., None, :]).flatten(-2)


def position_decode(probabilities: torch.Tensor, periods: tuple[float, float],
                    levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., levels, 9) to (mean (..., 2), spread (...))."""
    joint = probabilities.unflatten(-1, (3, 3))
    mean_x, spread_x = circular_decode(joint.sum(-1), periods[0], 3, levels)
    mean_y, spread_y = circular_decode(joint.sum(-2), periods[1], 3, levels)
    return torch.stack([mean_x, mean_y], -1), torch.sqrt(0.5 * (spread_x**2 + spread_y**2))


# --------------------------------------------------------------------------- scalar


def _scalar_coordinate(values: torch.Tensor, low: float, high: float, bins: int):
    spacing = (high - low) / (bins - 1)
    coordinate = torch.clamp((values - low) / spacing, 0.0, float(bins - 1))
    lower = torch.clamp_max(torch.floor(coordinate), float(bins - 2))
    return coordinate, lower, coordinate - lower, spacing


def _two_hot_at(lower: torch.Tensor, fraction: torch.Tensor, bins: int) -> torch.Tensor:
    lower = lower.long()
    return (1.0 - fraction)[..., None] * _one_hot(lower, bins, fraction.dtype) \
        + fraction[..., None] * _one_hot(lower + 1, bins, fraction.dtype)


def scalar_sharp(values: torch.Tensor, low: float, high: float, bins: int) -> torch.Tensor:
    """(...) to (..., bins): two-hot on the bins either side."""
    _, lower, fraction, _ = _scalar_coordinate(values, low, high, bins)
    return _two_hot_at(lower, fraction, bins)


def match_moments(probabilities: torch.Tensor, offsets: torch.Tensor, target_mean: torch.Tensor,
                  target_variance: torch.Tensor) -> torch.Tensor:
    """Move interior mass by one bin so the code has the target mean and variance about
    its lower bin (spec 3.3, "match moments")."""
    dtype = probabilities.dtype
    tolerance, tiny = _tolerance(dtype), torch.finfo(dtype).tiny
    bins = probabilities.shape[-1]
    index = torch.arange(bins, device=probabilities.device)
    interior = ((index > 0) & (index < bins - 1)).to(dtype)
    mean = (probabilities * offsets).sum(-1)
    second = (probabilities * offsets**2).sum(-1)
    source = probabilities * interior
    mass = torch.clamp_min(source.sum(-1), tiny)
    shift = (target_mean - mean) / mass
    shift = torch.where(shift.abs() <= tolerance, 0.0, shift)
    shifted_second = shift.abs() * (source * (2.0 * torch.sign(shift)[..., None] * offsets
                                              + 1.0)).sum(-1)
    widen = torch.clamp_min((target_variance + target_mean**2 - second - shifted_second)
                            / (2.0 * mass), 0.0)
    widen = torch.where(widen <= tolerance, 0.0, widen)
    excess = torch.clamp_min(2.0 * widen + shift.abs(), 1.0)
    widen, shift = widen / excess, shift / excess
    up = torch.clamp_min(shift, 0.0)[..., None] * source
    down = torch.clamp_min(-shift, 0.0)[..., None] * source
    spread = widen[..., None] * source
    rising, falling = up + spread, down + spread
    return probabilities - (up + down + 2.0 * spread) \
        + F.pad(rising[..., :-1], (1, 0)) + F.pad(falling[..., 1:], (0, 1))


def scalar_encode(mean: torch.Tensor, sigma: torch.Tensor, low: float, high: float,
                  bins: int) -> torch.Tensor:
    """(...) mean and sigma to (..., bins): a sampled Gaussian with folded tails, mixed with
    the sharp code and moment-matched."""
    coordinate, lower, fraction, spacing = _scalar_coordinate(mean, low, high, bins)
    last = bins - 1
    variance = torch.minimum((sigma / spacing) ** 2,
                             torch.clamp_min(coordinate * (last - coordinate)
                                             - fraction * (1.0 - fraction), 0.0))
    offsets = torch.arange(bins, dtype=mean.dtype, device=mean.device) - lower[..., None]
    width = torch.clamp_min(variance, 1.0)[..., None]
    weight = torch.clamp_max(variance, 1.0)[..., None]

    def gaussian(distance):
        return torch.exp(-0.5 * distance**2 / width) / torch.sqrt(2.0 * math.pi * width)

    keep, take = (1.0 - fraction)[..., None], fraction[..., None]
    smooth = keep * gaussian(offsets) + take * gaussian(offsets - 1.0)
    root = torch.sqrt(2.0 * width)[..., 0]
    low_tail = 0.5 * ((1.0 - fraction) * torch.erfc((lower + 0.5) / root)
                      + fraction * torch.erfc((lower + 1.5) / root))
    high_tail = 0.5 * ((1.0 - fraction) * torch.erfc((last - lower + 0.5) / root)
                       + fraction * torch.erfc((last - lower - 0.5) / root))
    smooth = smooth + F.pad(low_tail[..., None], (0, bins - 1)) \
        + F.pad(high_tail[..., None], (bins - 1, 0))
    code = (1.0 - weight) * _two_hot_at(lower, fraction, bins) + weight * smooth
    code = code / code.sum(-1, keepdim=True)
    return match_moments(code, offsets, fraction, variance + fraction * (1.0 - fraction))


def scalar_decode(probabilities: torch.Tensor, low: float, high: float,
                  bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., bins) to (mean, spread)."""
    spacing = (high - low) / (bins - 1)
    index = torch.arange(bins, dtype=probabilities.dtype, device=probabilities.device)
    coordinate = torch.clamp((probabilities * index).sum(-1), 0.0, float(bins - 1))
    variance = (probabilities * (index - coordinate[..., None]) ** 2).sum(-1)
    fraction = coordinate - torch.floor(coordinate)
    physical = variance - fraction * (1.0 - fraction)
    physical = torch.where(physical <= _tolerance(probabilities.dtype), 0.0, physical)
    return low + coordinate * spacing, torch.sqrt(physical) * spacing


# --------------------------------------------------------------------------- velocity

SQRT3 = math.sqrt(3.0)


def compress_velocity(velocity: torch.Tensor, knee: float) -> torch.Tensor:
    """(..., 2): v * knee ln(1 + |v|/knee) / |v|, and 0 at 0."""
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    ratio = knee * torch.log1p(speed / knee) / torch.where(speed > 0, speed, 1.0)
    return velocity * torch.where(speed > 0, ratio, 0.0)


def expand_velocity(compressed: torch.Tensor, knee: float) -> torch.Tensor:
    """The inverse of `compress_velocity`."""
    size = torch.linalg.vector_norm(compressed, dim=-1, keepdim=True)
    ratio = knee * torch.expm1(size / knee) / torch.where(size > 0, size, 1.0)
    return compressed * torch.where(size > 0, ratio, 0.0)


def _packed_jacobian(velocity: torch.Tensor, radial: torch.Tensor,
                     tangential: torch.Tensor) -> torch.Tensor:
    """(xx, xy, yy) of T I + (R - T) v^ v^T."""
    eps = torch.finfo(velocity.dtype).eps
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    unit = velocity / torch.clamp_min(speed, eps)[..., None]
    difference = radial - tangential
    return torch.stack([tangential + difference * unit[..., 0] ** 2,
                        difference * unit[..., 0] * unit[..., 1],
                        tangential + difference * unit[..., 1] ** 2], -1)


def velocity_jacobian(velocity: torch.Tensor, knee: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed Jacobian of `compress_velocity` at `velocity`, and its inverse."""
    eps, tiny = torch.finfo(velocity.dtype).eps, torch.finfo(velocity.dtype).tiny
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    logarithm = knee * torch.log1p(speed / knee)
    at_zero = speed == 0
    forward = _packed_jacobian(velocity, 1.0 / (1.0 + speed / knee),
                               torch.where(at_zero, 1.0, logarithm / torch.clamp_min(speed, eps)))
    inverse = _packed_jacobian(velocity, 1.0 + speed / knee,
                               torch.where(at_zero, 1.0, speed / torch.clamp_min(logarithm, tiny)))
    return forward, inverse


def sandwich(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    """A S A for packed symmetric 2x2 matrices (xx, xy, yy)."""
    outer_xx, outer_xy, outer_yy = outer.unbind(-1)
    inner_xx, inner_xy, inner_yy = inner.unbind(-1)
    return torch.stack([outer_xx * outer_xx * inner_xx + 2.0 * outer_xx * outer_xy * inner_xy
                        + outer_xy * outer_xy * inner_yy,
                        outer_xx * outer_xy * inner_xx
                        + (outer_xx * outer_yy + outer_xy * outer_xy) * inner_xy
                        + outer_xy * outer_yy * inner_yy,
                        outer_xy * outer_xy * inner_xx + 2.0 * outer_xy * outer_yy * inner_xy
                        + outer_yy * outer_yy * inner_yy], -1)


def _project(vector: torch.Tensor) -> torch.Tensor:
    """(..., 2) onto the axes at 0, 120 and 240 degrees: (..., 3)."""
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x, -0.5 * x + 0.5 * SQRT3 * y, -0.5 * x - 0.5 * SQRT3 * y], -1)


def _axis_range(bins: int, spacing: float) -> tuple[float, float]:
    half = 0.5 * (bins - 1) * spacing
    return -half, half


def velocity_sharp(velocity: torch.Tensor, bins: int, spacing: float,
                   knee: float) -> torch.Tensor:
    """(..., 2) to (..., 3, bins)."""
    return scalar_sharp(_project(compress_velocity(velocity, knee)),
                        *_axis_range(bins, spacing), bins)


def velocity_encode(mean: torch.Tensor, covariance: torch.Tensor, bins: int, spacing: float,
                    knee: float) -> torch.Tensor:
    """(..., 2) mean and packed (..., 3) covariance to (..., 3, bins)."""
    jacobian, _ = velocity_jacobian(mean, knee)
    xx, xy, yy = sandwich(jacobian, covariance).unbind(-1)
    variances = torch.stack([xx, 0.25 * xx - 0.5 * SQRT3 * xy + 0.75 * yy,
                             0.25 * xx + 0.5 * SQRT3 * xy + 0.75 * yy], -1)
    return scalar_encode(_project(compress_velocity(mean, knee)),
                         torch.sqrt(torch.clamp_min(variances, 0.0)),
                         *_axis_range(bins, spacing), bins)


def velocity_decode(probabilities: torch.Tensor, bins: int, spacing: float,
                    knee: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 3, bins) to (mean (..., 2), packed covariance (..., 3))."""
    projections, spreads = scalar_decode(probabilities, *_axis_range(bins, spacing), bins)
    p0, p1, p2 = projections.unbind(-1)
    compressed = torch.stack([(2.0 / 3.0) * (p0 - 0.5 * (p1 + p2)),
                              (2.0 / 3.0) * (0.5 * SQRT3 * (p1 - p2))], -1)
    v0, v1, v2 = (spreads**2).unbind(-1)
    xx = torch.clamp_min(v0, 0.0)
    yy = torch.clamp_min((2.0 * (v1 + v2) - v0) / 3.0, 0.0)
    bound = torch.sqrt(xx * yy)
    xy = torch.clamp((v2 - v1) / SQRT3, -bound, bound)
    mean = expand_velocity(compressed, knee)
    _, inverse = velocity_jacobian(mean, knee)
    return mean, sandwich(inverse, torch.stack([xx, xy, yy], -1))


# --------------------------------------------------------------------------- grouped


Runs = tuple[tuple[int, int], ...]  # (groups, size) per run, in vector order


def _split_runs(code: torch.Tensor, runs: Runs) -> list[torch.Tensor]:
    """The code vector as one (..., groups, size) tensor per run."""
    parts = code.split([groups * size for groups, size in runs], -1)
    return [part.unflatten(-1, (groups, size)) for part, (groups, size) in zip(parts, runs)]


def grouped_log_softmax(logits: torch.Tensor, runs: Runs) -> torch.Tensor:
    return torch.cat([F.log_softmax(part, -1).flatten(-2)
                      for part in _split_runs(logits, runs)], -1)


def grouped_cross_entropy(logits: torch.Tensor, target: torch.Tensor, runs: Runs) -> torch.Tensor:
    """(..., groups in all): -sum target log p within each group."""
    return torch.cat([-(target_part * F.log_softmax(part, -1)).sum(-1)
                      for part, target_part in zip(_split_runs(logits, runs),
                                                   _split_runs(target, runs))], -1)


def remove_baseline_floor(logits: torch.Tensor, runs: Runs,
                          floor: float = BASELINE_FLOOR) -> torch.Tensor:
    """Group softmax, then undo the floor's mixture with uniform (clip at zero, renormalise)."""
    tiny = torch.finfo(logits.dtype).tiny
    out = []
    for part, (_, size) in zip(_split_runs(logits, runs), runs):
        probabilities = torch.clamp_min((1.0 + size * floor) * torch.softmax(part, -1) - floor, 0.0)
        out.append((probabilities / torch.clamp_min(probabilities.sum(-1, keepdim=True),
                                                    tiny)).flatten(-2))
    return torch.cat(out, -1)


def baseline(code: torch.Tensor, floor: float = BASELINE_FLOOR) -> torch.Tensor:
    """The next-state head's starting logits: "nothing changes"."""
    return torch.log(code + floor)
