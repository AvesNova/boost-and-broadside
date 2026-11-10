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
BROAD = 0.9  # attenuation below which a circular level has broadened (the spread is read there)
SPREAD_FITS = ("v1", "coarsest")  # how a circular code's spread is decoded (`circular_decode`)


def _tolerance(dtype: torch.dtype) -> float:
    return 8.0 * torch.finfo(dtype).eps


def _one_hot(index: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot by comparison (no scatter), so it fuses and exports."""
    return (index[..., None] == torch.arange(size, device=index.device)).to(dtype)


# --------------------------------------------------------------------------- circular


def circular_encode(
    mean: torch.Tensor, sigma: torch.Tensor, period: float, colours: int, levels: int
) -> torch.Tensor:
    """(...) mean and sigma to (..., levels, colours) probabilities.

    The spec's coordinate is (m mod P) / p_l * c. Only its fraction and its floor mod c are
    used, so each level reduces the position modulo its own period first (m mod p_l, by
    remainders level after level) and the coordinate stays below c. That changes nothing in
    exact arithmetic, but in float32 a coordinate near c^(l + 1) (about 2e4 at the position
    code's finest level) keeps only 1e-3 of a colour, and fused and unfused kernels round
    it differently. The remainders are exact for the code's integer-metre periods.
    """
    reduced, coordinates = mean, []
    for level in range(levels):
        level_period = period / colours**level
        reduced = torch.remainder(reduced, level_period)
        coordinates.append(reduced * (colours / level_period))
    position = torch.stack(coordinates, -1)
    level_periods = period / colours ** torch.arange(levels, dtype=mean.dtype, device=mean.device)
    lower = torch.floor(position)
    fraction = position - lower
    colour = torch.remainder(lower, colours).long()
    sharp = (1.0 - fraction)[..., None] * _one_hot(colour, colours, mean.dtype) + fraction[
        ..., None
    ] * _one_hot(torch.remainder(colour + 1, colours), colours, mean.dtype)
    attenuation = torch.exp(-0.5 * (2.0 * math.pi * sigma[..., None] / level_periods) ** 2)
    return sharp * attenuation[..., None] + ((1.0 - attenuation) / colours)[..., None]


def circular_decode(
    probabilities: torch.Tensor,
    period: float,
    colours: int,
    levels: int,
    spread_fit: str = "coarsest",
) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., levels, colours) probabilities to (mean, spread), each (...). spread_fit (one of
    SPREAD_FITS): "coarsest", the spread read at the coarsest broadened level; "v1", v1's
    fit (`v1_spread_variance`), for the models trained with it."""
    if spread_fit not in SPREAD_FITS:
        raise ValueError(f"spread_fit {spread_fit!r} is not one of {SPREAD_FITS}")
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
    chord = torch.sqrt(
        (1.0 - fraction + fraction * math.cos(step)) ** 2 + (fraction * math.sin(step)) ** 2
    )
    attenuation = torch.clamp_max(torch.sqrt(real * real + imaginary * imaginary) / chord, 1.0)
    level_periods = [period / colours**level for level in range(levels)]
    # Per-level constants built on the device (no host copy, so eager graph capture works).
    refinement = colours ** torch.arange(levels, dtype=dtype, device=device)  # period / p_l
    local_mean = (sector + fraction) * (period / refinement) / colours
    precision = torch.where(attenuation > INFORMATIVE, attenuation**2, 0.0) * refinement**2

    mean, total = local_mean[..., 0], precision[..., 0]
    for level in range(1, levels):
        level_period = level_periods[level]
        unwrapped = (
            local_mean[..., level]
            + torch.round((mean - local_mean[..., level]) / level_period) * level_period
        )
        total = total + precision[..., level]
        mean = mean + precision[..., level] / torch.clamp_min(total, tiny) * (unwrapped - mean)
    mean = torch.remainder(mean, period)

    attenuation = torch.where(attenuation > 1.0 - 1e-6, 1.0, attenuation)
    informative = attenuation > INFORMATIVE
    curvature = 0.5 * (2.0 * math.pi * refinement / period) ** 2
    # Each level implies a variance, -ln a_l / kappa_l (a flat level: at least its floor's).
    # A Gaussian code implies the same one at every level. Any other code is read at the
    # coarsest level that has broadened (a < BROAD): its low frequency sees the whole
    # distribution, where finer levels see only its fine structure and would let a sharp
    # core or a stray peak decide.
    if spread_fit == "v1":
        variance = v1_spread_variance(attenuation, informative, curvature)
    else:
        implied = -torch.log(torch.clamp_min(attenuation, INFORMATIVE)) / curvature
        broad = attenuation < BROAD
        first = torch.where(broad.any(-1), torch.argmax(broad.to(torch.int32), -1), levels - 1)
        variance = torch.gather(implied, -1, first[..., None])[..., 0]
    two_hot = ((probabilities > 0).sum(-1) <= 2).all(-1)
    variance = torch.where(two_hot, 0.0, variance)
    spread = torch.where(informative.any(-1), torch.sqrt(variance), period)
    return mean, spread


def fourier_encode(
    mean: torch.Tensor, sigma: torch.Tensor, period: float, refinement: int, levels: int
) -> torch.Tensor:
    """(...) mean and sigma to (..., levels, 2): per level l of period p_l = period /
    refinement^l, a_l (cos, sin) of 2 pi m / p_l, a_l the circular code's attenuation
    exp(-(2 pi sigma / p_l)^2 / 2), so a blurred level fades toward 0 as the circular
    code's fades toward uniform. The phase reduces m modulo each level's period first, by
    remainders level after level as `circular_encode` does, so float32 keeps the finest
    level's phase."""
    reduced, phases = mean, []
    for level in range(levels):
        level_period = period / refinement**level
        reduced = torch.remainder(reduced, level_period)
        phases.append(reduced * (2.0 * math.pi / level_period))
    phase = torch.stack(phases, -1)
    level_periods = period / refinement ** torch.arange(
        levels, dtype=mean.dtype, device=mean.device
    )
    attenuation = torch.exp(-0.5 * (2.0 * math.pi * sigma[..., None] / level_periods) ** 2)
    return attenuation[..., None] * torch.stack([torch.cos(phase), torch.sin(phase)], -1)


def v1_spread_variance(
    attenuation: torch.Tensor, informative: torch.Tensor, curvature: torch.Tensor
) -> torch.Tensor:
    """v1's spread fit (spec 3.3; `spread_fit="v1"`, kept while a rated policy was trained
    with it): the levels' implied variances weighted by a^2 kappa^2, so the finest
    informative level decides. Per level (..., levels) in, (...) out."""
    tiny = torch.finfo(attenuation.dtype).tiny
    weight = informative.to(attenuation.dtype) * attenuation**2
    variance = -(weight * curvature * torch.log(torch.clamp_min(attenuation, INFORMATIVE))).sum(-1)
    return torch.clamp_min(variance / torch.clamp_min((weight * curvature**2).sum(-1), tiny), 0.0)


# --------------------------------------------------------------------------- position


def position_encode(
    mean: torch.Tensor, sigma: torch.Tensor, periods: tuple[float, float], levels: int
) -> torch.Tensor:
    """(..., 2) mean and sigma (...) (one for both axes) or (..., 2) (per axis) to
    (..., levels, 9): the outer product of a 3-colour circular code per axis, x-major."""
    if sigma.dim() != mean.dim():
        sigma = sigma[..., None].expand(mean.shape)
    if periods[0] == periods[1]:  # both axes in one call (one copy of the level loop)
        x, y = circular_encode(mean, sigma, periods[0], 3, levels).unbind(-3)
    else:
        x = circular_encode(mean[..., 0], sigma[..., 0], periods[0], 3, levels)
        y = circular_encode(mean[..., 1], sigma[..., 1], periods[1], 3, levels)
    return (x[..., :, None] * y[..., None, :]).flatten(-2)


def position_decode(
    probabilities: torch.Tensor,
    periods: tuple[float, float],
    levels: int,
    spread_fit: str = "coarsest",
) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., levels, 9) to (mean (..., 2), spread (...)); spread_fit as `circular_decode`."""
    joint = probabilities.unflatten(-1, (3, 3))
    if periods[0] == periods[1]:  # both axes in one call (one copy of the level loop)
        axes = torch.stack([joint.sum(-1), joint.sum(-2)], -3)
        mean, spread = circular_decode(axes, periods[0], 3, levels, spread_fit)
        return mean, torch.sqrt(0.5 * (spread**2).sum(-1))
    mean_x, spread_x = circular_decode(joint.sum(-1), periods[0], 3, levels, spread_fit)
    mean_y, spread_y = circular_decode(joint.sum(-2), periods[1], 3, levels, spread_fit)
    return torch.stack([mean_x, mean_y], -1), torch.sqrt(0.5 * (spread_x**2 + spread_y**2))


# --------------------------------------------------------------------------- scalar


# a scalar code's range end: a number, or a tensor over a scalar axis (several codes at once)
Bound = float | torch.Tensor


def _scalar_coordinate(values: torch.Tensor, low: Bound, high: Bound, bins: int):
    spacing = (high - low) / (bins - 1)
    coordinate = torch.clamp((values - low) / spacing, 0.0, float(bins - 1))
    lower = torch.clamp_max(torch.floor(coordinate), float(bins - 2))
    return coordinate, lower, coordinate - lower, spacing


def two_hot_at(lower: torch.Tensor, fraction: torch.Tensor, bins: int) -> torch.Tensor:
    """(...) to (..., bins): weight 1 - fraction on bin `lower` and fraction on the next."""
    lower = lower.long()
    return (1.0 - fraction)[..., None] * _one_hot(lower, bins, fraction.dtype) + fraction[
        ..., None
    ] * _one_hot(lower + 1, bins, fraction.dtype)


def scalar_sharp(values: torch.Tensor, low: Bound, high: Bound, bins: int) -> torch.Tensor:
    """(...) to (..., bins): two-hot on the bins either side (low and high: numbers, or
    tensors that broadcast against the values, one code call for several scalars)."""
    _, lower, fraction, _ = _scalar_coordinate(values, low, high, bins)
    return two_hot_at(lower, fraction, bins)


def match_moments(
    probabilities: torch.Tensor,
    offsets: torch.Tensor,
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
) -> torch.Tensor:
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
    shifted_second = shift.abs() * (
        source * (2.0 * torch.sign(shift)[..., None] * offsets + 1.0)
    ).sum(-1)
    widen = torch.clamp_min(
        (target_variance + target_mean**2 - second - shifted_second) / (2.0 * mass), 0.0
    )
    widen = torch.where(widen <= tolerance, 0.0, widen)
    excess = torch.clamp_min(2.0 * widen + shift.abs(), 1.0)
    widen, shift = widen / excess, shift / excess
    up = torch.clamp_min(shift, 0.0)[..., None] * source
    down = torch.clamp_min(-shift, 0.0)[..., None] * source
    spread = widen[..., None] * source
    rising, falling = up + spread, down + spread
    return (
        probabilities
        - (up + down + 2.0 * spread)
        + F.pad(rising[..., :-1], (1, 0))
        + F.pad(falling[..., 1:], (0, 1))
    )


def scalar_encode(
    mean: torch.Tensor, sigma: torch.Tensor, low: Bound, high: Bound, bins: int
) -> torch.Tensor:
    """(...) mean and sigma to (..., bins) (low and high: numbers, or tensors that broadcast
    against the values, one code call for several scalars): a sampled Gaussian with folded
    tails, mixed with
    the sharp code and moment-matched."""
    coordinate, lower, fraction, spacing = _scalar_coordinate(mean, low, high, bins)
    last = bins - 1
    variance = torch.minimum(
        (sigma / spacing) ** 2,
        torch.clamp_min(coordinate * (last - coordinate) - fraction * (1.0 - fraction), 0.0),
    )
    offsets = torch.arange(bins, dtype=mean.dtype, device=mean.device) - lower[..., None]
    width = torch.clamp_min(variance, 1.0)[..., None]
    weight = torch.clamp_max(variance, 1.0)[..., None]

    def gaussian(distance):
        return torch.exp(-0.5 * distance**2 / width) / torch.sqrt(2.0 * math.pi * width)

    keep, take = (1.0 - fraction)[..., None], fraction[..., None]
    smooth = keep * gaussian(offsets) + take * gaussian(offsets - 1.0)
    root = torch.sqrt(2.0 * width)[..., 0]
    low_tail = 0.5 * (
        (1.0 - fraction) * torch.erfc((lower + 0.5) / root)
        + fraction * torch.erfc((lower + 1.5) / root)
    )
    high_tail = 0.5 * (
        (1.0 - fraction) * torch.erfc((last - lower + 0.5) / root)
        + fraction * torch.erfc((last - lower - 0.5) / root)
    )
    smooth = (
        smooth
        + F.pad(low_tail[..., None], (0, bins - 1))
        + F.pad(high_tail[..., None], (bins - 1, 0))
    )
    code = (1.0 - weight) * two_hot_at(lower, fraction, bins) + weight * smooth
    code = code / code.sum(-1, keepdim=True)
    return match_moments(code, offsets, fraction, variance + fraction * (1.0 - fraction))


def scalar_decode(
    probabilities: torch.Tensor, low: Bound, high: Bound, bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _packed_jacobian(
    velocity: torch.Tensor, radial: torch.Tensor, tangential: torch.Tensor
) -> torch.Tensor:
    """(xx, xy, yy) of T I + (R - T) v^ v^T."""
    eps = torch.finfo(velocity.dtype).eps
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    unit = velocity / torch.clamp_min(speed, eps)[..., None]
    difference = radial - tangential
    return torch.stack(
        [
            tangential + difference * unit[..., 0] ** 2,
            difference * unit[..., 0] * unit[..., 1],
            tangential + difference * unit[..., 1] ** 2,
        ],
        -1,
    )


def velocity_jacobian(velocity: torch.Tensor, knee: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed Jacobian of `compress_velocity` at `velocity`, and its inverse."""
    eps, tiny = torch.finfo(velocity.dtype).eps, torch.finfo(velocity.dtype).tiny
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    logarithm = knee * torch.log1p(speed / knee)
    at_zero = speed == 0
    forward = _packed_jacobian(
        velocity,
        1.0 / (1.0 + speed / knee),
        torch.where(at_zero, 1.0, logarithm / torch.clamp_min(speed, eps)),
    )
    inverse = _packed_jacobian(
        velocity,
        1.0 + speed / knee,
        torch.where(at_zero, 1.0, speed / torch.clamp_min(logarithm, tiny)),
    )
    return forward, inverse


def sandwich(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    """A S A for packed symmetric 2x2 matrices (xx, xy, yy)."""
    outer_xx, outer_xy, outer_yy = outer.unbind(-1)
    inner_xx, inner_xy, inner_yy = inner.unbind(-1)
    return torch.stack(
        [
            outer_xx * outer_xx * inner_xx
            + 2.0 * outer_xx * outer_xy * inner_xy
            + outer_xy * outer_xy * inner_yy,
            outer_xx * outer_xy * inner_xx
            + (outer_xx * outer_yy + outer_xy * outer_xy) * inner_xy
            + outer_xy * outer_yy * inner_yy,
            outer_xy * outer_xy * inner_xx
            + 2.0 * outer_xy * outer_yy * inner_xy
            + outer_yy * outer_yy * inner_yy,
        ],
        -1,
    )


def _project(vector: torch.Tensor) -> torch.Tensor:
    """(..., 2) onto the axes at 0, 120 and 240 degrees: (..., 3)."""
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x, -0.5 * x + 0.5 * SQRT3 * y, -0.5 * x - 0.5 * SQRT3 * y], -1)


def _axis_range(bins: int, spacing: float) -> tuple[float, float]:
    half = 0.5 * (bins - 1) * spacing
    return -half, half


def velocity_encode(
    mean: torch.Tensor, covariance: torch.Tensor, bins: int, spacing: float, knee: float
) -> torch.Tensor:
    """(..., 2) mean and packed (..., 3) covariance to (..., 3, bins)."""
    jacobian, _ = velocity_jacobian(mean, knee)
    xx, xy, yy = sandwich(jacobian, covariance).unbind(-1)
    variances = torch.stack(
        [xx, 0.25 * xx - 0.5 * SQRT3 * xy + 0.75 * yy, 0.25 * xx + 0.5 * SQRT3 * xy + 0.75 * yy], -1
    )
    return scalar_encode(
        _project(compress_velocity(mean, knee)),
        torch.sqrt(torch.clamp_min(variances, 0.0)),
        *_axis_range(bins, spacing),
        bins,
    )


def velocity_decode(
    probabilities: torch.Tensor, bins: int, spacing: float, knee: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 3, bins) to (mean (..., 2), packed covariance (..., 3))."""
    projections, spreads = scalar_decode(probabilities, *_axis_range(bins, spacing), bins)
    p0, p1, p2 = projections.unbind(-1)
    compressed = torch.stack(
        [(2.0 / 3.0) * (p0 - 0.5 * (p1 + p2)), (2.0 / 3.0) * (0.5 * SQRT3 * (p1 - p2))], -1
    )
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


def split_runs(code: torch.Tensor, runs: Runs) -> list[torch.Tensor]:
    """The code vector as one (..., groups, size) tensor per run."""
    parts = code.split([groups * size for groups, size in runs], -1)
    return [part.unflatten(-1, (groups, size)) for part, (groups, size) in zip(parts, runs)]


def grouped_log_softmax(logits: torch.Tensor, runs: Runs) -> torch.Tensor:
    """Log-probabilities in the code's layout (..., width): a log-softmax over each group of each
    run."""
    return torch.cat([F.log_softmax(part, -1).flatten(-2) for part in split_runs(logits, runs)], -1)


def grouped_cross_entropy(logits: torch.Tensor, target: torch.Tensor, runs: Runs) -> torch.Tensor:
    """(..., groups in all): -sum target log p within each group."""
    return torch.cat(
        [
            -(target_part * F.log_softmax(part, -1)).sum(-1)
            for part, target_part in zip(split_runs(logits, runs), split_runs(target, runs))
        ],
        -1,
    )


def remove_baseline_floor(
    logits: torch.Tensor, runs: Runs, floor: float = BASELINE_FLOOR
) -> torch.Tensor:
    """Group softmax, then undo the floor's mixture with uniform (clip at zero, renormalise)."""
    tiny = torch.finfo(logits.dtype).tiny
    out = []
    for part, (_, size) in zip(split_runs(logits, runs), runs):
        probabilities = torch.clamp_min((1.0 + size * floor) * torch.softmax(part, -1) - floor, 0.0)
        out.append(
            (probabilities / torch.clamp_min(probabilities.sum(-1, keepdim=True), tiny)).flatten(-2)
        )
    return torch.cat(out, -1)


def baseline(code: torch.Tensor, floor: float = BASELINE_FLOOR) -> torch.Tensor:
    """The next-state head's starting logits: "nothing changes"."""
    return torch.log(code + floor)


# --------------------------------------------------------------------------- symlog HL-Gauss

HL_GAUSS_SIGMA = 0.75  # the symlog code's Gaussian, in bins
PADDING_BINS = 2  # bins past the real range on each side, for the Gaussian's spill


def symlog_sides(bins: int, below: int | None = None) -> tuple[int, int]:
    """(real bins below the zero centre, real bins above it) of a code of `bins` real bins:
    symmetric (bins odd) when below is None."""
    below = (bins - 1) // 2 if below is None else below
    return below, bins - 1 - below


def symlog_spacing(knee: float, bound: float, bins: int, below: int | None = None) -> float:
    """The bin spacing in symlog units, ln(1 + |x| / knee), of a code of `bins` real bins
    with an exact zero at a centre and the last centre above it at bound; `below` of them
    under the zero (None: as many as above, so the centres run from -bound to bound)."""
    _, above = symlog_sides(bins, below)
    return math.log1p(bound / knee) / above


def symlog_coordinate(values: torch.Tensor, knee: Bound, spacing: Bound) -> torch.Tensor:
    """values in bins from the zero centre: sign(x) ln(1 + |x| / knee) / spacing. knee and
    spacing: numbers, or tensors over a trailing component axis (several codes at once)."""
    return torch.sign(values) * torch.log1p(values.abs() / knee) / spacing


def symlog_value(coordinate: torch.Tensor, knee: Bound, spacing: Bound) -> torch.Tensor:
    """The inverse of `symlog_coordinate`."""
    return torch.sign(coordinate) * knee * torch.expm1(coordinate.abs() * spacing)


def symlog_smoothing(
    knee: Bound, spacing: Bound, low: int, high: int, sigma: float, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """((..., width) each bin centre's physical value, (..., width) the physical variance of
    the code of a value at that centre about it): the code's own smoothing, which
    `symlog_decode_at` takes off. knee, spacing: numbers, or (k, 1) tensors for k codes at
    once; like: the dtype and device."""
    index = torch.arange(-low, high + 1, dtype=like.dtype, device=like.device)
    centres = symlog_value(index, knee, spacing)
    own = hl_gauss(index, low, high, sigma)  # (width, width): each centre's own code
    smoothing = (own * (centres[..., None, :] - centres[..., :, None]) ** 2).sum(-1)
    return centres, smoothing


def hl_gauss(coordinate: torch.Tensor, low: int, high: int, sigma: float) -> torch.Tensor:
    """The Gaussian of `sigma` bins at coordinate (...) integrated over the bins centred at
    -low..high, divided by its sum: (..., low + high + 1)."""
    edges = torch.arange(-low, high + 2, dtype=coordinate.dtype, device=coordinate.device) - 0.5
    cumulative = torch.erf((edges - coordinate[..., None]) / (sigma * math.sqrt(2.0)))
    probabilities = 0.5 * (cumulative[..., 1:] - cumulative[..., :-1])
    return probabilities / probabilities.sum(-1, keepdim=True)


def symlog_encode(
    values: torch.Tensor,
    knee: float,
    bound: float,
    bins: int,
    padding: int = PADDING_BINS,
    sigma: float = HL_GAUSS_SIGMA,
    below: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(...) to ((..., bins + 2 padding) HL-Gauss probabilities, (...) clipped): the value's
    symlog coordinate clamped to the padded range (the outermost padding bin's centre;
    `clipped` marks the values that needed it), then the Gaussian of `sigma` bins integrated
    over each bin. Nothing folds: the mass past the outer edges is dropped and the code
    divided by its sum. below: the real bins under the zero (`symlog_spacing`)."""
    spacing = symlog_spacing(knee, bound, bins, below)
    under, over = symlog_sides(bins, below)
    return symlog_encode_at(values, under + padding, over + padding, knee, spacing, sigma)


def symlog_encode_at(
    values: torch.Tensor, low: int, high: int, knee: Bound, spacing: Bound, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`symlog_encode` of a code with `low` bins under its zero centre and `high` above it
    (padding included) at a given spacing: knee and spacing numbers, or tensors (k,) over
    values' last axis, k codes of one layout at once ((..., k, width), (..., k))."""
    coordinate = symlog_coordinate(values, knee, spacing)
    clipped = (coordinate < -low) | (coordinate > high)
    coordinate = torch.clamp(coordinate, -float(low), float(high))
    return hl_gauss(coordinate, low, high, sigma), clipped


INVERSE_STEPS = 64  # the mean inverse's grid steps per bin (`symlog_mean_inverse`)


def symlog_mean_inverse(
    bins: int,
    padding: int = PADDING_BINS,
    sigma: float = HL_GAUSS_SIGMA,
    below: int | None = None,
) -> tuple[torch.Tensor, float, float]:
    """(table (steps + 1,) float64, lowest, highest): the coordinate of the encoded value
    whose code has a given mean bin index, on a uniform grid of that mean from `lowest` (the
    code of the lowest padding centre's) to `highest`, INVERSE_STEPS per bin. Inside the
    code an encoded value's mean index is its coordinate to about 1e-5 bins; within a few
    bins of the padded ends the code loses the Gaussian's spill and its mean moves inward
    (by up to 0.3 bins at the ends): the table undoes that (`symlog_decode`)."""
    under, over = symlog_sides(bins, below)
    low, high = under + padding, over + padding
    coordinate = torch.linspace(
        -low, high, (low + high) * 16 * INVERSE_STEPS + 1, dtype=torch.float64
    )
    index = torch.arange(-low, high + 1, dtype=torch.float64)
    mean = (hl_gauss(coordinate, low, high, sigma) * index).sum(-1)  # strictly increasing
    lowest, highest = float(mean[0]), float(mean[-1])
    grid = torch.linspace(lowest, highest, (low + high) * INVERSE_STEPS + 1, dtype=torch.float64)
    right = torch.searchsorted(mean, grid).clamp(1, mean.numel() - 1)
    left = right - 1
    weight = ((grid - mean[left]) / (mean[right] - mean[left])).clamp(0.0, 1.0)
    table = coordinate[left] + weight * (coordinate[right] - coordinate[left])
    table[0], table[-1] = -low, high
    return table, lowest, highest


def invert_mean(
    mean: torch.Tensor, table: torch.Tensor, lowest: float, highest: float
) -> torch.Tensor:
    """The coordinate (...) of a mean bin index (...) by a `symlog_mean_inverse` table
    (linear between its steps; the ends clamp)."""
    steps = table.shape[0] - 1
    position = torch.clamp((mean - lowest) * (steps / (highest - lowest)), 0.0, float(steps))
    lower = torch.clamp_max(torch.floor(position), float(steps - 1))
    fraction = position - lower
    lower = lower.long()
    table = table.to(mean.dtype)
    return table[lower] + fraction * (table[lower + 1] - table[lower])


def symlog_decode(
    probabilities: torch.Tensor,
    knee: float,
    bound: float,
    bins: int,
    padding: int = PADDING_BINS,
    sigma: float = HL_GAUSS_SIGMA,
    below: int | None = None,
    inverse: tuple[torch.Tensor, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., bins + 2 padding) probabilities to (mean, variance), each (...), physical. The
    mean is the value at the mean coordinate: exact on an encoded value inside the code,
    and at the padded ends too given `inverse` (`symlog_mean_inverse` of the same code).
    The variance is the bins' physical variance about it less the code's own smoothing,
    each bin's mass times the smoothing of a value at its centre (by total variance: a
    mixture of encoded values decodes near their spread, an encoded value near 0), at
    least 0."""
    spacing = symlog_spacing(knee, bound, bins, below)
    under, over = symlog_sides(bins, below)
    low, high = under + padding, over + padding
    centres, smoothing = symlog_smoothing(knee, spacing, low, high, sigma, probabilities)
    return symlog_decode_at(probabilities, low, knee, spacing, centres, smoothing, inverse)


def symlog_decode_at(
    probabilities: torch.Tensor,
    low: int,
    knee: Bound,
    spacing: Bound,
    centres: torch.Tensor,
    smoothing: torch.Tensor,
    inverse: tuple[torch.Tensor, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`symlog_decode` of a code with `low` bins under its zero centre (padding included),
    given its `symlog_smoothing`: knee and spacing numbers with centres and smoothing
    (width,), or for k codes of one layout at once (probabilities (..., k, width)) tensors
    (k,) and (k, width)."""
    width = probabilities.shape[-1]
    index = torch.arange(-low, width - low, dtype=probabilities.dtype, device=probabilities.device)
    coordinate = (probabilities * index).sum(-1)
    if inverse is not None:
        coordinate = invert_mean(coordinate, *inverse)
    mean = symlog_value(coordinate, knee, spacing)
    spread = (probabilities * (centres - mean[..., None]) ** 2).sum(-1)
    return mean, torch.clamp_min(spread - (probabilities * smoothing).sum(-1), 0.0)


MEDIAN_WINDOW = 4  # bins either side of the median's bin that `symlog_median` averages


def symlog_median(
    probabilities: torch.Tensor,
    knee: float,
    bound: float,
    bins: int,
    padding: int = PADDING_BINS,
    below: int | None = None,
    inverse: tuple[torch.Tensor, float, float] | None = None,
    window: int = MEDIAN_WINDOW,
) -> torch.Tensor:
    """(..., bins + 2 padding) probabilities to the median value (...): the bin where the
    cumulative mass reaches 1/2, then the mean coordinate over the bins within `window` of
    it (`inverse` as in `symlog_decode`). An encoded value decodes to itself (its
    Gaussian lies inside the window to 5 sigma); a code with two well-separated modes
    decodes to the value of the mode that holds the median, so a jump of probability below
    1/2 is not taken (the most likely outcome, decisions D8)."""
    spacing = symlog_spacing(knee, bound, bins, below)
    low = symlog_sides(bins, below)[0] + padding
    return symlog_median_at(probabilities, low, knee, spacing, inverse, window)


def symlog_median_at(
    probabilities: torch.Tensor,
    low: int,
    knee: Bound,
    spacing: Bound,
    inverse: tuple[torch.Tensor, float, float] | None = None,
    window: int = MEDIAN_WINDOW,
) -> torch.Tensor:
    """`symlog_median` of a code with `low` bins under its zero centre (padding included):
    knee and spacing numbers, or tensors (k,) for k codes of one layout at once
    (probabilities (..., k, width))."""
    width = probabilities.shape[-1]
    index = torch.arange(width, device=probabilities.device)
    cumulative = torch.cumsum(probabilities, -1)
    crossing = torch.clamp_max((cumulative < 0.5).sum(-1), width - 1)
    near = ((index - crossing[..., None]).abs() <= window).to(probabilities.dtype)
    weight = probabilities * near
    position = (index - low).to(probabilities.dtype)
    tiny = torch.finfo(probabilities.dtype).tiny
    coordinate = (weight * position).sum(-1) / torch.clamp_min(weight.sum(-1), tiny)
    if inverse is not None:
        coordinate = invert_mean(coordinate, *inverse)
    return symlog_value(coordinate, knee, spacing)


SPREAD_LEVELS = (0.16, 0.5, 0.84)  # a Gaussian's median and one sigma either side
QUANTILE_FLOOR = 1e-6  # the cumulative mass held off 0 and 1 for its probit (float32-safe)


def _probit(cumulative: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * cumulative - 1.0)


def symlog_quantile(probabilities: torch.Tensor, low: int, level: float) -> torch.Tensor:
    """(..., width) probabilities to the coordinate (...) of their `level` quantile, in bins
    from the zero centre (low: the bins under it, padding included): the bin whose edges'
    cumulative masses straddle the level, then the cumulative mass interpolated across it
    linearly in its probit. A Gaussian's probit is linear, so the code of an encoded value
    (the HL-Gauss of `sigma` bins) gives its exact quantiles, and a mixture the quantiles of
    its smoothed distribution up to the curvature inside one bin."""
    floor = QUANTILE_FLOOR
    after = torch.cumsum(probabilities, -1)
    before = torch.cat([torch.zeros_like(after[..., :1]), after[..., :-1]], -1)
    width = probabilities.shape[-1]
    crossing = torch.clamp_max((after < level).sum(-1), width - 1)[..., None]
    lower = _probit(before.gather(-1, crossing)[..., 0].clamp(floor, 1.0 - floor))
    upper = _probit(after.gather(-1, crossing)[..., 0].clamp(floor, 1.0 - floor))
    target = _probit(after.new_full((), level))
    rise = upper - lower
    fraction = torch.where(
        rise > 0, (target - lower) / torch.where(rise > 0, rise, 1.0), 0.5
    ).clamp(0.0, 1.0)
    return (crossing[..., 0] - low).to(probabilities.dtype) - 0.5 + fraction


def symlog_spread(
    probabilities: torch.Tensor,
    knee: float,
    bound: float,
    bins: int,
    padding: int = PADDING_BINS,
    sigma: float = HL_GAUSS_SIGMA,
    below: int | None = None,
) -> torch.Tensor:
    """(..., bins + 2 padding) probabilities to a robust physical sigma (...): the 16th,
    50th and 84th percentiles c16, c50, c84 in the code's coordinate (`symlog_quantile`),
    each side's half-width less the code's own smoothing in quadrature, a = sqrt(max((c84 -
    c50)^2 - sigma^2, 0)) and b likewise below, and sigma = (v(c50 + a) - v(c50 - b)) / 2, v
    the code's value map. An encoded value decodes to 0; a Gaussian to its sigma within a
    few per cent across the knee (`docs/explanation/yemong.md`); a sliver of mass in a far
    bin moves a quantile by about its mass, not by its distance squared (the second
    moment's failure on a wide code). The sides apart: a symlog code folds near 0, so a
    distribution off the centre is lopsided in the coordinate."""
    spacing = symlog_spacing(knee, bound, bins, below)
    low = symlog_sides(bins, below)[0] + padding
    c16, c50, c84 = (symlog_quantile(probabilities, low, level) for level in SPREAD_LEVELS)

    def deconvolved(half: torch.Tensor) -> torch.Tensor:
        excess = half * half - sigma * sigma
        positive = excess > 0
        return torch.where(positive, torch.sqrt(torch.where(positive, excess, 1.0)), 0.0)

    upper = symlog_value(c50 + deconvolved(c84 - c50), knee, spacing)
    lower = symlog_value(c50 - deconvolved(c50 - c16), knee, spacing)
    return 0.5 * (upper - lower)
