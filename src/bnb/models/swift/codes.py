"""Categorical codes (spec 3.3): circular, position, scalar, velocity and grouped softmax codes.

Every function is vectorised over leading axes, works in any floating dtype, and is written
without data-dependent control flow so that `torch.compile(fullgraph=True)` can fuse it.
`eps` and `tiny` are the dtype's `torch.finfo` values.
"""

import math

import torch

TWO_PI = 2.0 * math.pi
SQRT3 = math.sqrt(3.0)


def machine_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


def smallest_normal(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).tiny


def level_periods(period: float, colours: int, levels: int, like: torch.Tensor) -> torch.Tensor:
    """(levels,) the level periods p_l = P / c^l."""
    return torch.tensor([period / colours**level for level in range(levels)], dtype=like.dtype,
                        device=like.device)


# --------------------------------------------------------------------------- circular


def circular_encode(period: float, colours: int, levels: int, mean: torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
    """mean, sigma (...) in the period's units -> probabilities (..., levels, colours)."""
    periods = level_periods(period, colours, levels, mean)
    position = torch.remainder(mean, period).unsqueeze(-1) / periods * colours  # (..., L)
    lower = torch.floor(position)
    fraction = (position - lower).unsqueeze(-1)
    colour = torch.remainder(lower, colours).unsqueeze(-1)
    index = torch.arange(colours, dtype=mean.dtype, device=mean.device)
    upper = torch.remainder(colour + 1, colours)
    sharp = (1 - fraction) * (index == colour) + fraction * (index == upper)
    attenuation = torch.exp(-0.5 * (TWO_PI * sigma.unsqueeze(-1) / periods) ** 2).unsqueeze(-1)
    return sharp * attenuation + (1 - attenuation) / colours


def circular_levels(period: float, colours: int, levels: int, probabilities: torch.Tensor):
    """Per-level decode: (local mean, attenuation, precision), each (..., levels)."""
    eps = machine_epsilon(probabilities)
    periods = level_periods(period, colours, levels, probabilities)
    theta = TWO_PI / colours
    angles = torch.arange(colours, dtype=probabilities.dtype, device=probabilities.device) * theta
    real = (probabilities * torch.cos(angles)).sum(-1)
    imaginary = (probabilities * torch.sin(angles)).sum(-1)
    phase = torch.remainder(torch.atan2(imaginary, real), TWO_PI)
    sector = torch.clamp_max(torch.floor(phase / theta), colours - 1)
    remainder = phase - sector * theta
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    sin_remainder = torch.sin(remainder)
    denominator = torch.clamp_min(sin_theta * torch.cos(remainder) + (1 - cos_theta) * sin_remainder,
                                  eps)
    fraction = torch.clamp(sin_remainder / denominator, 0, 1)
    chord = torch.sqrt((1 - fraction + fraction * cos_theta) ** 2 + (fraction * sin_theta) ** 2)
    attenuation = torch.clamp_max(torch.sqrt(real**2 + imaginary**2) / chord, 1)
    local_mean = (sector + fraction) * periods / colours
    precision = torch.where(attenuation > 1e-4, attenuation**2 * (period / periods) ** 2, 0.0)
    return local_mean, attenuation, precision, periods


def circular_decode(period: float, colours: int, levels: int, probabilities: torch.Tensor):
    """probabilities (..., levels, colours) -> (mean in [0, P), spread), each (...)."""
    tiny = smallest_normal(probabilities)
    local_mean, attenuation, precision, periods = circular_levels(period, colours, levels,
                                                                  probabilities)
    mean = local_mean[..., 0]
    total = precision[..., 0]
    for level in range(1, levels):
        level_period = period / colours**level
        level_mean = local_mean[..., level]
        unwrapped = level_mean + torch.round((mean - level_mean) / level_period) * level_period
        total = total + precision[..., level]
        mean = mean + precision[..., level] / torch.clamp_min(total, tiny) * (unwrapped - mean)
    mean = torch.remainder(mean, period)

    attenuation = torch.where(attenuation > 1 - 1e-6, 1.0, attenuation)
    informative = attenuation > 1e-4
    curvature = 0.5 * (TWO_PI / periods) ** 2
    weight = informative * attenuation**2
    numerator = -(weight * curvature * torch.log(torch.clamp_min(attenuation, 1e-4))).sum(-1)
    denominator = torch.clamp_min((weight * curvature**2).sum(-1), tiny)
    variance = torch.clamp_min(numerator / denominator, 0)
    two_hot = ((probabilities != 0).sum(-1) <= 2).all(-1)
    variance = torch.where(two_hot, 0.0, variance)
    spread = torch.where(informative.any(-1), torch.sqrt(variance), period)
    return mean, spread


# --------------------------------------------------------------------------- position


def position_encode(size, levels: int, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """mean (..., 2), sigma (...) -> (..., levels, 9): the x-major outer product of 3-colour
    circular codes, entry 3 i + j for x colour i and y colour j."""
    code_x = circular_encode(size[0], 3, levels, mean[..., 0], sigma)
    code_y = circular_encode(size[1], 3, levels, mean[..., 1], sigma)
    return (code_x.unsqueeze(-1) * code_y.unsqueeze(-2)).flatten(-2)


def position_decode(size, levels: int, probabilities: torch.Tensor):
    """(..., levels, 9) -> (mean (..., 2) in [0, P), spread (...))."""
    joint = probabilities.unflatten(-1, (3, 3))
    mean_x, spread_x = circular_decode(size[0], 3, levels, joint.sum(-1))
    mean_y, spread_y = circular_decode(size[1], 3, levels, joint.sum(-2))
    return torch.stack([mean_x, mean_y], -1), torch.sqrt((spread_x**2 + spread_y**2) / 2)


# --------------------------------------------------------------------------- scalar


def scalar_coordinate(low: float, high: float, bins: int, values: torch.Tensor):
    """(k, lower bin b, t) for values (...), each (...)."""
    spacing = (high - low) / (bins - 1)
    coordinate = torch.clamp((values - low) / spacing, 0, bins - 1)
    lower = torch.clamp_max(torch.floor(coordinate), bins - 2)
    return coordinate, lower, coordinate - lower


def scalar_sharp(low: float, high: float, bins: int, values: torch.Tensor) -> torch.Tensor:
    """values (...) -> the two-hot (..., bins)."""
    _, lower, fraction = scalar_coordinate(low, high, bins, values)
    index = torch.arange(bins, dtype=values.dtype, device=values.device)
    lower, fraction = lower.unsqueeze(-1), fraction.unsqueeze(-1)
    return (1 - fraction) * (index == lower) + fraction * (index == lower + 1)


def shift_up(values: torch.Tensor) -> torch.Tensor:
    """Moves the mass of bin i to bin i + 1 (the last bin's falls off)."""
    return torch.nn.functional.pad(values[..., :-1], (1, 0))


def shift_down(values: torch.Tensor) -> torch.Tensor:
    """Moves the mass of bin i to bin i - 1 (the first bin's falls off)."""
    return torch.nn.functional.pad(values[..., 1:], (0, 1))


def match_moments(probabilities, offsets, target_mean, target_variance):
    """Moves interior mass by one bin so the code has the target mean and variance about its
    offsets. probabilities, offsets (..., n); targets (...)."""
    tolerance = 8 * machine_epsilon(probabilities)
    tiny = smallest_normal(probabilities)
    bins = probabilities.shape[-1]
    index = torch.arange(bins, device=probabilities.device)
    interior = ((index > 0) & (index < bins - 1)).to(probabilities.dtype)
    mean = (probabilities * offsets).sum(-1)
    second = (probabilities * offsets**2).sum(-1)
    source = probabilities * interior
    mass = torch.clamp_min(source.sum(-1), tiny)
    shift = (target_mean - mean) / mass
    shift = torch.where(shift.abs() <= tolerance, 0.0, shift)
    shift_cost = shift.abs() * (source * (2 * torch.sign(shift).unsqueeze(-1) * offsets + 1)).sum(-1)
    widen = torch.clamp_min((target_variance + target_mean**2 - second - shift_cost) / (2 * mass), 0)
    widen = torch.where(widen <= tolerance, 0.0, widen)
    excess = torch.clamp_min(2 * widen + shift.abs(), 1)
    widen = (widen / excess).unsqueeze(-1)
    shift = (shift / excess).unsqueeze(-1)
    up = torch.clamp_min(shift, 0) * source
    down = torch.clamp_min(-shift, 0) * source
    spread = widen * source
    return (probabilities - (up + down + 2 * spread) + shift_up(up + spread)
            + shift_down(down + spread))


def scalar_encode(low: float, high: float, bins: int, mean: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
    """mean, sigma (...) in physical units -> probabilities (..., bins)."""
    spacing = (high - low) / (bins - 1)
    last = bins - 1
    coordinate, lower, fraction = scalar_coordinate(low, high, bins, mean)
    variance = torch.minimum((sigma / spacing) ** 2,
                             torch.clamp_min(coordinate * (last - coordinate)
                                             - fraction * (1 - fraction), 0))
    index = torch.arange(bins, dtype=mean.dtype, device=mean.device)
    offsets = index - lower.unsqueeze(-1)
    width = torch.clamp_min(variance, 1).unsqueeze(-1)
    weight = torch.clamp_max(variance, 1).unsqueeze(-1)
    t = fraction.unsqueeze(-1)
    b = lower.unsqueeze(-1)

    def gaussian(distance):
        return torch.exp(-0.5 * distance**2 / width) / torch.sqrt(TWO_PI * width)

    smooth = (1 - t) * gaussian(offsets) + t * gaussian(offsets - 1)
    root = torch.sqrt(2 * width)
    first_tail = 0.5 * ((1 - t) * torch.erfc((b + 0.5) / root) + t * torch.erfc((b + 1.5) / root))
    last_tail = 0.5 * ((1 - t) * torch.erfc((last - b + 0.5) / root)
                       + t * torch.erfc((last - b - 0.5) / root))
    first = index == 0
    smooth = smooth + first * first_tail + (index == last) * last_tail
    sharp = (1 - t) * (index == b) + t * (index == b + 1)
    probabilities = (1 - weight) * sharp + weight * smooth
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return match_moments(probabilities, offsets, fraction, variance + fraction * (1 - fraction))


def scalar_decode(low: float, high: float, bins: int, probabilities: torch.Tensor):
    """(..., bins) -> (mean, spread), each (...) in physical units."""
    tolerance = 8 * machine_epsilon(probabilities)
    spacing = (high - low) / (bins - 1)
    index = torch.arange(bins, dtype=probabilities.dtype, device=probabilities.device)
    coordinate = torch.clamp((probabilities * index).sum(-1), 0, bins - 1)
    variance = (probabilities * (index - coordinate.unsqueeze(-1)) ** 2).sum(-1)
    fraction = coordinate - torch.floor(coordinate)
    physical = variance - fraction * (1 - fraction)
    physical = torch.where(physical <= tolerance, 0.0, physical)
    return low + coordinate * spacing, torch.sqrt(physical) * spacing


# --------------------------------------------------------------------------- velocity


def compress_velocity(velocity: torch.Tensor, knee: float) -> torch.Tensor:
    """(..., 2) m/s -> compressed (..., 2): v kappa ln(1 + |v|/kappa) / |v|, 0 at 0."""
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    safe = torch.where(speed > 0, speed, 1.0)
    return torch.where(speed > 0, velocity * knee * torch.log1p(speed / knee) / safe, 0.0)


def expand_velocity(compressed: torch.Tensor, knee: float) -> torch.Tensor:
    """The inverse of compress_velocity: c kappa (e^{|c|/kappa} - 1) / |c|, 0 at 0."""
    size = torch.linalg.vector_norm(compressed, dim=-1, keepdim=True)
    safe = torch.where(size > 0, size, 1.0)
    return torch.where(size > 0, compressed * knee * torch.expm1(size / knee) / safe, 0.0)


def packed_radial(velocity: torch.Tensor, radial: torch.Tensor, tangential: torch.Tensor):
    """(xx, xy, yy) of T I + (R - T) v^ v^T, (..., 3)."""
    eps = machine_epsilon(velocity)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    direction = velocity / torch.clamp_min(speed, eps).unsqueeze(-1)
    difference = radial - tangential
    dx, dy = direction[..., 0], direction[..., 1]
    return torch.stack([tangential + difference * dx * dx, difference * dx * dy,
                        tangential + difference * dy * dy], -1)


def velocity_jacobian(velocity: torch.Tensor, knee: float):
    """The packed Jacobian of compress_velocity at velocity (..., 2), and its inverse."""
    eps = machine_epsilon(velocity)
    tiny = smallest_normal(velocity)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    logarithm = torch.log1p(speed / knee)
    zero = speed == 0
    radial = 1 / (1 + speed / knee)
    tangential = torch.where(zero, 1.0, knee * logarithm / torch.clamp_min(speed, eps))
    jacobian = packed_radial(velocity, radial, tangential)
    inverse_radial = 1 + speed / knee
    inverse_tangential = torch.where(zero, 1.0, speed / torch.clamp_min(knee * logarithm, tiny))
    return jacobian, packed_radial(velocity, inverse_radial, inverse_tangential)


def sandwich(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    """A S A for packed symmetric 2x2 matrices (..., 3) (xx, xy, yy)."""
    a, b, c = outer.unbind(-1)
    s, t, u = inner.unbind(-1)
    # A S = [[a s + b t, a t + b u], [b s + c t, b t + c u]]
    first_x, first_y = a * s + b * t, a * t + b * u
    second_x, second_y = b * s + c * t, b * t + c * u
    return torch.stack([first_x * a + first_y * b, first_x * b + first_y * c,
                        second_x * b + second_y * c], -1)


def project_three(vector: torch.Tensor) -> torch.Tensor:
    """(..., 2) -> (..., 3), projections onto axes at 0, 120 and 240 degrees."""
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x, -0.5 * x + (SQRT3 / 2) * y, -0.5 * x - (SQRT3 / 2) * y], -1)


def velocity_axis_range(bins: int, spacing: float):
    half = 0.5 * (bins - 1) * spacing
    return -half, half


def velocity_sharp(bins: int, spacing: float, knee: float, velocity: torch.Tensor) -> torch.Tensor:
    """(..., 2) m/s -> (..., 3, bins)."""
    low, high = velocity_axis_range(bins, spacing)
    return scalar_sharp(low, high, bins, project_three(compress_velocity(velocity, knee)))


def velocity_encode(bins: int, spacing: float, knee: float, mean: torch.Tensor,
                    covariance: torch.Tensor) -> torch.Tensor:
    """mean (..., 2) m/s, covariance packed (..., 3) (m/s)^2 -> (..., 3, bins)."""
    low, high = velocity_axis_range(bins, spacing)
    jacobian, _ = velocity_jacobian(mean, knee)
    xx, xy, yy = sandwich(jacobian, covariance).unbind(-1)
    variances = torch.stack([xx, 0.25 * xx - (SQRT3 / 2) * xy + 0.75 * yy,
                             0.25 * xx + (SQRT3 / 2) * xy + 0.75 * yy], -1)
    return scalar_encode(low, high, bins, project_three(compress_velocity(mean, knee)),
                         torch.sqrt(torch.clamp_min(variances, 0)))


def velocity_decode(bins: int, spacing: float, knee: float, probabilities: torch.Tensor):
    """(..., 3, bins) -> (mean (..., 2) m/s, covariance packed (..., 3))."""
    low, high = velocity_axis_range(bins, spacing)
    projection, spread = scalar_decode(low, high, bins, probabilities)
    p0, p1, p2 = projection.unbind(-1)
    compressed = torch.stack([(2 / 3) * (p0 - 0.5 * (p1 + p2)), (2 / 3) * (SQRT3 / 2) * (p1 - p2)],
                             -1)
    v0, v1, v2 = (spread**2).unbind(-1)
    xx = v0
    xy = (v2 - v1) / SQRT3
    yy = (2 * (v1 + v2) - xx) / 3
    xx = torch.clamp_min(xx, 0)
    yy = torch.clamp_min(yy, 0)
    bound = torch.sqrt(xx * yy)
    xy = torch.maximum(torch.minimum(xy, bound), -bound)
    mean = expand_velocity(compressed, knee)
    _, inverse = velocity_jacobian(mean, knee)
    return mean, sandwich(inverse, torch.stack([xx, xy, yy], -1))


# --------------------------------------------------------------------------- grouped


def run_slices(runs):
    """[(groups, size), ...] -> [(start, groups, size), ...]."""
    start, out = 0, []
    for groups, size in runs:
        out.append((start, int(groups), int(size)))
        start += int(groups) * int(size)
    return out


def grouped_log_softmax(logits: torch.Tensor, runs) -> torch.Tensor:
    pieces = []
    for start, groups, size in run_slices(runs):
        block = logits[..., start:start + groups * size].unflatten(-1, (groups, size))
        pieces.append(torch.log_softmax(block, -1).flatten(-2))
    return torch.cat(pieces, -1)


def grouped_cross_entropy(logits: torch.Tensor, target: torch.Tensor, runs) -> torch.Tensor:
    """(..., width) -> (..., total groups), -sum target log p within each group."""
    log_probabilities = grouped_log_softmax(logits, runs)
    pieces = []
    for start, groups, size in run_slices(runs):
        end = start + groups * size
        product = (target[..., start:end] * log_probabilities[..., start:end])
        pieces.append(-product.unflatten(-1, (groups, size)).sum(-1))
    return torch.cat(pieces, -1)


def remove_baseline_floor(logits: torch.Tensor, runs, floor: float = 1e-3) -> torch.Tensor:
    """Group softmax, then undo the floor's mixture with uniform: (..., width)."""
    tiny = smallest_normal(logits)
    pieces = []
    for start, groups, size in run_slices(runs):
        block = logits[..., start:start + groups * size].unflatten(-1, (groups, size))
        probabilities = torch.softmax(block, -1)
        lifted = torch.clamp_min((1 + size * floor) * probabilities - floor, 0)
        pieces.append((lifted / torch.clamp_min(lifted.sum(-1, keepdim=True), tiny)).flatten(-2))
    return torch.cat(pieces, -1)


def code_baseline(code: torch.Tensor) -> torch.Tensor:
    return torch.log(code + 1e-3)
