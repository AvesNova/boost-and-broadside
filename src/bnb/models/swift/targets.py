"""Critic distributions, the return scale, hex density, shot codes and scalar features
(spec 4, 8 and 9)."""

import math

import torch

from .codes import circular_encode, position_encode, scalar_encode, scalar_sharp

OUTCOME_WIN, OUTCOME_TIE, OUTCOME_LOSS, OUTCOME_UNRESOLVED = 0, 1, 2, 3


# --------------------------------------------------------------------------- critic


def symlog(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(values.abs())


def symexp(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(values.abs())


def value_bins(count: int, limit: float, dtype=torch.float64, device=None) -> torch.Tensor:
    """(count,) bins symexp(linspace(-limit, limit, count)), in raw return units."""
    return symexp(torch.linspace(-limit, limit, count, dtype=dtype, device=device))


def two_hot(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """values (...) -> (..., n) two-hot over the bins, clamped to the outer bins."""
    count = bins.shape[0]
    values = torch.clamp(values, bins[0], bins[-1])
    upper = torch.clamp(torch.searchsorted(bins, values.contiguous(), right=True), 1, count - 1)
    lower = upper - 1
    low, high = bins[lower], bins[upper]
    weight = torch.clamp((values - low) / (high - low), 0, 1)
    out = torch.zeros(*values.shape, count, dtype=values.dtype, device=values.device)
    out.scatter_add_(-1, lower.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), weight.unsqueeze(-1))
    return out


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return (torch.softmax(logits, -1) * bins).sum(-1)


def outcome_values(probabilities: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> P(win) - P(loss)."""
    return probabilities[..., OUTCOME_WIN] - probabilities[..., OUTCOME_LOSS]


def outcome_class(result: torch.Tensor) -> torch.Tensor:
    """win 0 if result > 0, loss 2 if < 0, else tie 1 (int64)."""
    return torch.where(result > 0, OUTCOME_WIN, torch.where(result < 0, OUTCOME_LOSS, OUTCOME_TIE))


def outcome_targets(probabilities, result, terminated, next_probabilities, gamma, lam):
    """The outcome's lambda targets, backwards over time.

    probabilities (L, B, 4) the predicted p(s_t); result, terminated (L, B); next_probabilities
    (B, 4) the bootstrap p(s_L). Returns (L, B, 4)."""
    steps = probabilities.shape[0]
    unresolved = torch.zeros_like(next_probabilities)
    unresolved[..., OUTCOME_UNRESOLVED] = 1
    following = next_probabilities
    target = next_probabilities
    out = []
    for t in reversed(range(steps)):
        target = gamma * ((1 - lam) * following + lam * target) + (1 - gamma) * unresolved
        ended = torch.nn.functional.one_hot(outcome_class(result[t]), 4).to(target.dtype)
        target = torch.where(terminated[t].unsqueeze(-1), ended, target)
        out.append(target)
        following = probabilities[t]
    return torch.stack(out[::-1])


def critic_values(local_logits, outcome_logits, bins, local_levels, outcome_level):
    """local_logits (..., N, K_local, bins), outcome_logits (..., 4) -> values (..., N, K)."""
    local = expectation(local_logits, bins)  # (..., N, K_local)
    outcome = outcome_values(torch.softmax(outcome_logits, -1))  # (...)
    levels = len(local_levels) + 1
    columns = [None] * levels
    for column, level in enumerate(local_levels):
        columns[level] = local[..., column]
    columns[outcome_level] = outcome.unsqueeze(-1).expand_as(local[..., 0])
    return torch.stack(columns, -1)


class ReturnScale:
    """EMAs (decay d) of the 5th and 95th percentiles of the masked returns; the scale is
    max(high - low, floor)."""

    def __init__(self, decay: float, floor: float):
        self.decay, self.floor = decay, floor
        self.low = 0.0
        self.high = 0.0

    def update(self, returns: torch.Tensor, mask: torch.Tensor) -> float:
        selected = returns[mask]
        if selected.numel():
            quantiles = torch.quantile(selected, torch.tensor([0.05, 0.95], dtype=selected.dtype,
                                                              device=selected.device))
            self.low = self.decay * self.low + (1 - self.decay) * quantiles[0].item()
            self.high = self.decay * self.high + (1 - self.decay) * quantiles[1].item()
        return self.scale

    @property
    def scale(self) -> float:
        return max(self.high - self.low, self.floor)


# --------------------------------------------------------------------------- density

HEX_STEPS = ((-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1))


def hex_axial(rings: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    for ring in range(1, rings + 1):
        q, r = ring, 0
        for step_q, step_r in HEX_STEPS:
            for _ in range(ring):
                cells.append((q, r))
                q, r = q + step_q, r + step_r
    return cells


def hex_cells(rings: int, dtype=torch.float64, device=None):
    """(axial (C, 2) int64, centres (C, 2) in arena radii)."""
    axial = torch.tensor(hex_axial(rings), dtype=torch.int64, device=device)
    q, r = axial[:, 0].to(dtype), axial[:, 1].to(dtype)
    centres = torch.stack([q + r / 2, r * math.sqrt(3) / 2], -1) / rings
    return axial, centres


def hex_lookup(rings: int, device=None) -> torch.Tensor:
    """(2R + 1, 2R + 1) int64: the cell index of axial (q, r) at [q + R, r + R], -1 outside."""
    table = torch.full((2 * rings + 1, 2 * rings + 1), -1, dtype=torch.int64)
    for index, (q, r) in enumerate(hex_axial(rings)):
        table[q + rings, r + rings] = index
    return table.to(device)


def hex_counts(x, y, is_ally, alive, arena_radius, rings: int, lookup=None) -> torch.Tensor:
    """Each living ship splits one unit over its triangle's vertices by barycentric weight.

    x, y (B, N) m; is_ally, alive (B, N) bool; arena_radius (B, 1) m. Returns (B, 2C): ally
    counts, then enemy counts."""
    if lookup is None:
        lookup = hex_lookup(rings, x.device)
    cells = 1 + 3 * rings * (rings + 1)
    scale = torch.clamp_min(arena_radius, 1e-6)
    x, y = x / scale, y / scale
    fractional_r = y * rings / (math.sqrt(3) / 2)
    fractional_q = x * rings - fractional_r / 2
    base_q, base_r = torch.floor(fractional_q), torch.floor(fractional_r)
    u, v = fractional_q - base_q, fractional_r - base_r
    lower = (u + v) <= 1
    vertex_q = torch.stack([torch.where(lower, 0.0, 1.0), torch.where(lower, 1.0, 0.0),
                            torch.where(lower, 0.0, 1.0)], -1)
    vertex_r = torch.stack([torch.where(lower, 0.0, 0.0), torch.where(lower, 0.0, 1.0),
                            torch.where(lower, 1.0, 1.0)], -1)
    weights = torch.stack([torch.where(lower, 1 - u - v, 1 - v), torch.where(lower, u, 1 - u),
                           torch.where(lower, v, u + v - 1)], -1)
    q = (base_q.unsqueeze(-1) + vertex_q).long()
    r = (base_r.unsqueeze(-1) + vertex_r).long()
    inside = (q.abs() <= rings) & (r.abs() <= rings) & ((q + r).abs() <= rings)
    side = lookup.shape[0]
    flat = (torch.clamp(q + rings, 0, side - 1) * side + torch.clamp(r + rings, 0, side - 1))
    index = lookup.flatten()[flat]
    keep = inside & (index >= 0) & alive.unsqueeze(-1)
    index = torch.where(keep, index, 0) + torch.where(is_ally, 0, cells).unsqueeze(-1)
    counts = torch.zeros(x.shape[0], 2 * cells, dtype=x.dtype, device=x.device)
    counts.scatter_add_(-1, index.flatten(1), (weights * keep).flatten(1))
    return counts


def poisson_terms(log_rate: torch.Tensor, counts: torch.Tensor):
    """(per-cell negative log-likelihood up to a constant, its saturated value)."""
    clipped = torch.clamp(log_rate, -20, 2)
    per_cell = torch.exp(clipped) - counts * clipped
    saturated = torch.where(counts > 0,
                            counts - counts * torch.log(torch.clamp_min(counts, 1e-12)), 0.0)
    return per_cell, saturated


# --------------------------------------------------------------------------- features


def fourier_time(values: torch.Tensor, horizon: float, harmonics: int) -> torch.Tensor:
    """values (...) -> (..., 1 + 2 harmonics): [s/H, sin(2 pi k s/H), cos(2 pi k s/H)], k = 1..H."""
    phase = values / horizon
    k = torch.arange(1, harmonics + 1, dtype=values.dtype, device=values.device)
    angle = 2 * math.pi * phase.unsqueeze(-1) * k
    return torch.cat([phase.unsqueeze(-1), torch.sin(angle), torch.cos(angle)], -1)


def respawn_features(seconds, horizon: float = 16.0, harmonics: int = 4):
    return fourier_time(seconds, horizon, harmonics)


def shot_time_features(ages, max_age: float, harmonics: int = 4):
    return fourier_time(ages, max_age, harmonics)


def local_presence(position, team, source, radius: float) -> torch.Tensor:
    """position (B, N, 2) m; team (B, N) 0/1 ships, 2 map tokens; source (B, N) bool, who
    may contribute. Returns (B, N, 2): ln(1 + Gaussian sums) over allies (same team, not
    self) and enemies (the other ship team)."""
    difference = position.unsqueeze(-2) - position.unsqueeze(-3)
    kernel = torch.exp(-(difference**2).sum(-1) / (2 * radius**2))  # (B, N, N)
    kernel = kernel * source.unsqueeze(-2)
    same = team.unsqueeze(-1) == team.unsqueeze(-2)
    count = team.shape[-1]
    self_pair = torch.eye(count, dtype=torch.bool, device=position.device)
    ally = (kernel * (same & ~self_pair)).sum(-1)
    other = (team.unsqueeze(-2) == (1 - team).unsqueeze(-1)) & (team.unsqueeze(-1) < 2)
    enemy = (kernel * other).sum(-1)
    return torch.log1p(torch.stack([ally, enemy], -1))


# --------------------------------------------------------------------------- shots


def shot_trajectory_code(parameters, snapshots: torch.Tensor) -> torch.Tensor:
    """snapshots (..., 5): displacement x, y (m), velocity x, y (m/s), refractive index.
    Returns (..., 8 * 9 + 5 * 4 + speed bins + index bins)."""
    period = parameters["displacement_period"]
    zero = torch.zeros_like(snapshots[..., 0])
    displacement = position_encode((period, period), int(parameters["displacement_levels"]),
                                   snapshots[..., 0:2], zero).flatten(-2)
    direction = circular_encode(2 * math.pi, 4, int(parameters["direction_levels"]),
                                torch.atan2(snapshots[..., 3], snapshots[..., 2]), zero).flatten(-2)
    speed = scalar_sharp(0.0, parameters["top_speed"], int(parameters["speed_bins"]),
                         torch.hypot(snapshots[..., 2], snapshots[..., 3]))
    span = parameters["index_span"]
    index = scalar_sharp(-span, span, int(parameters["index_bins"]),
                         torch.log(torch.clamp_min(snapshots[..., 4], 1e-6)))
    return torch.cat([displacement, direction, speed, index], -1)


def shot_outcome_code(parameters, hit, distance, side) -> torch.Tensor:
    """hit 0 none, 1 enemy, 2 ally; distance (m) of the closest pass; side 1 enemy, 2 ally.
    Returns (..., 3 + 2 near bins): [enemy hit, ally hit, enemy near, ally near, clear]."""
    radius = parameters["collision_radius"]
    near = int(parameters["near_bins"])
    low = math.log(radius)
    spacing = math.log(parameters["near_span_radii"]) / (near - 1)
    high = low + near * spacing
    clear = math.exp(high)
    bins = near + 1
    distance = distance.to(torch.float64) if not distance.is_floating_point() else distance
    logarithm = torch.log(torch.clamp_min(distance, radius))
    code = scalar_encode(low, high, bins, logarithm,
                         torch.full_like(logarithm, parameters["near_sigma_bins"] * spacing))
    far = (distance >= clear).unsqueeze(-1)
    one_hot_clear = torch.zeros_like(code)
    one_hot_clear[..., near] = 1
    code = torch.where(far, one_hot_clear, code)
    ally_side = (side == 2).unsqueeze(-1)
    near_part = code[..., :near]
    zeros = torch.zeros_like(near_part)
    out = torch.cat([torch.zeros_like(code[..., :2]), torch.where(ally_side, zeros, near_part),
                     torch.where(ally_side, near_part, zeros), code[..., near:]], -1)
    hit_code = torch.zeros_like(out)
    hit_code[..., 0] = 1
    ally_hit = torch.zeros_like(out)
    ally_hit[..., 1] = 1
    out = torch.where((hit == 1).unsqueeze(-1), hit_code, out)
    return torch.where((hit == 2).unsqueeze(-1), ally_hit, out)
