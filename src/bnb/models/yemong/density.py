"""The hex density target (spec 9): where the living ships of each side are, on a hex grid
that scales with the arena, and the Poisson terms of a predicted rate."""

import math

import torch

SQRT3_HALF = 0.5 * math.sqrt(3.0)
RING_STEPS = ((-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1))
LOG_RATE_RANGE = (-20.0, 2.0)


def cell_count(rings: int) -> int:
    return 1 + 3 * rings * (rings + 1)


def hex_cells(rings: int, dtype=torch.float32, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial coordinates (C, 2) int64 in grid order (the centre, then ring by ring, each
    from (r, 0) along RING_STEPS), and the cell centres (C, 2) in arena radii."""
    axial = [(0, 0)]
    for ring in range(1, rings + 1):
        q, r = ring, 0
        for step_q, step_r in RING_STEPS:
            for _ in range(ring):
                axial.append((q, r))
                q, r = q + step_q, r + step_r
    axial = torch.tensor(axial, dtype=torch.int64, device=device)
    q, r = axial[:, 0].to(dtype), axial[:, 1].to(dtype)
    centres = torch.stack([q + 0.5 * r, SQRT3_HALF * r], -1) / max(rings, 1)
    return axial, centres


def _cell_lookup(rings: int, device) -> torch.Tensor:
    """(2R+1, 2R+1) int64: the cell index of axial (q, r) at [q + R, r + R], or C if off-grid."""
    axial, _ = hex_cells(rings, device=device)
    side = 2 * rings + 1
    lookup = torch.full((side, side), cell_count(rings), dtype=torch.int64, device=device)
    lookup[axial[:, 0] + rings, axial[:, 1] + rings] = torch.arange(axial.shape[0], device=device)
    return lookup


def hex_counts(x: torch.Tensor, y: torch.Tensor, is_ally: torch.Tensor, alive: torch.Tensor,
               arena_radius: torch.Tensor, rings: int) -> torch.Tensor:
    """(B, 2C): each living ship splits one unit over its grid triangle by barycentric weight;
    ally cells, then enemy cells. x, y, is_ally, alive (B, N); arena_radius broadcasts to them."""
    cells, side = cell_count(rings), 2 * rings + 1
    scale = rings / torch.clamp_min(arena_radius, 1e-6)
    axis_r = y * scale / SQRT3_HALF
    axis_q = x * scale - 0.5 * axis_r
    base_q, base_r = torch.floor(axis_q), torch.floor(axis_r)
    u, v = axis_q - base_q, axis_r - base_r
    lower = u + v <= 1.0
    # Three vertices per ship: the lower triangle (0,0),(1,0),(0,1) or the upper (1,0),(0,1),(1,1).
    device = x.device
    vertex = torch.arange(3, device=device)
    offset_q = torch.where(lower[..., None], vertex == 1, vertex != 1).long()
    offset_r = torch.where(lower[..., None], vertex == 2, vertex >= 1).long()
    weight = torch.where(lower[..., None], torch.stack([1.0 - u - v, u, v], -1),
                         torch.stack([1.0 - v, 1.0 - u, u + v - 1.0], -1))
    q = base_q.long()[..., None] + offset_q + rings
    r = base_r.long()[..., None] + offset_r + rings
    inside = (q >= 0) & (q < side) & (r >= 0) & (r < side)
    lookup = _cell_lookup(rings, device)
    index = lookup[q.clamp(0, side - 1), r.clamp(0, side - 1)]
    index = torch.where(inside, index, cells)  # off-grid: the dropped cell
    index = index + torch.where(is_ally, 0, cells + 1)[..., None]
    weight = weight * alive[..., None].to(weight.dtype)
    counts = torch.zeros(x.shape[:-1] + (2 * (cells + 1),), dtype=x.dtype, device=x.device)
    counts.scatter_add_(-1, index.flatten(-2), weight.flatten(-2))
    return torch.cat([counts[..., :cells], counts[..., cells + 1:-1]], -1)


def poisson_terms(log_rate: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor,
                                                                          torch.Tensor]:
    """The per-cell Poisson negative log-likelihood (without log y!) of a clipped log-rate,
    and its saturated value (the best any rate could do)."""
    clipped = torch.clamp(log_rate, *LOG_RATE_RANGE)
    per_cell = torch.exp(clipped) - counts * clipped
    saturated = torch.where(counts > 0,
                            counts - counts * torch.log(torch.clamp_min(counts, 1e-12)), 0.0)
    return per_cell, saturated
