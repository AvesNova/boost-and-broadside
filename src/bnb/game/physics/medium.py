"""Refractive fields (game spec 4): ln n and its gradient at a batch of points.

Written as reductions over the field axis: each field's profile is elementwise over
(points, fields), and the union composition needs only sums, one product and one
leave-one-out sum, which torch.compile fuses into a single reduction kernel for any M,
without materializing the (points, fields) intermediate. (Unrolling a loop over the fields
instead makes the generated code grow with M until the compiler splits it into many
kernels that store their intermediates.)
"""

import torch

from ..rules import LN_SQRT2
from ..state import Fields


def medium(fields: Fields, x: torch.Tensor, y: torch.Tensor):
    """(ln n, d/dx ln n, d/dy ln n) at points (B, P) for fields (B, M)."""
    def per_field(v):  # (B, M) as (B, 1, M), against points (B, P, 1)
        return v[:, None, :]

    level = per_field(fields.level.to(torch.float32) * LN_SQRT2)  # target log index L_i
    band = per_field(fields.width)
    dx, dy = x[..., None] - per_field(fields.x), y[..., None] - per_field(fields.y)
    d = torch.sqrt(dx * dx + dy * dy)
    zeta = 0.5 - (d - per_field(fields.radius)) / band
    z = zeta.clamp(0.0, 1.0)
    w = 1.0 - z
    alpha = z * z * z * (z * (6.0 * z - 15.0) + 10.0)
    # 1 - alpha in factored form (the profile is symmetric: 1 - alpha(z) = alpha(1 - z)), so
    # it is exactly positive wherever the gradient is non-zero, even where alpha rounds to 1.
    open_ = w * w * w * (w * (6.0 * w - 15.0) + 10.0)
    in_band = (zeta > 0.0) & (zeta < 1.0) & (d > 1e-6)
    slope = torch.where(in_band, -30.0 * (z * w) ** 2 / (band * d.clamp_min(1e-6)), 0.0)
    gx, gy = slope * dx, slope * dy

    weight = alpha.sum(-1)  # W
    weighted = (alpha * level).sum(-1)
    clear = open_.prod(-1)  # prod (1 - alpha_i): the coverage is A = 1 - clear
    # grad A = sum_i grad alpha_i prod_{j != i} (1 - alpha_j) = clear sum_i grad alpha_i /
    # (1 - alpha_i), dividing only where grad alpha_i != 0, where 1 - alpha_i > 0.
    inv_open = torch.where(slope != 0.0, 1.0 / torch.where(slope != 0.0, open_, 1.0), 0.0)
    gcx, gcy = clear * (gx * inv_open).sum(-1), clear * (gy * inv_open).sum(-1)
    gwx, gwy = gx.sum(-1), gy.sum(-1)
    gsx, gsy = (gx * level).sum(-1), (gy * level).sum(-1)

    cover = 1.0 - clear
    covered = weight > 1e-12
    safe_w = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, weighted / safe_w, 0.0)
    gmx = torch.where(covered, (gsx - mean * gwx) / safe_w, 0.0)
    gmy = torch.where(covered, (gsy - mean * gwy) / safe_w, 0.0)
    return cover * mean, mean * gcx + cover * gmx, mean * gcy + cover * gmy
