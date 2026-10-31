"""Refractive fields (game spec 4): ln n and its gradient at a batch of points.

The fields are folded in one at a time. Each needs only its own profile and the running
sums, so with the loop unrolled under torch.compile the whole evaluation is one
elementwise kernel over the points, with no (points x fields) intermediate.
"""

import torch

from ..rules import LN_SQRT2
from ..state import Fields


def medium(fields: Fields, x: torch.Tensor, y: torch.Tensor):
    """(ln n, d/dx ln n, d/dy ln n) at points (B, P) for fields (B, M)."""
    level = fields.level.to(torch.float32) * LN_SQRT2  # target log index L_i
    weight = torch.zeros_like(x)  # W = sum alpha_i
    weighted = torch.zeros_like(x)  # sum alpha_i L_i
    clear = torch.ones_like(x)  # prod (1 - alpha_i): the coverage is A = 1 - clear
    gwx, gwy = torch.zeros_like(x), torch.zeros_like(x)  # sum grad alpha_i
    gsx, gsy = torch.zeros_like(x), torch.zeros_like(x)  # sum grad alpha_i L_i
    gcx, gcy = torch.zeros_like(x), torch.zeros_like(x)  # grad A
    for i in range(fields.x.shape[1]):
        dx = x - fields.x[:, i, None]
        dy = y - fields.y[:, i, None]
        band = fields.width[:, i, None]
        d = torch.sqrt(dx * dx + dy * dy)
        zeta = 0.5 - (d - fields.radius[:, i, None]) / band
        z = zeta.clamp(0.0, 1.0)
        alpha = z * z * z * (z * (6.0 * z - 15.0) + 10.0)
        in_band = (zeta > 0.0) & (zeta < 1.0) & (d > 1e-6)
        slope = torch.where(in_band, -30.0 * (z * (z - 1.0)) ** 2 / (band * d.clamp_min(1e-6)), 0.0)
        gx, gy = slope * dx, slope * dy
        lvl = level[:, i, None]
        weight = weight + alpha
        weighted = weighted + alpha * lvl
        gwx, gwy = gwx + gx, gwy + gy
        gsx, gsy = gsx + gx * lvl, gsy + gy * lvl
        # Product rule on the running product: grad A gains grad alpha_i times the product
        # of the other factors, without dividing by (1 - alpha_i), which may be zero.
        gcx, gcy = gcx * (1.0 - alpha) + gx * clear, gcy * (1.0 - alpha) + gy * clear
        clear = clear * (1.0 - alpha)
    cover = 1.0 - clear
    covered = weight > 1e-12
    safe_w = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, weighted / safe_w, 0.0)
    gmx = torch.where(covered, (gsx - mean * gwx) / safe_w, 0.0)
    gmy = torch.where(covered, (gsy - mean * gwy) / safe_w, 0.0)
    return cover * mean, mean * gcx + cover * gmx, mean * gcy + cover * gmy
