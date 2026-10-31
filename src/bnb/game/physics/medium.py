"""Refractive fields (SPEC 4): ln n and its gradient at a batch of points.

The fields are folded in one at a time. Each field needs only its own profile and the
running sums, so with the loop unrolled under torch.compile the whole evaluation is a
single elementwise kernel over the points, with no (points x fields) intermediate.
"""

import torch

from .constants import LN_SQRT2


def target_log_index(flevel: torch.Tensor) -> torch.Tensor:
    """L_i = level * ln sqrt(2)."""
    return flevel.to(torch.float32) * LN_SQRT2


def medium(fields: dict, x: torch.Tensor, y: torch.Tensor):
    """ln n, d/dx ln n, d/dy ln n at points (B, P) for fields fx fy fr fw flevel (B, M)."""
    level = target_log_index(fields["flevel"])
    weight = torch.zeros_like(x)  # W = sum alpha_i
    weighted = torch.zeros_like(x)  # sum alpha_i L_i
    clear = torch.ones_like(x)  # prod (1 - alpha_i), so the coverage is A = 1 - clear
    grad_w = (torch.zeros_like(x), torch.zeros_like(x))  # sum grad alpha_i
    grad_weighted = (torch.zeros_like(x), torch.zeros_like(x))  # sum grad alpha_i L_i
    grad_cover = (torch.zeros_like(x), torch.zeros_like(x))  # grad A
    for i in range(fields["fx"].shape[1]):
        dx = x - fields["fx"][:, i, None]
        dy = y - fields["fy"][:, i, None]
        band = fields["fw"][:, i, None]
        d = torch.sqrt(dx * dx + dy * dy)
        zeta = 0.5 - (d - fields["fr"][:, i, None]) / band
        z = zeta.clamp(0.0, 1.0)
        alpha = z * z * z * (z * (6.0 * z - 15.0) + 10.0)
        in_band = (zeta > 0.0) & (zeta < 1.0) & (d > 1e-6)
        slope = torch.where(in_band, -30.0 * (z * (z - 1.0)) ** 2 / (band * d.clamp_min(1e-6)), 0.0)
        gx, gy = slope * dx, slope * dy
        lvl = level[:, i, None]
        weight = weight + alpha
        weighted = weighted + alpha * lvl
        grad_w = (grad_w[0] + gx, grad_w[1] + gy)
        grad_weighted = (grad_weighted[0] + gx * lvl, grad_weighted[1] + gy * lvl)
        # Product rule on the running product: grad A gains grad alpha_i times the product of
        # the other factors, without dividing by (1 - alpha_i), which may be zero.
        grad_cover = (grad_cover[0] * (1.0 - alpha) + gx * clear, grad_cover[1] * (1.0 - alpha) + gy * clear)
        clear = clear * (1.0 - alpha)
    cover = 1.0 - clear
    covered = weight > 1e-12
    safe_w = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, weighted / safe_w, 0.0)
    grad_mean = [torch.where(covered, (gs - mean * gw) / safe_w, 0.0) for gs, gw in zip(grad_weighted, grad_w)]
    logn = cover * mean
    return logn, mean * grad_cover[0] + cover * grad_mean[0], mean * grad_cover[1] + cover * grad_mean[1]
