"""Small normative formulas shared by flight and bullets (SPEC 3.3)."""

import torch


def cs(theta: torch.Tensor):
    """(cos, sin) of a small angle by the normative truncated series."""
    t2 = theta * theta
    cos = 1.0 - t2 * (0.5 - t2 * (1.0 / 24.0 - t2 / 720.0))
    sin = theta * (1.0 - t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 / 5040.0)))
    return cos, sin


def rotate(x: torch.Tensor, y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """The vector (x, y) turned counter-clockwise by the angle with this cos and sin."""
    return x * cos - y * sin, x * sin + y * cos


def softmin(a: torch.Tensor, b, q: float) -> torch.Tensor:
    """a (1 + (a/b)^q)^(-1/q) for a, b >= 0.

    Written as lo (1 + (lo/hi)^q)^(-1/q) with lo, hi the smaller and larger argument (the
    same value, since the formula is symmetric in a and b), so the power never overflows.
    """
    if isinstance(b, torch.Tensor):
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    else:
        lo, hi = a.clamp_max(b), a.clamp_min(b)
    return lo * torch.pow(1.0 + torch.pow(lo / hi, q), -1.0 / q)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """t^2 (3 - 2t), the engine fall phi of SPEC 3.3 after clipping."""
    return t * t * (3.0 - 2.0 * t)
