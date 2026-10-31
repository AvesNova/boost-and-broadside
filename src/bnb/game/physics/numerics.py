"""Small normative formulas shared by flight and bullets (game spec 3.3)."""

import torch


def small_angle_cos_sin(angle: torch.Tensor):
    """(cos, sin) of a small angle by the normative truncated series (spec: cs)."""
    square = angle * angle
    cos = 1.0 - square * (0.5 - square * (1.0 / 24.0 - square / 720.0))
    sin = angle * (1.0 - square * (1.0 / 6.0 - square * (1.0 / 120.0 - square / 5040.0)))
    return cos, sin


def rotate(x: torch.Tensor, y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """The vector (x, y) turned counter-clockwise by the angle with this cos and sin."""
    return x * cos - y * sin, x * sin + y * cos


def softmin(a: torch.Tensor, b, exponent: float) -> torch.Tensor:
    """a (1 + (a/b)^q)^(-1/q) for a, b >= 0.

    Written with the smaller and the larger argument (the formula is symmetric in a and b),
    so the power never overflows.
    """
    if isinstance(b, torch.Tensor):
        low, high = torch.minimum(a, b), torch.maximum(a, b)
    else:
        low, high = a.clamp_max(b), a.clamp_min(b)
    return low * torch.pow(1.0 + torch.pow(low / high, exponent), -1.0 / exponent)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """t^2 (3 - 2t), the engine fall of game spec 3.3 after clipping."""
    return t * t * (3.0 - 2.0 * t)


def lookup(values: tuple, index: torch.Tensor) -> torch.Tensor:
    """values[index] for a small constant table, as a chain of selects: no host-to-device
    copy, so it is safe inside CUDA-graph capture in eager mode, and it fuses when
    compiled. Indices outside the table give values[0]."""
    dtype = torch.int32 if all(isinstance(v, int) for v in values) else torch.float32
    out = torch.full_like(index, values[0], dtype=dtype)
    for i, value in enumerate(values[1:], 1):
        out = torch.where(index == i, value, out)
    return out
