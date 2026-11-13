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


def flight_path_velocity(
    speed: torch.Tensor,
    heading_cos: torch.Tensor,
    heading_sin: torch.Tensor,
    angle_cos: torch.Tensor,
    angle_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) a velocity of this speed along the flight path: the nose turned back by the
    angle of attack (its cos and sin)."""
    return (
        speed * (heading_cos * angle_cos + heading_sin * angle_sin),
        speed * (heading_sin * angle_cos - heading_cos * angle_sin),
    )


def softmin(a: torch.Tensor, b, exponent: float) -> torch.Tensor:
    """a (1 + (a/b)^q)^(-1/q) for a, b >= 0.

    Written with the smaller and the larger argument (the formula is symmetric in a and b),
    so the power never overflows. softmin(0, 0) is 0 (a turn budget of exactly 0 with the
    stick neutral), not 0/0.
    """
    if isinstance(b, torch.Tensor):
        low, high = torch.minimum(a, b), torch.maximum(a, b)
    else:
        low, high = a.clamp_max(b), a.clamp_min(b)
    positive = high > 0.0
    ratio = torch.where(positive, low / torch.where(positive, high, 1.0), 0.0)
    return low * torch.pow(1.0 + torch.pow(ratio, exponent), -1.0 / exponent)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """t^2 (3 - 2t), the engine fall of game spec 3.3 after clipping."""
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: torch.Tensor) -> torch.Tensor:
    """t^3 (t (6t - 15) + 10), the quintic step of t in [0, 1] (zero slope and curvature
    at both ends): the fields' profile (game spec 4) and the boundary's damage ramp."""
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def lookup(values: tuple, index: torch.Tensor) -> torch.Tensor:
    """values[index] for a small constant table, as a chain of selects: no host-to-device
    copy, so it is safe inside CUDA-graph capture in eager mode, and it fuses when
    compiled. Indices outside the table give values[0]."""
    dtype = torch.int32 if all(isinstance(v, int) for v in values) else torch.float32
    out = torch.full_like(index, values[0], dtype=dtype)
    for i, value in enumerate(values[1:], 1):
        out = torch.where(index == i, value, out)
    return out
