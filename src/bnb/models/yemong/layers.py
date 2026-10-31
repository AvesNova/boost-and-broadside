"""Shared layers: rotary position tables, their rotation, and the gated MLP (spec 6.2-6.3)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(width: int) -> nn.RMSNorm:
    """RMSNorm with a learned scale and torch's default epsilon (the dtype's machine eps)."""
    return nn.RMSNorm(width, eps=None)


def rotary_tables(position: torch.Tensor, heading: torch.Tensor, periods_x: torch.Tensor,
                  periods_y: torch.Tensor, frequencies_heading: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each (..., tokens, 1, pairs), of the angles [x f_x, y f_y, heading f_h],
    with the position frequencies given as periods p = 2 pi / f.

    position (..., tokens, 2) in metres; heading (..., tokens) in radians (0 for a token
    without one). A position angle is 2 pi (x mod p) / p: the same rotation, but the angle
    stays below 2 pi, where x f would reach thousands of radians in the shifted frame and
    keep float32 only 1e-4 rad. For power-of-two periods the remainder is exact.
    """

    def phase(coordinate: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        return 2.0 * math.pi * (torch.remainder(coordinate, periods) / periods)

    angles = torch.cat([phase(position[..., 0:1], periods_x), phase(position[..., 1:2], periods_y),
                        heading[..., None] * frequencies_heading], -1)[..., None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first 2R dims of each head (..., tokens, heads, width) as interleaved
    (even, odd) pairs by the tables' angles; the other dims pass through."""
    rotated = 2 * cos.shape[-1]
    pairs = values[..., :rotated].unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    turned = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    return torch.cat([turned, values[..., rotated:]], -1)


class GatedMLP(nn.Module):
    """down(gelu(gate x) * up x), no biases. The gate and up projections are one matrix
    (gate rows first), so they cost one matrix product."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        with torch.no_grad():  # each half as nn.Linear(width, hidden) would start
            for half in self.gate_up.weight.chunk(2):
                nn.init.kaiming_uniform_(half, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(inputs).chunk(2, -1)
        return self.down(F.gelu(gate) * up)
