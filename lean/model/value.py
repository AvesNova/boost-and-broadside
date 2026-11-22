"""Categorical value targets: uniform bin centres, HL-Gauss targets and expectation decode.

HL-Gauss (Farebrother et al. 2024, "Stop Regressing"): the target for a scalar is a
Gaussian centred on it, integrated over each bin and renormalised over the support.
"""

import math

import torch
from torch import Tensor


def value_bin_centres(low: Tensor | float, high: Tensor | float, bins: int) -> Tensor:
    """Uniformly spaced bin centres from `low` to `high` inclusive, in value units.

    `low` and `high` are numbers or tensors of one shape (...); the result is (..., bins)
    float32. `low` and `high` are the outermost centres, so the bins cover half a bin
    width beyond them.
    """
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)
    fraction = torch.linspace(0.0, 1.0, bins)
    return low[..., None] + (high - low)[..., None] * fraction


def hl_gauss_targets(values: Tensor, centres: Tensor, sigma_bins: float = 0.75) -> Tensor:
    """HL-Gauss target probabilities for scalar values.

    values: (...) float in value units. centres: (..., bins) uniform bin centres that
    broadcast against values[..., None] (for per-level values (..., LEVELS) pass
    (LEVELS, bins)). sigma_bins: the Gaussian's standard deviation in bin widths.
    Returns (..., bins) float32 that sums to 1. Values are clamped to the outermost
    centres. The expectation of the target equals the value when it lies at least three
    bin widths inside the outermost centres; nearer the edge the truncated tail pulls
    the expectation inward, so supports should be set that much wider than the returns.
    """
    centres = centres.float()
    width = centres[..., 1:2] - centres[..., 0:1]
    edges = torch.cat([centres - 0.5 * width, centres[..., -1:] + 0.5 * width], dim=-1)
    values = torch.maximum(torch.minimum(values.float(), centres[..., -1]), centres[..., 0])
    scaled = (edges - values[..., None]) / (sigma_bins * width * math.sqrt(2.0))
    cumulative = torch.erf(scaled)
    mass = cumulative[..., 1:] - cumulative[..., :-1]
    return mass / (cumulative[..., -1:] - cumulative[..., :1])


def decode_value(logits: Tensor, centres: Tensor) -> Tensor:
    """Expected value of categorical value logits.

    logits: (..., bins). centres: (..., bins) broadcastable bin centres in value units.
    Returns (...) float32 in value units. The softmax runs in float32.
    """
    return (torch.softmax(logits.float(), dim=-1) * centres).sum(dim=-1)
