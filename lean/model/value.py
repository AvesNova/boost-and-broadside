"""Categorical supports: bin centres, symlog spacing, HL-Gauss targets and expectation decode.

Every categorical head (the critic's levels and the next-state columns) has COLUMNS bins,
uniform in its bin space: s(x) = sign(x) knee ln(1 + |x| / knee) for a symlog support (knee
in target units), x itself for a linear one (knee 0). HL-Gauss (Farebrother et al. 2024,
"Stop Regressing"): the target for a scalar is a Gaussian centred on s(scalar) in bin space,
integrated over each bin and renormalised over the support; the decode is the expectation
over the centres mapped back to target units.
"""

import math

import torch
from torch import Tensor

# Every support spans its measured range with MEASURED_COLUMNS columns (the outermost centres
# at the measured ends, so the range sets the column width in bin space), then BUFFER_COLUMNS
# more of that width at each end: room for the HL-Gauss target near the measured ends and for
# outliers. 63 in all, an odd count inside the range, so 0 is a centre of a symmetric range
# (author 2026-11-25).
MEASURED_COLUMNS = 57
BUFFER_COLUMNS = 3
COLUMNS = MEASURED_COLUMNS + 2 * BUFFER_COLUMNS


def symlog(x: Tensor, knee: Tensor | float) -> Tensor:
    """Bin space of target units x (...): sign(x) knee ln(1 + |x| / knee) where knee > 0, x
    where knee is 0 (linear). knee: a number or a tensor broadcasting against x (per column)."""
    knee = torch.as_tensor(knee, dtype=torch.float32, device=x.device)
    safe = torch.where(knee > 0, knee, torch.ones_like(knee))
    return torch.where(knee > 0, torch.sign(x) * safe * torch.log1p(x.abs() / safe), x)


def symexp(s: Tensor, knee: Tensor | float) -> Tensor:
    """The inverse of symlog: bin space back to target units."""
    knee = torch.as_tensor(knee, dtype=torch.float32, device=s.device)
    safe = torch.where(knee > 0, knee, torch.ones_like(knee))
    return torch.where(knee > 0, torch.sign(s) * safe * torch.expm1(s.abs() / safe), s)


def buffered_centres(
    low: Tensor | float, high: Tensor | float, knee: Tensor | float = 0.0, bins: int = COLUMNS
) -> Tensor:
    """Bin centres, in bin space, of a support measured from `low` to `high` (target units;
    numbers or tensors of one shape (...), `knee` broadcasting against them): (..., bins)
    float32, uniform in bin space, the column width (s(high) - s(low)) / (bins - 2
    BUFFER_COLUMNS - 1), with BUFFER_COLUMNS columns beyond each measured end."""
    low = symlog(torch.as_tensor(low, dtype=torch.float32), knee)
    high = symlog(torch.as_tensor(high, dtype=torch.float32), knee)
    width = (high - low) / (bins - 2 * BUFFER_COLUMNS - 1)
    return value_bin_centres(low - BUFFER_COLUMNS * width, high + BUFFER_COLUMNS * width, bins)


def value_bin_centres(low: Tensor | float, high: Tensor | float, bins: int) -> Tensor:
    """Uniformly spaced bin centres from `low` to `high` inclusive.

    `low` and `high` are numbers or tensors of one shape (...); the result is (..., bins)
    float32. `low` and `high` are the outermost centres, so the bins cover half a bin
    width beyond them.
    """
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)
    fraction = torch.linspace(0.0, 1.0, bins)
    return low[..., None] + (high - low)[..., None] * fraction


def hl_gauss_targets(
    values: Tensor, centres: Tensor, knee: Tensor | float = 0.0, sigma_bins: float = 0.75
) -> Tensor:
    """HL-Gauss target probabilities for scalar values.

    values: (...) float in target units. centres: (..., bins) bin-space centres
    (buffered_centres) that broadcast against values[..., None] (for per-level values
    (..., LEVELS) pass (LEVELS, bins)); knee: the supports' knees, broadcasting against
    values. sigma_bins: the Gaussian's standard deviation in bin widths (of bin space).
    Returns (..., bins) float32 that sums to 1. Values are clamped to the outermost
    centres. In bin space the expectation of the target equals s(value) when it lies at
    least three bin widths inside the outermost centres; nearer the edge the truncated tail
    pulls it inward (what the buffer columns are for).
    """
    centres = centres.float()
    width = centres[..., 1:2] - centres[..., 0:1]
    edges = torch.cat([centres - 0.5 * width, centres[..., -1:] + 0.5 * width], dim=-1)
    values = symlog(values.float(), knee)
    values = torch.maximum(torch.minimum(values, centres[..., -1]), centres[..., 0])
    scaled = (edges - values[..., None]) / (sigma_bins * width * math.sqrt(2.0))
    cumulative = torch.erf(scaled)
    mass = cumulative[..., 1:] - cumulative[..., :-1]
    return mass / (cumulative[..., -1:] - cumulative[..., :1])


def decode_value(logits: Tensor, centres: Tensor, knee: Tensor | float = 0.0) -> Tensor:
    """Expected value of categorical logits in target units.

    logits: (..., bins). centres: (..., bins) broadcastable bin-space centres; knee: their
    knees, broadcasting against centres[..., 0]. The expectation is over the centres mapped
    back to target units. Returns (...) float32. The softmax runs in float32.
    """
    knee = torch.as_tensor(knee, dtype=torch.float32, device=centres.device)
    real = symexp(centres.float(), knee[..., None])
    return (torch.softmax(logits.float(), dim=-1) * real).sum(dim=-1)
