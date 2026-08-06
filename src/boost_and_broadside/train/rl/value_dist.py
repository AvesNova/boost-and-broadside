"""Categorical value distributions: bin geometry, targets, and decoding.

The critic predicts a distribution over a fixed grid of scalar values rather than
a single number, and is trained by cross-entropy against a target distribution
built from the observed return. The scalar the rest of the system needs — for
GAE, for bootstrapping, for diagnostics — is the distribution's mean.

Why this beats a regression head here: the return targets this critic fits are
sparse and heavy-tailed. Terminal components fire once per episode and are zero
everywhere else, so a squared-error head is pulled toward the mean of a bimodal
target and its gradient scales with the residual, which the per-component scale
then has to keep in bounds. Cross-entropy's gradient is bounded by construction
and does not care about the target's scale at all, which is what lets
``ReturnScaler.min_span`` go back to being a pure divide-by-zero epsilon.

**Binning happens in normalized space** — after :class:`ReturnScaler` maps each
component's returns to roughly [-1, 1]. A fixed grid is only meaningful if the
quantity it covers has a stable scale, and the scaler is what provides that. It
also means one support serves all K components, whose natural scales differ by
orders of magnitude.

``support`` is a half-span in those normalized units, so the grid covers
[-support, +support]. Returns outside it are clamped to the end bins: the
distribution cannot represent them, and its mean saturates there. Measured on a
live checkpoint the worst-case |z| was 6.3 with p99.9 at 4.1, so ±5 clips about
3 in 100,000 targets.
"""

import torch
import torch.nn.functional as F

# Numerical floor for the HL-Gauss smoothing width, in bin widths. Below this the
# Gaussian is narrower than float error on the CDF differences and the target
# degenerates into noise rather than into the two-hot limit.
_MIN_SIGMA_BINS = 1e-3


def bin_centers(
    num_bins: int, support: float, *, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The (num_bins,) grid of scalar values the distribution is defined over.

    Endpoints sit exactly on ±support, so a target at the edge of the support is
    represented exactly rather than one half-width inside it.
    """
    return torch.linspace(-support, support, num_bins, device=device, dtype=dtype)


def expected_value(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Mean of the categorical distribution: (..., K, bins) logits → (..., K).

    Computed in fp32 regardless of the autocast dtype in force. This value feeds
    GAE, whose recursion accumulates over the rollout horizon, and bf16's ~3
    decimal digits are not enough to carry a probability-weighted sum through it.

    ``dtype=`` on the softmax rather than ``.float()`` on the input: the logits are
    the largest tensor in the critic path, and casting first materializes a second
    full-size copy before the softmax allocates its own.
    """
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs @ centers.to(probs.dtype)


def two_hot_targets(
    values: torch.Tensor, centers: torch.Tensor, sigma: float = 0.0
) -> torch.Tensor:
    """Target distribution over ``centers`` for scalar ``values`` (..., K) → (..., K, bins).

    ``sigma`` is the smoothing width **in bin widths**:

    * ``0`` — two-hot. All mass on the two bins bracketing the value, split by
      linear interpolation, so the target's mean is the value exactly.
    * ``> 0`` — HL-Gauss. Mass from a Gaussian of that width, integrated over
      each bin. Trades a little target sharpness for a smoother loss surface;
      whether that helps is empirical, which is why it is a knob and not a
      rewrite.

    Values outside the support are clamped to the end bins.
    """
    num_bins = centers.numel()
    lo, hi = centers[0], centers[-1]
    values = values.float().clamp(lo, hi)

    if num_bins == 1:
        return torch.ones(*values.shape, 1, device=values.device, dtype=values.dtype)

    width = (hi - lo) / (num_bins - 1)

    if sigma > _MIN_SIGMA_BINS:
        # Integrate the Gaussian over each bin: P(bin i) = Φ(edge_{i+1}) - Φ(edge_i),
        # with the outer edges at ±inf so all the tail mass lands in the end bins
        # and the target stays a proper distribution.
        edges = torch.cat(
            [
                torch.full_like(centers[:1], float("-inf")),
                (centers[:-1] + centers[1:]) * 0.5,
                torch.full_like(centers[-1:], float("inf")),
            ]
        )
        z = (edges - values.unsqueeze(-1)) / (sigma * width)
        cdf = torch.special.ndtr(z)
        return (cdf[..., 1:] - cdf[..., :-1]).clamp(min=0.0)

    # Two-hot: linear interpolation onto the two bracketing bins.
    pos = ((values - lo) / width).clamp(0, num_bins - 1)
    lower = pos.floor()
    upper_weight = pos - lower
    lower_idx = lower.long()
    # At the top edge pos == num_bins-1 exactly, so lower_idx would index one past
    # the last bin. Pull it back and hand all the weight to the lower bin.
    at_top = lower_idx >= num_bins - 1
    lower_idx = lower_idx.clamp(max=num_bins - 2)
    upper_weight = torch.where(at_top, torch.ones_like(upper_weight), upper_weight)

    target = torch.zeros(*values.shape, num_bins, device=values.device, dtype=values.dtype)
    target.scatter_(-1, lower_idx.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    target.scatter_add_(-1, (lower_idx + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-(…, K) cross-entropy between a target distribution and predicted logits.

    In fp32: ``log_softmax`` over a 51-way support under bf16 loses enough
    precision in the small-probability bins to matter for a loss that is summed
    over K components and a full rollout. Upcasting inside the softmax rather
    than before it avoids a second full-size copy of the logits.
    """
    return -(targets * F.log_softmax(logits, dim=-1, dtype=torch.float32)).sum(dim=-1)
