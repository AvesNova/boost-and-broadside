"""Critic distributions and the return scale (spec 8): categorical values over symlog bins,
the outcome classes and their lambda targets, and the percentile scale of returns.
"""

import torch

WIN, TIE, LOSS, UNRESOLVED = range(4)
OUTCOMES = 4


def symlog(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(values.abs())


def symexp(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(values.abs())


def value_bins(count: int, limit: float, dtype=torch.float32, device=None) -> torch.Tensor:
    """symexp(linspace(-limit, limit, count))."""
    return symexp(torch.linspace(-limit, limit, count, dtype=dtype, device=device))


def two_hot(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """(...) to (..., bins): the two neighbouring bins, linearly weighted."""
    count = bins.shape[0]
    values = torch.clamp(values, bins[0], bins[-1])
    upper = torch.clamp(torch.searchsorted(bins, values.contiguous(), right=True), 1, count - 1)
    lower = upper - 1
    weight = torch.clamp((values - bins[lower]) / (bins[upper] - bins[lower]), 0.0, 1.0)
    index = torch.arange(count, device=values.device)
    return (1.0 - weight)[..., None] * (index == lower[..., None]) \
        + weight[..., None] * (index == upper[..., None])


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return (torch.softmax(logits, -1) * bins).sum(-1)


def outcome_values(probabilities: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss)."""
    return probabilities[..., WIN] - probabilities[..., LOSS]


def outcome_class(result: torch.Tensor) -> torch.Tensor:
    """int64: win if the result is positive, loss if negative, else tie."""
    return torch.where(result > 0, WIN, torch.where(result < 0, LOSS, TIE)).long()


def outcome_targets(probabilities: torch.Tensor, result: torch.Tensor, terminated: torch.Tensor,
                    next_probabilities: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Lambda targets for the outcome distribution, (L, ..., 4).

    probabilities (L, ..., 4) are p(s_t); next_probabilities (..., 4) bootstrap p(s_L);
    result and terminated (L, ...) say how transition t ended.
    """
    unresolved = torch.zeros(OUTCOMES, dtype=probabilities.dtype, device=probabilities.device)
    unresolved[UNRESOLVED] = 1.0
    final = (outcome_class(result)[..., None] == torch.arange(OUTCOMES, device=result.device))
    final = final.to(probabilities.dtype)
    following = next_probabilities
    target = next_probabilities
    out = []
    for t in reversed(range(probabilities.shape[0])):
        target = gamma * ((1.0 - lam) * following + lam * target) + (1.0 - gamma) * unresolved
        target = torch.where(terminated[t][..., None], final[t], target)
        out.append(target)
        following = probabilities[t]
    return torch.stack(out[::-1])


class ReturnScale:
    """EMAs of the 5th and 95th percentiles of the masked returns; the scale is their gap,
    at least `floor`. The learner owns one and calls `update` once per batch."""

    def __init__(self, decay: float, floor: float, device=None, dtype=torch.float32) -> None:
        self.decay, self.floor = decay, floor
        self.low = torch.zeros((), dtype=dtype, device=device)
        self.high = torch.zeros((), dtype=dtype, device=device)

    def update(self, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fold one batch in (an empty batch is skipped) and return the scale. Reads the mask
        on the host, so it is not for captured code."""
        selected = returns[mask]
        if selected.numel():
            quantiles = torch.quantile(selected, torch.tensor([0.05, 0.95], dtype=returns.dtype,
                                                              device=returns.device))
            self.low = self.decay * self.low + (1.0 - self.decay) * quantiles[0]
            self.high = self.decay * self.high + (1.0 - self.decay) * quantiles[1]
        return self.scale()

    def scale(self) -> torch.Tensor:
        return torch.clamp_min(self.high - self.low, self.floor)
