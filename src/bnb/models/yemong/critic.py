"""Critic distributions and the return scale (spec 8): categorical values over symlog bins,
the outcome classes and their lambda targets, and the percentile scale of returns.
"""

import torch

from .codes import two_hot_at

WIN, TIE, LOSS, UNRESOLVED = range(4)
OUTCOMES = 4


def symlog(values: torch.Tensor) -> torch.Tensor:
    """sign(x) log(1 + |x|), elementwise."""
    return torch.sign(values) * torch.log1p(values.abs())


def symexp(values: torch.Tensor) -> torch.Tensor:
    """The inverse of symlog: sign(x) (exp|x| - 1), elementwise."""
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
    return two_hot_at(lower, weight, count)


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """The mean (...) of the categorical distribution softmax(logits) (..., bins) over the bin
    values."""
    return (torch.softmax(logits, -1) * bins).sum(-1)


def outcome_values(probabilities: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss)."""
    return probabilities[..., WIN] - probabilities[..., LOSS]


def final_outcome(win_loss: torch.Tensor) -> torch.Tensor:
    """(..., 4): the outcome distribution of a match that ended with win_loss (..., 2), the
    view's weights of a win and of a loss (`bnb.contracts.Events.win_loss`): win, draw (the
    rest, 1 - win - loss), loss, and unresolved 0. One-hot for a win, draw or loss; team
    deathmatch's (share, 0, 1 - share, 0) of the view's kill share, whose value
    (`outcome_values`) is 2 share - 1, the outcome reward's."""
    win, loss = win_loss.unbind(-1)
    return torch.stack([win, 1.0 - win - loss, loss, torch.zeros_like(win)], -1)


def outcome_targets(
    probabilities: torch.Tensor,
    win_loss: torch.Tensor,
    terminated: torch.Tensor,
    next_probabilities: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Lambda targets for the outcome distribution, (L, ..., 4).

    probabilities (L, ..., 4) are p(s_t); next_probabilities (..., 4) bootstrap p(s_L);
    win_loss (L, ..., 2) and terminated (L, ...) say how transition t ended (an ended
    transition's target is `final_outcome`, soft in team deathmatch).
    """
    unresolved = torch.zeros(OUTCOMES, dtype=probabilities.dtype, device=probabilities.device)
    unresolved[UNRESOLVED] = 1.0
    final = final_outcome(win_loss).to(probabilities.dtype)
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
        self.levels = torch.tensor([0.05, 0.95], dtype=dtype, device=device)

    def update(self, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fold one batch in (an empty batch is skipped) and return the scale. No host sync:
        masked-out returns are NaN to nanquantile, and an empty batch leaves the EMAs."""
        masked = torch.where(mask, returns, torch.nan).flatten()
        quantiles = torch.nanquantile(masked, self.levels)
        some = mask.any()
        decay = self.decay
        self.low = torch.where(some, decay * self.low + (1.0 - decay) * quantiles[0], self.low)
        self.high = torch.where(some, decay * self.high + (1.0 - decay) * quantiles[1], self.high)
        return self.scale()

    def scale(self) -> torch.Tensor:
        """The current scale, a scalar tensor: the gap of the percentile EMAs, at least floor."""
        return torch.clamp_min(self.high - self.low, self.floor)
