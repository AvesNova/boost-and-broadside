"""Critic distributions and the return scale (spec 8): categorical values over symlog bins,
the outcome classes and their lambda targets, and the percentile scale of returns.

The critic's value levels are named (`bnb.contracts.VALUE_LEVELS`: each reward level's
offence and defence); the outcome level's are read off the outcome head (`outcome_part`),
every other level has a categorical value per ship. A build from before the split names the
reward levels themselves, and its outcome level is P(win) - P(loss) whole.
"""

import torch

from bnb.contracts import level_part

from .codes import HL_GAUSS_SIGMA, hl_gauss

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


def value_targets(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """(...) to (..., bins): the critic's HL-Gauss target (Farebrother et al. 2024, "Stop
    Regressing"), the Gaussian of `codes.HL_GAUSS_SIGMA` bin widths at the value's position
    on the symlog bins, integrated over each bin and renormalised over them
    (`codes.hl_gauss`). The bins are evenly spaced in symlog (`value_bins`); a value past
    the outer bins is clamped to them."""
    count = bins.shape[0]
    low, high = symlog(bins[0]), symlog(bins[-1])
    position = (symlog(values).clamp(low, high) - low) * ((count - 1) / (high - low))
    return hl_gauss(position, 0, count - 1, HL_GAUSS_SIGMA)


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """The mean (...) of the categorical distribution softmax(logits) (..., bins) over the bin
    values."""
    return (torch.softmax(logits, -1) * bins).sum(-1)


def outcome_values(probabilities: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss)."""
    return probabilities[..., WIN] - probabilities[..., LOSS]


def outcome_part(level: str, outcome_level: str | None) -> str | None:
    """What the outcome head gives the value level `level`: "offence" (P(win), the outcome's
    offence), "defence" (-P(loss)), "whole" (P(win) - P(loss), a build's level named
    `outcome_level` itself), or None, a per-ship level (every level without an outcome
    level)."""
    if outcome_level is None:
        return None
    if level == outcome_level:
        return "whole"
    for part in ("offence", "defence"):
        if level == level_part(outcome_level, part):
            return part
    return None


def outcome_part_values(probabilities: torch.Tensor, part: str) -> torch.Tensor:
    """(...) the value of an outcome part (`outcome_part`) from the outcome probabilities
    (..., 4): P(win), -P(loss) or P(win) - P(loss). The parts sum to the whole."""
    if part == "offence":
        return probabilities[..., WIN]
    if part == "defence":
        return -probabilities[..., LOSS]
    return outcome_values(probabilities)


def local_levels(levels: tuple[str, ...], outcome_level: str | None) -> tuple[int, ...]:
    """The indices into levels of the per-ship levels, every level the outcome head does not
    give (`outcome_part`): the levels with a value distribution."""
    return tuple(i for i, name in enumerate(levels) if outcome_part(name, outcome_level) is None)


def check_levels(levels: tuple[str, ...], outcome_level: str | None) -> None:
    """Refuse a critic layout: levels not distinct or none; an outcome level the head gives
    no level of; no per-ship level."""
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"the critic's levels {levels} must be distinct, and some")
    given = [name for name in levels if outcome_part(name, outcome_level) is not None]
    if outcome_level is not None and not given:
        raise ValueError(f"outcome level {outcome_level!r} gives none of {levels}")
    if len(given) == len(levels):
        raise ValueError("the critic needs at least one per-ship level besides the outcome")


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
