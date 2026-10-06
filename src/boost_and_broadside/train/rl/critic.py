"""Categorical critic: symlog-spaced value bins, the outcome classes, the return scale.

Rewards, returns and advantages stay scalar and raw; only the critic's output is
a distribution. Each per-ship reward level is read off logits over fixed bins
whose locations are symlog-spaced, and its value is the expectation in raw
units. The target is the realised lambda-return two-hot over the bins
(DreamerV3), trained by cross-entropy; the target's mean equals the return, so
what the value means does not change, only how it is fitted.

The match outcome is read per environment off the global token as four classes,
win, tie, loss and unresolved. The discount is modelled as probability mass
leaking into "unresolved", worth zero, so the categorical target's expectation
is exactly the scalar lambda-return GAE uses (:func:`outcome_targets`).

Advantages are normalised once, after the levels are summed: divided by the EMA
5th-95th percentile spread of the summed return, floored at one win
(DreamerV3's ``perc`` normaliser; the mean is not subtracted).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from boost_and_broadside.constants import (
    NUM_OUTCOME_CLASSES,
    OUTCOME_LOSS_INDEX,
    OUTCOME_TIE_INDEX,
    OUTCOME_UNRESOLVED_INDEX,
    OUTCOME_WIN_INDEX,
)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + |x|)``."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(x) * torch.expm1(x.abs())


class CriticOutput(NamedTuple):
    """What the value heads return.

    ``value`` is the expected return of every reward level in raw units, with
    one exception: the outcome column is ``P(win) - P(loss)`` in the observer's
    frame (team 0 of the canonical view), unweighted and unsigned per ship. The
    trainer signs it by each ship's authoritative team and scales it by the
    outcome level's weight (see ``PPOTrainer._ship_values``).
    """

    value: torch.Tensor  # (..., N, K)
    local_logits: torch.Tensor | None  # (..., N, K_local, bins) or None at rollout
    outcome_logits: torch.Tensor | None  # (..., 4) per environment, or None


def value_bins(num_bins: int, symlog_limit: float) -> torch.Tensor:
    """Bin locations ``symexp(linspace(-L, L, num_bins))`` in raw units."""
    return symexp(torch.linspace(-symlog_limit, symlog_limit, num_bins))


def two_hot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Spread each value over its two neighbouring bins so the mean is exact.

    Values beyond the outermost bins land on the edge bin.

    Args:
        x: (...) raw values.
        bins: (n,) ascending bin locations.

    Returns:
        (..., n) probabilities whose expectation over ``bins`` is ``x`` clamped
        to the support.
    """
    x = x.clamp(bins[0], bins[-1])
    upper = torch.searchsorted(bins, x.contiguous(), right=True).clamp(1, bins.numel() - 1)
    lower = upper - 1
    low_value, high_value = bins[lower], bins[upper]
    weight_high = ((x - low_value) / (high_value - low_value)).clamp(0.0, 1.0)
    target = torch.zeros(*x.shape, bins.numel(), dtype=x.dtype, device=x.device)
    target.scatter_(-1, lower.unsqueeze(-1), (1.0 - weight_high).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), weight_high.unsqueeze(-1))
    return target


def expectation(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value of a categorical over ``bins``: (..., n) -> (...)."""
    return (F.softmax(logits.float(), dim=-1) * bins).sum(-1)


def outcome_values(probabilities: torch.Tensor) -> torch.Tensor:
    """``P(win) - P(loss)``: the outcome on the +-1 scale, (..., 4) -> (...)."""
    return probabilities[..., OUTCOME_WIN_INDEX] - probabilities[..., OUTCOME_LOSS_INDEX]


def outcome_class_of(result: torch.Tensor) -> torch.Tensor:
    """The realised class of a signed result: + win, - loss, 0 tie. (...) -> (...) long."""
    return torch.where(
        result > 0.0,
        OUTCOME_WIN_INDEX,
        torch.where(result < 0.0, OUTCOME_LOSS_INDEX, OUTCOME_TIE_INDEX),
    ).long()


@torch.no_grad()
def outcome_targets(
    probabilities: torch.Tensor,
    result: torch.Tensor,
    terminated: torch.Tensor,
    next_probabilities: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """The categorical lambda-return of the match outcome, per environment.

    ``Q_t = gamma * [(1 - lambda) * p(s_{t+1}) + lambda * Q_{t+1}]
    + (1 - gamma) * e_unresolved`` and ``Q_t = onehot(result)`` on a terminal
    transition. Valued at win +1, loss -1, tie and unresolved 0, its
    expectation is the scalar lambda-return with ``V = P(win) - P(loss)``.

    Args:
        probabilities: (T, B, 4) rollout-time class probabilities p(s_t).
        result: (T, B) signed match result on each transition (zero until the
            terminal one).
        terminated: (T, B) bool, the episode ended on transition t.
        next_probabilities: (B, 4) p(s_T) for the bootstrap.
        next_done: (B,) bool, the episode ended on the last transition.
        gamma, gae_lambda: The outcome level's discount and GAE lambda.

    Returns:
        (T, B, 4) target distributions.
    """
    steps = probabilities.shape[0]
    unresolved = torch.zeros(NUM_OUTCOME_CLASSES, device=probabilities.device)
    unresolved[OUTCOME_UNRESOLVED_INDEX] = 1.0
    realised = F.one_hot(outcome_class_of(result), NUM_OUTCOME_CLASSES).float()  # (T, B, 4)
    targets = torch.empty_like(probabilities, dtype=torch.float32)
    following = next_probabilities.float()  # Q_{t+1}, bootstrapped from p(s_T)
    for t in reversed(range(steps)):
        nxt = next_probabilities.float() if t == steps - 1 else probabilities[t + 1].float()
        target = (
            gamma * ((1.0 - gae_lambda) * nxt + gae_lambda * following) + (1.0 - gamma) * unresolved
        )
        ended = terminated[t].unsqueeze(-1)  # (B, 1)
        target = torch.where(ended, realised[t], target)
        targets[t] = target
        following = target
    del next_done  # an episode ending on the last transition is ``terminated[-1]``
    return targets


class ReturnNormalizer:
    """DreamerV3's return scale: ``max(floor, EMA p95 - EMA p5)`` of the summed return.

    Advantages are divided by it, without subtracting a mean, so they are never
    amplified while the return spread is under one win.

    Args:
        decay: EMA decay of the percentiles (DreamerV3: 0.99).
        floor: Minimum scale (DreamerV3: 1.0, one win).
        device: Where the running statistics live.
    """

    LOW_PERCENTILE = 0.05
    HIGH_PERCENTILE = 0.95

    def __init__(self, decay: float, floor: float, device: torch.device) -> None:
        self.decay = decay
        self.floor = floor
        self.low = torch.zeros((), device=device)
        self.high = torch.zeros((), device=device)

    @torch.no_grad()
    def update(self, returns: torch.Tensor, mask: torch.Tensor) -> None:
        """Fold one batch's percentiles into the EMA.

        Args:
            returns: (...) summed returns.
            mask: (...) bool, which entries count (living actors).
        """
        values = returns.float()[mask]
        if values.numel() == 0:
            return
        quantiles = torch.quantile(
            values,
            torch.tensor([self.LOW_PERCENTILE, self.HIGH_PERCENTILE], device=values.device),
        )
        rate = 1.0 - self.decay
        self.low = self.decay * self.low + rate * quantiles[0]
        self.high = self.decay * self.high + rate * quantiles[1]

    @property
    def scale(self) -> torch.Tensor:
        """The divisor for the summed advantage, () float."""
        return (self.high - self.low).clamp(min=self.floor)

    def state_dict(self) -> dict:
        return {"low": self.low.clone(), "high": self.high.clone()}

    def load_state_dict(self, state: dict) -> None:
        self.low = state["low"].to(self.low.device)
        self.high = state["high"].to(self.high.device)


#: Equal-width bins of the predicted win probability for the reliability diagram.
OUTCOME_CALIBRATION_BINS = 10


def outcome_calibration(
    probabilities: torch.Tensor, result: torch.Tensor, terminated: torch.Tensor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``P(win | resolved)`` against realised wins, as additive ``(total, count)`` sums.

    Every transition whose episode ends inside the rollout is scored against
    that episode's result: a win if the terminal result is positive, otherwise
    a loss or a tie. Transitions whose episode is still running at the end of the
    rollout carry no realised outcome and are left out.

    Args:
        probabilities: (T, B, 4) rollout-time outcome-class probabilities.
        result: (T, B) signed result on each transition, zero until the terminal one.
        terminated: (T, B) bool, the episode ended on transition t.

    Returns:
        ``outcome/calibration_brier``, ``outcome/predicted_win``,
        ``outcome/realised_win``, and per bin ``b`` of the prediction
        ``outcome/calibration_bin_<b>/{predicted,realised}`` -- a reliability
        diagram, whose two series agree when the head is calibrated.
    """

    steps = probabilities.shape[0]
    realised = torch.full_like(result, float("nan"), dtype=torch.float32)
    pending = torch.full_like(result[0], float("nan"), dtype=torch.float32)
    for t in range(steps - 1, -1, -1):
        pending = torch.where(terminated[t], result[t].float(), pending)
        realised[t] = pending
    resolved = ~torch.isnan(realised)
    count = resolved.sum().float()
    win = (realised > 0).float()
    resolved_mass = probabilities[..., :OUTCOME_UNRESOLVED_INDEX].sum(-1)
    predicted = probabilities[..., OUTCOME_WIN_INDEX] / resolved_mass.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    mask = resolved.float()
    diagnostics = {
        "outcome/calibration_brier": (((predicted - win).square() * mask).sum(), count),
        "outcome/predicted_win": ((predicted * mask).sum(), count),
        "outcome/realised_win": ((win * mask).sum(), count),
    }
    bins = (predicted * OUTCOME_CALIBRATION_BINS).long().clamp(0, OUTCOME_CALIBRATION_BINS - 1)
    for b in range(OUTCOME_CALIBRATION_BINS):
        in_bin = (bins == b).float() * mask
        bin_count = in_bin.sum()
        diagnostics[f"outcome/calibration_bin_{b}/predicted"] = (
            (predicted * in_bin).sum(),
            bin_count,
        )
        diagnostics[f"outcome/calibration_bin_{b}/realised"] = ((win * in_bin).sum(), bin_count)
    return diagnostics
