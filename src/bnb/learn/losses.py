"""The loss terms (learner spec 4), each a masked sum over its tokens divided by a given
denominator: the mask summed over the whole minibatch, so micro-batches add up to the
unsplit minibatch exactly. v1's math; fixtures `learner_policy`, `learner_critic` and
`learner_shot_cross_entropy`. Every reduction is float32 or wider (the inputs' dtype).
"""

import torch
import torch.nn.functional as F

from bnb.models.yemong import codes, critic


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The sum of values where mask is true: a scalar in the values' dtype."""
    return torch.where(mask, values, 0.0).sum()


def phasor_square_error(readout: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The rotary read-out's error per token: the squared distance between its (cos, sin)
    pair and the true one, averaged over the frequencies. readout and target
    (..., 2, pairs); (...). Its minimum over a distribution of true angles is the expected
    phasor, whose length shrinks with the spread; an all-zero read-out scores 1."""
    return (readout - target).square().sum(-2).mean(-1)


def surrogate(
    log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    return_scale: torch.Tensor | float,
    clip: float,
    clip_high: float | None = None,
) -> torch.Tensor:
    """The clipped surrogate per token, max(-A r, -A clip(r, 1 - clip, 1 + clip_high)), with
    A the advantage over the return scale (no mean subtracted) and r the probability ratio,
    exp(log_probability - old_log_probability). clip_high: the upper bound's margin, `clip`
    when None (the symmetric clip); a larger one is DAPO's clip-higher."""
    scaled = advantage / return_scale
    ratio = torch.exp(log_probability - old_log_probability)
    high = clip if clip_high is None else clip_high
    return torch.maximum(-scaled * ratio, -scaled * torch.clamp(ratio, 1.0 - clip, 1.0 + high))


def total_variation(old_probabilities: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """(...) the exact total variation between the rollout's joint action distribution
    `old_probabilities` (..., NUM_JOINT) and the current policy's, softmax of `logits`
    (..., NUM_JOINT): half the summed absolute difference, detached, float32."""
    current = torch.softmax(logits.detach().float(), -1)
    return 0.5 * (current - old_probabilities.float()).abs().sum(-1)


def binary_total_variation(
    log_probability: torch.Tensor, old_log_probability: torch.Tensor
) -> torch.Tensor:
    """(...) DPPO's binary approximation: the total variation of the two-way split {the
    sampled action, the rest}, |mu(a) - pi(a)|, detached. A lower bound of the exact form
    (the total variation is the largest difference over every event)."""
    return (log_probability.detach().exp() - old_log_probability.exp()).abs()


def dppo_keep(
    divergence: torch.Tensor, ratio: torch.Tensor, advantage: torch.Tensor, delta: float
) -> torch.Tensor:
    """(...) bool, DPPO's mask (Qi et al. 2026, eq. 12): False where the divergence is past
    `delta` and the update moves further away (A > 0 and r > 1, or A < 0 and r < 1); a move
    back toward ratio 1 is never masked."""
    ratio = ratio.detach()
    away = ((advantage > 0) & (ratio > 1.0)) | ((advantage < 0) & (ratio < 1.0))
    return ~((divergence > delta) & away)


def dppo_surrogate(
    log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    return_scale: torch.Tensor | float,
    keep: torch.Tensor,
) -> torch.Tensor:
    """DPPO's surrogate per token: the unclipped -A r where `keep` (`dppo_keep`), else 0
    (no gradient)."""
    scaled = advantage / return_scale
    ratio = torch.exp(log_probability - old_log_probability)
    return torch.where(keep, -scaled * ratio, 0.0)


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-sum target log softmax(logits) over the last axis."""
    return -(target * F.log_softmax(logits, -1)).sum(-1)


def index_cross_entropy(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """-log softmax(logits)[index], per token."""
    return -F.log_softmax(logits, -1).gather(-1, index.long()[..., None])[..., 0]


def critic_losses(
    local_logits: torch.Tensor,
    outcome_logits: torch.Tensor | None,
    local_returns: torch.Tensor,
    outcome_targets: torch.Tensor | None,
    valued: torch.Tensor,
    bins: torch.Tensor,
    local_levels: tuple[int, ...],
    outcome_level: int | None,
    valued_denominator: torch.Tensor | float,
    step_denominator: torch.Tensor | float,
) -> torch.Tensor:
    """(levels,) one loss per level, in level order.

    Per-ship levels (local_logits (..., S, per-ship levels, bins), local_returns
    (..., S, per-ship levels)): the cross-entropy against the two-hot of the realised return
    over the valued ships (alive, or waiting to respawn), divided by valued_denominator. The
    outcome (outcome_logits and outcome_targets (..., 4)): the cross-entropy against its
    categorical targets per environment-step, divided by step_denominator. Without an
    outcome level (outcome_level, its logits and targets None) every level is per-ship.
    """
    local = cross_entropy(local_logits, critic.two_hot(local_returns, bins))  # (..., S, levels)
    local = torch.where(valued[..., None], local, 0.0).flatten(0, -2).sum(0) / valued_denominator
    if outcome_level is None:
        return local
    outcome = cross_entropy(outcome_logits, outcome_targets).sum() / step_denominator
    columns = dict(zip(local_levels, local.unbind(-1), strict=True))
    columns[outcome_level] = outcome
    return torch.stack([columns[level] for level in range(len(columns))])


def shot_cross_entropy(
    trajectory_logits: torch.Tensor,
    trajectory_target: torch.Tensor,
    outcome_logits: torch.Tensor,
    outcome_target: torch.Tensor,
    runs: codes.Runs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trajectory cross-entropy per part (..., parts), each summed over its softmax
    groups (runs: (groups, size) per part), and the outcome code's (...)."""
    per_group = codes.grouped_cross_entropy(trajectory_logits, trajectory_target, runs)
    parts = torch.stack(
        [part.sum(-1) for part in per_group.split([groups for groups, _ in runs], -1)], -1
    )
    return parts, cross_entropy(outcome_logits, outcome_target)
