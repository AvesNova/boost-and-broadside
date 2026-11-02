"""The loss terms (learner spec 4), each a masked sum over its tokens divided by a given
denominator: the mask summed over the whole minibatch, so micro-batches add up to the
unsplit minibatch exactly. v1's math; fixtures `learner_policy`, `learner_critic` and
`learner_shot_cross_entropy`. Every reduction is float32 or wider (the inputs' dtype).
"""

import torch
import torch.nn.functional as F

from bnb.models.yemong import codes, critic


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, 0.0).sum()


def surrogate(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    return_scale: torch.Tensor | float,
    clip: float,
) -> torch.Tensor:
    """The clipped surrogate per token, max(-A r, -A clip(r, 1 +- clip)), with A the
    advantage over the return scale (no mean subtracted) and r = exp(logp - old_logp)."""
    scaled = advantage / return_scale
    ratio = torch.exp(logp - old_logp)
    return torch.maximum(-scaled * ratio, -scaled * torch.clamp(ratio, 1.0 - clip, 1.0 + clip))


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-sum target log softmax(logits) over the last axis."""
    return -(target * F.log_softmax(logits, -1)).sum(-1)


def index_cross_entropy(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """-log softmax(logits)[index], per token."""
    return -F.log_softmax(logits, -1).gather(-1, index.long()[..., None])[..., 0]


def critic_losses(
    local_logits: torch.Tensor,
    outcome_logits: torch.Tensor,
    local_returns: torch.Tensor,
    outcome_targets: torch.Tensor,
    valued: torch.Tensor,
    bins: torch.Tensor,
    local_levels: tuple[int, ...],
    outcome_level: int,
    valued_denominator: torch.Tensor | float,
    step_denominator: torch.Tensor | float,
) -> torch.Tensor:
    """(K,) one loss per level, in level order.

    Per-ship levels (local_logits (..., S, K - 1, bins), local_returns (..., S, K - 1)): the
    cross-entropy against the two-hot of the realised return over the valued ships (alive,
    or waiting to respawn), divided by valued_denominator. The outcome (outcome_logits and
    outcome_targets (..., 4)): the cross-entropy against its categorical targets per
    environment-step, divided by step_denominator.
    """
    local = cross_entropy(local_logits, critic.two_hot(local_returns, bins))  # (..., S, K - 1)
    local = torch.where(valued[..., None], local, 0.0).flatten(0, -2).sum(0) / valued_denominator
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
