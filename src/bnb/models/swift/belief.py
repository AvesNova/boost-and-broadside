"""The belief store (spec 5.2 steps 2 and 5): assimilating sight, storing the forecast.

Both functions are width-generic (any number of mean, spread and action channels) and
branch-free, so they fuse under `torch.compile`.
"""

import torch


def observe(valid, age, means, spreads, action, truth, visible, spawned):
    """Assimilate one observation.

    valid, visible, spawned (..., N) bool; age (..., N) int ticks; means, truth (..., N, C);
    spreads (..., N, U); action (..., N, A) probabilities.
    Returns (valid, age, means, spreads, action), the belief after sight."""
    valid = (valid & ~spawned) | visible
    seen = visible.unsqueeze(-1)
    kept = valid.unsqueeze(-1)
    means = torch.where(seen, truth, torch.where(kept, means, 0.0))
    spreads = torch.where(seen | ~kept, 0.0, spreads)
    age = torch.where(visible, 0, torch.where(valid, age + 1, 0)).to(age.dtype)
    action = torch.where(kept, action, 1.0 / action.shape[-1])
    return valid, age, means, spreads, action


def advance(valid, means, spreads, action, prediction, enemy_action_logits):
    """Store the forecast.

    prediction (..., N, C + U): decoded means, then spreads. enemy_action_logits (..., N, A)
    or None (no forecast yet: the action becomes uniform). Returns (means, spreads, action,
    nonfinite), nonfinite the count over every
    prediction entry, shaped like `valid` without its last axis (per view)."""
    finite = torch.isfinite(prediction)
    nonfinite = (~finite).flatten(-2).sum(-1)
    prediction = torch.where(finite, prediction, 0.0)
    channels = means.shape[-1]
    kept = valid.unsqueeze(-1)
    means = torch.where(kept, prediction[..., :channels], means)
    spreads = torch.where(kept, prediction[..., channels:], spreads)
    if enemy_action_logits is None:
        action = torch.full_like(action, 1.0 / action.shape[-1])
    else:
        action = torch.softmax(enemy_action_logits, -1)
    return means, spreads, action, nonfinite
