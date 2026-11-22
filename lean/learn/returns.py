"""Returns: categorical value helpers, GAE per reward level, advantage normalisation."""

import torch
from torch import Tensor

from lean.learn.config import LearnConfig
from lean.model.value import value_bin_centres


def value_centres(config: LearnConfig, bins: int, device=None) -> Tensor:
    """Bin centres of the categorical critic, (LEVELS, bins), evenly spaced from
    config.value_low to config.value_high of each level."""
    return value_bin_centres(torch.tensor(config.value_low), torch.tensor(config.value_high), bins).to(device)


def generalised_advantages(
    reward: Tensor, value: Tensor, bootstrap: Tensor, done: Tensor, discounts: Tensor, gae_lambda: float
) -> tuple[Tensor, Tensor]:
    """GAE per reward level.

    reward, value: (T, E, N, LEVELS); bootstrap: (E, N, LEVELS), the value after the last
    tick; done: (T, E) bool, the episode ended on that tick (a time-limit end is a true
    termination: nothing is bootstrapped across it); discounts: (LEVELS,) per tick.
    `value` and `bootstrap` must already be zero on rows outside the valued mask, so a
    ship's death bootstraps from zero while later rewards of its episode (the outcome)
    still reach its earlier ticks through the lambda-return.

    Returns (advantage, value target), both (T, E, N, LEVELS).
    """
    ticks = reward.shape[0]
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for tick in range(ticks - 1, -1, -1):
        alive = (~done[tick]).to(reward.dtype)[:, None, None]
        delta = reward[tick] + discounts * next_value * alive - value[tick]
        running = delta + discounts * gae_lambda * alive * running
        advantage[tick] = running
        next_value = value[tick]
    return advantage, advantage + value


def normalise_advantages(advantage: Tensor, mask: Tensor, config: LearnConfig) -> Tensor:
    """Sum advantages over levels and scale them over the whole update.

    advantage (T, E, N, LEVELS), mask (T, E, N) bool of valued tokens. Returns (T, E, N),
    zero outside the mask. The mean is not subtracted. "std" divides by the standard
    deviation over masked tokens; "percentile" divides by the range between the two
    configured quantiles.
    """
    total = advantage.sum(-1)
    chosen = total[mask]
    if chosen.numel() < 2:
        return total * mask
    if config.advantage_normaliser == "percentile":
        low, high = config.advantage_percentiles
        quantiles = torch.quantile(chosen, torch.tensor([low, high], device=chosen.device))
        scale = quantiles[1] - quantiles[0]
    else:
        scale = chosen.std()
    return total / scale.clamp(min=1e-8) * mask
