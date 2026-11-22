"""Returns: categorical value helpers, GAE per reward level, advantage normalisation and the
supports' range monitoring."""

import torch
from torch import Tensor

from lean.learn.config import LEVEL_NAMES, LearnConfig
from lean.model.value import value_bin_centres


def value_centres(config: LearnConfig, bins: int, device=None) -> Tensor:
    """Bin centres of the categorical critic, (LEVELS, bins), evenly spaced from
    config.value_low to config.value_high of each level."""
    return value_bin_centres(torch.tensor(config.value_low), torch.tensor(config.value_high), bins).to(device)


def generalised_advantages(
    reward: Tensor, value: Tensor, bootstrap: Tensor, done: Tensor, discounts: Tensor, gae_lambda: Tensor | float
) -> tuple[Tensor, Tensor]:
    """GAE per reward level.

    reward, value: (T, E, N, LEVELS); bootstrap: (E, N, LEVELS), the value after the last
    tick; done: (T, E) bool, the episode ended on that tick (a time-limit end is a true
    termination: nothing is bootstrapped across it); discounts: (LEVELS,) per tick;
    gae_lambda: (LEVELS,) or one number.
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


def support_usage(target: Tensor, reward: Tensor, mask: Tensor, centres: Tensor, config: LearnConfig) -> dict[str, float]:
    """How each level's value targets sit on its support, as metrics.

    target (T, E, N, LEVELS) in return units; reward (T, E, N, LEVELS) the rollout's; mask
    (T, E, N) bool of valued tokens; centres (LEVELS, bins). Per level: `range_clamped_<level>`, the share of targets beyond the
    outermost centres (HL-Gauss clamps them); `range_bins_<level>`, the bins holding at
    least config.range_bin_share of the targets (nearest centre); and two warning flags,
    1.0 or 0.0: `range_warning_clamped_<level>` when the clamped share exceeds
    config.range_clamped_warning, `range_warning_narrow_<level>` when fewer than
    config.range_narrow_bins bins are occupied, unless the level paid no reward anywhere in
    the rollout (an objective level in elimination: its targets are only the critic's
    own values).
    """
    chosen = target[mask].float()  # (count, LEVELS)
    metrics: dict[str, float] = {}
    if chosen.shape[0] == 0:
        return metrics
    centres = centres.to(chosen.device).float()
    bins = centres.shape[-1]
    width = centres[:, 1] - centres[:, 0]
    clamped = ((chosen < centres[:, 0]) | (chosen > centres[:, -1])).float().mean(0)
    nearest = ((chosen - centres[:, 0]) / width).round().clamp(0, bins - 1).long()
    counts = torch.zeros(len(LEVEL_NAMES), bins, device=chosen.device)
    counts.scatter_add_(1, nearest.T, torch.ones_like(nearest.T, dtype=counts.dtype))
    occupied = (counts / chosen.shape[0] >= config.range_bin_share).sum(1)
    silent = (reward == 0).flatten(0, -2).all(0)
    for level, name in enumerate(LEVEL_NAMES):
        metrics[f"range_clamped_{name}"] = float(clamped[level])
        metrics[f"range_bins_{name}"] = float(occupied[level])
        metrics[f"range_warning_clamped_{name}"] = float(clamped[level] > config.range_clamped_warning)
        metrics[f"range_warning_narrow_{name}"] = float((occupied[level] < config.range_narrow_bins) & ~silent[level])
    return metrics
