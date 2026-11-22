"""Returns: categorical value helpers, GAE per reward level, advantage normalisation."""

import math

import torch
from torch import Tensor

from lean.learn.config import LearnConfig


def value_centres(config: LearnConfig, bins: int, device=None) -> Tensor:
    """Bin centres of the categorical critic, (LEVELS, bins), evenly spaced over
    [-half_width, +half_width] of each level."""
    half = torch.tensor(config.value_half_widths, dtype=torch.float32, device=device)
    unit = torch.linspace(-1.0, 1.0, bins, device=device)
    return half[:, None] * unit[None, :]


def expected_value(value_logits: Tensor, centres: Tensor) -> Tensor:
    """Scalar value per level, (..., LEVELS), from logits (..., LEVELS, BINS)."""
    return (torch.softmax(value_logits, dim=-1) * centres).sum(-1)


def hl_gauss(target: Tensor, centres: Tensor, sigma_ratio: float) -> Tensor:
    """Gaussian-binned target distribution (HL-Gauss), (..., LEVELS, BINS).

    `target` (..., LEVELS) is clamped to the support; each bin gets the mass of a normal
    centred on the target with standard deviation sigma_ratio * bin width, renormalised
    over the support.
    """
    width = centres[:, 1] - centres[:, 0]
    edges = torch.cat([centres - width[:, None] / 2, centres[:, -1:] + width[:, None] / 2], dim=1)
    clamped = torch.maximum(torch.minimum(target, centres[:, -1]), centres[:, 0])
    sigma = (sigma_ratio * width).clamp(min=1e-8)
    cumulative = 0.5 * (1.0 + torch.erf((edges - clamped[..., None]) / (sigma[:, None] * math.sqrt(2.0))))
    mass = cumulative[..., 1:] - cumulative[..., :-1]
    return mass / mass.sum(-1, keepdim=True).clamp(min=1e-8)


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
