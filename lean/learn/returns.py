"""Returns: categorical value helpers, GAE per reward level, advantage normalisation and the
supports' range monitoring."""

import torch
from torch import Tensor

from lean.contract import LEVEL_NAMES
from lean.learn.config import LearnConfig
from lean.model.temporal import doubling_scan
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

    A_t = delta_t + discount * lambda * (1 - done_t) * A_(t+1) runs backwards in time as
    lean.model's doubling scan over the reversed ticks (log2 T whole-tensor rounds, no
    Python loop over T); equal to the tick-by-tick recursion within float32 rounding.
    """
    alive = (~done).to(reward.dtype)[:, :, None, None]
    next_value = torch.cat([value[1:], bootstrap[None]])
    delta = reward + discounts * next_value * alive - value
    decay = discounts * gae_lambda * alive  # (T, E, 1, LEVELS)
    reversed_advantage, _ = doubling_scan(decay.flip(0), delta.flip(0), torch.zeros_like(bootstrap))
    advantage = reversed_advantage.flip(0)
    return advantage, advantage + value


def join_segments(records: Tensor, segments: int) -> Tensor:
    """A target-pass micro-batch's records (T, U x k, ...), segment-major
    (RolloutStore.chain_index), as each seat's consecutive ticks (U x T, k, ...)."""
    ticks, views = records.shape[:2]
    seats, rest = views // segments, records.shape[2:]
    return records.reshape((ticks, segments, seats) + rest).transpose(0, 1).reshape((segments * ticks, seats) + rest)


def split_segments(records: Tensor, segments: int) -> Tensor:
    """join_segments undone: (U x T, k, ...) back to (T, U x k, ...)."""
    ticks, seats, rest = records.shape[0] // segments, records.shape[1], records.shape[2:]
    return records.reshape((segments, ticks, seats) + rest).transpose(0, 1).reshape((ticks, segments * seats) + rest)


def normalise_advantages(total: Tensor, mask: Tensor, config: LearnConfig) -> Tensor:
    """Scale the advantages (summed over levels) over the whole update.

    total (T, E, N), mask (T, E, N) bool of valued tokens. Returns (T, E, N), zero outside
    the mask. The mean is not subtracted. "std" divides by the standard deviation over
    masked tokens, from masked sums (no read-back); "percentile" divides by the range
    between the two configured quantiles (selecting the masked tokens reads their count
    back). Fewer than two masked tokens leave the advantages unscaled.
    """
    weight = mask.to(total.dtype)
    count = weight.sum()
    if config.advantage_normaliser == "percentile":
        chosen = total[mask]
        low, high = config.advantage_percentiles
        quantiles = torch.quantile(chosen, torch.tensor([low, high], device=chosen.device)) if chosen.numel() >= 2 else None
        scale = quantiles[1] - quantiles[0] if quantiles is not None else torch.ones((), device=total.device)
    else:
        mean = (total * weight).sum() / count.clamp(min=1.0)
        variance = (((total - mean) * weight) ** 2).sum() / (count - 1.0).clamp(min=1.0)
        scale = torch.where(count >= 2, variance.sqrt(), 1.0)
    return total / scale.clamp(min=1e-8) * mask


def usage_counts(target: Tensor, mask: Tensor, centres: Tensor) -> Tensor:
    """The tallies behind support_usage for one batch of targets, to be summed over batches:
    (LEVELS, BINS + 2), per level the count of valued targets nearest each bin centre, then
    the count beyond the outermost centres, then the count of valued targets. target
    (..., LEVELS) in return units, mask (...) bool, centres (LEVELS, BINS)."""
    levels, bins = centres.shape
    chosen = target.float().reshape(-1, levels)
    weight = mask.reshape(-1).float()
    centres = centres.to(chosen.device).float()
    width = centres[:, 1] - centres[:, 0]
    clamped = (((chosen < centres[:, 0]) | (chosen > centres[:, -1])).float() * weight[:, None]).sum(0)
    nearest = ((chosen - centres[:, 0]) / width).round().clamp(0, bins - 1).long()
    counts = torch.zeros(levels, bins, device=chosen.device)
    counts.scatter_add_(1, nearest.T, weight.expand(levels, -1).contiguous())
    return torch.cat([counts, clamped[:, None], weight.sum().expand(levels, 1)], 1)


def usage_flags(counts: Tensor, paid: Tensor, config: LearnConfig) -> Tensor:
    """From summed usage_counts (LEVELS, BINS + 2) and `paid` (LEVELS,) bool (the level paid
    a reward somewhere in the rollout): (4, LEVELS), the rows of support_usage's measures
    (clamped share, occupied bins, the clamped warning, the narrow warning) as a tensor."""
    total = counts[:, -1].clamp(min=1.0)
    clamped = counts[:, -2] / total
    occupied = (counts[:, :-2] / total[:, None] >= config.range_bin_share).sum(1).float()
    return torch.stack([
        clamped, occupied, (clamped > config.range_clamped_warning).float(),
        ((occupied < config.range_narrow_bins) & paid).float(),
    ])


def usage_metrics(rows: list[list[float]]) -> dict[str, float]:
    """usage_flags read back (a list of 4 rows of LEVELS numbers) as named metrics."""
    metrics: dict[str, float] = {}
    for level, name in enumerate(LEVEL_NAMES):
        metrics[f"range_clamped_{name}"] = rows[0][level]
        metrics[f"range_bins_{name}"] = rows[1][level]
        metrics[f"range_warning_clamped_{name}"] = rows[2][level]
        metrics[f"range_warning_narrow_{name}"] = rows[3][level]
    return metrics


def support_usage(target: Tensor, reward: Tensor, mask: Tensor, centres: Tensor, config: LearnConfig) -> dict[str, float]:
    """How each level's value targets sit on its support, as metrics.

    target (T, E, N, LEVELS) in return units; reward (T, E, N, LEVELS) the rollout's; mask
    (T, E, N) bool of valued tokens; centres (LEVELS, bins). Per level: `range_clamped_<level>`,
    the share of targets beyond the outermost centres (HL-Gauss clamps them);
    `range_bins_<level>`, the bins holding at least config.range_bin_share of the targets
    (nearest centre); and two warning flags, 1.0 or 0.0: `range_warning_clamped_<level>` when
    the clamped share exceeds config.range_clamped_warning, `range_warning_narrow_<level>`
    when fewer than config.range_narrow_bins bins are occupied, unless the level paid no
    reward anywhere in the rollout (an objective level in elimination: its targets are only
    the critic's own values). No valued token gives no metrics. The update sums
    usage_counts over its batches instead; this is the same on whole tensors.
    """
    if not mask.any():
        return {}
    paid = (reward != 0).flatten(0, -2).any(0)
    return usage_metrics(usage_flags(usage_counts(target, mask, centres), paid, config).tolist())
