"""Returns: categorical value helpers, GAE per reward level, advantage normalisation and the
supports' range monitoring."""

import functools
from typing import NamedTuple

import torch
from torch import Tensor

from lean.contract import LEVEL_NAMES
from lean.learn.config import LearnConfig
from lean.model.scan import linear_scan
from lean.model.value import buffered_centres, symlog


def value_knees(config: LearnConfig, device=None) -> Tensor:
    """The critic levels' symlog knees, (LEVELS,) float32 in return units (0 linear)."""
    return torch.tensor(config.value_knee, dtype=torch.float32, device=device)


def value_centres(config: LearnConfig, bins: int, device=None) -> Tensor:
    """Bin centres of the categorical critic, (LEVELS, bins), in each level's bin space
    (value_knees): config.value_low to config.value_high of each level (the measured range)
    plus the buffer columns (lean.model.value.buffered_centres)."""
    return buffered_centres(
        torch.tensor(config.value_low), torch.tensor(config.value_high), value_knees(config), bins
    ).to(device)


class ValueSupport(NamedTuple):
    """The critic's support on the device: `centres` (LEVELS, BINS) (value_centres) and
    `knees` (LEVELS,) (value_knees). Built once outside the compiled graphs and passed in:
    a table made with torch.tensor inside a graph is a pageable host-to-device copy (a host
    wait) at every call."""

    centres: Tensor
    knees: Tensor


def value_support(config: LearnConfig, bins: int, device=None) -> ValueSupport:
    """The critic's support (ValueSupport) on `device`: built on the host and copied once per
    process for each support (the measured ranges, knees, bins and device), then reused, so a
    collect or an update makes no pageable copy for it. Read-only: callers never write into
    it."""
    return _support(
        tuple(config.value_low),
        tuple(config.value_high),
        tuple(config.value_knee),
        bins,
        torch.device("cpu" if device is None else device),
    )


@functools.cache
def _support(low: tuple, high: tuple, knee: tuple, bins: int, device: torch.device):
    knees = torch.tensor(knee, dtype=torch.float32)
    centres = buffered_centres(torch.tensor(low), torch.tensor(high), knees, bins)
    return ValueSupport(centres.to(device), knees.to(device))


def generalised_advantages(
    reward: Tensor,
    value: Tensor,
    bootstrap: Tensor,
    done: Tensor,
    discounts: Tensor,
    gae_lambda: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """GAE per reward level.

    reward, value: (T, E, N, LEVELS); bootstrap: (E, N, LEVELS), the value after the last
    tick; done: (T, E) bool, the episode ended on that tick (a time-limit end is a true
    termination: nothing is bootstrapped across it); discounts: (LEVELS,) per tick;
    gae_lambda: (LEVELS,) or one number.
    `value` and `bootstrap` are the critic's values on the valued rows (`own`), its living
    team-mates' mean value on a fallen ship's rows (fallen_team_values) and zero elsewhere.

    Returns (advantage, value target), both (T, E, N, LEVELS).

    A_t = delta_t + discount * lambda * (1 - done_t) * A_(t+1) runs backwards in time as
    lean.model's linear scan over the reversed ticks (one pass, float32).
    """
    alive = (~done).to(reward.dtype)[:, :, None, None]
    next_value = torch.cat([value[1:], bootstrap[None]])
    delta = reward + discounts * next_value * alive - value
    decay = discounts * gae_lambda * alive  # (T, E, 1, LEVELS)
    reversed_advantage, _ = linear_scan(decay.flip(0), delta.flip(0), torch.zeros_like(bootstrap))
    advantage = reversed_advantage.flip(0)
    return advantage, advantage + value


def fallen_team_values(value: Tensor, own: Tensor, fallen: Tensor) -> Tensor:
    """A fallen ship's value for the returns (author 2026-11-28, option C): its living
    team-mates' mean value at the same tick and level.

    value (..., N, LEVELS), zero outside `own`; own (..., N) bool, the view's living ships;
    fallen (..., N) bool, the view's team's dead ships still in the match (no respawn: it is
    masked out of observation and acting). A fallen row takes the mean over the own rows of
    the same view and tick (zero when none lives: the team is out and the episode has
    ended). A perfect critic on every team-mate predicts the same match outcome, so the
    outcome reaches a ship's ticks before its death at the discount (gamma^k), not at
    (gamma lambda)^k as through zero values. Its death penalty is paid as before."""
    weight = own.to(value.dtype)[..., None]
    mean = (value * weight).sum(-2, keepdim=True) / weight.sum(-2, keepdim=True).clamp(min=1.0)
    return torch.where(fallen[..., None], mean, value)


def join_segments(records: Tensor, segments: int) -> Tensor:
    """A target-pass micro-batch's records (T, U x k, ...), segment-major
    (RolloutStore.chain_index), as each seat's consecutive ticks (U x T, k, ...)."""
    ticks, views = records.shape[:2]
    seats, rest = views // segments, records.shape[2:]
    return (
        records.reshape((ticks, segments, seats) + rest)
        .transpose(0, 1)
        .reshape((segments * ticks, seats) + rest)
    )


def split_segments(records: Tensor, segments: int) -> Tensor:
    """join_segments undone: (U x T, k, ...) back to (T, U x k, ...)."""
    ticks, seats, rest = records.shape[0] // segments, records.shape[1], records.shape[2:]
    return (
        records.reshape((segments, ticks, seats) + rest)
        .transpose(0, 1)
        .reshape((ticks, segments * seats) + rest)
    )


def normalise_advantages(total: Tensor, mask: Tensor) -> Tensor:
    """Scale the advantages (summed over levels) over the whole update.

    total (T, E, N), mask (T, E, N) bool of valued tokens. Returns (T, E, N), zero outside
    the mask: divided by the standard deviation over the masked tokens, the mean not
    subtracted, from masked sums (no read-back). Fewer than two masked tokens leave the
    advantages unscaled.
    """
    weight = mask.to(total.dtype)
    count = weight.sum()
    mean = (total * weight).sum() / count.clamp(min=1.0)
    variance = (((total - mean) * weight) ** 2).sum() / (count - 1.0).clamp(min=1.0)
    scale = torch.where(count >= 2, variance.sqrt(), 1.0)
    return total / scale.clamp(min=1e-8) * mask


def advantage_sums(advantage: Tensor, mask: Tensor) -> Tensor:
    """The tallies behind advantage_shares for one batch, to be summed over batches:
    advantage (..., LEVELS) per level, mask (...) bool of valued tokens. Returns (3 + 2
    LEVELS,) float64: count, sum and sum of squares of the summed advantage, then per level
    the sum and the sum of its product with the summed advantage."""
    weight = mask.to(advantage.dtype)
    total = advantage.sum(-1)
    weighted = advantage * weight[..., None]
    per_level = weighted.flatten(0, -2)
    return torch.cat(
        [
            torch.stack([weight.sum(), (total * weight).sum(), (total * total * weight).sum()]),
            per_level.sum(0),
            (per_level * total.flatten()[:, None]).sum(0),
        ]
    ).double()


def advantage_shares(sums: Tensor) -> tuple[Tensor, Tensor]:
    """From summed advantage_sums: each level's share of the summed advantage's variance,
    Cov(A_level, A) / Var(A) (LEVELS,), which add up to 1 (covariances between levels
    split between them), and the summed advantage's standard deviation before scaling.
    Zero without at least two valued tokens."""
    levels = (sums.shape[0] - 3) // 2
    count, total, squares = sums[0], sums[1], sums[2]
    level_sum, product = sums[3 : 3 + levels], sums[3 + levels :]
    count = count.clamp(min=1.0)
    mean = total / count
    variance = squares / count - mean * mean
    covariance = product / count - level_sum / count * mean
    valid = (sums[0] >= 2) & (variance > 0)
    shares = torch.where(valid, covariance / torch.where(valid, variance, 1.0), 0.0)
    deviation = torch.where(
        valid, (variance.clamp(min=0.0) * count / (count - 1.0).clamp(min=1.0)).sqrt(), 0.0
    )
    return shares.float(), deviation.float()


def usage_counts(target: Tensor, mask: Tensor, centres: Tensor, knee: Tensor) -> Tensor:
    """How the value targets sit on each level's support, for one batch of targets, to be
    summed over batches (the update does; usage_flags and usage_metrics then read the sums):
    (LEVELS, BINS + 2), per level the count of valued targets nearest each bin centre, then
    the count beyond the outermost centres, then the count of valued targets. target
    (..., LEVELS) in return units, mask (...) bool, centres (LEVELS, BINS) in bin space,
    knee (LEVELS,) (value_knees)."""
    levels, bins = centres.shape
    chosen = symlog(target.float().reshape(-1, levels), knee.to(target.device))
    weight = mask.reshape(-1).float()
    centres = centres.to(chosen.device).float()
    width = centres[:, 1] - centres[:, 0]
    clamped = (
        ((chosen < centres[:, 0]) | (chosen > centres[:, -1])).float() * weight[:, None]
    ).sum(0)
    nearest = ((chosen - centres[:, 0]) / width).round().clamp(0, bins - 1).long()
    counts = torch.zeros(levels, bins, device=chosen.device)
    counts.scatter_add_(1, nearest.T, weight.expand(levels, -1).contiguous())
    return torch.cat([counts, clamped[:, None], weight.sum().expand(levels, 1)], 1)


def usage_flags(counts: Tensor, paid: Tensor, config: LearnConfig) -> Tensor:
    """From summed usage_counts (LEVELS, BINS + 2) and `paid` (LEVELS,) bool (the level paid
    a reward somewhere in the rollout): (4, LEVELS), the clamped share (targets beyond the
    outermost centres, which HL-Gauss clamps), the occupied bins (holding at least
    config.range_bin_share of the targets, nearest centre), the clamped warning (the share
    above config.range_clamped_warning) and the narrow warning (fewer than
    config.range_narrow_bins bins, unless the level paid no reward anywhere in the rollout:
    an objective level in elimination, whose targets are only the critic's own values)."""
    total = counts[:, -1].clamp(min=1.0)
    clamped = counts[:, -2] / total
    occupied = (counts[:, :-2] / total[:, None] >= config.range_bin_share).sum(1).float()
    return torch.stack(
        [
            clamped,
            occupied,
            (clamped > config.range_clamped_warning).float(),
            ((occupied < config.range_narrow_bins) & paid).float(),
        ]
    )


def usage_metrics(rows: list[list[float]]) -> dict[str, float]:
    """usage_flags read back (a list of 4 rows of LEVELS numbers) as named metrics, per level:
    `range_clamped_<level>`, `range_bins_<level>`, `range_warning_clamped_<level>` and
    `range_warning_narrow_<level>` (the warnings 1.0 or 0.0)."""
    metrics: dict[str, float] = {}
    for level, name in enumerate(LEVEL_NAMES):
        metrics[f"range_clamped_{name}"] = rows[0][level]
        metrics[f"range_bins_{name}"] = rows[1][level]
        metrics[f"range_warning_clamped_{name}"] = rows[2][level]
        metrics[f"range_warning_narrow_{name}"] = rows[3][level]
    return metrics
