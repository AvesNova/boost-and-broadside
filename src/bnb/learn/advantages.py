"""How the value levels share the policy's advantage (diagnostics): per level, its part of
the summed advantage in the units the policy term reads, and its part of the policy
gradient.

The policy term reads A / scale, A = sum_k A_k with A_k a level's team-mixed advantage
(`bnb.learn.returns.team_mixed_levels`: the level's weighted reward's GAE, plus its team
spirit times the living teammates' mean) and scale the return scale. Each series below is
of A_k / scale over the learner's living own ships (the policy term's tokens), every update:
- `advantage/mean_absolute/{level}`: mean |A_k|;
- `advantage/standard_deviation/{level}`: its standard deviation (population);
- `advantage/share/{level}`: mean |A_k| / sum_j mean |A_j|, the level's share of the
  advantage's size (the shares sum to 1);
- `advantage/correlation/{level}`: the correlation of A_k with A over the tokens;
- `advantage/raw_standard_deviation/{level}`: the standard deviation of the level's GAE
  before the reward weight, the team mixing and the return scale (A_k's own GAE divided by
  the level's reward weight), so a weight or a scale out of proportion shows;
and, but the raw one, each per mode of the batch's games as `.../{level}/{mode}`.

At the diagnostics cadence (`policy_level_series`), over the cadence's minibatch, the
policy term's gradient with each level's advantage alone (A = A_k, every other level 0),
every parameter the policy reaches (the trunk and the policy head):
- `gradient/policy_level_norm/{level}`: its norm;
- `gradient/policy_level_share/{level}`: its projection on the whole policy gradient g,
  <g_k, g> / |g|^2 (the shares sum to 1 where no token is clipped: the clipped surrogate
  is linear in A elsewhere);
- `gradient/policy_level_cosine/{a}/{b}`: every pair of levels' cosine, and each level's
  with the whole policy gradient (`b` = `policy`).
"""

import itertools

import torch

from bnb.contracts.series import Mean

# The level sums' columns, per slice and level: count is per slice.
LEVEL_COLUMNS = ("absolute", "first", "second", "cross")
TOTAL_COLUMNS = ("count", "first", "second")


def _per_slice(values: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
    """values (L, V, S, ...) summed over steps and ships, then over each slice's views:
    (slices, ...). member (V, slices)."""
    per_view = values.sum((0, 2))  # (V, ...)
    return torch.einsum("v...,vm->m...", per_view, member)


def level_sums(
    levels: torch.Tensor, total: torch.Tensor, mask: torch.Tensor, member: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sums the advantage series read, per slice: ((slices, LEVEL_COLUMNS, levels): the
    sum of |A_k|, of A_k, of A_k^2 and of A_k A; (slices, TOTAL_COLUMNS): the token count,
    the sum of A and of A^2). levels (L, V, S, levels) the per-level advantages, total
    (L, V, S) their sum, both in the policy's units; mask (L, V, S) the tokens; member
    (V, slices) each view's slices (a 0/1 matrix)."""
    weight = mask.to(levels.dtype)
    levels = levels * weight[..., None]
    total = total * weight
    per_level = torch.stack(
        [levels.abs(), levels, levels.square(), levels * total[..., None]], -2
    )  # (L, V, S, LEVEL_COLUMNS, levels)
    whole = torch.stack([weight, total, total.square()], -1)  # (L, V, S, TOTAL_COLUMNS)
    return _per_slice(per_level, member), _per_slice(whole, member)


def _deviation(first: torch.Tensor, second: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """The population standard deviation from a sum and a sum of squares (0 without
    tokens)."""
    tokens = count.clamp_min(1.0)
    return (second / tokens - (first / tokens).square()).clamp_min(0.0).sqrt()


def level_series(
    per_level: torch.Tensor,
    whole: torch.Tensor,
    level_names: tuple[str, ...],
    slice_names: tuple[str | None, ...],
) -> dict:
    """The advantage series of the module's docstring from `level_sums`' sums; a slice named
    None is the whole batch (no suffix)."""
    out = {}
    for index, slice_name in enumerate(slice_names):
        suffix = "" if slice_name is None else f"/{slice_name}"
        absolute, first, second, cross = per_level[index].unbind(0)  # (levels,) each
        count, total_first, total_second = whole[index].unbind(0)
        deviation = _deviation(first, second, count)
        total_deviation = _deviation(total_first, total_second, count)
        tokens = count.clamp_min(1.0)
        covariance = cross / tokens - first / tokens * (total_first / tokens)
        spread = deviation * total_deviation
        correlation = torch.where(spread > 0, covariance / spread.clamp_min(1e-30), 0.0)
        size = absolute.sum()
        for level, name in enumerate(level_names):
            out[f"advantage/mean_absolute/{name}{suffix}"] = Mean(absolute[level], count)
            out[f"advantage/standard_deviation/{name}{suffix}"] = deviation[level]
            out[f"advantage/share/{name}{suffix}"] = Mean(absolute[level], size)
            out[f"advantage/correlation/{name}{suffix}"] = correlation[level]
    return out


def raw_sums(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(3, levels): the token count, the sum and the sum of squares of each level's
    advantage (..., levels) over the tokens of mask (...)."""
    weight = mask.to(advantages.dtype)[..., None]
    count = weight.expand_as(advantages)
    flat = torch.stack([count, advantages * weight, advantages.square() * weight])
    return flat.flatten(1, -2).sum(1)


def raw_series(sums: torch.Tensor, weights: tuple[float, ...], level_names) -> dict:
    """`advantage/raw_standard_deviation/{level}`: from `raw_sums` of the levels' GAE in
    reward units, each over its reward weight."""
    count, first, second = sums.unbind(0)
    deviation = _deviation(first, second, count)
    return {
        f"advantage/raw_standard_deviation/{name}": deviation[index] / weight
        for index, (name, weight) in enumerate(zip(level_names, weights, strict=True))
    }


def policy_level_series(
    level_gradients: torch.Tensor, policy_gradient: torch.Tensor, level_names: tuple[str, ...]
) -> dict:
    """The cadence's series of the module's docstring from each level's policy gradient
    alone (levels, P) and the whole policy gradient (P,), flat over the same parameters."""
    stacked = torch.cat([level_gradients, policy_gradient[None]]).float()
    gram = stacked @ stacked.T
    norms = gram.diagonal().clamp_min(0.0).sqrt()
    whole = len(level_names)
    step = gram[whole, whole]
    out = {}
    names = (*level_names, "policy")
    for index, name in enumerate(level_names):
        out[f"gradient/policy_level_norm/{name}"] = norms[index]
        out[f"gradient/policy_level_share/{name}"] = Mean(gram[index, whole], step)
    for a, b in itertools.combinations(range(len(names)), 2):
        product = norms[a] * norms[b]
        out[f"gradient/policy_level_cosine/{names[a]}/{names[b]}"] = Mean(gram[a, b], product)
    return out
