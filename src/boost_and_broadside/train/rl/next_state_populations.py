"""Who the next-state head learns from, and how much each population teaches it.

Every supervised next-state token belongs to one of three populations whose
targets mean different things:

* **ally** -- always in sight, so the input is sharp and the target one step of
  real dynamics.
* **enemy_visible** -- in sight now, so the same kind of target as an ally's.
* **enemy_hidden** -- out of sight, so the input is the belief's smoothed code
  and the target is where the ship really is, however far that has drifted.

One cross-entropy over all three decides how much each population moves the
shared trunk, and the answer is not readable from the aggregate loss. This
module supplies the two measurements that make it readable.

**Population sums**, cheap enough for every update: per population and code
group, the mean cross-entropy and the mean cross-entropy of the zero-residual
head ("nothing changes" from the code the head read). The head is worth its
cost on a group only where the first sits below the second.

**Gradient decomposition**, as a gradient diagnostic: the next-state term split
by population, and each population split again into two disjoint halves of the
environments. The halves are independent samples of the same population's
gradient, so their inner product estimates the squared norm of the *coherent*
gradient -- the part a larger batch would keep -- while the norm of their sum
also carries per-token noise. A population can dominate the gradient norm and
still contribute little a step can use, and only the halves tell those apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from boost_and_broadside.train.rl.grad_diagnostics import (
    TermGradientAccumulator,
    scope_metric_records,
    scope_statistics,
)
from boost_and_broadside.train.rl.ship_codes import CODE_GROUP_NAMES

#: Supervised next-state populations, in a fixed order every tensor here uses.
NEXT_STATE_POPULATIONS: tuple[str, ...] = ("ally", "enemy_visible", "enemy_hidden")
#: The two disjoint environment halves each population is split into.
ENVIRONMENT_HALVES: tuple[str, ...] = ("a", "b")
#: The objectives decomposed: the trained cross-entropy.
NEXT_STATE_LIKELIHOODS: tuple[str, ...] = ("ce",)

#: Rows of :func:`population_moments`' second axis.
MOMENT_NAMES: tuple[str, ...] = ("ce", "baseline_ce")

_EPS = 1e-12


def population_masks(
    supervised: torch.Tensor, team_id: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """Partition the supervised tokens into the three populations.

    Allies are classified by team alone: allied visibility is guaranteed by
    perception, so an ally that read as out of sight would still be supervised
    on a dynamics label and belongs with the others.

    Args:
        supervised: (T, B, N) bool, tokens the next-state loss supervises.
        team_id:    (T, B, N) raw team id; 0 is the observing team.
        visible:    (T, B, N) bool, whether the observer sees the ship now.

    Returns:
        (3, T, B, N) bool, disjoint, and summing to ``supervised``.
    """

    ally = team_id == 0
    enemy = ~ally
    return torch.stack(
        [
            supervised & ally,
            supervised & enemy & visible,
            supervised & enemy & ~visible,
        ]
    )


def environment_halves(num_envs: int, device: torch.device) -> torch.Tensor:
    """(2, B) bool: even and odd environment slots.

    Environments are independent streams, so any fixed partition of them gives
    two independent samples. Parity is used because it needs no randomness and
    keeps both halves populated in every micro-batch of two or more envs.
    """

    even = torch.arange(num_envs, device=device) % 2 == 0
    return torch.stack([even, ~even])


def gradient_terms(
    losses: Mapping[str, torch.Tensor],
    masks: torch.Tensor,
    halves: torch.Tensor,
    denominator: torch.Tensor,
    coef: float,
) -> dict[str, torch.Tensor]:
    """One scalar term per objective, population, and environment half.

    Each term is normalized exactly as the training loss is, so the six terms
    of the trained objective sum to the ``next_state`` term's gradient.

    Args:
        losses:      Objective name → (T, B, N, G) per-group loss.
        masks:       (3, T, B, N) bool population masks.
        halves:      (2, B) bool environment halves.
        denominator: The training loss's divisor: supervised tokens times groups.
        coef:        The next-state loss coefficient.

    Returns:
        ``"ns_<objective>/<population>/<half>"`` → scalar loss.
    """

    scale = coef / denominator
    terms: dict[str, torch.Tensor] = {}
    for likelihood, per_group in losses.items():
        per_token = per_group.sum(-1)  # (T, B, N)
        for p, population in enumerate(NEXT_STATE_POPULATIONS):
            for h, half in enumerate(ENVIRONMENT_HALVES):
                weight = (masks[p] & halves[h].view(1, -1, 1)).float()  # (T, B, N)
                terms[f"ns_{likelihood}/{population}/{half}"] = (per_token * weight).sum() * scale
    return terms


def population_moments(
    cross_entropy: torch.Tensor,
    baseline_cross_entropy: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive per-population sums for one micro-batch.

    Args:
        cross_entropy:          (T, B, N, G) the head's per-group cross-entropy.
        baseline_cross_entropy: (T, B, N, G) the zero-residual head's.
        masks:                  (3, T, B, N) bool population masks.

    Returns:
        ``(sums, counts)``: sums is (3, 2, G) in :data:`MOMENT_NAMES` order and
        counts is (3,), both float64 so they can be summed over a whole update
        without losing the small populations to roundoff.
    """

    with torch.no_grad():
        per_token = torch.stack(
            [cross_entropy.detach().float(), baseline_cross_entropy.detach().float()]
        )  # (2, T, B, N, G)
        weights = masks.float()  # (3, T, B, N)
        sums = torch.einsum("mtbng,ptbn->pmg", per_token, weights)  # (3, 2, G)
        counts = weights.sum((1, 2, 3))  # (3,)
    return sums.double(), counts.double()


def population_metric_records(sums: torch.Tensor, counts: torch.Tensor) -> dict[str, float]:
    """Finalize accumulated population sums into logger keys.

    Args:
        sums:   (3, 2, G) accumulated :func:`population_moments` sums.
        counts: (3,) accumulated token counts.

    Returns:
        Metric name → value. A population with no tokens in the update logs its
        share as zero and nothing else, rather than a mean over nothing.
    """

    sums = sums.cpu()
    counts = counts.cpu()
    total = float(counts.sum())
    records: dict[str, float] = {}
    for p, population in enumerate(NEXT_STATE_POPULATIONS):
        count = float(counts[p])
        records[f"next_state_population/{population}/token_share"] = count / max(total, 1.0)
        if count <= 0.0:
            continue
        means = sums[p] / count  # (2, G)
        for m, moment in enumerate(MOMENT_NAMES):
            records[f"next_state_population/{population}/{moment}"] = float(means[m].mean())
        for g, group in enumerate(CODE_GROUP_NAMES):
            records[f"next_state_ce/{population}/{group}"] = float(means[0, g])
            records[f"next_state_baseline_ce/{population}/{group}"] = float(means[1, g])
    return records


def _merge_gram(gram: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Gram matrix of group sums, from the Gram matrix of their parts.

    ``<sum_{i in A} g_i, sum_{j in B} g_j> = sum_{i in A, j in B} G[i, j]``, so
    the population gradients' statistics come from the halves' Gram matrix
    without a second pass over the parameters.
    """

    indicator = torch.zeros(len(groups), gram.shape[0], dtype=gram.dtype, device=gram.device)
    for row, members in enumerate(groups):
        indicator[row, list(members)] = 1.0
    return indicator @ gram @ indicator.T


def gradient_metric_records(accumulator: TermGradientAccumulator) -> dict[str, float]:
    """The population decomposition's metrics for one diagnosed minibatch.

    For the objective and both parameter scopes (whole model and shared trunk),
    under group ``next_state_ce`` or ``trunk_next_state_ce``:

    * ``grad_norm`` / ``grad_share`` / ``grad_cos`` per population, and the
      group's ``total_norm`` and ``agreement``, exactly as the other groups
      publish them.
    * ``grad_halves_cos/<group>/<population>`` -- cosine between the two
      environment halves' gradients. Near 1: the population's gradient is
      signal. Near 0: it is per-token noise at this batch size.
    * ``grad_coherent_norm/<group>/<population>`` -- ``2 sqrt(<g_a, g_b>)``,
      an estimate of the population's expected gradient norm with the noise
      removed.
    * ``grad_coherent_fraction/<group>/<population>`` -- ``4 <g_a, g_b> /
      ||g_a + g_b||**2``, the fraction of the population's squared gradient norm
      that is coherent. An unbiased estimator, so it can read slightly below
      zero for a population that is all noise.
    * ``grad_coherent_share/<group>/<population>`` -- the population's share of
      the summed coherent norms: which population the optimizer actually
      learns from, as opposed to which one is loudest.

    Args:
        accumulator: The minibatch's accumulated term gradients.

    Returns:
        Metric name → value; empty when the split was not accumulated.
    """

    accumulated = set(accumulator.term_names)
    records: dict[str, float] = {}
    for likelihood in NEXT_STATE_LIKELIHOODS:
        names = [
            f"ns_{likelihood}/{population}/{half}"
            for population in NEXT_STATE_POPULATIONS
            for half in ENVIRONMENT_HALVES
        ]
        if not all(name in accumulated for name in names):
            continue
        groups = [
            [2 * p, 2 * p + 1] for p in range(len(NEXT_STATE_POPULATIONS))
        ]  # indices of each population's halves in ``names``
        for trunk in (False, True):
            group = f"trunk_next_state_{likelihood}" if trunk else f"next_state_{likelihood}"
            gram = accumulator.gram(names, trunk=trunk).detach().cpu().double()  # (6, 6)
            merged = _merge_gram(gram, groups)  # (3, 3)
            records.update(
                scope_metric_records(group, scope_statistics(NEXT_STATE_POPULATIONS, merged))
            )
            coherent: dict[str, float] = {}
            for p, population in enumerate(NEXT_STATE_POPULATIONS):
                a, b = groups[p]
                cross = float(gram[a, b])
                norm_a = math.sqrt(max(float(gram[a, a]), 0.0))
                norm_b = math.sqrt(max(float(gram[b, b]), 0.0))
                summed_sq = max(float(merged[p, p]), 0.0)
                coherent[population] = 2.0 * math.sqrt(max(cross, 0.0))
                records[f"grad_halves_cos/{group}/{population}"] = cross / (norm_a * norm_b + _EPS)
                records[f"grad_coherent_norm/{group}/{population}"] = coherent[population]
                records[f"grad_coherent_fraction/{group}/{population}"] = (
                    4.0 * cross / (summed_sq + _EPS)
                )
            coherent_total = sum(coherent.values())
            for population, value in coherent.items():
                records[f"grad_coherent_share/{group}/{population}"] = value / (
                    coherent_total + _EPS
                )
    return records


__all__ = [
    "ENVIRONMENT_HALVES",
    "MOMENT_NAMES",
    "NEXT_STATE_LIKELIHOODS",
    "NEXT_STATE_POPULATIONS",
    "environment_halves",
    "gradient_metric_records",
    "gradient_terms",
    "population_masks",
    "population_metric_records",
    "population_moments",
]
