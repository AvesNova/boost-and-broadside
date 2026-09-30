"""Who the next-state head learns from, and who it would learn from instead.

Every supervised next-state token belongs to one of three populations whose
labels mean different things:

* **ally** -- always in sight, so the label is one step of real dynamics.
* **enemy_visible** -- in sight now, so the same kind of label as an ally's.
* **enemy_hidden** -- out of sight, so the label pairs a *believed* current
  state with the true next one. It is one step of dynamics plus the correction
  of however far the belief has drifted, and it is wider by that drift.

A single likelihood over all three decides how much each population moves the
shared trunk, and the answer is not readable from the aggregate loss. This
module supplies the two measurements that make it readable.

**Calibration moments**, cheap enough for every update: per population and
channel, the mean likelihood, the mean squared standardized residual (1.0 when
the head's spread is honest), the mean log spread, and the mean squared error.
Whether one population swamps another under a given likelihood depends mostly
on whether the spread has caught up with its labels, so this is the reading
that says whether a gradient imbalance is structural or merely early.

**Gradient decomposition**, as a gradient diagnostic: the next-state term split
by population, and each population split again into two disjoint halves of the
environments. The halves are independent samples of the same population's
gradient, so their inner product estimates the squared norm of the *coherent*
gradient -- the part a larger batch would keep -- while the norm of their sum
also carries per-token noise. A population can dominate the gradient norm and
still contribute little a step can use, and only the halves tell those apart.

The decomposition is repeated for three likelihoods at the same weights: the
Gaussian NLL being trained, beta-NLL at beta 0.5, and squared error on the
means. A run therefore trains one objective and still reports how the
alternatives would have divided the trunk between populations once the spread
is trained, which a short run under each alternative cannot, because under
those the spread does not train the same way or at all.
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
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PhysicalNextState,
)

#: Supervised next-state populations, in a fixed order every tensor here uses.
NEXT_STATE_POPULATIONS: tuple[str, ...] = ("ally", "enemy_visible", "enemy_hidden")
#: The two disjoint environment halves each population is split into.
ENVIRONMENT_HALVES: tuple[str, ...] = ("a", "b")
#: Candidate likelihoods, the trained one first.
NEXT_STATE_LIKELIHOODS: tuple[str, ...] = ("nll", "beta_nll", "mse")
#: The beta of the beta-NLL candidate (Seitzer et al., 2022).
BETA_NLL_BETA = 0.5

#: Rows of :func:`population_moments`' second axis.
MOMENT_NAMES: tuple[str, ...] = ("nll", "z2", "log_sigma", "sq_err")

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


def candidate_losses(
    model: PhysicalNextState,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    nll: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-channel loss under each candidate likelihood, ``(..., 11)`` each.

    Args:
        model:      The next-state model the trainer uses.
        prediction: (..., 24) head output, means then uncertainty.
        labels:     (..., 11) normalized labels.
        nll:        (..., 11) the trained likelihood, already computed.

    Returns:
        Likelihood name → per-channel loss. ``beta_nll`` weights each channel's
        NLL by a stop-gradient ``sigma**(2*beta)``; ``mse`` is half the squared
        normalized residual on the means alone, so it sends no gradient to the
        spread.
    """

    weight = torch.exp(2.0 * BETA_NLL_BETA * model.log_sigma(prediction).detach())
    return {
        "nll": nll,
        "beta_nll": nll * weight,
        "mse": 0.5 * model.residual(prediction, labels).pow(2),
    }


def gradient_terms(
    losses: Mapping[str, torch.Tensor],
    masks: torch.Tensor,
    halves: torch.Tensor,
    denominator: torch.Tensor,
    coef: float,
) -> dict[str, torch.Tensor]:
    """One scalar term per likelihood, population, and environment half.

    Each term is normalized exactly as the training loss is -- the minibatch
    supervised-token count times the channel count -- so the trained
    likelihood's six terms sum to the ``next_state`` term's gradient, and every
    likelihood's terms are on the scale that likelihood would train at.

    Args:
        losses:      Likelihood name → (T, B, N, 11) per-channel loss.
        masks:       (3, T, B, N) bool population masks.
        halves:      (2, B) bool environment halves.
        denominator: Minibatch supervised-token count (scalar tensor).
        coef:        The next-state loss coefficient.

    Returns:
        ``"ns_<likelihood>/<population>/<half>"`` → scalar loss.
    """

    scale = coef / (denominator * PHYSICAL_MEAN_DIM)
    terms: dict[str, torch.Tensor] = {}
    for likelihood, per_dim in losses.items():
        per_token = per_dim.sum(-1)  # (T, B, N)
        for p, population in enumerate(NEXT_STATE_POPULATIONS):
            for h, half in enumerate(ENVIRONMENT_HALVES):
                weight = (masks[p] & halves[h].view(1, -1, 1)).float()  # (T, B, N)
                terms[f"ns_{likelihood}/{population}/{half}"] = (per_token * weight).sum() * scale
    return terms


def population_moments(
    model: PhysicalNextState,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    nll: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive per-population calibration sums for one micro-batch.

    Args:
        model:      The next-state model.
        prediction: (T, B, N, 24) head output.
        labels:     (T, B, N, 11) labels.
        nll:        (T, B, N, 11) per-channel NLL.
        masks:      (3, T, B, N) bool population masks.

    Returns:
        ``(sums, counts)``: sums is (3, 4, 11) in :data:`MOMENT_NAMES` order and
        counts is (3,), both float64 so they can be summed over a whole update
        without losing the small populations to roundoff.
    """

    with torch.no_grad():
        prediction = prediction.detach().float()
        per_token = torch.stack(
            [
                nll.detach().float(),
                model.standardized_square(prediction, labels),
                model.log_sigma(prediction),
                model.residual(prediction, labels).pow(2),
            ]
        )  # (4, T, B, N, 11)
        weights = masks.float()  # (3, T, B, N)
        sums = torch.einsum("mtbnc,ptbn->pmc", per_token, weights)  # (3, 4, 11)
        counts = weights.sum((1, 2, 3))  # (3,)
    return sums.double(), counts.double()


def population_metric_records(sums: torch.Tensor, counts: torch.Tensor) -> dict[str, float]:
    """Finalize accumulated calibration sums into logger keys.

    Args:
        sums:   (3, 4, 11) accumulated :func:`population_moments` sums.
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
        means = sums[p] / count  # (4, 11)
        for m, moment in enumerate(MOMENT_NAMES):
            records[f"next_state_population/{population}/{moment}"] = float(means[m].mean())
        for c, channel in enumerate(PHYSICAL_MEAN_NAMES):
            records[f"next_state_z2/{population}/{channel}"] = float(means[1, c])
            records[f"next_state_log_sigma/{population}/{channel}"] = float(means[2, c])
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

    For every likelihood and both parameter scopes (whole model and shared
    trunk), under group ``next_state_<likelihood>`` or
    ``trunk_next_state_<likelihood>``:

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
    "BETA_NLL_BETA",
    "ENVIRONMENT_HALVES",
    "MOMENT_NAMES",
    "NEXT_STATE_LIKELIHOODS",
    "NEXT_STATE_POPULATIONS",
    "candidate_losses",
    "environment_halves",
    "gradient_metric_records",
    "gradient_terms",
    "population_masks",
    "population_metric_records",
    "population_moments",
]
