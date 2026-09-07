"""Gradient decomposition over one complete optimizer minibatch.

The PPO update sends several named losses into one shared trunk, and
``max_grad_norm`` renormalizes whatever arrives together. What each term
contributes, and whether two terms pull the trunk the same way, is therefore not
recoverable from the total gradient norm — it has to be measured.

This module measures it. :class:`TermGradientAccumulator` takes named scalar
loss terms one micro-batch at a time, differentiates each against the model
parameters, and accumulates the results parameter-by-parameter. Norms and
cosines are taken only once the whole optimizer minibatch has been accumulated,
so what they describe is the gradient direction of a real optimizer step:

    cos(sum_microbatches g_a, sum_microbatches g_b)

and never the average of per-micro-batch cosines, which is a different quantity
and is not the one the step follows.

Every statistic for one group of terms comes from a single Gram matrix
``G[i][j] = <g_i, g_j>``: the diagonal gives norms, the off-diagonal gives
cosines, and the full sum gives ``||sum_i g_i||`` without ever materializing the
summed vector. Gradients are never flattened into one large tensor — they are
held and reduced per parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

# Divide-by-zero guard for cosines and shares. A term with no gradient at all
# reads as zero rather than NaN, which is the honest answer: it did not move the
# parameters.
_EPS = 1e-12


@dataclass(frozen=True)
class ScopeStatistics:
    """Norms, cosines, and agreement for one group of terms over one parameter scope.

    Attributes:
        norms:      Term name → L2 norm of its accumulated minibatch gradient.
        cosines:    Unordered term pair → cosine between their gradients.
        shares:     Term name → its norm divided by the summed term norms. The
                    share of the pre-clip gradient budget that term is asking for.
        total_norm: ``||sum_i g_i||`` — the norm of the combined gradient.
        agreement:  ``||sum_i g_i|| / sum_i ||g_i||``. One means every term points
                    the same way; near zero means they are cancelling.
    """

    norms: dict[str, float]
    cosines: dict[tuple[str, str], float]
    shares: dict[str, float]
    total_norm: float
    agreement: float


class TermGradientAccumulator:
    """Accumulates named term gradients across the micro-batches of one minibatch.

    Args:
        params:     Parameters to differentiate against, in a stable order.
        trunk_mask: Parallel to ``params``; True where the parameter belongs to
                    the shared trunk. Task-specific heads have disjoint
                    parameters, so a whole-model cosine between two heads' terms
                    is dominated by that disjointness; the trunk scope is where
                    terms actually compete.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], trunk_mask: Sequence[bool]) -> None:
        if len(params) != len(trunk_mask):
            raise ValueError("params and trunk_mask must describe the same parameters")
        self._params = list(params)
        self._trunk_mask = list(trunk_mask)
        # Term name → per-parameter fp32 gradient sums, aligned with _params.
        # None entries are parameters the term does not reach.
        self._sums: dict[str, list[torch.Tensor | None]] = {}
        self._order: list[str] = []
        self._microbatches = 0

    @property
    def microbatches(self) -> int:
        """Micro-batches accumulated so far."""

        return self._microbatches

    @property
    def term_names(self) -> tuple[str, ...]:
        """Terms that have contributed at least one gradient, in first-seen order."""

        return tuple(self._order)

    def accumulate(self, terms: Mapping[str, torch.Tensor], scale: float = 1.0) -> None:
        """Add one micro-batch's contribution to every named term.

        Terms that do not participate in the graph (a disabled coefficient, a
        constant zero) are skipped rather than differentiated.

        Args:
            terms: Term name → scalar loss, already weighted exactly as it enters
                the training loss.
            scale: Factor the training backward applies to this micro-batch's
                loss (the per-scale divisor), so the accumulated gradients sum to
                the gradient the optimizer step actually receives.
        """
        for name, term in terms.items():
            if not term.requires_grad:
                continue
            scaled = term if scale == 1.0 else term * scale
            grads = torch.autograd.grad(
                scaled,
                self._params,
                retain_graph=True,
                allow_unused=True,
            )
            self._add(name, grads)
        self._microbatches += 1

    def _add(self, name: str, grads: Sequence[torch.Tensor | None]) -> None:
        """Fold one gradient tuple into the running fp32 sum for ``name``."""
        slot = self._sums.get(name)
        if slot is None:
            self._order.append(name)
            self._sums[name] = [None if g is None else g.detach().float() for g in grads]
            return
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            existing = slot[index]
            slot[index] = grad.detach().float() if existing is None else existing + grad.float()

    def _dot(self, left: str, right: str, trunk: bool) -> torch.Tensor:
        """Inner product of two accumulated gradients over the selected scope."""
        left_grads = self._sums[left]
        right_grads = self._sums[right]
        total: torch.Tensor | None = None
        for index, (a, b) in enumerate(zip(left_grads, right_grads, strict=True)):
            if a is None or b is None:
                continue
            if trunk and not self._trunk_mask[index]:
                continue
            partial = (a * b).sum()
            total = partial if total is None else total + partial
        if total is None:
            return torch.zeros((), device=self._params[0].device, dtype=torch.float32)
        return total

    def gram(self, names: Sequence[str], *, trunk: bool) -> torch.Tensor:
        """The (n, n) Gram matrix of the named terms' accumulated gradients.

        Args:
            names: Terms to include; every name must have been accumulated.
            trunk: Restrict the inner products to shared-trunk parameters.

        Returns:
            (n, n) float32 symmetric matrix on the parameter device.
        """
        n = len(names)
        entries = [
            self._dot(names[i], names[j], trunk) for i in range(n) for j in range(i, n)
        ]  # (n*(n+1)/2,) upper triangle, row-major
        if not entries:
            return torch.zeros((0, 0), device=self._params[0].device, dtype=torch.float32)
        packed = torch.stack(entries)  # (n*(n+1)/2,)
        matrix = torch.zeros(n, n, device=packed.device, dtype=packed.dtype)  # (n, n)
        rows, cols = torch.triu_indices(n, n, device=packed.device)
        matrix[rows, cols] = packed
        return matrix + torch.triu(matrix, diagonal=1).T

    def statistics(self, names: Sequence[str], *, trunk: bool) -> ScopeStatistics:
        """Norms, cosines, shares, and agreement for one group of terms.

        Args:
            names: Terms forming the group; the summed-gradient statistics
                describe exactly this subset.
            trunk: Restrict to shared-trunk parameters.

        Returns:
            The group's statistics, already transferred off the device.
        """
        return scope_statistics(names, self.gram(names, trunk=trunk))


def scope_statistics(names: Sequence[str], gram: torch.Tensor) -> ScopeStatistics:
    """Derive every published statistic from one group's Gram matrix.

    Args:
        names: Term names, ordered to match ``gram``'s rows and columns.
        gram:  (n, n) inner products of the terms' accumulated gradients.

    Returns:
        Host-side statistics for the group.
    """
    if len(names) == 0:
        return ScopeStatistics({}, {}, {}, 0.0, 0.0)
    values = gram.detach().cpu().double()  # (n, n) one transfer per group
    norms_t = values.diagonal().clamp(min=0.0).sqrt()  # (n,)
    norms = {name: float(norms_t[i]) for i, name in enumerate(names)}
    norm_sum = float(norms_t.sum())
    total_norm = float(values.sum().clamp(min=0.0).sqrt())
    cosines = {
        (names[i], names[j]): float(values[i, j] / (norms_t[i] * norms_t[j] + _EPS))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    shares = {name: norm / (norm_sum + _EPS) for name, norm in norms.items()}
    return ScopeStatistics(
        norms=norms,
        cosines=cosines,
        shares=shares,
        total_norm=total_norm,
        agreement=total_norm / (norm_sum + _EPS),
    )


def scope_metric_records(group: str, stats: ScopeStatistics) -> dict[str, float]:
    """Flatten one group's statistics into logger keys.

    Args:
        group: Metric-namespace segment, e.g. ``top_level`` or
            ``trunk_reward_policy``.
        stats: The group's statistics.

    Returns:
        Metric name → value.
    """
    records: dict[str, float] = {}
    for name, value in stats.norms.items():
        records[f"grad_norm/{group}/{name}"] = value
    for name, value in stats.shares.items():
        records[f"grad_share/{group}/{name}"] = value
    for (left, right), value in stats.cosines.items():
        records[f"grad_cos/{group}/{left}__{right}"] = value
    records[f"grad_diag/total_norm/{group}"] = stats.total_norm
    records[f"grad_diag/agreement/{group}"] = stats.agreement
    return records


__all__ = [
    "ScopeStatistics",
    "TermGradientAccumulator",
    "scope_metric_records",
    "scope_statistics",
]
