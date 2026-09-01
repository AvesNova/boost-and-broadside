"""Choosing which ladder games to play next.

Phase 4 of ``docs/internal/live-elo-plan.md``. The evaluator's slot 4 plays the
floating checkpoint against one anchor per episode, and until now it drew that
anchor by local information alone — ``p(1−p)``, which peaks on the most evenly
matched opponent available. That rule answers "which single game is most
informative about *some* rating", which is not the question. The question is
which game most reduces the uncertainty in the rung's offset **from the floor**,
and a game can be individually informative while telling us nothing about that.

The correction comes from two facts checked numerically in ``elo_diagnostics``:
the Fisher information of a Bradley-Terry model is a weighted graph Laplacian,
and ``Var(r_i − r_j)`` is the effective resistance between them. Under
Sherman-Morrison the marginal value of one more game on edge (i, j) toward
``Var(r_protagonist − r_anchor)`` is

    gain_ij = c²·p_ij(1−p_ij) · b_ij²    where   b = φ_i − φ_j,
                                                 L φ = e_protagonist − e_anchor

which is *local information times global position*: the potential drop across an
edge under unit current injected at the protagonist and drawn off at the anchor.
Play the matches that carry the most current between the rung and the floor.
The old rule is the first factor with the second silently set to one.

Two things this buys that a threshold could not. A dead-end branch falls out on
its own, because no current flows through it — the random agent scores orders of
magnitude below a mid-ladder rung without anyone having to decide it is too weak
to matter. And the direct rung-to-scripted edge wins early because ``b²`` is
large, then hands its budget to the chain by itself as the rung strengthens and
``p → 1`` kills ``info`` faster than ``b²`` grows. "When does the anchor
saturate" stops being a question anyone has to answer in advance.

The floor mixed in at the end is not a hedge against the rule being wrong. It is
insurance against the failure mode information-weighted allocation *causes*:
starve an edge completely and the graph can split, at which point the ratings
either side of the split are unidentified and swing on nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from boost_and_broadside.train.rl.bradley_terry import RATING_SCALE, fisher_information
from boost_and_broadside.train.rl.elo_diagnostics import potential
from boost_and_broadside.train.rl.match_matrix import MatchMatrix

_C = float(np.log(10.0) / RATING_SCALE)

# Share of the budget spread evenly over the candidates regardless of score.
# Small enough not to blunt the rule, large enough that no edge can be starved to
# the point of disconnecting the graph. A game costs nothing extra to play — the
# slot runs either way — so the only price of the floor is a slightly less
# optimal split, against a failure mode that is unrecoverable within a run.
DEFAULT_FLOOR_FRACTION = 0.15


def current_flow_scores(
    matrix: MatchMatrix,
    ratings: Mapping[str, float],
    *,
    protagonist: str,
    anchor: str,
    candidates: Sequence[str],
) -> np.ndarray | None:
    """Score each candidate opponent for the protagonist by c-optimal gain.

    Returns None when the target difference is not identified — the protagonist
    and the anchor sit in different components of the accumulated graph, which
    is the normal state early in a run before anything has been played. The
    caller should fall back to its previous rule rather than treat a flat score
    as a preference.
    """
    labels = sorted({*ratings, protagonist, anchor, *candidates})
    if protagonist not in ratings or anchor not in ratings:
        return None
    index = {label: position for position, label in enumerate(labels)}
    rating_array = np.array([ratings[label] for label in labels], dtype=np.float64)
    laplacian = fisher_information(matrix.pair_games(labels), rating_array)
    phi = potential(laplacian, index[protagonist], index[anchor])
    if phi is None:
        return None
    # gain = local information at one game x squared potential drop.
    gap = rating_array[index[protagonist]] - np.array(
        [ratings[label] for label in candidates], dtype=np.float64
    )
    probability = 1.0 / (1.0 + 10.0 ** (-gap / RATING_SCALE))
    drop = phi[index[protagonist]] - np.array(
        [phi[index[label]] for label in candidates], dtype=np.float64
    )
    return _C**2 * probability * (1.0 - probability) * drop**2


def allocation_weights(
    matrix: MatchMatrix,
    ratings: Mapping[str, float],
    *,
    protagonist: str,
    anchor: str,
    candidates: Sequence[str],
    floor_fraction: float = DEFAULT_FLOOR_FRACTION,
) -> np.ndarray | None:
    """Return a probability over ``candidates``, or None to keep the old rule.

    The scores are used as weights directly rather than by taking the single
    best edge. One episode is a tiny fraction of an update's budget, and a greedy
    argmax would pile every game of the update onto one pair — at which point
    that pair's information saturates and the next games are wasted. Sampling in
    proportion spreads the batch for free, which is the cheap stand-in for the
    Sherman-Morrison re-scoring a proper sequential allocation would do.
    """
    if not candidates:
        return None
    if not 0.0 <= floor_fraction < 1.0:
        raise ValueError(f"floor_fraction must be in [0, 1), got {floor_fraction}")
    scores = current_flow_scores(
        matrix, ratings, protagonist=protagonist, anchor=anchor, candidates=candidates
    )
    if scores is None or not np.all(np.isfinite(scores)):
        return None
    total = scores.sum()
    if total <= 0.0:
        return None
    uniform = np.full(len(candidates), 1.0 / len(candidates))
    return (1.0 - floor_fraction) * (scores / total) + floor_fraction * uniform
