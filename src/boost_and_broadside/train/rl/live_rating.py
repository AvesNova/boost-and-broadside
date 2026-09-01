"""Two-stage live rating: refit the ladder, then solve for the live policy.

Phase 2 of ``docs/internal/live-elo-plan.md``, rescoped by the measurement in
that document's Phase 0 section. Two findings from replaying 719's recorded
counts against its post-hoc calibrated curve shaped what is here, and both
contradict the original design:

*Fit against the whole pool, not the floor.* Scored over 1004 updates, a fit
using every opponent reaches 15.4 RMS against the calibrated curve where a fit
using only the gauge's defined references reaches 65.9 and one using only the
scripted controller reaches 63.0. Bradley-Terry pinning every difference from a
single shared anchor is a property of the MLE over a *connected graph*; it says
nothing useful about the one saturated edge a floor-only fit collapses to once
the policy wins almost every game against the floor. The run's own rungs are
what carry the rating, so the ladder's quality is the binding constraint — which
is what stage 1 is for.

*Use one update, not a window.* Every window length tested was monotonically
worse: 15.4 at one update, 21.8 at two, 36.7 at eight. The policy improves fast
enough that pooling costs more in lag than it buys in variance.

The structural gain is separate from the RMS number and is not visible in it.
A K-factor filter carries state, so it settles where competing pulls cancel and a
shift that gets in stays in — run 727 held a +56 Elo shift for 270M steps after a
resume. A per-update solve has no memory of its own previous value, so that
mechanism is gone. It is not gone from the *ladder*: stage 1's ratings are
accumulated, deliberately, because those players do not move.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry, fit_single_rating
from boost_and_broadside.train.rl.match_matrix import MatchMatrix

# Virtual decisive games per player, split for and against the anchor. Without
# it a player that wins every game it plays has an infinite MLE, which is routine
# here — a late rung beats the random agent every time — and the iteration then
# stops wherever its cap lands, so refits on near-identical counts land hundreds
# of points apart. This is the single most likely cause of the meaningless swings
# earlier attempts at an online fitter produced. Matches ELO_CALIBRATE's value.
DEFAULT_PRIOR_GAMES = 1.0

# Players excluded from every fit here because their strength changes under the
# record. A count matrix cannot say when a game was played, so pooling a moving
# player's results fits the average of something that was never the same twice.
NON_STATIONARY = frozenset({"avg"})


def fit_ladder(
    matrix: MatchMatrix,
    *,
    anchor_label: str,
    anchor_elo: float,
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> dict[str, float]:
    """Refit every accumulated player's rating, with the anchor held fixed.

    The anchor is pinned rather than centred. Centring on a pool mean, or on
    whichever player happens to be first, makes every rating jump when the pool
    changes — and the pool changes at every milestone. Shifting so the anchor
    reads its defined gauge value keeps the scale still across promotions, which
    is what lets ratings from different points in a run be compared at all.

    Returns an empty mapping when the anchor has no accumulated games, since
    without it the fit has no gauge and the caller should keep using the ratings
    it already has.
    """
    labels = [label for label in matrix.labels() if label not in NON_STATIONARY]
    if anchor_label not in labels or len(labels) < 2:
        return {}
    wins = matrix.scored_wins(labels)
    anchor = labels.index(anchor_label)
    if wins[anchor].sum() + wins[:, anchor].sum() <= 0.0:
        return {}
    fit = fit_bradley_terry(wins, anchor=anchor, prior_games=prior_games)
    # fit_bradley_terry pins its anchor at 0, so one shift puts the whole ladder
    # on the gauge.
    return {
        label: float(rating) + anchor_elo
        for label, rating in zip(labels, fit.ratings)
    }


def rate_live(
    counts: Mapping[str, tuple[float, float, float]],
    ratings: Mapping[str, float],
) -> tuple[float, float]:
    """Solve for the rating that best explains one update's live record.

    Args:
        counts:  This update's win/loss/tie per opponent, from the live policy's
                 perspective.
        ratings: Opponent ratings, ideally stage 1's refit output.

    Returns:
        ``(rating, stderr)``. A clean sweep in either direction has no finite
        maximum likelihood and comes back as ``±inf`` with infinite error rather
        than as a large finite number the caller might mistake for a measurement.
    """
    usable = [
        label
        for label, (wins, losses, ties) in counts.items()
        if label in ratings and label not in NON_STATIONARY and (wins or losses or ties)
    ]
    if not usable:
        return float("nan"), float("inf")
    opponents = np.array([ratings[label] for label in usable], dtype=np.float64)
    # Half-win draws, the convention every other caller of the fitter uses.
    wins = np.array([counts[label][0] + 0.5 * counts[label][2] for label in usable])
    losses = np.array([counts[label][1] + 0.5 * counts[label][2] for label in usable])
    return fit_single_rating(opponents, wins, losses)


class TwoStageRating:
    """Stage 1 and stage 2 run together, once per update.

    Holds the last finite live rating so a single saturated update — the live
    policy sweeping every opponent it happened to draw — reports the previous
    value rather than a gap or an infinity. That is a display convenience, not a
    filter: it never mixes the two, and the next solvable update replaces it
    outright.
    """

    def __init__(self, *, anchor_label: str, anchor_elo: float) -> None:
        self.anchor_label = anchor_label
        self.anchor_elo = float(anchor_elo)
        self.ladder: dict[str, float] = {}
        self.live_elo: float | None = None
        self.live_stderr: float = float("inf")

    def update(
        self,
        matrix: MatchMatrix,
        counts: Mapping[str, tuple[float, float, float]],
        fallback_ratings: Mapping[str, float],
    ) -> dict[str, float]:
        """Refit the ladder and the live policy, and report both as metrics.

        Args:
            matrix:            Accumulated record among weight-frozen players.
            counts:            This update's live-policy record by opponent.
            fallback_ratings:  Ratings to use for opponents the matrix has no
                               games for. The gauge's defined values for the
                               stationary references, and the filter's current
                               numbers for anything else.
        """
        self.ladder = fit_ladder(
            matrix, anchor_label=self.anchor_label, anchor_elo=self.anchor_elo
        )
        ratings = {**fallback_ratings, **self.ladder}
        rating, stderr = rate_live(counts, ratings)
        metrics: dict[str, float] = {}
        if np.isfinite(rating):
            self.live_elo = rating
            self.live_stderr = stderr
            metrics["two_stage/live_elo"] = rating
            metrics["two_stage/live_stderr"] = stderr
        elif self.live_elo is not None:
            metrics["two_stage/live_elo"] = self.live_elo
        metrics["two_stage/ladder_players"] = float(len(self.ladder))
        for label, value in self.ladder.items():
            metrics[f"two_stage/ladder/{label}"] = value
        return metrics
