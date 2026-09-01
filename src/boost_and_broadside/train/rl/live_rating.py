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
    fixed: Mapping[str, float],
    prior_games: float = DEFAULT_PRIOR_GAMES,
    passes: int = 64,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """Rate the checkpoints against references whose ratings are already known.

    ``fixed`` holds the gauge's defined players — the random agent at 0, each
    semi-random rung at 1000·p, the scripted controller at 1000. These are not
    estimates and are not refit. Everything else in the matrix is free.

    **Fitting them jointly instead is a trap, and it was observed failing on real
    data before this signature grew a ``fixed`` argument.** Eleven updates into
    run 728 the accumulated record held two clean sweeps by the same checkpoint,
    15–0 over the random agent and 27–0 over the 0.2 rung. Complete separation
    sends both opponents' maximum likelihood to −∞, so ``prior_games`` alone
    decides where they land — and its shrinkage weakens with sample size, so the
    agent with *fewer* games came out higher. A joint fit ranked random above the
    0.2 rung by 100 points, on 42 games that agreed perfectly with the gauge.
    Pinning the defined players removes the failure at its source rather than
    waiting for game counts to grow out of it.

    The trade is deliberate. The linear rung assignment is known to sit up to
    ~106 Elo from a fitted ladder at the weak end (see ``config/live_elo``), so
    pinning propagates a bias into the checkpoint ratings. That bias is bounded,
    known, and identical in every run under the same gauge, which is what matters
    for comparing runs; an unpinned fit at low connectivity is unbounded and
    different every time.

    Solved by coordinate ascent — each free player in turn against everyone
    else's current rating — rather than by the joint MM iteration, which has no
    way to hold a player still. The likelihood is concave in each coordinate and
    every free player is fit exactly given the rest, so the passes converge
    monotonically. On this pool it takes a handful.

    Returns an empty mapping when there is nothing to fit, in which case the
    caller keeps the ratings it already has.
    """
    labels = [label for label in matrix.labels() if label not in NON_STATIONARY]
    free = [label for label in labels if label not in fixed]
    known = {label: float(fixed[label]) for label in labels if label in fixed}
    if not free or not known:
        return dict(known)
    games = {label: _opponent_counts(matrix, label) for label in free}
    # Start every free player at the strongest defined reference. A checkpoint
    # on the ladder has beaten its way up to at least the rungs it plays, so this
    # is closer than the pool mean and costs a pass or two less.
    ratings = {**known, **dict.fromkeys(free, max(known.values()))}
    for _ in range(passes):
        movement = 0.0
        for label in free:
            counts = {
                opponent: record
                for opponent, record in games[label].items()
                if opponent in ratings
            }
            rating, _ = rate_live(
                counts, ratings, prior_games=prior_games, prior_rating=max(known.values())
            )
            if np.isfinite(rating):
                movement = max(movement, abs(rating - ratings[label]))
                ratings[label] = rating
        if movement < tolerance:
            break
    return {label: ratings[label] for label in labels}


def _opponent_counts(
    matrix: MatchMatrix, player: str
) -> dict[str, tuple[float, float, float]]:
    """One player's raw win/loss/tie record against each opponent it has met."""
    counts: dict[str, tuple[float, float, float]] = {}
    for record in matrix.as_records():
        low, high = str(record["a"]), str(record["b"])
        if player == low:
            counts[high] = (record["wins_a"], record["wins_b"], record["ties"])
        elif player == high:
            counts[low] = (record["wins_b"], record["wins_a"], record["ties"])
    return counts


def rate_live(
    counts: Mapping[str, tuple[float, float, float]],
    ratings: Mapping[str, float],
    *,
    prior_games: float = 0.0,
    prior_rating: float = 0.0,
) -> tuple[float, float]:
    """Solve for the rating that best explains one update's live record.

    Args:
        counts:  This update's win/loss/tie per opponent, from the live policy's
                 perspective.
        ratings: Opponent ratings, ideally stage 1's refit output.

        prior_games: Virtual games split evenly for and against
                 ``prior_rating``. Zero for the live policy, where an unbounded
                 record should be *reported* as unbounded; positive when fitting
                 a ladder player, where a bounded answer is needed every update
                 and the shrinkage is under a rating point once real games
                 accumulate.
        prior_rating: Where those virtual games are played.

    Returns:
        ``(rating, stderr)``. With no prior, a clean sweep in either direction
        has no finite maximum likelihood and comes back as ``±inf`` with infinite
        error rather than as a large finite number a caller might mistake for a
        measurement.
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
    if prior_games > 0.0:
        opponents = np.append(opponents, prior_rating)
        wins = np.append(wins, 0.5 * prior_games)
        losses = np.append(losses, 0.5 * prior_games)
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
        gauge: Mapping[str, float],
    ) -> dict[str, float]:
        """Refit the ladder and the live policy, and report both as metrics.

        Args:
            matrix:            Accumulated record among weight-frozen players.
            counts:            This update's live-policy record by opponent.
            fallback_ratings:  Ratings to use for opponents the matrix has no
                               games for. The gauge's defined values for the
                               stationary references, and the filter's current
                               numbers for anything else.
            gauge:             The defined players, held fixed by stage 1.
        """
        self.ladder = fit_ladder(matrix, fixed=gauge)
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
