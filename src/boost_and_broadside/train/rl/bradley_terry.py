"""Bradley-Terry rating estimation from raw match counts.

The in-training evaluator maintains ratings by nudging them after each result
with a fixed K-factor. That is a tracking filter for a moving target, not an
estimator: its fixed point depends on K, on the order results arrive, and on
where the rating happened to start. For a roster of frozen checkpoints — none
of which move — the ratings are a static estimation problem with a maximum
likelihood answer, and this module computes it.

Draws are supplied to this module already scored, as fractional wins — the
caller decides the convention (see TIE_MODES in evaluation/tournament.py). The MM
iteration is weight-based, so fractional counts need no special handling.

What this module deliberately does not do is fit a draw *parameter*. Davidson
and Rao-Kupper, the standard tie-aware models, are both scale-invariant in the
strengths, so they can only express draw frequency as a function of the rating
gap. Measurement here contradicts that: draws track the absolute level of a
matchup, with near-random pairs stalemating to the horizon ~30% of the time
while evenly matched strong pairs draw well under 1%. Fitting either model would
bias every rating touching a tie-heavy pair. Scoring a draw as half a win avoids
this entirely, because it models no draw process at all — it just fits the
expected score, which is what Elo has always meant.
"""

from dataclasses import dataclass

import numpy as np

# Rating points per factor-of-10 in odds, matching expected_score in elo_eval.
RATING_SCALE = 400.0
# d(win probability)/d(rating difference) scaling: p is logistic in c * gap.
_C = np.log(10.0) / RATING_SCALE


@dataclass(frozen=True)
class RatingFit:
    """Maximum likelihood ratings for one set of match counts."""

    ratings: np.ndarray  # (K,) rating points, anchor pinned at 0
    stderr: np.ndarray  # (K,) standard errors; 0 at the anchor by construction
    games: np.ndarray  # (K,) decisive games each player contributed to
    iterations: int
    converged: bool

    def max_stderr(self, exclude_gameless: bool = True) -> float:
        """Largest standard error over rated players, in rating points."""
        mask = self.games > 0 if exclude_gameless else np.ones_like(self.games, dtype=bool)
        return float(self.stderr[mask].max()) if mask.any() else float("inf")


def win_probability(rating: np.ndarray | float, opponent: np.ndarray | float) -> np.ndarray | float:
    """Logistic probability that `rating` beats `opponent`, given a decisive result."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(opponent) - np.asarray(rating)) / RATING_SCALE))


def fit_bradley_terry(
    wins: np.ndarray,
    anchor: int = 0,
    prior_games: float = 1.0,
    max_iterations: int = 10_000,
    tolerance: float = 1e-9,
) -> RatingFit:
    """Fit ratings to a decisive-win count matrix by maximum likelihood.

    Uses Hunter's MM iteration on the strength scale, which is monotonic in the
    likelihood and cannot diverge, then converts to rating points.

    Args:
        wins:        (K, K) matrix; wins[i, j] is i's decisive wins over j.
                     Draws must already be excluded.
        anchor:      Index pinned at rating 0. Bradley-Terry ratings are only
                     identified up to a shared additive constant, so some player
                     has to define the origin.
        prior_games: Virtual decisive games per player, split evenly for and
                     against the anchor. Without this a player that wins (or
                     loses) every game it plays has an infinite MLE, which is
                     routine here — early checkpoints are genuinely hopeless
                     against late ones. At the default one game against the
                     thousands a converged suite collects, the shrinkage toward
                     the anchor is well under a rating point.

    Returns:
        The fitted ratings and their standard errors.
    """
    wins = np.asarray(wins, dtype=np.float64)
    assert wins.ndim == 2 and wins.shape[0] == wins.shape[1], "wins must be square"
    assert (wins >= 0).all(), "win counts must be non-negative"
    size = wins.shape[0]
    assert 0 <= anchor < size, f"anchor {anchor} out of range for {size} players"

    counts = wins.copy()
    np.fill_diagonal(counts, 0.0)
    if prior_games > 0:
        half = prior_games / 2.0
        counts[:, anchor] += half
        counts[anchor, :] += half
        np.fill_diagonal(counts, 0.0)

    total_wins = counts.sum(axis=1)  # (K,)
    pair_games = counts + counts.T  # (K, K), symmetric decisive games

    strength = np.ones(size, dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        # π_i ← W_i / Σ_j n_ij / (π_i + π_j)
        denominator = (pair_games / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = np.where(denominator > 0, total_wins / np.maximum(denominator, 1e-300), strength)
        updated = np.maximum(updated, 1e-300)
        updated /= updated[anchor]  # re-gauge every step to keep the scale pinned
        shift = np.abs(np.log(updated) - np.log(strength)).max()
        strength = updated
        if shift < tolerance:
            converged = True
            break

    ratings = RATING_SCALE * np.log10(strength)
    ratings -= ratings[anchor]
    return RatingFit(
        ratings=ratings,
        stderr=rating_stderr(pair_games, ratings, anchor),
        games=(wins + wins.T).sum(axis=1),
        iterations=iteration,
        converged=converged,
    )


def fisher_information(pair_games: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Fisher information of the ratings given a symmetric game-count matrix.

    Diagonal entries accumulate each player's total information; off-diagonal
    entries carry the negative coupling that makes a rating chain's error
    accumulate along its length.
    """
    probability = win_probability(ratings[:, None], ratings[None, :])  # (K, K)
    weight = _C**2 * pair_games * probability * (1.0 - probability)
    np.fill_diagonal(weight, 0.0)
    information = -weight.copy()
    np.fill_diagonal(information, weight.sum(axis=1))
    return information


def _reduced_covariance(
    pair_games: np.ndarray, ratings: np.ndarray, anchor: int
) -> tuple[np.ndarray, list[int]]:
    """Covariance of the non-anchor ratings, and the indices it covers.

    The full information matrix is singular by construction — shifting every
    rating by a constant changes nothing — so the anchor's row and column are
    dropped before inverting, which is exactly the gauge the fit pins.
    """
    size = ratings.shape[0]
    kept = [index for index in range(size) if index != anchor]
    information = fisher_information(pair_games, ratings)[np.ix_(kept, kept)]
    # Ridge keeps a player with no games from making the whole matrix singular;
    # its own variance blows up (correctly) instead of taking the fit with it.
    information += np.eye(len(kept)) * 1e-12
    return np.linalg.inv(information), kept


def rating_stderr(pair_games: np.ndarray, ratings: np.ndarray, anchor: int) -> np.ndarray:
    """Standard error of each rating, in rating points. The anchor's is 0."""
    covariance = rating_covariance(pair_games, ratings, anchor)
    stderr = np.zeros_like(ratings)
    stderr[:] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return stderr


def rating_covariance(
    pair_games: np.ndarray, ratings: np.ndarray, anchor: int
) -> np.ndarray:
    """Full rating covariance matrix in a gauge where ``anchor`` is fixed.

    Returning the full matrix makes uncertainty transformable along with the
    ratings. In particular, a two-landmark normalization depends on correlated
    differences and cannot be assigned an honest error bar from marginal
    standard errors alone.
    """
    reduced, kept = _reduced_covariance(pair_games, ratings, anchor)
    covariance = np.zeros((ratings.size, ratings.size), dtype=np.float64)
    covariance[np.ix_(kept, kept)] = reduced
    return covariance


def allocation_value(pair_games: np.ndarray, ratings: np.ndarray, anchor: int) -> np.ndarray:
    """Marginal reduction in total rating variance per additional game, per pair.

    Returns a symmetric (K, K) matrix whose (i, j) entry is how much one more
    decisive game between i and j would shrink the summed variance of every
    rating — the A-optimal criterion, d(trace F⁻¹)/dn.

    A-optimality on the absolute ratings is the deliberate choice over
    D-optimality on pairwise differences. Pure information-greedy selection
    chases near-equal opponents, which here are neighbours on the training
    timeline, and builds a chain: excellent relative precision, with absolute
    error random-walking along its length. Because trace(F⁻¹) is computed with
    the anchor's gauge already imposed, it prices that accumulated drift in and
    spends some of the budget on long-range links that pin the chain down.
    """
    covariance, kept = _reduced_covariance(pair_games, ratings, anchor)
    size = ratings.shape[0]
    # Column of the covariance for each player's unit vector; the anchor's is 0
    # because its rating is fixed and carries no variance to reduce.
    columns = np.zeros((size, len(kept)), dtype=np.float64)
    columns[kept] = covariance
    probability = win_probability(ratings[:, None], ratings[None, :])
    curvature = _C**2 * probability * (1.0 - probability)  # (K, K)
    difference = columns[:, None, :] - columns[None, :, :]  # (K, K, K-1)
    value = curvature * np.einsum("ijk,ijk->ij", difference, difference)
    np.fill_diagonal(value, 0.0)
    return value


def allocate_games(
    pair_games: np.ndarray,
    ratings: np.ndarray,
    anchor: int,
    budget: int,
    minimum_per_pair: int = 0,
) -> np.ndarray:
    """Split a batch of `budget` games across pairs by A-optimal value.

    Returns a symmetric (K, K) integer matrix of games to play per pair. Pairs
    are unordered here; the caller is expected to split each pair's games evenly
    between the two team roles so that any first-player advantage cancels rather
    than being absorbed into the ratings.
    """
    if np.asarray(pair_games).sum() <= 0:
        # Seed batch: with no results in hand there is no basis to prefer any
        # pair, so spread evenly. Deriving weights from the empty information
        # matrix instead would just rank pairs by the regularizing ridge.
        value = np.triu(np.ones_like(pair_games, dtype=np.float64), k=1)
    else:
        value = np.triu(allocation_value(pair_games, ratings, anchor), k=1)
    total = value.sum()
    if total <= 0:  # no information anywhere — spread evenly
        value = np.triu(np.ones_like(value), k=1)
        total = value.sum()
    share = value / total
    allocation = np.floor(share * budget).astype(np.int64)
    if minimum_per_pair > 0:
        allocation = np.maximum(allocation, minimum_per_pair * np.triu(np.ones_like(allocation), 1))
    # Hand out the remainder to the most valuable pairs, largest share first.
    remaining = budget - int(allocation.sum())
    if remaining > 0:
        order = np.argsort(share, axis=None)[::-1]
        for flat in order[:remaining]:
            row, column = np.unravel_index(flat, share.shape)
            if row < column:
                allocation[row, column] += 1
    return allocation + allocation.T


def fit_single_rating(
    opponent_ratings: np.ndarray,
    wins: np.ndarray,
    losses: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    bracket: float = 4_000.0,
) -> tuple[float, float]:
    """Fit one player's rating against opponents whose ratings are already known.

    This is how a non-stationary player is rated. The live policy exists in one
    form for a single update, so it can never be re-measured; all that survives
    is its win/loss record from that update. Solving for the rating that best
    explains that record against calibrated opponents recovers what the policy
    was actually worth at that moment, independent of wherever the in-training
    filter had drifted to.

    Args:
        bracket: How far beyond the opponents' own range to search, in rating
                 points. A record that implies a rating outside this window is
                 reported as unbounded rather than as a huge finite number.

    Returns:
        (rating, standard error). The rating is +/- inf when the record is a
        clean sweep in either direction, with infinite error to match.
    """
    opponent_ratings = np.asarray(opponent_ratings, dtype=np.float64)
    wins = np.asarray(wins, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    decisive = wins + losses
    total = decisive.sum()
    if total <= 0:
        return float("nan"), float("inf")
    if losses.sum() <= 0:
        return float("inf"), float("inf")
    if wins.sum() <= 0:
        return float("-inf"), float("inf")

    def gradient(rating: float) -> float:
        probability = win_probability(rating, opponent_ratings)
        return float((wins * (1.0 - probability) - losses * probability).sum())

    # Bisection rather than Newton. The log-likelihood is strictly concave, so
    # its gradient decreases monotonically and a bracket cannot fail — whereas
    # Newton diverges on lopsided records: as the rating runs ahead of every
    # opponent the curvature underflows toward zero while the gradient stays
    # finite, so the step explodes. That produced ratings around 1e144 on real
    # early-training records, which are near-sweeps against the random anchor.
    low = float(opponent_ratings.min()) - bracket
    high = float(opponent_ratings.max()) + bracket
    if gradient(low) < 0.0:
        return float("-inf"), float("inf")
    if gradient(high) > 0.0:
        return float("inf"), float("inf")
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        if gradient(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    rating = 0.5 * (low + high)
    probability = win_probability(rating, opponent_ratings)
    information = _C**2 * (decisive * probability * (1.0 - probability)).sum()
    stderr = float(1.0 / np.sqrt(information)) if information > 0 else float("inf")
    return float(rating), stderr
