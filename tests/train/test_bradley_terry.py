"""Tests for offline Bradley-Terry rating estimation."""

import numpy as np
import pytest

from boost_and_broadside.train.rl.bradley_terry import (
    RATING_SCALE,
    allocate_games,
    allocation_value,
    fit_bradley_terry,
    fit_single_rating,
    rating_stderr,
    win_probability,
)


def _synthetic_wins(
    ratings: np.ndarray, games_per_pair: int, seed: int = 0, noisy: bool = False
) -> np.ndarray:
    """Decisive win counts generated from known ratings."""
    size = len(ratings)
    rng = np.random.default_rng(seed)
    wins = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            probability = win_probability(ratings[i], ratings[j])
            if noisy:
                won = rng.binomial(games_per_pair, probability)
            else:
                won = probability * games_per_pair
            wins[i, j] = won
            wins[j, i] = games_per_pair - won
    return wins


class TestFit:
    def test_recovers_known_ratings_from_exact_counts(self):
        """With counts set to their expected values the MLE is the truth."""
        truth = np.array([0.0, 150.0, 400.0, 620.0])
        fit = fit_bradley_terry(_synthetic_wins(truth, 2_000), anchor=0, prior_games=0.0)
        assert fit.converged
        assert np.allclose(fit.ratings, truth, atol=1.0)

    def test_anchor_is_pinned_at_zero(self):
        truth = np.array([0.0, 300.0, 700.0])
        fit = fit_bradley_terry(_synthetic_wins(truth, 500), anchor=0)
        assert fit.ratings[0] == pytest.approx(0.0, abs=1e-9)

    def test_ratings_are_identified_only_up_to_the_anchor(self):
        """Shifting the anchor shifts every rating by a constant, preserving gaps."""
        truth = np.array([0.0, 300.0, 700.0])
        wins = _synthetic_wins(truth, 500)
        first = fit_bradley_terry(wins, anchor=0, prior_games=0.0).ratings
        second = fit_bradley_terry(wins, anchor=2, prior_games=0.0).ratings
        assert np.allclose(first - first[2], second, atol=1e-6)

    def test_undefeated_player_stays_finite_under_the_prior(self):
        """A checkpoint that never loses has an infinite MLE; the prior bounds it
        so one saturated matchup cannot take the whole fit with it."""
        wins = np.array(
            [[0.0, 0.0, 0.0], [40.0, 0.0, 20.0], [40.0, 20.0, 0.0]]
        )  # player 0 never wins
        fit = fit_bradley_terry(wins, anchor=0, prior_games=1.0)
        assert np.isfinite(fit.ratings).all()
        assert fit.ratings[1] > 0 and fit.ratings[2] > 0

    def test_prior_shrinkage_is_negligible_against_real_volume(self):
        truth = np.array([0.0, 250.0, 500.0])
        exact = _synthetic_wins(truth, 4_000)
        with_prior = fit_bradley_terry(exact, anchor=0, prior_games=1.0).ratings
        without = fit_bradley_terry(exact, anchor=0, prior_games=0.0).ratings
        assert np.abs(with_prior - without).max() < 1.0

    def test_noisy_counts_land_within_a_few_standard_errors(self):
        truth = np.array([0.0, 200.0, 400.0, 600.0, 800.0])
        fit = fit_bradley_terry(_synthetic_wins(truth, 3_000, noisy=True), anchor=0)
        error = np.abs(fit.ratings - truth)
        assert (error[1:] < 4.0 * fit.stderr[1:]).all(), (error, fit.stderr)


class TestStandardError:
    def test_error_shrinks_as_the_square_root_of_games(self):
        truth = np.array([0.0, 100.0, 200.0])
        few = fit_bradley_terry(_synthetic_wins(truth, 250), anchor=0)
        many = fit_bradley_terry(_synthetic_wins(truth, 4_000), anchor=0)
        ratio = few.stderr[1] / many.stderr[1]
        assert ratio == pytest.approx(4.0, rel=0.15)  # 16x games -> 4x tighter

    def test_saturated_matchups_carry_almost_no_information(self):
        """Games nobody can lose are nearly worthless: the same count of games
        against a hopeless opponent leaves the rating far less determined."""
        even = np.array([0.0, 0.0])
        lopsided = np.array([0.0, 1200.0])
        even_error = rating_stderr(np.full((2, 2), 500.0), even, anchor=0)[1]
        lopsided_error = rating_stderr(np.full((2, 2), 500.0), lopsided, anchor=0)[1]
        assert lopsided_error > 10.0 * even_error

    def test_chain_error_accumulates_along_its_length(self):
        """A rating measured only through intermediaries is less certain than one
        measured against the anchor directly — this is the drift A-optimal
        allocation exists to control."""
        pair_games = np.zeros((4, 4))
        for i in range(3):  # 0 - 1 - 2 - 3, links only between neighbours
            pair_games[i, i + 1] = pair_games[i + 1, i] = 400.0
        ratings = np.zeros(4)
        stderr = rating_stderr(pair_games, ratings, anchor=0)
        assert stderr[1] < stderr[2] < stderr[3]


class TestAllocation:
    def test_prefers_the_pair_that_reduces_variance_most(self):
        """The far end of a chain is the least determined, so the next games
        should go to the link that pins it down."""
        pair_games = np.zeros((3, 3))
        pair_games[0, 1] = pair_games[1, 0] = 1_000.0
        pair_games[1, 2] = pair_games[2, 1] = 20.0
        value = allocation_value(pair_games, np.zeros(3), anchor=0)
        assert value[1, 2] > value[0, 1]

    def test_prefers_closer_matchups_at_equal_connectivity(self):
        """Among opponents a player is equally well connected to, the nearer in
        rating is worth more — that is where the score variance lives."""
        pair_games = np.full((4, 4), 500.0)
        np.fill_diagonal(pair_games, 0.0)
        value = allocation_value(pair_games, np.array([0.0, 100.0, 200.0, 300.0]), anchor=0)
        assert value[0, 1] > value[0, 2] > value[0, 3]

    def test_budget_flows_to_a_poorly_pinned_outlier(self):
        """A player far above the field is barely determined however many games
        it has played, because every one of them was a foregone conclusion. The
        allocator spends on it, and picks its least hopeless opponent to do it —
        this is the chain-drift control that A-optimality buys."""
        pair_games = np.full((4, 4), 500.0)
        np.fill_diagonal(pair_games, 0.0)
        ratings = np.array([0.0, 100.0, 200.0, 1_500.0])
        value = allocation_value(pair_games, ratings, anchor=0)
        outlier_pairs = [value[0, 3], value[1, 3], value[2, 3]]
        assert min(outlier_pairs) > max(value[0, 1], value[0, 2], value[1, 2])
        assert value[2, 3] > value[1, 3] > value[0, 3]

    def test_allocation_spends_the_whole_budget(self):
        pair_games = np.full((4, 4), 100.0)
        np.fill_diagonal(pair_games, 0.0)
        allocation = allocate_games(pair_games, np.zeros(4), anchor=0, budget=1_000)
        assert allocation.sum() // 2 == 1_000
        assert (allocation == allocation.T).all()

    def test_allocation_is_even_when_no_games_exist_yet(self):
        allocation = allocate_games(np.zeros((3, 3)), np.zeros(3), anchor=0, budget=300)
        upper = np.triu(allocation, k=1)
        assert upper[upper > 0].std() <= 1.0


class TestSingleRating:
    def test_recovers_a_known_rating_against_fixed_opponents(self):
        opponents = np.array([0.0, 300.0, 600.0])
        truth = 400.0
        probability = win_probability(truth, opponents)
        games = 20_000
        rating, stderr = fit_single_rating(
            opponents, probability * games, (1 - probability) * games
        )
        assert rating == pytest.approx(truth, abs=2.0)
        assert stderr < 5.0

    def test_matches_the_closed_form_for_a_single_opponent(self):
        """Against one opponent the MLE is just the log-odds of the win rate."""
        rating, _ = fit_single_rating(np.array([100.0]), np.array([750.0]), np.array([250.0]))
        expected = 100.0 + RATING_SCALE * np.log10(750.0 / 250.0)
        assert rating == pytest.approx(expected, abs=1e-6)

    def test_sweeps_report_infinite_rating_and_error(self):
        opponents = np.array([0.0, 100.0])
        assert fit_single_rating(opponents, np.array([10.0, 10.0]), np.zeros(2))[0] == float("inf")
        assert fit_single_rating(opponents, np.zeros(2), np.array([10.0, 10.0]))[0] == -float("inf")

    def test_lopsided_records_stay_bounded(self):
        """A near-sweep implies a high rating, not an astronomical one. Newton
        diverges here — curvature underflows while the gradient stays finite —
        and produced ~1e144 on real early-training records against random."""
        rating, _ = fit_single_rating(np.array([0.0]), np.array([4_999.0]), np.array([1.0]))
        assert np.isfinite(rating)
        assert 1_000.0 < rating < 4_000.0

    def test_a_record_beyond_the_bracket_is_unbounded(self):
        rating, stderr = fit_single_rating(
            np.array([0.0]), np.array([1e9]), np.array([1.0]), bracket=200.0
        )
        assert rating == float("inf")
        assert stderr == float("inf")

    def test_no_games_is_not_a_rating(self):
        rating, stderr = fit_single_rating(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert np.isnan(rating)
        assert stderr == float("inf")

    def test_agrees_with_the_joint_fit_when_opponents_are_known(self):
        truth = np.array([0.0, 200.0, 500.0, 900.0])
        wins = _synthetic_wins(truth, 5_000)
        joint = fit_bradley_terry(wins, anchor=0, prior_games=0.0)
        # Re-fit player 3 alone against the other three held at their fitted values.
        others = [0, 1, 2]
        rating, _ = fit_single_rating(joint.ratings[others], wins[3, others], wins[others, 3])
        assert rating == pytest.approx(joint.ratings[3], abs=1.0)
