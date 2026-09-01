"""Tests for the read-only live-Elo instrumentation (live-elo-plan Phase 0)."""

import numpy as np
import pytest

from boost_and_broadside.train.rl.bradley_terry import (
    fisher_information,
    fit_bradley_terry,
)
from boost_and_broadside.train.rl.elo_diagnostics import (
    DEFAULT_WINDOW_UPDATES,
    LiveEloDiagnostics,
    effective_resistance,
    fiedler_value,
    pair_information,
    potential,
)

# The gauge's defined references, as config/live_elo pins them.
GAUGE = {"random": 0.0, "semi_scripted:0.5": 500.0, "scripted": 1000.0}
STATIONARY = dict.fromkeys(GAUGE, True)


def two_node(games: float, gap: float) -> np.ndarray:
    """Laplacian for a single edge between players separated by ``gap``."""
    return fisher_information(
        np.array([[0.0, games], [games, 0.0]]), np.array([0.0, -gap])
    )


class TestEffectiveResistance:
    """Var(r_i − r_j) read off the graph, against closed forms."""

    def test_a_single_edge_has_resistance_one_over_its_weight(self):
        laplacian = two_node(games=400.0, gap=0.0)
        # Evenly matched, so p(1−p) = 1/4 and the weight is games·c²/4.
        weight = 400.0 * (np.log(10.0) / 400.0) ** 2 * 0.25
        assert effective_resistance(laplacian, 0, 1) == pytest.approx(1.0 / weight)

    def test_a_two_hop_path_adds_its_resistances_in_series(self):
        # 0 — 1 — 2, all evenly matched. This is why a rating carried up a
        # chain is known less precisely than one measured directly.
        games = np.zeros((3, 3))
        games[0, 1] = games[1, 0] = 400.0
        games[1, 2] = games[2, 1] = 400.0
        laplacian = fisher_information(games, np.zeros(3))
        hop = effective_resistance(laplacian, 0, 1)
        assert effective_resistance(laplacian, 0, 2) == pytest.approx(2.0 * hop)

    def test_a_saturated_edge_is_nearly_useless(self):
        """The mechanism the whole plan is about: p → 1 kills information."""
        even = effective_resistance(two_node(games=400.0, gap=0.0), 0, 1)
        saturated = effective_resistance(two_node(games=400.0, gap=800.0), 0, 1)
        assert saturated > 20.0 * even

    def test_disconnected_players_have_no_identified_difference(self):
        games = np.zeros((3, 3))
        games[0, 1] = games[1, 0] = 400.0  # 2 plays nobody
        laplacian = fisher_information(games, np.zeros(3))
        assert effective_resistance(laplacian, 0, 2) == float("inf")
        assert potential(laplacian, 0, 2) is None

    def test_it_reproduces_the_fitters_own_standard_errors(self):
        """The claim the design rests on, checked against the existing fitter.

        ``rating_stderr`` inverts the reduced information matrix directly; the
        resistance route never forms that inverse. Agreeing to eight digits
        means the graph picture is the same estimator, not an approximation.
        """
        wins = np.array(
            [
                [0.0, 300.0, 420.0],
                [200.0, 0.0, 380.0],
                [80.0, 120.0, 0.0],
            ]
        )
        fit = fit_bradley_terry(wins, anchor=0, prior_games=0.0)
        laplacian = fisher_information(wins + wins.T, fit.ratings)
        for player in (1, 2):
            resistance = effective_resistance(laplacian, player, 0)
            assert np.sqrt(resistance) == pytest.approx(fit.stderr[player], rel=1e-8)


class TestFiedlerValue:
    def test_it_is_zero_when_the_pool_splits(self):
        games = np.zeros((4, 4))
        games[0, 1] = games[1, 0] = 400.0
        games[2, 3] = games[3, 2] = 400.0
        laplacian = fisher_information(games, np.zeros(4))
        assert fiedler_value(laplacian) == pytest.approx(0.0, abs=1e-12)

    def test_it_is_positive_when_every_player_is_reachable(self):
        games = np.full((4, 4), 400.0)
        np.fill_diagonal(games, 0.0)
        assert fiedler_value(fisher_information(games, np.zeros(4))) > 0.0


class TestPairInformation:
    def test_it_peaks_at_an_even_matchup(self):
        even = pair_information(1000.0, 1000.0, games=100.0)
        lopsided = pair_information(1000.0, 400.0, games=100.0)
        assert even > lopsided > 0.0

    def test_draws_count_as_evidence(self):
        """Games are total episodes; the caller must not pre-filter ties."""
        assert pair_information(1000.0, 1000.0, games=200.0) == pytest.approx(
            2.0 * pair_information(1000.0, 1000.0, games=100.0)
        )


class TestDriftDetector:
    """The instrument that would have caught run 727 at its resume seam."""

    def test_a_consistent_rating_shows_no_drift(self):
        diagnostics = LiveEloDiagnostics()
        # 500 vs scripted-at-1000 means winning about 5% of the time.
        metrics = diagnostics.update(
            live_elo=500.0,
            match_counts={"scripted": (50, 950, 0)},
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert metrics["elo_diag/drift_vs_scripted"] == pytest.approx(0.0, abs=15.0)

    def test_an_inflated_rating_shows_positive_drift(self):
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=800.0,  # the filter's claim
            match_counts={"scripted": (50, 950, 0)},  # the record's story: ~500
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert metrics["elo_diag/drift_vs_scripted"] == pytest.approx(300.0, abs=20.0)

    def test_the_gauge_fit_pools_every_defined_reference(self):
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=500.0,
            match_counts={
                "random": (300, 0, 0),
                "semi_scripted:0.5": (100, 100, 0),
                "scripted": (5, 95, 0),
            },
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        # Pooling more opponents cannot be less certain than one of them alone.
        assert metrics["elo_diag/implied_gauge_stderr"] < (
            metrics["elo_diag/implied_scripted_stderr"]
        )

    def test_self_generated_opponents_never_enter_the_gauge_fit(self):
        """A rung the run rated itself cannot testify about the run's rating."""
        ratings = {**GAUGE, "ckpt_1024000": 1400.0}
        stationary = {**STATIONARY, "ckpt_1024000": False}
        diagnostics = LiveEloDiagnostics()
        with_checkpoint = diagnostics.update(
            live_elo=500.0,
            match_counts={"scripted": (50, 950, 0), "ckpt_1024000": (500, 500, 0)},
            ratings=ratings,
            stationary=stationary,
        )
        diagnostics = LiveEloDiagnostics()
        without = diagnostics.update(
            live_elo=500.0,
            match_counts={"scripted": (50, 950, 0)},
            ratings=ratings,
            stationary=stationary,
        )
        assert with_checkpoint["elo_diag/implied_gauge_elo"] == pytest.approx(
            without["elo_diag/implied_gauge_elo"]
        )

    def test_a_sweep_is_dropped_rather_than_logged_as_a_spike(self):
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=2000.0,
            match_counts={"scripted": (400, 0, 0)},
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert "elo_diag/drift_vs_scripted" not in metrics
        assert all(np.isfinite(value) for value in metrics.values())

    def test_draws_are_scored_as_half_a_win(self):
        """Dropping them would throw away most tie-heavy matchups' evidence."""
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=1000.0,
            match_counts={"scripted": (0, 0, 400)},  # every game a draw
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        # All draws against scripted reads as an even matchup, not as no data.
        assert metrics["elo_diag/implied_scripted_elo"] == pytest.approx(1000.0, abs=1.0)


class TestMovement:
    def test_the_first_update_reports_no_movement(self):
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=500.0,
            match_counts={"scripted": (50, 950, 0)},
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert "elo_diag/movement_z" not in metrics

    def test_a_still_rating_scores_zero(self):
        diagnostics = LiveEloDiagnostics()
        counts = {"scripted": (50, 950, 0)}
        diagnostics.update(
            live_elo=500.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        metrics = diagnostics.update(
            live_elo=500.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        assert metrics["elo_diag/movement_z"] == pytest.approx(0.0)

    def test_a_jump_beyond_the_updates_evidence_scores_high(self):
        """The 727 seam signature: an 85-point step a full update cannot buy."""
        diagnostics = LiveEloDiagnostics()
        counts = {"scripted": (50, 950, 0)}
        diagnostics.update(
            live_elo=500.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        metrics = diagnostics.update(
            live_elo=585.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        assert metrics["elo_diag/live_elo_delta"] == pytest.approx(85.0)
        assert metrics["elo_diag/movement_z"] > 3.0

    def test_the_same_jump_is_unremarkable_on_ten_games(self):
        """Normalising by evidence, not by size: z answers 'could games do this'.

        Against a saturated opponent ten episodes locate the rating only to a
        few hundred points, so an 85-point step is genuinely within noise. The
        alarm has to be scale-free or it fires on every early update.
        """
        diagnostics = LiveEloDiagnostics()
        counts = {"scripted": (0, 10, 0)}
        diagnostics.update(
            live_elo=500.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        metrics = diagnostics.update(
            live_elo=585.0, match_counts=counts, ratings=GAUGE, stationary=STATIONARY
        )
        assert metrics["elo_diag/movement_z"] < 1.0


class TestWindow:
    def test_counts_pool_across_updates(self):
        diagnostics = LiveEloDiagnostics(window_updates=3)
        for _ in range(3):
            metrics = diagnostics.update(
                live_elo=500.0,
                match_counts={"scripted": (5, 95, 0)},
                ratings=GAUGE,
                stationary=STATIONARY,
            )
        assert metrics["elo_diag/window_games"] == pytest.approx(300.0)

    def test_the_window_forgets_beyond_its_length(self):
        diagnostics = LiveEloDiagnostics(window_updates=2)
        for _ in range(5):
            metrics = diagnostics.update(
                live_elo=500.0,
                match_counts={"scripted": (5, 95, 0)},
                ratings=GAUGE,
                stationary=STATIONARY,
            )
        assert metrics["elo_diag/window_games"] == pytest.approx(200.0)

    def test_opponents_missing_from_the_roster_are_ignored(self):
        """A promotion can retire a label while the window still holds it."""
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=500.0,
            match_counts={"scripted": (5, 95, 0), "ckpt_retired": (10, 10, 0)},
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert metrics["elo_diag/window_games"] == pytest.approx(100.0)

    def test_a_gameless_update_is_survivable(self):
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=500.0, match_counts={}, ratings=GAUGE, stationary=STATIONARY
        )
        assert metrics["elo_diag/window_games"] == pytest.approx(0.0)
        assert "elo_diag/drift_vs_scripted" not in metrics

    def test_the_default_window_is_short_enough_to_track_a_moving_policy(self):
        assert 1 < DEFAULT_WINDOW_UPDATES <= 16


class TestPoolGraph:
    def test_the_reported_error_grows_as_the_policy_outgrows_the_anchor(self):
        """Names the ceiling: a floor anchor gets costlier to measure against."""
        errors = []
        for live_elo in (1000.0, 1400.0, 1800.0):
            diagnostics = LiveEloDiagnostics()
            metrics = diagnostics.update(
                live_elo=live_elo,
                match_counts={"scripted": (500, 500, 0)},
                ratings=GAUGE,
                stationary=STATIONARY,
            )
            errors.append(metrics["elo_diag/se_live_vs_scripted"])
        assert errors[0] < errors[1] < errors[2]

    def test_the_graph_is_a_star_on_the_live_policy_today(self):
        """Stated so Phase 1 has something to break when it adds real edges."""
        diagnostics = LiveEloDiagnostics()
        metrics = diagnostics.update(
            live_elo=1000.0,
            match_counts={"scripted": (500, 500, 0), "random": (500, 0, 0)},
            ratings=GAUGE,
            stationary=STATIONARY,
        )
        assert metrics["elo_diag/pool_size"] == pytest.approx(3.0)
        # With no rung-vs-rung games the only path to scripted is the direct
        # edge, so the resistance is exactly that edge's reciprocal weight.
        direct = 1.0 / pair_information(1000.0, 1000.0, games=1000.0)
        assert metrics["elo_diag/se_live_vs_scripted"] == pytest.approx(np.sqrt(direct))
