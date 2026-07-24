"""Tests for the post-training calibration mode's pure logic."""

import numpy as np
import pytest

from boost_and_broadside.modes.elo_calibrate import calibrate_live_curve
from boost_and_broadside.train.rl.bradley_terry import win_probability


def _record(update: int, step: int, counts: dict, live: float = 0.0, avg: float = 0.0) -> dict:
    return {"update": update, "global_step": step, "live": live, "avg": avg, "counts": counts}


class TestCalibrateLiveCurve:
    def test_recovers_the_rating_that_explains_the_record(self):
        """The live policy's rating comes from its own results, not from whatever
        the in-training filter had drifted to."""
        opponents = {"random": 0.0, "ckpt_1": 500.0}
        games = 4_000
        truth = 300.0
        counts = {}
        for label, rating in opponents.items():
            probability = win_probability(truth, rating)
            counts[label] = [probability * games, (1 - probability) * games, 0]
        curve = calibrate_live_curve([_record(1, 100, counts, live=-999.0)], opponents)
        assert curve[0]["live_calibrated"] == pytest.approx(truth, abs=5.0)
        assert curve[0]["live_training"] == -999.0  # untouched, for comparison

    def test_ignores_opponents_with_no_calibrated_rating(self):
        counts = {"random": [90, 10, 0], "mystery_agent": [50, 50, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        assert curve[0]["games"] == 100

    def test_avg_is_excluded_from_the_live_fit(self):
        """The averaged policy moves every update, so it cannot serve as a fixed
        reference for the live policy that update."""
        counts = {"random": [90, 10, 0], "avg": [500, 500, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0, "avg": 700.0})
        assert curve[0]["games"] == 100, "avg games must not enter the live fit"

    def test_avg_is_rated_against_the_calibrated_live_policy(self):
        """Second stage: avg's only opponent is live, so it is placed once live is."""
        counts = {"random": [990, 10, 0], "avg": [500, 500, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        # An even record against live puts avg at live's rating.
        assert curve[0]["avg_calibrated"] == pytest.approx(curve[0]["live_calibrated"], abs=1.0)

    def test_avg_record_is_inverted_from_the_live_perspective(self):
        """Counts are stored as live's wins/losses; avg beating live must raise
        avg above live, not sink it."""
        counts = {"random": [990, 10, 0], "avg": [100, 900, 0]}  # live loses to avg
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        assert curve[0]["avg_calibrated"] > curve[0]["live_calibrated"]

    def test_updates_without_usable_counts_are_dropped(self):
        history = [_record(1, 10, {}), _record(2, 20, {"random": [5, 5, 0]})]
        curve = calibrate_live_curve(history, {"random": 0.0})
        assert [point["update"] for point in curve] == [2]

    def test_ties_do_not_enter_the_fit(self):
        """Draws are excluded from the likelihood, so adding them to a record
        must not move the rating it implies."""
        ratings = {"random": 0.0}
        without = calibrate_live_curve([_record(1, 10, {"random": [800, 200, 0]})], ratings)
        with_ties = calibrate_live_curve([_record(1, 10, {"random": [800, 200, 500]})], ratings)
        assert without[0]["live_calibrated"] == pytest.approx(with_ties[0]["live_calibrated"])

    def test_drawing_every_game_matches_the_opponents_rating(self):
        """Drawing every game against a known opponent is exactly the evidence
        that you are that opponent's equal, and half-win scoring says so. This is
        the case decisive-only cannot answer at all: with no decisive games it
        has nothing to fit, and reports the rating as undefined."""
        opponent = {"stockfish": 3_600.0}
        counts = {"stockfish": [0, 0, 1_000]}
        half = calibrate_live_curve([_record(1, 10, counts)], opponent, "half_win")
        assert half[0]["live_calibrated"] == pytest.approx(3_600.0, abs=1.0)

        decisive = calibrate_live_curve([_record(1, 10, counts)], opponent, "decisive")
        assert np.isnan(decisive[0]["live_calibrated"])

    def test_half_win_extracts_more_information_from_a_draw_heavy_record(self):
        """Draws carry information about parity. Dropping them throws it away,
        which is most costly exactly where draws are common."""
        ratings = {"random": 0.0}
        record = _record(1, 10, {"random": [2794, 10, 1120]})
        half = calibrate_live_curve([record], ratings, "half_win")[0]
        decisive = calibrate_live_curve([record], ratings, "decisive")[0]
        assert half["live_stderr"] < decisive["live_stderr"] / 4.0

    def test_half_win_mode_counts_draws_as_half(self):
        """Under the half-win convention a drawn game is half a win and half a
        loss, so a record of pure draws sits level with its opponent."""
        curve = calibrate_live_curve(
            [_record(1, 10, {"random": [0, 0, 400]})], {"random": 0.0}, tie_mode="half_win"
        )
        assert curve[0]["live_calibrated"] == pytest.approx(0.0, abs=1.0)

    def test_half_win_and_decisive_disagree_exactly_where_draws_are(self):
        """A tie-free record is identical under both conventions; adding draws
        pulls the half-win rating toward the opponent and leaves the other put."""
        ratings = {"random": 0.0}
        clean = {"random": [900, 100, 0]}
        drawn = {"random": [900, 100, 1000]}
        decisive_clean = calibrate_live_curve([_record(1, 10, clean)], ratings)[0]
        decisive_drawn = calibrate_live_curve([_record(1, 10, drawn)], ratings)[0]
        half_clean = calibrate_live_curve([_record(1, 10, clean)], ratings, "half_win")[0]
        half_drawn = calibrate_live_curve([_record(1, 10, drawn)], ratings, "half_win")[0]

        assert decisive_clean["live_calibrated"] == pytest.approx(decisive_drawn["live_calibrated"])
        assert half_clean["live_calibrated"] == pytest.approx(decisive_clean["live_calibrated"])
        assert 0.0 < half_drawn["live_calibrated"] < half_clean["live_calibrated"]

    def test_a_clean_sweep_is_reported_as_unbounded(self):
        curve = calibrate_live_curve([_record(1, 10, {"random": [100, 0, 0]})], {"random": 0.0})
        assert not np.isfinite(curve[0]["live_calibrated"])
        assert not np.isfinite(curve[0]["live_stderr"])
