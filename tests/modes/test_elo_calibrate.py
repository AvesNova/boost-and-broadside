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

    def test_a_clean_sweep_is_reported_as_unbounded(self):
        curve = calibrate_live_curve([_record(1, 10, {"random": [100, 0, 0]})], {"random": 0.0})
        assert not np.isfinite(curve[0]["live_calibrated"])
        assert not np.isfinite(curve[0]["live_stderr"])
