"""Tests for the two-stage live rating (live-elo-plan Phase 2)."""

import numpy as np
import pytest

from boost_and_broadside.train.rl.live_rating import (
    TwoStageRating,
    fit_ladder,
    rate_live,
)
from boost_and_broadside.train.rl.match_matrix import MatchMatrix

SCRIPTED = 1000.0
# The gauge's defined players: not estimates, and never refit by stage 1.
GAUGE = {"random": 0.0, "semi_scripted_0p5": 500.0, "scripted": SCRIPTED}


def ladder_matrix() -> MatchMatrix:
    """A three-rung chain: random below scripted, two checkpoints above."""
    matrix = MatchMatrix()
    matrix.record("scripted", "random", 900, 20, 80)
    matrix.record("ckpt_1", "random", 950, 10, 40)
    matrix.record("ckpt_1", "scripted", 600, 380, 20)
    matrix.record("ckpt_2", "ckpt_1", 620, 360, 20)
    matrix.record("ckpt_2", "scripted", 700, 290, 10)
    return matrix


class TestFitLadder:
    def test_the_anchor_reads_its_defined_value(self):
        """The gauge has to hold still across promotions, or nothing compares."""
        ladder = fit_ladder(ladder_matrix(), fixed=GAUGE)
        assert ladder["scripted"] == pytest.approx(SCRIPTED)

    def test_the_order_follows_the_record(self):
        ladder = fit_ladder(ladder_matrix(), fixed=GAUGE)
        assert ladder["ckpt_2"] > ladder["ckpt_1"] > ladder["scripted"] > ladder["random"]

    def test_the_anchor_holds_still_when_the_pool_grows(self):
        """A promotion must not move every existing rating."""
        matrix = ladder_matrix()
        before = fit_ladder(matrix, fixed=GAUGE)
        matrix.record("ckpt_3", "ckpt_2", 700, 280, 20)
        after = fit_ladder(matrix, fixed=GAUGE)
        assert after["scripted"] == pytest.approx(before["scripted"])
        assert after["ckpt_1"] == pytest.approx(before["ckpt_1"], abs=1.0)

    def test_a_sweep_stays_bounded(self):
        """Complete separation is routine here and must not run away.

        A late rung beats the random agent every single time. Without the
        virtual prior games that record has an infinite maximum likelihood, and
        an iteration cap decides the answer instead of the data.
        """
        matrix = MatchMatrix()
        matrix.record("scripted", "random", 1000, 0, 0)
        matrix.record("ckpt_1", "random", 1000, 0, 0)
        matrix.record("ckpt_1", "scripted", 600, 400, 0)
        ladder = fit_ladder(matrix, fixed=GAUGE)
        assert all(np.isfinite(value) for value in ladder.values())
        assert abs(ladder["random"]) < 10_000.0

    def test_a_moving_player_never_enters_the_fit(self):
        matrix = ladder_matrix()
        matrix.record("avg", "scripted", 500, 500, 0)
        assert "avg" not in fit_ladder(matrix, fixed=GAUGE)

    def test_an_ungauged_matrix_yields_nothing(self):
        """Without the anchor there is no scale, so the caller keeps what it has."""
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "ckpt_2", 5, 5, 0)
        assert fit_ladder(matrix, fixed=GAUGE) == {}

    def test_an_empty_matrix_yields_nothing(self):
        assert fit_ladder(MatchMatrix(), fixed=GAUGE) == {}


    def test_the_defined_players_are_never_moved(self):
        """They are the scale. Refitting them is refitting the ruler."""
        ladder = fit_ladder(ladder_matrix(), fixed=GAUGE)
        for label, elo in GAUGE.items():
            if label in ladder:
                assert ladder[label] == pytest.approx(elo)

    def test_sweeps_cannot_rank_a_reference_by_its_sample_size(self):
        """The failure this signature exists to prevent, observed on run 728.

        A checkpoint swept both the random agent (15-0) and the 0.2 rung (27-0).
        Under a joint fit both opponents separate completely, so only the prior
        bounds them -- and its shrinkage weakens with sample size, which put
        random a hundred points *above* a rung it is plainly worse than.
        """
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "random", 15, 0, 0)
        matrix.record("ckpt_1", "semi_scripted_0p2", 27, 0, 0)
        matrix.record("ckpt_1", "scripted", 2, 7, 0)
        gauge = {"random": 0.0, "semi_scripted_0p2": 200.0, "scripted": SCRIPTED}
        ladder = fit_ladder(matrix, fixed=gauge)
        assert ladder["random"] < ladder["semi_scripted_0p2"] < ladder["scripted"]

    def test_a_swept_checkpoint_still_gets_a_finite_rating(self):
        """A number is needed every update, so the prior has to bind here."""
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "random", 200, 0, 0)
        matrix.record("ckpt_1", "semi_scripted_0p5", 200, 0, 0)
        ladder = fit_ladder(matrix, fixed=GAUGE)
        assert np.isfinite(ladder["ckpt_1"])
        assert ladder["ckpt_1"] > GAUGE["semi_scripted_0p5"]

    def test_a_checkpoint_is_placed_by_the_references_it_splits(self):
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "random", 300, 0, 0)
        matrix.record("ckpt_1", "semi_scripted_0p5", 150, 150, 0)
        matrix.record("ckpt_1", "scripted", 20, 280, 0)
        assert fit_ladder(matrix, fixed=GAUGE)["ckpt_1"] == pytest.approx(500.0, abs=40.0)

    def test_checkpoints_are_rated_against_each_other_too(self):
        """The chain: a rung above every reference is placed by the rung below."""
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "scripted", 700, 300, 0)
        matrix.record("ckpt_2", "ckpt_1", 700, 300, 0)
        ladder = fit_ladder(matrix, fixed={"scripted": SCRIPTED})
        assert ladder["ckpt_2"] > ladder["ckpt_1"] > SCRIPTED

    def test_it_converges_to_the_same_answer_from_a_different_start(self):
        """Coordinate ascent on a concave likelihood has one fixed point."""
        matrix = ladder_matrix()
        assert fit_ladder(matrix, fixed=GAUGE)["ckpt_2"] == pytest.approx(
            fit_ladder(matrix, fixed=GAUGE, passes=512)["ckpt_2"], abs=0.5
        )

class TestRateLive:
    def test_it_recovers_a_known_rating(self):
        ratings = {"scripted": 1000.0, "ckpt_1": 1400.0}
        # Even against ckpt_1 means the live policy is worth about 1400.
        rating, stderr = rate_live({"ckpt_1": (500, 500, 0)}, ratings)
        assert rating == pytest.approx(1400.0, abs=15.0)
        assert 0.0 < stderr < 50.0

    def test_pooling_opponents_beats_any_one_of_them(self):
        ratings = {"scripted": 1000.0, "ckpt_1": 1400.0, "random": 0.0}
        _, one = rate_live({"ckpt_1": (250, 250, 0)}, ratings)
        _, many = rate_live(
            {"ckpt_1": (250, 250, 0), "scripted": (240, 10, 0), "random": (250, 0, 0)},
            ratings,
        )
        assert many < one

    def test_a_sweep_is_reported_as_unbounded(self):
        """A large finite number here would be mistaken for a measurement."""
        rating, stderr = rate_live({"scripted": (400, 0, 0)}, {"scripted": 1000.0})
        assert rating == float("inf")
        assert stderr == float("inf")

    def test_an_empty_record_is_not_a_rating(self):
        rating, _ = rate_live({}, {"scripted": 1000.0})
        assert np.isnan(rating)

    def test_unrated_opponents_are_skipped(self):
        rating, _ = rate_live(
            {"ckpt_1": (500, 500, 0), "mystery": (900, 100, 0)}, {"ckpt_1": 1400.0}
        )
        assert rating == pytest.approx(1400.0, abs=15.0)

    def test_the_moving_average_is_excluded(self):
        ratings = {"ckpt_1": 1400.0, "avg": 1400.0}
        with_avg = rate_live({"ckpt_1": (500, 500, 0), "avg": (999, 1, 0)}, ratings)
        without = rate_live({"ckpt_1": (500, 500, 0)}, ratings)
        assert with_avg[0] == pytest.approx(without[0])


class TestTwoStageRating:
    def build(self) -> TwoStageRating:
        return TwoStageRating(anchor_label="scripted", anchor_elo=SCRIPTED)

    @staticmethod
    def call(stage, matrix, counts, fallback, gauge=None):
        return stage.update(matrix, counts, fallback, GAUGE if gauge is None else gauge)

    def test_it_reports_both_stages(self):
        stage = self.build()
        metrics = self.call(stage, ladder_matrix(), {"ckpt_2": (500, 500, 0)}, {"ckpt_2": 0.0})
        assert metrics["two_stage/ladder_players"] == pytest.approx(4.0)
        assert metrics["two_stage/live_elo"] == pytest.approx(
            stage.ladder["ckpt_2"], abs=20.0
        )

    def test_the_refit_ladder_overrides_the_fallback_ratings(self):
        """Stage 1's whole point: stop rating the live policy off filter output."""
        stage = self.build()
        metrics = self.call(
            stage, ladder_matrix(), {"ckpt_2": (500, 500, 0)}, {"ckpt_2": 9999.0}
        )
        assert metrics["two_stage/live_elo"] < 5000.0

    def test_the_fallback_covers_opponents_the_matrix_has_never_seen(self):
        stage = self.build()
        metrics = self.call(
            stage, MatchMatrix(), {"semi_scripted_0p5": (500, 500, 0)},
            {"semi_scripted_0p5": 500.0},
        )
        assert metrics["two_stage/live_elo"] == pytest.approx(500.0, abs=20.0)

    def test_a_saturated_update_holds_the_last_value_rather_than_spiking(self):
        stage = self.build()
        first = self.call(stage, ladder_matrix(), {"ckpt_2": (500, 500, 0)}, {"ckpt_2": 0.0})
        second = self.call(stage, ladder_matrix(), {"ckpt_2": (500, 0, 0)}, {"ckpt_2": 0.0})
        assert second["two_stage/live_elo"] == pytest.approx(first["two_stage/live_elo"])
        assert "two_stage/live_stderr" not in second

    def test_it_carries_no_state_of_its_own_across_a_solvable_update(self):
        """The seam property: a stateless solve cannot hold a shift.

        Two instances fed the same update report the same rating regardless of
        the history behind them, which is what a K-factor filter cannot do.
        """
        fresh = self.build()
        used = self.build()
        for _ in range(5):
            self.call(used, ladder_matrix(), {"scripted": (900, 100, 0)}, {})
        counts = {"ckpt_2": (500, 500, 0)}
        assert self.call(fresh, ladder_matrix(), counts, {})["two_stage/live_elo"] == (
            pytest.approx(self.call(used, ladder_matrix(), counts, {})["two_stage/live_elo"])
        )
