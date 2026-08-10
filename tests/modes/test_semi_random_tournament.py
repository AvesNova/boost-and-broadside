"""Tests for semi-random ladder configuration."""

import pytest

from boost_and_broadside.modes.semi_random_tournament import (
    _greatest_divisor_at_most,
    _live_gauge_error,
    parse_probabilities,
)


def test_probability_ladder_is_sorted_and_deduplicated() -> None:
    assert parse_probabilities("1,0.2,0,0.2") == [0.0, 0.2, 1.0]


@pytest.mark.parametrize(
    "text",
    ["0,0.5", "0.5,1", "0,-0.1,1", "0,1.1,1", "0,nan,1", "0,inf,1"],
)
def test_probability_ladder_requires_valid_endpoints(text: str) -> None:
    with pytest.raises(ValueError):
        parse_probabilities(text)


def test_batch_chunk_divides_target_without_exceeding_capacity() -> None:
    assert _greatest_divisor_at_most(256, 11) == 8
    assert _greatest_divisor_at_most(256, 200) == 128


class TestLiveGaugeError:
    """The ladder validates training's derived gauge instead of supplying it."""

    _PROBABILITIES = [0.0, 0.5, 1.0]
    _LABELS = ["random", "semi_scripted_0p5", "scripted"]

    def _views(self, fitted: list[float]) -> dict:
        return {"random_zero_scripted_1000": {"ratings": fitted, "stderr": [0.0, 7.5, 0.0]}}

    def test_the_residual_is_live_minus_fitted(self) -> None:
        rows = _live_gauge_error(
            self._PROBABILITIES, self._LABELS, self._views([0.0, 420.0, 1000.0])
        )

        assert [row["label"] for row in rows] == self._LABELS
        assert [row["live_elo"] for row in rows] == [0.0, 500.0, 1000.0]
        # The gauge rates the rung 80 points above where it actually played.
        assert [row["live_elo_error"] for row in rows] == [0.0, 80.0, 0.0]
        assert rows[1]["fitted_regauged_stderr"] == 7.5

    def test_both_endpoints_are_reported_and_are_zero_by_construction(self) -> None:
        rows = _live_gauge_error(
            self._PROBABILITIES, self._LABELS, self._views([0.0, 500.0, 1000.0])
        )

        assert rows[0]["live_elo_error"] == 0.0
        assert rows[-1]["live_elo_error"] == 0.0
        assert all(row["live_elo_error"] == 0.0 for row in rows)

    def test_a_non_finite_fit_reports_no_residual_rather_than_a_nan(self) -> None:
        rows = _live_gauge_error(
            self._PROBABILITIES, self._LABELS, self._views([0.0, float("inf"), 1000.0])
        )

        assert rows[1]["live_elo_error"] is None
        assert rows[1]["live_elo"] == 500.0
