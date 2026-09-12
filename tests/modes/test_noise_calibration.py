"""Tests for noise_calibration mode's output-building logic."""

import datetime
import warnings

import numpy as np
import pytest

from boost_and_broadside.modes.noise_calibration import (
    _REPORT_FEATURES,
    _build_output,
    _report_layout,
)


def _make_phase1(target_dim: int) -> dict:
    return {
        "err_count": 100.0,
        "err_sq_sum": np.full(target_dim, 0.04),
        "err_sum": np.zeros(target_dim),
        "lag1_sq_sum": np.full(target_dim, 1.0),
        "lag1_cross_sum": np.full(target_dim, 0.5),
        "team_err_sq_sum": np.full((2, target_dim), 0.04),
        "team_count": np.full(2, 50.0),
        "combat_err_sq_sum": np.full((2, target_dim), 0.04),
        "combat_count": np.full(2, 50.0),
    }


def _make_phase2(target_dim: int, ar_window: int) -> dict:
    return {
        "ar_sq_sum": np.full((ar_window, target_dim), 0.04),
        "ar_count": np.full(ar_window, 20.0),
    }


class TestBuildOutputTimestamp:
    """AUDIT-023: datetime.utcnow() is deprecated (Python 3.13+); the fix must
    keep producing an ISO-8601 UTC timestamp that a downstream JSON consumer
    can parse."""

    def test_timestamp_is_a_parseable_iso8601_utc_string(self):
        output = _build_output(
            phase1=_make_phase1(target_dim=1),
            phase2=_make_phase2(target_dim=1, ar_window=20),
            checkpoint_path="dummy.pt",
            num_envs=4,
            num_steps=4,
            num_ar_envs=4,
            num_ar_windows=1,
            feature_groups={"pos": ([0], "position")},
        )

        timestamp = output["metadata"]["timestamp"]
        parsed = datetime.datetime.fromisoformat(timestamp)

        assert parsed.tzinfo is not None, "timestamp must carry explicit UTC offset info"
        assert parsed.utcoffset() == datetime.timedelta(0)


class TestReportLayout:
    """Every target dimension the coordinator predicts must be named.

    The report groups target dimensions by hand. A predictor added to the
    coordinator and not to that table was still measured, but reached the
    published figure as an untitled panel over a dimension nobody could
    identify — so the layout now refuses to build instead.
    """

    def test_the_layout_names_every_dimension_the_coordinator_produces(self):
        from boost_and_broadside.config.defaults import SHIP_CONFIG
        from boost_and_broadside.train.rl.features import build_standard_coordinator

        coordinator = build_standard_coordinator(SHIP_CONFIG)

        groups, dim_names = _report_layout(coordinator)

        assert len(dim_names) == coordinator.total_target_dimension
        assert all(dim_names), dim_names
        grouped = sorted(index for dims, _ in groups.values() for index in dims)
        assert grouped == list(range(coordinator.total_target_dimension))

    def test_a_dimension_the_layout_forgets_is_refused(self, monkeypatch):
        from boost_and_broadside.config.defaults import SHIP_CONFIG
        from boost_and_broadside.train.rl.features import build_standard_coordinator

        coordinator = build_standard_coordinator(SHIP_CONFIG)
        forgetful = {
            name: entry for name, entry in _REPORT_FEATURES.items() if name != "local_log_index"
        }
        monkeypatch.setattr(
            "boost_and_broadside.modes.noise_calibration._REPORT_FEATURES", forgetful
        )

        with pytest.raises(ValueError, match="names no channel"):
            _report_layout(coordinator)


class TestLagOneOnARunTooShortToMeasureIt:
    """A short run need not produce a single ship valid on two consecutive steps.

    ``lag1_sq_sum`` is a sum of squares, so a dimension nothing was measured on
    twice in a row leaves it at exactly zero -- and ``lag1_cross_sum`` at zero
    with it, since both are accumulated over the same mask. The quotient is
    0/0.

    ``np.where`` does not save the caller from that: it selects between two
    arrays that have *both* already been computed, so the guarded branch still
    evaluates the division, still produces a nan, and still warns before the
    result is thrown away. The answer was never wrong; it was arrived at
    noisily, and the noise showed up in every short run and every smoke case.
    """

    @staticmethod
    def _output(phase1: dict, target_dim: int = 2) -> dict:
        return _build_output(
            phase1=phase1,
            phase2=_make_phase2(target_dim=target_dim, ar_window=20),
            checkpoint_path="dummy.pt",
            num_envs=2,
            num_steps=2,
            num_ar_envs=2,
            num_ar_windows=1,
            feature_groups={"a": ([0], "first"), "b": ([1], "second")},
        )

    def test_an_unmeasured_lag_warns_about_nothing(self, target_dim: int = 2) -> None:
        phase1 = _make_phase1(target_dim)
        phase1["lag1_sq_sum"] = np.zeros(target_dim)
        phase1["lag1_cross_sum"] = np.zeros(target_dim)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            output = self._output(phase1)

        assert output["features"]["a"]["rho_lag1"] == 0.0
        assert output["features"]["b"]["rho_lag1"] == 0.0

    def test_an_unmeasured_lag_reports_no_correlation_rather_than_a_nan(self) -> None:
        """A nan here would reach the report and the recommended noise, where it
        is a number nobody can act on rather than an absent measurement."""

        phase1 = _make_phase1(target_dim=2)
        phase1["lag1_sq_sum"] = np.zeros(2)
        phase1["lag1_cross_sum"] = np.zeros(2)

        recommended = self._output(phase1)["recommended_noise"]

        assert all(np.isfinite(entry["rho"]) for entry in recommended.values())
        assert [entry["rho"] for entry in recommended.values()] == [0.0, 0.0]

    def test_a_measured_lag_is_still_the_ratio_it_always_was(self) -> None:
        """The guard must not flatten the dimensions that were measured."""

        phase1 = _make_phase1(target_dim=2)
        phase1["lag1_sq_sum"] = np.array([2.0, 4.0])
        phase1["lag1_cross_sum"] = np.array([1.0, 1.0])

        features = self._output(phase1)["features"]

        assert features["a"]["rho_lag1"] == pytest.approx(0.5)
        assert features["b"]["rho_lag1"] == pytest.approx(0.25)

    def test_one_measured_dimension_beside_one_unmeasured_one(self) -> None:
        """The mask is per dimension, not per run: a whole-array guard would
        take the measured dimension down with the unmeasured one."""

        phase1 = _make_phase1(target_dim=2)
        phase1["lag1_sq_sum"] = np.array([0.0, 4.0])
        phase1["lag1_cross_sum"] = np.array([0.0, 3.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            features = self._output(phase1)["features"]

        assert features["a"]["rho_lag1"] == 0.0
        assert features["b"]["rho_lag1"] == pytest.approx(0.75)

    def test_the_correlation_stays_within_its_own_range(self) -> None:
        """Few samples make a ratio that is arithmetically above 1; a
        correlation above 1 is not a reading, so it is clipped."""

        phase1 = _make_phase1(target_dim=2)
        phase1["lag1_sq_sum"] = np.array([1.0, 1.0])
        phase1["lag1_cross_sum"] = np.array([9.0, -9.0])

        features = self._output(phase1)["features"]

        assert features["a"]["rho_lag1"] == 1.0
        assert features["b"]["rho_lag1"] == -1.0
