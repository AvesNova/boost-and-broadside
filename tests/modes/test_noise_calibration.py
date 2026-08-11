"""Tests for noise_calibration mode's output-building logic."""

import datetime

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
            name: entry
            for name, entry in _REPORT_FEATURES.items()
            if name != "local_log_index"
        }
        monkeypatch.setattr(
            "boost_and_broadside.modes.noise_calibration._REPORT_FEATURES", forgetful
        )

        with pytest.raises(ValueError, match="names no channel"):
            _report_layout(coordinator)
