"""Tests for noise_calibration mode's output-building logic."""

import datetime

import numpy as np

from boost_and_broadside.modes.noise_calibration import _build_output


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
