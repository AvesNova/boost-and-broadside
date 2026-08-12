"""Tests for the unified (W&B-format) history reader.

The point of the format is that in-training and calibrated data are read the same
way, so the core extractors are exercised on both a W&B-shaped and a
calibrated-shaped row list."""

import numpy as np

from boost_and_broadside.viz.history import (
    checkpoint_points,
    combine_series,
    series,
    series_band,
)

# A W&B-shaped slice: sparse rows, an absent point, a None, and out-of-order steps.
WANDB_ROWS = [
    {"_step": 3_000, "overview/kl": 0.03, "next_state/pos_x_dphase": 0.2},
    {"_step": 1_000, "overview/kl": 0.01, "next_state/pos_x_dphase": 0.5,
     "next_state/pos_y_dphase": 0.6},
    {"_step": 2_000, "overview/kl": None},  # logged the step, not this metric
    {"_step": 4_000, "next_state/pos_x_dphase": float("nan")},  # non-finite
]

# A calibrated-shaped slice: the exact keys modes/elo_calibrate_history.py emits.
CALIB_ROWS = [
    {"_step": 100, "calibrated_elo/live": 950.0, "calibrated_elo/live_stderr": 12.0},
    {"_step": 200, "calibrated_elo/live": 1200.0, "calibrated_elo/live_stderr": 8.0},
    {"_step": 300, "calibrated_elo/live": float("inf")},  # clean sweep, no stderr
]


class TestSeries:
    def test_drops_absent_none_and_nonfinite_and_sorts(self):
        x, y = series(WANDB_ROWS, "overview/kl")
        assert list(x) == [1_000.0, 3_000.0]  # 2000 is None, 4000 absent; sorted
        assert list(y) == [0.01, 0.03]

    def test_works_identically_on_calibrated_rows(self):
        x, y = series(CALIB_ROWS, "calibrated_elo/live")
        assert list(x) == [100.0, 200.0]  # inf dropped
        assert list(y) == [950.0, 1200.0]

    def test_missing_key_yields_empty(self):
        x, y = series(WANDB_ROWS, "does/not/exist")
        assert x.size == 0 and y.size == 0


class TestSeriesBand:
    def test_band_aligns_and_needs_both_value_and_error(self):
        x, y, lo, hi = series_band(CALIB_ROWS, "calibrated_elo/live", "calibrated_elo/live_stderr")
        assert list(x) == [100.0, 200.0]  # step 300 has no stderr → excluded
        assert list(lo) == [938.0, 1192.0]
        assert list(hi) == [962.0, 1208.0]

    def test_n_sigma_scales_the_band(self):
        _, _, lo, hi = series_band(
            CALIB_ROWS, "calibrated_elo/live", "calibrated_elo/live_stderr", n_sigma=2.0
        )
        assert lo[0] == 950.0 - 24.0 and hi[0] == 950.0 + 24.0


class TestCombineSeries:
    def test_averages_symmetric_dims(self):
        keys = ["next_state/pos_x_dphase", "next_state/pos_y_dphase"]
        x, y = combine_series(WANDB_ROWS, keys)
        # Only the 1000 step has both members finite; 3000 lacks pos_y.
        assert list(x) == [1_000.0]
        assert y[0] == np.mean([0.5, 0.6])

    def test_skips_step_missing_any_member(self):
        keys = ["next_state/pos_x_dphase", "next_state/pos_y_dphase"]
        rows = [{"_step": 5, "next_state/pos_x_dphase": 1.0}]  # no pos_y
        x, y = combine_series(rows, keys)
        assert x.size == 0 and y.size == 0


class TestCheckpointPoints:
    def test_reads_ckpt_ratings_and_stderr(self):
        summary = {
            "calibrated_elo/ckpt_200": 1050.0, "calibrated_elo/ckpt_200_stderr": 7.0,
            "calibrated_elo/ckpt_100": 900.0, "calibrated_elo/ckpt_100_stderr": 5.0,
            "calibrated_elo/live": 1200.0,  # not a checkpoint
            "calibrated_elo/scripted": 800.0,
        }
        steps, ratings, errs = checkpoint_points(summary)
        assert list(steps) == [100.0, 200.0]  # sorted, live/scripted excluded
        assert list(ratings) == [900.0, 1050.0]
        assert list(errs) == [5.0, 7.0]

    def test_missing_stderr_is_nan(self):
        steps, ratings, errs = checkpoint_points({"calibrated_elo/ckpt_50": 500.0})
        assert list(steps) == [50.0] and np.isnan(errs[0])
