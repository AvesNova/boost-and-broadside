"""Tests for reducing a calibration result to the W&B export format."""

import math

from boost_and_broadside.modes.elo_calibrate_history import to_history_rows, to_summary


def _result() -> dict:
    """A small, structurally complete calibration result."""
    return {
        "run": "test-run-1",
        "reference": "scripted",
        "tie_mode": "half_win",
        "tie_mode_alt": "decisive",
        "converged": True,
        "target_stderr": 10.0,
        "anchor_offset_stderr": 42.0,
        "players": [
            {"label": "random", "training_elo": 0.0, "calibrated_elo": 0.0, "stderr": 0.0,
             "global_step": 0},
            {"label": "scripted", "training_elo": None, "calibrated_elo": 822.0, "stderr": 5.0,
             "global_step": None},
            {"label": "ckpt_100", "training_elo": 480.0, "calibrated_elo": 975.0, "stderr": 7.5,
             "global_step": 100},
        ],
        "curve": [
            {"update": 1, "global_step": 100, "live_training": 10.0, "live_calibrated": 950.0,
             "live_stderr": 12.0, "avg_calibrated": 900.0, "avg_stderr": 20.0,
             "live_calibrated_alt": 1100.0, "live_stderr_alt": 13.0},
            {"update": 2, "global_step": 200, "live_training": 20.0, "live_calibrated": 1200.0,
             "live_stderr": 8.0},
        ],
    }


class TestToHistoryRows:
    def test_rows_are_keyed_by_step_in_order(self):
        rows = to_history_rows(_result())
        assert [r["_step"] for r in rows] == [100, 200]

    def test_curve_maps_to_wandb_ladder_keys(self):
        first = next(r for r in to_history_rows(_result()) if r["_step"] == 100)
        assert first["ladder/elo/live"] == 950.0
        assert first["ladder/elo/live_stderr"] == 12.0
        assert first["ladder/elo/avg"] == 900.0
        assert first["ladder/elo/live_decisive"] == 1100.0

    def test_checkpoint_rating_lands_on_its_own_step(self):
        first = next(r for r in to_history_rows(_result()) if r["_step"] == 100)
        assert first["ladder/elo/ckpt_100"] == 975.0
        assert first["ladder/elo/ckpt_100_stderr"] == 7.5

    def test_non_finite_values_are_omitted_not_written_as_nan(self):
        result = _result()
        result["curve"][1]["live_calibrated"] = float("inf")  # a clean sweep
        result["curve"][1]["live_stderr"] = float("nan")
        result["curve"][1]["avg_calibrated"] = 1300.0  # keeps the row alive
        second = next(r for r in to_history_rows(result) if r["_step"] == 200)
        assert "ladder/elo/live" not in second
        assert "ladder/elo/live_stderr" not in second
        assert second["ladder/elo/avg"] == 1300.0
        assert all(math.isfinite(v) for row in to_history_rows(result) for v in row.values())

    def test_rows_with_no_finite_metric_are_dropped(self):
        result = _result()
        for point in result["curve"]:  # nothing plottable anywhere
            for key in list(point):
                if key not in ("update", "global_step"):
                    point[key] = float("nan")
        result["players"] = [p for p in result["players"] if p["label"] == "scripted"]
        assert to_history_rows(result) == []

    def test_scripted_and_random_do_not_produce_ckpt_rows(self):
        # scripted has no step; random has a step but is the zero anchor with no
        # training_elo distinction — only real ladder rungs get ckpt keys.
        rows = to_history_rows(_result())
        assert not any(k.startswith("ladder/elo/ckpt_0") for r in rows for k in r)


class TestToSummary:
    def test_exposes_scripted_landmark(self):
        assert to_summary(_result())["elo/scripted"] == 822.0

    def test_exposes_every_checkpoint_final(self):
        summary = to_summary(_result())
        assert summary["ladder/elo/ckpt_100"] == 975.0
        assert summary["ladder/elo/ckpt_100_stderr"] == 7.5

    def test_final_live_and_step(self):
        summary = to_summary(_result())
        assert summary["ladder/elo/live"] == 1200.0  # last curve point
        assert summary["_step"] == 200

    def test_carries_calibration_metadata(self):
        summary = to_summary(_result())
        assert summary["reference"] == "scripted"
        assert summary["converged"] is True
        assert summary["target_stderr"] == 10.0
        assert summary["anchor_offset_stderr"] == 42.0


class TestFinalCheckpointInclusion:
    def test_step_bearing_player_without_training_rating_lands_on_the_axis(self):
        """The run's final checkpoint was never a roster entry, so it has no
        training rating — but its tournament rating pins the curve's endpoint."""
        result = _result()
        result["players"].append(
            {"label": "ckpt_999", "training_elo": None, "calibrated_elo": 1810.0,
             "stderr": 4.0, "global_step": 999}
        )
        row = next(r for r in to_history_rows(result) if r["_step"] == 999)
        assert row["ladder/elo/ckpt_999"] == 1810.0
        summary = to_summary(result)
        assert summary["ladder/elo/ckpt_999"] == 1810.0
        assert summary["_step"] == 999
