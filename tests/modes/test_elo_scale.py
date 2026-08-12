"""Tests for the scale-Elo player field and its rating views.

The reference-ladder join and the figure it feeds moved to the publication
renderer; both are covered there.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.evaluation.tournament import parallel_envs_for, rating_views
from boost_and_broadside.modes.elo_scale import _build_scale_players, _player_metadata


def _views() -> dict[str, dict[str, list[float]]]:
    ratings = np.array([0.0, 500.0, 800.0])
    games = np.full((3, 3), 1_000.0)
    np.fill_diagonal(games, 0.0)
    return rating_views(ratings, games, ["random", "scripted", "final"])


def test_rating_views_apply_the_three_anchor_conventions() -> None:
    views = _views()

    assert views["random_zero"]["ratings"] == pytest.approx([0.0, 500.0, 800.0])
    assert views["scripted_1000"]["ratings"] == pytest.approx([500.0, 1000.0, 1300.0])
    assert views["random_zero_scripted_1000"]["ratings"] == pytest.approx(
        [0.0, 1000.0, 1600.0]
    )


def test_dual_anchor_uncertainty_pins_both_landmarks() -> None:
    dual_error = _views()["random_zero_scripted_1000"]["stderr"]

    assert dual_error[0] == pytest.approx(0.0)
    assert dual_error[1] == pytest.approx(0.0)
    assert dual_error[2] > 0.0


def test_parallel_width_respects_quadratic_collision_budget() -> None:
    assert parallel_envs_for(total_ships=8, maximum=16_384) == 16_384
    assert parallel_envs_for(total_ships=128, maximum=16_384) == 244


def test_scale_field_has_one_final_when_final_step_is_absent_from_roster(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    final = run / "step_000000000100.pt"
    torch.save({"global_step": 100}, final)
    roster = {"entries": []}
    monkeypatch.setattr(
        "boost_and_broadside.evaluation.tournament.load_ladder_policy",
        lambda *args, **kwargs: object(),
    )

    metadata = _player_metadata(run, roster, final)
    players = _build_scale_players(run, roster, None, ShipConfig(), 8, "cpu")

    assert [record["label"] for record in metadata] == ["random", "scripted", "final"]
    assert [player.label for player in players] == ["random", "scripted", "final"]
    assert metadata[-1]["global_step"] == players[-1].global_step == 100
