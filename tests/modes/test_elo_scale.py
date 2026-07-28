"""Tests for scale-ELO transforms and rendering."""

from pathlib import Path

import numpy as np
import pytest

from boost_and_broadside.modes.elo_scale import parallel_envs_for, rating_views
from boost_and_broadside.modes.elo_scale_plots import write_scale_plots


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


def test_scale_plots_write_every_anchor_view(tmp_path: Path) -> None:
    views = _views()
    result = {
        "player_labels": ["random", "scripted", "final"],
        "scales": {
            str(size): {"team_size": size, "ratings": views}
            for size in (1, 2, 4, 8)
        },
    }

    paths = write_scale_plots(result, tmp_path)

    assert len(paths) == 3
    assert all(path.exists() for path in paths)
