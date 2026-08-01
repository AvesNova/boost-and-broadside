"""Tests for scale-Elo transforms and rendering."""

from pathlib import Path

import numpy as np
import pytest

from boost_and_broadside.modes.elo_scale import (
    combine_reference_ladder,
    parallel_envs_for,
    rating_views,
)
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


def test_scale_plot_writes_selected_anchor_view(tmp_path: Path) -> None:
    views = _views()
    result = {
        "player_labels": ["random", "scripted", "final"],
        "scales": {
            str(size): {"team_size": size, "ratings": views}
            for size in (1, 2, 4, 8)
        },
    }

    paths = write_scale_plots(result, tmp_path)

    assert len(paths) == 1
    assert all(path.exists() for path in paths)


def test_reference_ladder_is_joined_through_shared_endpoints() -> None:
    checkpoint_result = {
        "run": "example",
        "player_labels": ["random", "scripted", "final"],
        "scales": {
            "4": {
                "team_size": 4,
                "tie_mode": "half_win",
                "wins_matrix": [[0, 2, 0], [8, 0, 4], [10, 6, 0]],
                "ties_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
        },
    }
    reference_result = {
        "run": "example",
        "labels": ["random", "semi_scripted_0p5", "scripted"],
        "probabilities": [0.0, 0.5, 1.0],
        "games_per_pair": 10,
        "scales": {
            "4": {
                "team_size": 4,
                "wins_matrix": [[0, 2, 1], [8, 0, 3], [9, 7, 0]],
                "ties_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
        },
    }

    combined = combine_reference_ladder(checkpoint_result, reference_result)

    assert combined["player_labels"] == [
        "random",
        "scripted",
        "final",
        "semi_scripted_0p5",
    ]
    assert combined["scales"]["4"]["reference_ladder_games"] == 30
    assert combined["scales"]["4"]["ratings"]["scripted_1000"]["ratings"][1] \
        == pytest.approx(1000.0)
