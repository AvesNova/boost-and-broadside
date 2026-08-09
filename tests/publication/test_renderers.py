"""Every registered renderer, driven from a fixture artifact.

These cover the rendering contract rather than the look of a chart: the exact
filenames each entry owns, that re-rendering the same measurement is
byte-identical (``publish --check`` depends on it), and the numeric helpers that
carry the AR report's stated metric definitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore, Invocation
from boost_and_broadside.evaluation.tournament import rating_views
from boost_and_broadside.publication.renderer_api import RenderInputs, get_renderer
from boost_and_broadside.publication.renderers.ar_report import (
    _calc_toroidal_euclidean,
    _clamp_alive_prob,
    _toroidal_center_of_mass,
    _unwrap_1d,
)
from boost_and_broadside.publication.renderers.elo_scale import combine_reference_ladder


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        checkpoint_root=tmp_path / "checkpoints",
        standalone_root=tmp_path / "artifacts",
        invocation=Invocation(argv=("bnb", "fixture"), command="fixture"),
    )


def _artifact(tmp_path, artifact_type: str, result: dict, arrays: dict | None = None):
    store = _store(tmp_path)
    artifact = store.create(
        ArtifactRecipe(artifact_type, 2 if artifact_type == "crossover" else 1),
        store.standalone_owner(),
    )
    artifact.write_json(result)
    if arrays is not None:
        artifact.write_npz(arrays)
    artifact.complete()
    return artifact


def _render(name: str, sources: dict, out_dir: Path) -> list[str]:
    """Render as publication does: into an existing, empty directory."""

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = get_renderer(name)
    renderer.render(RenderInputs(artifacts=sources), out_dir)
    return sorted(
        str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()
    )


# --- fixture measurements -------------------------------------------------


def _crossover_result() -> dict:
    return {
        "schema_version": 2,
        "run": "fixture-run",
        "num_envs": 8,
        "max_total_ships": 64,
        "rows": [
            {
                "trained": size,
                "crossover": size * 2,
                "beats_up_to": size * 2 - 1,
                "capped": False,
                "win_rate_at_beats_up_to": 0.6,
                "win_rate_at_crossover": 0.4,
                "curve": {},
            }
            for size in (1, 2, 4, 8)
        ],
    }


def _scale_result() -> dict:
    ratings = np.array([0.0, 500.0, 800.0])
    games = np.full((3, 3), 1_000.0)
    np.fill_diagonal(games, 0.0)
    views = rating_views(ratings, games, ["random", "scripted", "final"])
    return {
        "schema_version": 1,
        "run": "fixture-run",
        "player_labels": ["random", "scripted", "final"],
        "scales": {
            str(size): {
                "team_size": size,
                "tie_mode": "half_win",
                "ratings": views,
                "wins_matrix": [[0, 2, 0], [8, 0, 4], [10, 6, 0]],
                "ties_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
            for size in (1, 2, 4, 8)
        },
    }


def _ladder_result() -> dict:
    return {
        "schema_version": 1,
        "run": "fixture-run",
        "probabilities": [0.0, 0.5, 1.0],
        "labels": ["random", "semi_scripted_0p5", "scripted"],
        "games_per_pair": 10,
        "scales": {
            "4": {
                "team_size": 4,
                "ratings": rating_views(
                    np.array([0.0, 400.0, 900.0]),
                    np.full((3, 3), 100.0),
                    ["random", "semi_scripted_0p5", "scripted"],
                ),
                "adjacent_matchups": [
                    {
                        "lower": "random",
                        "higher": "semi_scripted_0p5",
                        "higher_expected_score": 0.8,
                    },
                    {
                        "lower": "semi_scripted_0p5",
                        "higher": "scripted",
                        "higher_expected_score": 0.7,
                    },
                ],
                "wins_matrix": [[0, 2, 1], [8, 0, 3], [9, 7, 0]],
                "ties_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
        },
    }


def _ar_arrays(steps: int = 6, ships: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    arrays = {}
    for prefix in ("gt", "cl", "ol"):
        arrays[f"{prefix}_pos"] = rng.uniform(0, 100, (steps, ships, 2)).astype(np.float32)
        arrays[f"{prefix}_vel"] = rng.normal(0, 1, (steps, ships, 2)).astype(np.float32)
        arrays[f"{prefix}_att"] = rng.normal(0, 1, (steps, ships, 2)).astype(np.float32)
        for field in ("ang_vel", "health", "power", "cooldown"):
            arrays[f"{prefix}_{field}"] = rng.normal(0, 1, (steps, ships, 1)).astype(np.float32)
        arrays[f"{prefix}_alive"] = np.ones((steps, ships), dtype=np.float32)
        arrays[f"{prefix}_alive_prob"] = np.ones((steps, ships), dtype=np.float32)
    return arrays


def _noise_result() -> dict:
    features = {
        name: {
            "aux_dims": [index * 2, index * 2 + 1],
            "sigma": 0.1 + index / 100,
            "bias": 0.001,
            "rho_lag1": 0.2,
            "sigma_team0": 0.1,
            "sigma_team1": 0.1,
            "team_symmetry_ok": True,
            "sigma_combat": 0.12,
            "sigma_noncombat": 0.09,
        }
        for index, name in enumerate(("pos_x", "velocity"))
    }
    return {
        "metadata": {"checkpoint": "fixture.pt", "num_envs": 4},
        "features": features,
        "ar_growth": {
            "depth": [1, 2, 3],
            "rmse_per_feature": {name: [0.1, 0.2, 0.3] for name in features},
        },
        "recommended_noise": {name: {"sigma": 0.1, "rho": 0.2} for name in features},
        "dim_names": ["pos_sin_x", "pos_cos_x", "vel_vx_norm", "vel_vy_norm"],
    }


# --- renderers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [
        ("crossover-phase-v1", ["crossover_phase.png"]),
        ("crossover-ratio-v1", ["crossover_ratio.png"]),
        ("crossover-data-v1", ["crossover.json"]),
    ],
)
def test_crossover_renderers_own_their_filenames(tmp_path, renderer, expected) -> None:
    artifact = _artifact(tmp_path, "crossover", _crossover_result())

    assert _render(renderer, {"crossover": artifact}, tmp_path / renderer) == expected


def test_the_published_crossover_data_is_the_measurement(tmp_path) -> None:
    artifact = _artifact(tmp_path, "crossover", _crossover_result())
    out_dir = tmp_path / "out"
    _render("crossover-data-v1", {"crossover": artifact}, out_dir)

    assert json.loads((out_dir / "crossover.json").read_text()) == _crossover_result()


def test_elo_scale_renders_the_scripted_anchored_view(tmp_path) -> None:
    artifact = _artifact(tmp_path, "elo-scale", _scale_result())

    written = _render("elo-scale-v1", {"scale": artifact}, tmp_path / "out")

    assert written == ["elo_scale_scripted_1000.png"]


def test_elo_scale_joins_an_optional_reference_ladder(tmp_path) -> None:
    scale = _artifact(tmp_path, "elo-scale", _scale_result())
    ladder = _artifact(tmp_path, "semi-random-ladder", _ladder_result())

    written = _render(
        "elo-scale-v1", {"scale": scale, "reference": ladder}, tmp_path / "out"
    )

    assert written == ["elo_scale_scripted_1000.png"]


def test_reference_ladder_is_joined_through_shared_endpoints() -> None:
    combined = combine_reference_ladder(
        {
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
        },
        {
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
        },
    )

    assert combined["player_labels"] == ["random", "scripted", "final", "semi_scripted_0p5"]
    assert combined["scales"]["4"]["reference_ladder_games"] == 30
    assert combined["scales"]["4"]["ratings"]["scripted_1000"]["ratings"][1] == pytest.approx(
        1000.0
    )


def test_semi_random_renders_the_connectivity_diagnostic(tmp_path) -> None:
    artifact = _artifact(tmp_path, "semi-random-ladder", _ladder_result())

    written = _render("semi-random-connectivity-v1", {"ladder": artifact}, tmp_path / "out")

    assert written == ["semi_random_connectivity.png"]


def test_the_ar_report_renders_its_complete_published_set(tmp_path) -> None:
    artifact = _artifact(
        tmp_path,
        "ar-report",
        {
            "schema_version": 1,
            "num_ships": 4,
            "num_steps": 6,
            "world_size": [100.0, 100.0],
            "agents": {"team0": "fixture.pt", "team1": "scripted"},
            "rollouts": ["gt", "cl", "ol"],
            "fields": ["pos"],
        },
        _ar_arrays(),
    )

    written = _render("ar-report-v1", {"ar_report": artifact}, tmp_path / "out")

    assert "ar_report.md" in written
    assert {"2d_map.png", "2d_map_ship0.png", "2d_vel_map.png"} <= set(written)
    assert sum(name.startswith("mae_") for name in written) == 6
    assert sum(name.startswith("feature_") for name in written) == 9


def test_noise_calibration_renders_its_report(tmp_path) -> None:
    artifact = _artifact(tmp_path, "noise-calibration", _noise_result())

    written = _render("noise-calibration-v1", {"noise": artifact}, tmp_path / "out")

    assert written == [
        "ar_growth.png",
        "autocorrelation.png",
        "error_distributions.png",
        "noise_params.json",
        "team_symmetry.png",
    ]


@pytest.mark.parametrize(
    ("renderer", "artifact_type", "source", "result"),
    [
        ("crossover-phase-v1", "crossover", "crossover", _crossover_result()),
        ("elo-scale-v1", "elo-scale", "scale", _scale_result()),
        ("semi-random-connectivity-v1", "semi-random-ladder", "ladder", _ladder_result()),
    ],
)
def test_rendering_the_same_measurement_twice_is_byte_identical(
    tmp_path, renderer, artifact_type, source, result
) -> None:
    artifact = _artifact(tmp_path, artifact_type, result)
    first, second = tmp_path / "first", tmp_path / "second"

    names = _render(renderer, {source: artifact}, first)
    _render(renderer, {source: artifact}, second)

    assert [(first / name).read_bytes() for name in names] == [
        (second / name).read_bytes() for name in names
    ]


def test_a_renderer_writes_only_inside_the_directory_it_is_given(tmp_path) -> None:
    artifact = _artifact(tmp_path, "crossover", _crossover_result())
    out_dir = tmp_path / "out"
    before = {path for path in tmp_path.rglob("*") if path.is_file()}

    _render("crossover-phase-v1", {"crossover": artifact}, out_dir)

    written = {path for path in tmp_path.rglob("*") if path.is_file()} - before
    assert all(path.is_relative_to(out_dir) for path in written)


# --- the AR report's stated metric definitions ----------------------------


def test_unwrap_1d_makes_boundary_crossing_continuous() -> None:
    # A ship drifting off the right edge (98 -> 2) reappears on the left after wrapping;
    # unwrapping must extend the trajectory past the edge rather than jump back across it.
    W = 100.0
    unwrapped = _unwrap_1d(np.array([95.0, 98.0, 2.0, 5.0]), W)
    steps = np.diff(unwrapped)

    assert np.all(np.abs(steps) < W / 2)
    assert np.all(steps > 0)  # motion stays monotonic, no phantom reversal at the seam


def test_toroidal_center_of_mass_anchors_near_the_wrap_seam() -> None:
    # Two ships hugging opposite edges (x=1 and x=99) are adjacent on the torus; their CoM
    # belongs at the seam (~0/100), not the naive arithmetic mean of 50.
    W_x = W_y = 100.0
    com_x, com_y = _toroidal_center_of_mass(np.array([[1.0, 50.0], [99.0, 50.0]]), W_x, W_y)

    wrapped_x = com_x % W_x
    assert min(wrapped_x, W_x - wrapped_x) < 5.0
    assert abs(com_y - 50.0) < 1e-6


def test_toroidal_euclidean_uses_short_way_and_masks_dead_pairs() -> None:
    W_x = W_y = 100.0
    pos1 = np.array([[[1.0, 10.0], [50.0, 50.0]]])  # (1 step, 2 ships, 2)
    pos2 = np.array([[[99.0, 10.0], [50.0, 50.0]]])
    alive1 = np.array([[True, True]])
    alive2 = np.array([[True, False]])  # ship 1 dead in method 2

    dist = _calc_toroidal_euclidean(pos1, pos2, W_x, W_y, alive1, alive2)

    assert dist[0, 0] == 2.0  # short way across the seam, not 98
    assert np.isnan(dist[0, 1])  # dead pair masked out


def test_clamp_alive_prob_zeros_from_first_death_onward() -> None:
    # A ship that dies at step 2 must read prob 0 for every later step, even if the raw
    # predicted alive-prob "revives" it — deaths are permanent within a rollout.
    alive_prob = np.array([[0.9], [0.8], [0.3], [0.7], [0.6]])
    alive = np.array([[True], [True], [False], [False], [False]])

    clamped = _clamp_alive_prob(alive_prob, alive, plot_N=1)

    assert np.array_equal(clamped[:2, 0], np.array([0.9, 0.8]))
    assert np.all(clamped[2:, 0] == 0.0)
