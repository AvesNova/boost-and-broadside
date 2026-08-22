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
from boost_and_broadside.modes.elo_calibrate_history import to_history_rows, to_summary
from boost_and_broadside.publication import registered_renderers
from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    RenderInputs,
    get_renderer,
)
from boost_and_broadside.publication.renderers.ar_report import (
    _calc_toroidal_euclidean,
    _clamp_alive_prob,
    _toroidal_center_of_mass,
    _unwrap_1d,
)
from boost_and_broadside.publication.renderers.elo_scale import combine_reference_ladder
from boost_and_broadside.publication.renderers.training import _NEXT_STATE


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


def _calibration_result() -> dict:
    """A calibration in the shape ``modes/elo_calibrate.py`` persists."""

    curve = []
    for update, step in enumerate(range(0, 2_000_000, 400_000), start=1):
        live = 200.0 * update
        curve.append(
            {
                "update": update,
                "global_step": step,
                "live_elo": live + 50.0,
                "live_calibrated": live,
                "live_stderr": 12.0,
                "games": 400,
                "avg_live_elo": live + 20.0,
                "avg_calibrated": live - 30.0,
                "avg_stderr": 15.0,
                "live_calibrated_alt": live + 10.0,
                "live_stderr_alt": 13.0,
            }
        )
    players = [
        {"label": "random", "live_elo": 0.0, "global_step": 0},
        {"label": "scripted", "live_elo": None, "global_step": None},
        {"label": "ckpt_400000", "live_elo": 900.0, "global_step": 400_000},
        {"label": "ckpt_1200000", "live_elo": 1300.0, "global_step": 1_200_000},
    ]
    for player, (calibrated, stderr) in zip(
        players, [(0.0, 8.0), (1000.0, 6.0), (850.0, 9.0), (1250.0, 10.0)], strict=True
    ):
        player["calibrated_elo"] = calibrated
        player["stderr"] = stderr
    tie_rates = [
        {
            "a": "random",
            "b": "scripted",
            "games": 400,
            "tie_rate": 0.04,
            "mean_rating": 500.0,
            "rating_gap": 1000.0,
        },
        {
            "a": "ckpt_400000",
            "b": "ckpt_1200000",
            "games": 400,
            "tie_rate": 0.01,
            "mean_rating": 1050.0,
            "rating_gap": 400.0,
        },
    ]
    return {
        "schema_version": 1,
        "run": "fixture-run",
        "anchor": "scripted",
        "anchor_elo": 1000.0,
        "anchor_offset_stderr": 6.0,
        "reference": "scripted",
        "tie_mode": "half_win",
        "tie_mode_alt": "decisive",
        "target_stderr": 10.0,
        "converged": True,
        "players": players,
        "player_labels": [player["label"] for player in players],
        "curve": curve,
        "batches": [
            {
                "batch": index + 1,
                "games": 400,
                "cumulative_games": 400 * (index + 1),
                "max_stderr": 20.0 - 5.0 * index,
                "mean_stderr": 14.0 - 3.0 * index,
                "seconds": 2.0,
                "ratings": [0.0, 1000.0, 850.0, 1250.0],
            }
            for index in range(3)
        ],
        "wins_matrix": [
            [0.0, 10.0, 20.0, 5.0],
            [190.0, 0.0, 120.0, 60.0],
            [180.0, 80.0, 0.0, 40.0],
            [195.0, 140.0, 160.0, 0.0],
        ],
        "ties_matrix": [[0.0] * 4 for _ in range(4)],
        "tie_rates": tie_rates,
        "training_tie_rates": [dict(row, update=index + 1) for index, row in enumerate(tie_rates)],
    }


def _calibration_artifact(tmp_path):
    """A calibration artifact carrying the chart pair the mode writes beside it."""

    store = _store(tmp_path)
    artifact = store.create(
        ArtifactRecipe("elo-calibration", 1, subjects={"run": "fixture-run"}),
        store.standalone_owner(),
    )
    result = _calibration_result()
    artifact.write_json(result)
    artifact.write_jsonl(to_history_rows(result), "chart_history.jsonl")
    artifact.write_json(to_summary(result), "chart_summary.json")
    artifact.complete()
    return artifact


def _wandb_history(spelling: int = 0) -> list[dict]:
    """Sampled training history in the sparse shape W&B logs and the export keeps.

    ``spelling`` picks which of the trainer's next-state metric namings the
    export carries: 0 is the one the landmark run logged, 1 the one the current
    trainer derives from the feature coordinator.
    """

    rows = []
    for index, step in enumerate(range(0, 2_000_000, 400_000)):
        row = {
            "_step": step,
            "overview/win_rate_vs_scripted": 0.1 + 0.15 * index,
            "overview/explained_variance": 0.2 + 0.1 * index,
            "overview/reward_mean": -1.0 + 0.5 * index,
            "overview/kl": 0.02 - 0.002 * index,
            "overview/clip_fraction": 0.15 - 0.01 * index,
        }
        for _, spellings in _NEXT_STATE:
            for offset, key in enumerate(spellings[spelling]):
                row[key] = 0.5 / (index + 1) + 0.01 * offset
        if index == 2:
            # W&B logs sparsely; a metric absent from one row is normal.
            del row["overview/kl"]
        rows.append(row)
    return rows


def _wandb_export(tmp_path, rows: list[dict] | None = None):
    """A ``wandb-export`` artifact in the shape ``scripts/export_wandb_run.py`` writes."""

    store = _store(tmp_path)
    artifact = store.create(
        ArtifactRecipe(
            "wandb-export",
            1,
            subjects={"wandb_run": "fixture/boost-and-broadside/abc123", "run": "fixture-run"},
            parameters={"samples": 2000},
        ),
        store.standalone_owner(),
    )
    artifact.write_json({"profile": "rl"}, "config.json")
    artifact.write_json({"overview/win_rate_vs_scripted": 0.85}, "summary.json")
    artifact.write_json(
        {"id": "abc123", "name": "fixture-run", "state": "finished"}, "run_meta.json"
    )
    (artifact.path / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (rows if rows is not None else _wandb_history()))
    )
    artifact.attach("history.jsonl")
    artifact.complete()
    return artifact


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


def test_the_elo_curve_renders_from_a_stored_calibration(tmp_path) -> None:
    artifact = _calibration_artifact(tmp_path)

    written = _render("training-elo-curve-v1", {"calibration": artifact}, tmp_path / "out")

    assert written == ["elo_curve.png"]


def test_the_calibration_diagnostics_render_both_draw_conventions(tmp_path) -> None:
    artifact = _calibration_artifact(tmp_path)

    written = _render(
        "elo-calibration-diagnostics-v1", {"calibration": artifact}, tmp_path / "out"
    )

    assert written == [
        "avg_curve.png",
        "calibrated_decisive.png",
        "calibrated_half_win.png",
        "checkpoint_ratings.png",
        "convergence.png",
        "live_and_avg.png",
        "live_curve.png",
        "tie_conventions.png",
        "tie_rates.png",
    ]


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [
        ("training-win-rate-v1", ["win_rate_vs_scripted.png"]),
        ("training-health-v1", ["training_health.png"]),
        ("next-state-error-v1", ["next_state_error.png"]),
    ],
)
def test_the_training_figures_render_from_a_stored_export(tmp_path, renderer, expected) -> None:
    artifact = _wandb_export(tmp_path)

    assert _render(renderer, {"wandb_export": artifact}, tmp_path / renderer) == expected


@pytest.mark.parametrize(
    "renderer", ["training-win-rate-v1", "training-health-v1", "next-state-error-v1"]
)
def test_rendering_the_same_export_twice_is_byte_identical(tmp_path, renderer) -> None:
    artifact = _wandb_export(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"

    names = _render(renderer, {"wandb_export": artifact}, first)
    _render(renderer, {"wandb_export": artifact}, second)

    assert [(first / name).read_bytes() for name in names] == [
        (second / name).read_bytes() for name in names
    ]


def test_the_next_state_figure_reads_either_metric_spelling(tmp_path) -> None:
    """The rename at `afdf406` renamed every predictor; both spellings render."""

    landmark = _wandb_export(tmp_path, _wandb_history(spelling=0))
    current = _wandb_export(tmp_path, _wandb_history(spelling=1))
    first, second = tmp_path / "landmark", tmp_path / "current"

    assert _render("next-state-error-v1", {"wandb_export": landmark}, first) == [
        "next_state_error.png"
    ]
    assert _render("next-state-error-v1", {"wandb_export": current}, second) == [
        "next_state_error.png"
    ]
    # The same measurements under two names are the same figure.
    assert (first / "next_state_error.png").read_bytes() == (
        second / "next_state_error.png"
    ).read_bytes()


@pytest.mark.parametrize(
    ("renderer", "dropped"),
    [
        ("next-state-error-v1", "next_state/"),
        ("training-win-rate-v1", "overview/win_rate_vs_scripted"),
        ("training-health-v1", "overview/kl"),
    ],
)
def test_a_figure_with_no_measurements_fails_instead_of_coming_out_blank(
    tmp_path, renderer, dropped
) -> None:
    """A metric this build cannot find is a refusal, not an empty canvas."""

    rows = [
        {key: value for key, value in row.items() if not key.startswith(dropped)}
        for row in _wandb_history()
    ]
    artifact = _wandb_export(tmp_path, rows)

    with pytest.raises(PublicationError, match="would be blank"):
        _render(renderer, {"wandb_export": artifact}, tmp_path / renderer)


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


def test_promoted_media_is_copied_under_its_own_name(tmp_path) -> None:
    clip = tmp_path / "out" / "duel-4v4.gif"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"GIF89a-curated")
    out_dir = tmp_path / "published"
    out_dir.mkdir()

    get_renderer("media-copy-v1").render(RenderInputs(files={"clip": clip}), out_dir)

    assert (out_dir / "duel-4v4.gif").read_bytes() == clip.read_bytes()


def test_an_external_asset_is_verified_rather_than_rendered(tmp_path) -> None:
    renderer = get_renderer("external-asset-v1")

    assert renderer.external
    with pytest.raises(PublicationError, match="never rendered"):
        renderer.render(RenderInputs(), tmp_path)


def _figures_artifact(tmp_path):
    """A run's rendered figure set: one file, one directory, as the real one has."""

    store = _store(tmp_path)
    artifact = store.create(ArtifactRecipe("figures", 1), store.standalone_owner())
    (artifact.path / "elo_curve.png").write_bytes(b"\x89PNG chart bytes")
    (artifact.path / "ar_report_4v4").mkdir()
    (artifact.path / "ar_report_4v4" / "report.md").write_text("# report\n")
    (artifact.path / "ar_report_4v4" / "panel.png").write_bytes(b"\x89PNG panel")
    artifact.attach("elo_curve.png")
    artifact.attach("ar_report_4v4/report.md")
    artifact.attach("ar_report_4v4/panel.png")
    artifact.complete()
    return artifact


def test_publishing_a_figure_copies_the_bytes_the_run_already_rendered(tmp_path) -> None:
    """Publication must not re-render. Two renders are equal only by convention.

    Copying is what makes the published chart and the run's own evidence the
    same artifact rather than two that happen to agree today.
    """

    artifact = _figures_artifact(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    get_renderer("figure-copy-v1").render(
        RenderInputs(artifacts={"figures": artifact}, figure="elo_curve.png"), out_dir
    )

    assert (out_dir / "elo_curve.png").read_bytes() == (
        artifact.path / "elo_curve.png"
    ).read_bytes()


def test_publishing_a_figure_tree_copies_every_file_under_it(tmp_path) -> None:
    artifact = _figures_artifact(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = get_renderer("figure-tree-copy-v1").render(
        RenderInputs(artifacts={"figures": artifact}, figure="ar_report_4v4"), out_dir
    )

    assert sorted(path.relative_to(out_dir).as_posix() for path in written) == [
        "panel.png",
        "report.md",
    ]
    assert (out_dir / "report.md").read_text() == "# report\n"


def test_publishing_a_figure_the_run_does_not_have_says_what_it_does_have(tmp_path) -> None:
    artifact = _figures_artifact(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(PublicationError, match="holds no figure named 'nope.png'"):
        get_renderer("figure-copy-v1").render(
            RenderInputs(artifacts={"figures": artifact}, figure="nope.png"), out_dir
        )


@pytest.mark.parametrize(
    ("renderer", "figure", "expected"),
    (
        ("figure-copy-v1", "ar_report_4v4", "publish it with figure-tree-copy-v1"),
        ("figure-tree-copy-v1", "elo_curve.png", "publish it with figure-copy-v1"),
    ),
)
def test_a_figure_copied_by_the_wrong_renderer_names_the_right_one(
    tmp_path, renderer: str, figure: str, expected: str
) -> None:
    artifact = _figures_artifact(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(PublicationError, match=expected):
        get_renderer(renderer).render(
            RenderInputs(artifacts={"figures": artifact}, figure=figure), out_dir
        )


def test_every_registered_renderer_is_named_by_a_test() -> None:
    """A tripwire for the claim that these tests cover the renderer inventory.

    S09's handoff recorded that every renderer was covered from fixture
    artifacts while three had none at all, and nothing checked. This is coarse —
    a name in a comment would satisfy it — but it fails the moment a renderer is
    registered with no test naming it.
    """

    sources = " ".join(path.read_text() for path in Path(__file__).parent.glob("test_*.py"))
    assert [name for name in registered_renderers() if name not in sources] == []


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
