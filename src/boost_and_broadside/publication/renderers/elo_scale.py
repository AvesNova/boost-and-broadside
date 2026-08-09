"""Render the selected fleet-scale checkpoint rating view.

The reference-ladder join lives here rather than in the measurement: attaching a
checkpoint tournament to an independently measured semi-random ladder through
their shared random and scripted endpoints is a reporting view over two stored
artifacts, and it replays no match.
"""

from pathlib import Path

import numpy as np

from boost_and_broadside.evaluation.tournament import rating_views
from boost_and_broadside.publication.renderer_api import Renderer, RenderInputs, register
from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry
from boost_and_broadside.viz import style


def _available(result: dict) -> list[dict]:
    return [
        result["scales"][str(team_size)]
        for team_size in sorted(int(key) for key in result.get("scales", {}))
        if result["scales"][str(team_size)].get("ratings")
    ]


def _series(scales: list[dict], view: str, player_index: int, key: str) -> np.ndarray:
    return np.asarray(
        [scale["ratings"][view][key][player_index] for scale in scales], dtype=float
    )


def _plot_view(result: dict, path: Path) -> Path:
    scales = _available(result)
    labels = result["player_labels"]
    final_index = labels.index("final")
    x = np.arange(len(scales), dtype=float)
    team_sizes = [scale["team_size"] for scale in scales]
    view = "scripted_1000"

    figure = style.new_figure((10.0, 5.8))
    axes = figure.add_subplot(111)
    style.style_axes(axes, "Zero-shot checkpoint strength across fleet sizes")

    final = _series(scales, view, final_index, "ratings")
    final_error = _series(scales, view, final_index, "stderr")
    measured = np.isfinite(final) & np.isfinite(final_error)
    axes.fill_between(
        x[measured],
        final[measured] - final_error[measured],
        final[measured] + final_error[measured],
        color=style.BLUE,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    axes.plot(
        x,
        final,
        color=style.BLUE,
        linewidth=2.4,
        marker="o",
        markersize=5,
        markeredgecolor=style.SURFACE,
        markeredgewidth=1.0,
        label="final checkpoint (±1 SE)",
        zorder=4,
    )

    axes.axhline(
        1000.0,
        color=style.ORANGE,
        linewidth=1.1,
        linestyle=(0, (5, 4)),
        label="scripted (1000)",
    )

    if 4 in team_sizes:
        training_x = float(team_sizes.index(4))
        axes.axvline(
            training_x, color=style.BASELINE, linewidth=1.1, linestyle=(0, (3, 4)), zorder=1
        )
        axes.annotate(
            "training scale",
            (training_x, 0.99),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            color=style.INK_MUTED,
            fontsize=9,
            va="top",
        )

    axes.set_xticks(x, [f"{size}v{size}" for size in team_sizes])
    axes.set_xlabel("ships per team", color=style.INK_SECONDARY, fontsize=10)
    axes.set_ylabel("Elo (scripted = 1000)", color=style.INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=style.INK_SECONDARY, fontsize=9, loc="best")
    axes.margins(x=0.05)
    return style.save(figure, path)


def write_scale_plots(result: dict, output_dir: Path) -> list[Path]:
    """Write the scripted-anchored view used in project documentation."""
    if not _available(result):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    return [_plot_view(result, output_dir / "elo_scale_scripted_1000.png")]


def combine_reference_ladder(result: dict, reference_result: dict) -> dict:
    """Refit scale ratings after joining an independently measured reference ladder.

    The checkpoint and reference tournaments share the same random and scripted
    controllers. Joining their outcome matrices at those players adds intermediate
    comparisons without replaying checkpoint matches. The returned object is a derived
    reporting view; both input artifacts remain the sources of raw outcomes.
    """
    if result.get("run") != reference_result.get("run"):
        raise ValueError("checkpoint and reference tournaments belong to different runs")

    checkpoint_labels = list(result["player_labels"])
    reference_labels = list(reference_result["labels"])
    for endpoint in ("random", "scripted"):
        if endpoint not in checkpoint_labels or endpoint not in reference_labels:
            raise ValueError(f"both tournaments must contain {endpoint!r}")

    labels = checkpoint_labels + [
        label for label in reference_labels if label not in checkpoint_labels
    ]
    label_indices = {label: index for index, label in enumerate(labels)}

    def add_matrix(target: np.ndarray, values: list[list[float]], source_labels: list[str]) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        expected = (len(source_labels), len(source_labels))
        if matrix.shape != expected:
            raise ValueError("stored tournament matrix does not match its player labels")
        indices = [label_indices[label] for label in source_labels]
        target[np.ix_(indices, indices)] += matrix

    scales = {}
    for key, checkpoint_scale in result.get("scales", {}).items():
        reference_scale = reference_result.get("scales", {}).get(key)
        if reference_scale is None:
            continue
        if checkpoint_scale["team_size"] != reference_scale["team_size"]:
            raise ValueError(f"team-size mismatch for scale {key}")
        if checkpoint_scale.get("tie_mode", "half_win") != "half_win":
            raise ValueError("reference-ladder reporting requires half-win tie scoring")

        shape = (len(labels), len(labels))
        wins = np.zeros(shape, dtype=np.float64)
        ties = np.zeros(shape, dtype=np.float64)
        add_matrix(wins, checkpoint_scale["wins_matrix"], checkpoint_labels)
        add_matrix(ties, checkpoint_scale["ties_matrix"], checkpoint_labels)
        add_matrix(wins, reference_scale["wins_matrix"], reference_labels)
        add_matrix(ties, reference_scale["ties_matrix"], reference_labels)

        scored_wins = wins + 0.5 * ties
        pair_games = wins + wins.T + ties + ties.T
        fit = fit_bradley_terry(
            scored_wins,
            anchor=labels.index("scripted"),
            prior_games=1.0,
        )
        scale = dict(checkpoint_scale)
        scale["ratings"] = rating_views(fit.ratings, pair_games, labels)
        scale["reference_ladder_games"] = int(
            np.asarray(reference_scale["wins_matrix"], dtype=float).sum()
            + np.asarray(reference_scale["ties_matrix"], dtype=float).sum()
        )
        scales[key] = scale

    return {
        "run": result["run"],
        "player_labels": labels,
        "team_sizes": sorted(int(key) for key in scales),
        "reference_ladder": {
            "probabilities": reference_result["probabilities"],
            "games_per_pair": reference_result["games_per_pair"],
        },
        "scales": scales,
    }


def _render(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    result = inputs.artifact("scale").read_json()
    reference = inputs.optional("reference")
    if reference is not None:
        result = combine_reference_ladder(result, reference.read_json())
    return write_scale_plots(result, out_dir)


register(
    Renderer(
        name="elo-scale-v1",
        description="Checkpoint strength across symmetric fleet sizes, scripted-anchored.",
        render=_render,
        required_artifacts=("scale",),
        optional_artifacts=("reference",),
        supported_schemas={"scale": (1,), "reference": (1,)},
    )
)
