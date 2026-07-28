"""Render the three reporting-anchor views of fleet-scale ratings."""

from pathlib import Path

import numpy as np

from boost_and_broadside.viz import style

_VIEWS = (
    (
        "random_zero",
        "elo_scale_random_zero.png",
        "Fleet-scale ELO — random anchored at 0",
        "ELO (random = 0)",
    ),
    (
        "scripted_1000",
        "elo_scale_scripted_1000.png",
        "Fleet-scale ELO — scripted anchored at 1000",
        "ELO (scripted = 1000)",
    ),
    (
        "random_zero_scripted_1000",
        "elo_scale_dual_anchor.png",
        "Fleet-scale rating — random 0, scripted 1000",
        "baseline-normalized rating",
    ),
)


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


def _plot_view(result: dict, view: str, path: Path, title: str, ylabel: str) -> Path:
    scales = _available(result)
    labels = result["player_labels"]
    final_index = labels.index("final")
    random_index = labels.index("random")
    scripted_index = labels.index("scripted")
    x = np.arange(len(scales), dtype=float)
    team_sizes = [scale["team_size"] for scale in scales]

    figure = style.new_figure((10.0, 5.8))
    axes = figure.add_subplot(111)
    style.style_axes(axes, title)

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

    if view == "random_zero":
        baseline = _series(scales, view, scripted_index, "ratings")
        baseline_error = _series(scales, view, scripted_index, "stderr")
        axes.fill_between(
            x,
            baseline - baseline_error,
            baseline + baseline_error,
            color=style.ORANGE,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        axes.plot(x, baseline, color=style.ORANGE, linewidth=1.8, label="scripted")
        axes.axhline(0.0, color=style.INK_MUTED, linewidth=1.1, linestyle=(0, (5, 4)))
    elif view == "scripted_1000":
        baseline = _series(scales, view, random_index, "ratings")
        baseline_error = _series(scales, view, random_index, "stderr")
        axes.fill_between(
            x,
            baseline - baseline_error,
            baseline + baseline_error,
            color=style.INK_MUTED,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        axes.plot(x, baseline, color=style.INK_MUTED, linewidth=1.6, label="random")
        axes.axhline(1000.0, color=style.ORANGE, linewidth=1.1, linestyle=(0, (5, 4)))
    else:
        axes.axhline(0.0, color=style.INK_MUTED, linewidth=1.1, linestyle=(0, (5, 4)))
        axes.axhline(1000.0, color=style.ORANGE, linewidth=1.1, linestyle=(0, (5, 4)))

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
    axes.set_ylabel(ylabel, color=style.INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=style.INK_SECONDARY, fontsize=9, loc="best")
    axes.margins(x=0.05)
    return style.save(figure, path)


def write_scale_plots(result: dict, output_dir: Path) -> list[Path]:
    """Write every anchor view that has at least one completed batch."""
    if not _available(result):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_view(result, view, output_dir / filename, title, ylabel)
        for view, filename, title, ylabel in _VIEWS
    ]
