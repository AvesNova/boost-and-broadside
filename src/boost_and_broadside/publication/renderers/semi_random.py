"""Plots for the random-to-scripted calibration ladder."""

from pathlib import Path

import numpy as np

from boost_and_broadside.publication.renderer_api import Renderer, RenderInputs, register
from boost_and_broadside.viz import style


def _scales(result: dict) -> list[dict]:
    return [
        result["scales"][key]
        for key in sorted(result.get("scales", {}), key=int)
        if result["scales"][key].get("ratings")
    ]


def _write_connectivity_plot(result: dict, output_dir: Path) -> Path:
    figure = style.new_figure((10.0, 5.8))
    axes = figure.add_subplot(111)
    style.style_axes(axes, "Reference-ladder connectivity")
    probabilities = np.asarray(result["probabilities"], dtype=float)
    x = 100 * probabilities[1:]

    axes.axhspan(0.2, 0.8, color=style.AQUA, alpha=0.08, linewidth=0)
    axes.axhline(0.5, color=style.BASELINE, linestyle="--", linewidth=1.2)
    for index, scale in enumerate(_scales(result)):
        scores = np.asarray(
            [row["higher_expected_score"] for row in scale["adjacent_matchups"]],
            dtype=float,
        )
        color = style.CATEGORICAL[index % len(style.CATEGORICAL)]
        axes.plot(
            x,
            scores,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=f"{scale['team_size']}v{scale['team_size']}",
        )

    axes.set_xlabel("Scripted-action probability of higher rung (%)", color=style.INK_SECONDARY)
    axes.set_ylabel("Expected score against previous rung", color=style.INK_SECONDARY)
    axes.set_xlim(float(x.min()), 100)
    axes.set_ylim(0, 1.02)
    axes.legend(frameon=False, ncol=2, labelcolor=style.INK_SECONDARY)
    path = output_dir / "semi_random_connectivity.png"
    return style.save(figure, path)


def write_plots(result: dict, output_dir: Path) -> list[Path]:
    """Render the reference-ladder connectivity diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not _scales(result):
        return []
    return [_write_connectivity_plot(result, output_dir)]


def _render(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    return write_plots(inputs.artifact("ladder").read_json(), out_dir)


register(
    Renderer(
        name="semi-random-connectivity-v1",
        description="How informative each step of the random-to-scripted ladder is.",
        render=_render,
        required_artifacts=("ladder",),
        supported_schemas={"ladder": (1,)},
    )
)
