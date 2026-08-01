"""Render the policy-vs-scripted crossover figures from crossover.json.

Two views of the same data (produced by ``--mode crossover``):
  phase   — a square phase diagram: policy-controlled ships on x, scripted-controlled
            ships on y, with the win/lose boundary against the equal-count parity line.
  ratio   — scripted-controlled ships beaten per policy-controlled ship.

Usage:
    uv run scripts/render_crossover.py            # docs/crossover -> docs/results
"""

import argparse
import json
from pathlib import Path

import numpy as np

from boost_and_broadside.viz import style


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [r for r in json.loads(path.read_text())["rows"] if r["beats_up_to"] is not None]
    rows.sort(key=lambda r: r["trained"])
    trained = np.array([r["trained"] for r in rows], dtype=float)
    beats = np.array([r["beats_up_to"] for r in rows], dtype=float)
    crossover = np.array([r["crossover"] for r in rows], dtype=float)
    return trained, beats, crossover


def phase_diagram(trained, beats, crossover, out: Path) -> Path:
    """Scripted-count phase boundary with equal data scales and a 45-degree parity line."""
    figure = style.new_figure((8.4, 8.4))
    axes = figure.add_subplot(111)
    style.style_axes(axes, "Zero-shot crossover against the scripted controller")

    axis_max = 64
    # The 50% boundary sits between "still beaten" and the crossover; draw it there.
    boundary = (beats + crossover) / 2.0

    axes.fill_between(trained, 0, boundary, color=style.BLUE, alpha=0.14, linewidth=0)
    axes.fill_between(trained, boundary, axis_max, color=style.ORANGE, alpha=0.14, linewidth=0)
    axes.plot(trained, boundary, color=style.SHADE_DARK, linewidth=2.2, zorder=4)
    axes.plot(
        [0, axis_max],
        [0, axis_max],
        color=style.INK_MUTED,
        linewidth=1.3,
        linestyle=(0, (5, 4)),
        zorder=3,
    )
    axes.annotate(
        "equal ship counts (1:1)",
        xy=(40, 40),
        color=style.INK_MUTED,
        fontsize=9,
        ha="center",
        va="center",
        rotation=45,
        rotation_mode="anchor",
        zorder=5,
        bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 1.0, "alpha": 0.8},
    )

    axes.annotate(
        "Policy-controlled team wins",
        xy=(40, 23),
        color=style.SHADE_DARK,
        fontsize=12,
        fontweight="semibold",
        ha="center",
    )
    axes.annotate(
        "Scripted-controlled team wins",
        xy=(19, 48),
        color=style.ORANGE,
        fontsize=12,
        fontweight="semibold",
        ha="center",
    )

    ticks = np.arange(0, 61, 10)
    axes.set_xlim(0, axis_max)
    axes.set_ylim(0, axis_max)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("policy-controlled ships", color=style.INK_SECONDARY, fontsize=10)
    axes.set_ylabel("scripted-controlled ships", color=style.INK_SECONDARY, fontsize=10)
    return style.save(figure, out)


def ratio_chart(trained, beats, crossover, out: Path) -> Path:
    """Scripted-controlled ships beaten per policy-controlled ship."""
    figure = style.new_figure((10.0, 5.8))
    axes = figure.add_subplot(111)
    style.style_axes(
        axes,
        "Numerical advantage per policy-controlled ship",
        "Scripted-controlled ships defeated for each policy-controlled ship — the edge peaks for "
        "mid-sized teams (~1.5x), then eases toward ~1.35x",
    )
    ratio = beats / trained

    axes.axhline(1.0, color=style.INK_MUTED, linewidth=1.3, linestyle=(0, (5, 4)), zorder=2)
    axes.annotate("parity (1:1)", xy=(trained.max(), 1.0), xytext=(0, -12),
                  textcoords="offset points", color=style.INK_MUTED, fontsize=9, ha="right")
    axes.fill_between(trained, 1.0, ratio, where=ratio >= 1.0, color=style.BLUE,
                      alpha=0.13, linewidth=0, zorder=2)
    axes.plot(trained, ratio, color=style.BLUE, linewidth=2.0, marker="o", markersize=5,
              markeredgecolor=style.SURFACE, markeredgewidth=1.2, zorder=4)

    axes.set_xlabel("policy-controlled ships", color=style.INK_SECONDARY, fontsize=10)
    axes.set_ylabel(
        "scripted-controlled ships beaten per policy-controlled ship",
        color=style.INK_SECONDARY,
        fontsize=10,
    )
    axes.margins(x=0.03)
    return style.save(figure, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/crossover/crossover.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    trained, beats, crossover = _load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in (
        phase_diagram(trained, beats, crossover, args.out / "crossover_phase.png"),
        ratio_chart(trained, beats, crossover, args.out / "crossover_ratio.png"),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
