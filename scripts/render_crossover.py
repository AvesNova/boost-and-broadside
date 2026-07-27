"""Render the trained-vs-scripted crossover figures from crossover.json.

Two views of the same data (produced by ``--mode crossover``):
  phase   — a phase diagram: scripted count on y, trained on x, the win/lose
            boundary drawn and its two regions shaded, against the y=x unity line.
  ratio   — scripted agents beaten per trained agent, against the 1:1 parity line.

Usage:
    uv run --no-sync scripts/render_crossover.py            # docs/crossover -> docs/results
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
    """Scripted-count phase boundary with shaded win regions and the unity line."""
    figure = style.new_figure((9.0, 7.2))
    axes = figure.add_subplot(111)
    style.style_axes(
        axes,
        "How many scripted agents does it take to win?",
        "The boundary is the largest scripted team each trained team still beats "
        "(>50% of games)",
    )

    x_max = trained.max()
    y_max = crossover.max() * 1.06
    # The 50% boundary sits between "still beaten" and the crossover; draw it there.
    boundary = (beats + crossover) / 2.0

    axes.fill_between(trained, 0, boundary, color=style.BLUE, alpha=0.14, linewidth=0)
    axes.fill_between(trained, boundary, y_max, color=style.ORANGE, alpha=0.14, linewidth=0)
    axes.plot(trained, boundary, color=style.SHADE_DARK, linewidth=2.2, zorder=4)
    axes.plot([0, x_max], [0, x_max], color=style.INK_MUTED, linewidth=1.3,
              linestyle=(0, (5, 4)), zorder=3)
    axes.annotate("equal numbers (1:1)", xy=(x_max, x_max), xytext=(-6, 8),
                  textcoords="offset points", color=style.INK_MUTED, fontsize=9,
                  ha="right", va="bottom", rotation=0)

    axes.annotate("Trained team wins", xy=(0.62 * x_max, 0.30 * y_max), color=style.SHADE_DARK,
                  fontsize=13, fontweight="semibold", ha="center")
    axes.annotate("Scripted team wins", xy=(0.28 * x_max, 0.86 * y_max), color=style.ORANGE,
                  fontsize=13, fontweight="semibold", ha="center")

    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("trained agents", color=style.INK_SECONDARY, fontsize=10)
    axes.set_ylabel("scripted agents", color=style.INK_SECONDARY, fontsize=10)
    return style.save(figure, out)


def ratio_chart(trained, beats, crossover, out: Path) -> Path:
    """Scripted agents beaten per trained agent, against the 1:1 parity line."""
    figure = style.new_figure((10.0, 5.8))
    axes = figure.add_subplot(111)
    style.style_axes(
        axes,
        "Advantage per trained agent",
        "Scripted agents defeated for each trained agent — the edge peaks for "
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

    axes.set_xlabel("trained agents", color=style.INK_SECONDARY, fontsize=10)
    axes.set_ylabel("scripted beaten per trained agent", color=style.INK_SECONDARY, fontsize=10)
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
