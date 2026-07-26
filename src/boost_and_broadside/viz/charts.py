"""Figure builders that turn W&B-format history into paper-quality plots.

One builder covers the README's needs: ``trend`` plots any metric over training —
a single series in the accent hue (the emphasis form, no legend, the title names
it) or several in the categorical palette with direct-labelled ends. It reads the
arrays produced by ``viz.history`` and paints with ``viz.style``, so the
in-training and calibrated sources are drawn by the very same code.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter

from boost_and_broadside.viz import style

_X_LABEL = "environment steps (millions)"
_MILLION = 1e6


@dataclass
class Line:
    """One series to plot: step array (env steps), value array, and its name."""

    x: np.ndarray
    y: np.ndarray
    label: str


def trend(
    lines: list[Line],
    out: Path,
    *,
    title: str,
    subtitle: str = "",
    ylabel: str,
    log_y: bool = False,
    percent: bool = False,
    reference_lines: list[tuple[str, float]] | None = None,
    size: tuple[float, float] = (10.0, 5.8),
) -> Path:
    """A metric (or a few) over training.

    A single line is the emphasis form — the accent hue, no legend, the title
    names it. Several lines take the categorical palette in fixed order and are
    direct-labelled at their right ends, which carries identity without a legend
    box and satisfies the relief rule for the low-contrast slots.

    ``reference_lines`` are landmark values — a fixed opponent's rating, a target
    — drawn as recessive dashed rules behind the data.
    """
    figure = style.new_figure(size)
    axes = figure.add_subplot(111)
    style.style_axes(axes, title, subtitle)
    if reference_lines:
        style.draw_reference_lines(axes, reference_lines)

    multi = len(lines) > 1
    ends: list[tuple[float, float, str, str]] = []
    for index, line in enumerate(lines):
        if line.x.size == 0:
            continue
        color = style.CATEGORICAL[index % len(style.CATEGORICAL)] if multi else style.BLUE
        x = line.x / _MILLION
        axes.plot(x, line.y, color=color, linewidth=2.0, zorder=3 + index)
        if multi:
            ends.append((x[-1], float(line.y[-1]), color, line.label))

    axes.set_ylabel(ylabel, color=style.INK_SECONDARY, fontsize=10)
    axes.set_xlabel(_X_LABEL, color=style.INK_SECONDARY, fontsize=10)
    if percent:
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if log_y:
        axes.set_yscale("log")
    axes.margins(x=0.14 if multi else 0.02)

    # Place the direct labels last, and only after a draw has settled the scale
    # and limits — label_series_ends maps data to axes-fraction, which is wrong
    # if the y-scale (e.g. log) or the autoscaled limits are not yet final.
    if multi:
        figure.canvas.draw()
        style.label_series_ends(axes, ends)
    return style.save(figure, out)
