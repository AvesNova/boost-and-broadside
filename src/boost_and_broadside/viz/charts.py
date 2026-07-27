"""Figure builders that turn W&B-format history into paper-quality plots.

``trend`` plots one metric over training as a standalone figure; ``grid`` packs
several such panels into one figure (a 2xN quad). Both share ``_draw_panel`` — the
same per-axes drawing — so a metric looks identical whether it stands alone or
sits in a grid. They read the arrays from ``viz.history`` and paint with
``viz.style``, so in-training and calibrated sources are drawn by the same code.
"""

import math
from dataclasses import dataclass, field
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


@dataclass
class Panel:
    """One metric's worth of a grid: its lines and how to scale/label them."""

    lines: list[Line]
    title: str
    ylabel: str
    log_y: bool = False
    percent: bool = False
    reference_lines: list[tuple[str, float]] = field(default_factory=list)


def _draw_panel(
    axes,
    lines: list[Line],
    *,
    title: str,
    ylabel: str,
    subtitle: str = "",
    log_y: bool = False,
    percent: bool = False,
    reference_lines: list[tuple[str, float]] | None = None,
    xlabel: bool = True,
) -> list[tuple[float, float, str, str]]:
    """Draw one metric onto ``axes``; return its right-end label entries.

    A single line is the emphasis form (the accent hue). Several take the
    categorical palette in fixed order and are direct-labelled at their ends —
    the caller places those labels after a draw, once the scale is final. Landmark
    ``reference_lines`` sit behind the data as recessive dashed rules.
    """
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
    if xlabel:
        axes.set_xlabel(_X_LABEL, color=style.INK_SECONDARY, fontsize=10)
    if percent:
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if log_y:
        axes.set_yscale("log")
    axes.margins(x=0.14 if multi else 0.02)
    return ends


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
    """One metric over training as a standalone figure."""
    figure = style.new_figure(size)
    axes = figure.add_subplot(111)
    ends = _draw_panel(
        axes, lines, title=title, subtitle=subtitle, ylabel=ylabel,
        log_y=log_y, percent=percent, reference_lines=reference_lines,
    )
    # Place direct labels last, after a draw settles the scale and limits —
    # label_series_ends maps data to axes-fraction, wrong if the y-scale (log) or
    # autoscaled limits are not yet final.
    if ends:
        figure.canvas.draw()
        style.label_series_ends(axes, ends)
    return style.save(figure, out)


def grid(
    panels: list[Panel],
    out: Path,
    *,
    ncols: int = 2,
    size: tuple[float, float] = (12.0, 8.4),
) -> Path:
    """Several metrics as panels in one figure — a 2xN quad.

    Each panel is drawn by the same ``_draw_panel`` a standalone ``trend`` uses,
    so a metric reads the same either way. Only the bottom row carries the shared
    x-axis label, keeping the interior uncluttered.
    """
    nrows = math.ceil(len(panels) / ncols)
    figure = style.new_figure(size)
    # A constrained layout spaces the panels (titles, ticks, labels) without the
    # tight_layout/annotation conflict; save() then skips tight_layout.
    figure.set_layout_engine("constrained")
    spec = figure.add_gridspec(nrows, ncols)

    deferred: list[tuple] = []
    for index, panel in enumerate(panels):
        axes = figure.add_subplot(spec[divmod(index, ncols)])
        on_bottom_row = index >= len(panels) - ncols
        ends = _draw_panel(
            axes, panel.lines, title=panel.title, ylabel=panel.ylabel,
            log_y=panel.log_y, percent=panel.percent,
            reference_lines=panel.reference_lines, xlabel=on_bottom_row,
        )
        if ends:
            deferred.append((axes, ends))

    figure.canvas.draw()  # settle every panel's scale/limits before labelling
    for axes, ends in deferred:
        style.label_series_ends(axes, ends)
    return style.save(figure, out, tight=False)
