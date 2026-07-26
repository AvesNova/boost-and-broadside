"""Figure builders that turn W&B-format history into paper-quality plots.

Two builders cover the README's needs: ``trend`` for any metric over training
(one series in the accent hue, or several in the categorical palette with
direct-labelled ends), and ``elo_curve`` for the headline rating figure, which
overlays the calibrated curve on the in-training one and marks the frozen ladder.

Both read the arrays produced by ``viz.history`` and paint with ``viz.style``, so
the in-training and calibrated sources are drawn by the very same code.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter

from boost_and_broadside.viz import history, style

_X_LABEL = "environment steps (millions)"
_MILLION = 1e6


@dataclass
class Line:
    """One series to plot: step array (env steps), value array, and its name."""

    x: np.ndarray
    y: np.ndarray
    label: str


def _finish(axes, out: Path, figure, *, percent: bool, log_y: bool) -> Path:
    axes.set_xlabel(_X_LABEL, color=style.INK_SECONDARY, fontsize=10)
    if percent:
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if log_y:
        axes.set_yscale("log")
    return style.save(figure, out)


def trend(
    lines: list[Line],
    out: Path,
    *,
    title: str,
    subtitle: str = "",
    ylabel: str,
    log_y: bool = False,
    percent: bool = False,
    size: tuple[float, float] = (10.0, 5.8),
) -> Path:
    """A metric (or a few) over training.

    A single line is the emphasis form — the accent hue, no legend, the title
    names it. Several lines take the categorical palette in fixed order and are
    direct-labelled at their right ends, which carries identity without a legend
    box and satisfies the relief rule for the low-contrast slots.
    """
    figure = style.new_figure(size)
    axes = figure.add_subplot(111)
    style.style_axes(axes, title, subtitle)

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
    if multi:
        style.label_series_ends(axes, ends)
        axes.margins(x=0.14)  # room for the end labels
    else:
        axes.margins(x=0.02)

    axes.set_ylabel(ylabel, color=style.INK_SECONDARY, fontsize=10)
    return _finish(axes, out, figure, percent=percent, log_y=log_y)


def elo_curve(
    out: Path,
    *,
    calib_rows: list[dict],
    calib_summary: dict,
    wandb_rows: list[dict] | None = None,
    run: str = "",
) -> Path:
    """The headline ELO figure: calibrated rating over training, in context.

    The calibrated live curve is the subject — accent hue with a ±1 SE band — and
    the in-training estimate rides behind it in muted ink as context. The frozen
    ladder's calibrated checkpoints are scattered as points, and the scripted
    agent (the one opponent shared across runs) is a dashed landmark, so the
    reader can place where the policy actually got to.
    """
    figure = style.new_figure((11.0, 6.4))
    axes = figure.add_subplot(111)
    subtitle = "Each update's rating refit against post-hoc measured opponents"
    style.style_axes(axes, f"ELO over training  —  {run}" if run else "ELO over training", subtitle)

    scripted = calib_summary.get("elo/scripted")
    if history._finite(scripted):
        style.draw_reference_lines(axes, [("scripted agent", float(scripted))])

    ends: list[tuple[float, float, str, str]] = []

    # In-training estimate — context, behind everything, in muted ink.
    if wandb_rows is not None:
        wx, wy = history.series(wandb_rows, "ladder/elo/live")
        if wx.size:
            wx = wx / _MILLION
            axes.plot(wx, wy, color=style.INK_MUTED, linewidth=1.8, zorder=2)
            ends.append((wx[-1], float(wy[-1]), style.INK_MUTED, "in-training"))

    # Calibrated live curve — the subject, with its ±1 SE band.
    bx, by, lo, hi = history.series_band(
        calib_rows, "ladder/elo/live", "ladder/elo/live_stderr"
    )
    if bx.size:
        bx = bx / _MILLION
        axes.fill_between(bx, lo, hi, color=style.BLUE, alpha=0.16, linewidth=0, zorder=3)
        axes.plot(bx, by, color=style.BLUE, linewidth=2.2, zorder=5)
        ends.append((bx[-1], float(by[-1]), style.BLUE, "calibrated"))

    # Frozen ladder checkpoints — calibrated ratings as points with error bars.
    steps, ratings, errs = history.checkpoint_points(calib_summary)
    if steps.size:
        axes.errorbar(
            steps / _MILLION, ratings, yerr=np.where(np.isfinite(errs), errs, 0.0),
            fmt="o", markersize=6.5, color=style.SHADE_DARK, ecolor=style.SHADE_DARK,
            elinewidth=1.4, capsize=3.0, markeredgecolor=style.SURFACE, markeredgewidth=1.5,
            zorder=6, label="ladder checkpoints",
        )

    style.label_series_ends(axes, ends)
    axes.set_ylabel("ELO vs random anchor", color=style.INK_SECONDARY, fontsize=10)
    axes.set_xlabel(_X_LABEL, color=style.INK_SECONDARY, fontsize=10)
    if steps.size:
        axes.legend(frameon=False, labelcolor=style.INK_SECONDARY, fontsize=9.5, loc="lower right")
    axes.margins(x=0.1)
    return style.save(figure, out)
