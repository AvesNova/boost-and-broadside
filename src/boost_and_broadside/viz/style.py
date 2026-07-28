"""The house style for every figure — one palette, one set of chrome.

Colours come from the validated categorical palette (see the data-viz skill's
`references/palette.md`). The eight hues are fixed in order; that ordering is the
colour-vision-safety mechanism, not decoration — it clears the adjacent-pair CVD
and contrast gates on the light paper surface. Three of the slots sit below 3:1
against the surface, so any chart that seats four or more series must direct-label
them (the "relief" rule) rather than lean on colour alone.

Everything a figure needs — surface, ink, grid, the palette, and the axis/label
helpers — lives here so the look is defined in exactly one place. Both the
calibration-mode diagnostics and the README figures import it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a training container
import matplotlib.pyplot as plt  # noqa: E402

# --- Surfaces and ink (paper/light) ----------------------------------------
SURFACE = "#fcfcfb"  # chart surface
PAGE = "#f9f9f7"  # page plane behind the axes
INK = "#0b0b0b"  # primary ink — titles
INK_SECONDARY = "#52514e"  # axis labels, legends
INK_MUTED = "#898781"  # ticks, landmark labels, de-emphasised marks
GRID = "#e1e0d9"  # hairline gridlines
BASELINE = "#c3c2b7"  # baseline / axis spines

# --- Categorical palette (fixed order; do not cycle or re-order) ------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)

# Single-hue sequential shades (blue ramp), for before/after marks and points.
SHADE_LIGHT = "#86b6ef"  # step 250
SHADE_DARK = "#184f95"  # step 600

# Semantic roles for the calibration diagnostics, mapped onto palette hues so
# those plots share this one source of truth.
TRAINING = BLUE  # the in-training estimate
CALIBRATED = GREEN  # the post-hoc calibrated rating
AVG = HALF_WIN = MAGENTA  # averaged policy / secondary draw convention

# A rating this uncertain is not a measurement — a +/-150 band spans a quarter of
# the ladder. Such points stay in the data with their error and are simply not
# drawn (they arise early, on a game or two, and would stretch the axis wildly).
MAX_PLOT_STDERR = 150.0


def new_figure(size=(11.0, 6.0)):
    """A figure on the paper surface at print resolution."""
    return plt.figure(figsize=size, dpi=160, facecolor=SURFACE)


def style_axes(axes, title: str = "", subtitle: str = "") -> None:
    """Recessive grid and axes; titles in ink, never in a series colour."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if title:
        # Pad leaves room for the subtitle, offset in points rather than axes
        # fractions so the gap does not shrink as the panel gets shorter.
        axes.set_title(
            title, color=INK, fontsize=13, fontweight="semibold", loc="left",
            pad=30 if subtitle else 12,
        )
    if subtitle:
        axes.annotate(
            subtitle, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 7),
            textcoords="offset points", color=INK_MUTED, fontsize=9.5, va="bottom",
        )


def label_series_ends(axes, entries: list[tuple[float, float, str, str]]) -> None:
    """Direct-label each series past its right end, nudging apart any collisions.

    Curves that converge would otherwise stack their labels and render both
    unreadable. The de-collision runs in axes-fraction space, so it spaces labels
    by how far apart they *look* — which stays correct on a log axis, where equal
    pixel gaps span very different data distances. Call after all series are drawn
    so the limits are settled. Each entry is (x, y, colour, text).
    """
    if not entries:
        return
    to_fraction = (axes.transData + axes.transAxes.inverted()).transform
    converted = [(x, *to_fraction((x, y)), color, text) for x, y, color, text in entries]
    minimum = 0.034  # a labelline-height gap, in fraction of the panel
    placed: list[float] = []
    for x, _fx, fy, color, text in sorted(converted, key=lambda entry: entry[2]):
        target = fy if not placed else max(fy, placed[-1] + minimum)
        placed.append(target)
        axes.annotate(
            text, (x, target), xycoords=("data", "axes fraction"),
            xytext=(7, 0), textcoords="offset points",
            color=color, fontsize=9.5, va="center", fontweight="medium",
        )


def draw_reference_lines(axes, lines: list[tuple[str, float]]) -> None:
    """Landmark values as recessive dashed rules, right-labelled on a surface plate.

    A landmark spans the full width, so no anchor position is reliably clear of
    the data; the label sits on a surface-coloured plate to stay legible over
    whatever it lands on.
    """
    for name, value in lines:
        axes.axhline(value, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
        axes.annotate(
            f"{name}  {value:.0f}",
            xy=(0.992, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=9,
            ha="right",
            va="bottom",
            zorder=6,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        )


def save(figure, path: Path, tight: bool = True) -> Path:
    """Write a figure to disk on the paper surface.

    ``tight`` runs ``tight_layout`` for a single-panel figure; a multi-panel grid
    that manages its own spacing (e.g. via a constrained layout) passes
    ``tight=False`` so the two do not fight.
    """
    if tight:
        figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path
