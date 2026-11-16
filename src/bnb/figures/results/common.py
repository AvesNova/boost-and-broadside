"""What every results chart shares: the fixed colours of the modes and the ratios, direct
labels, the writer (a PNG with its provenance in its text chunks, plus a vector copy for the
paper) and the values file.

The values file (`docs/results/values.json` by default) is one flat JSON object of named
numbers, {name: number}, that a docs page quotes as `{{name}}` (or `{{pending: name}}` before
the number exists). Every chart merges the numbers it shows into it (`write_values`): the
names it writes replace the same names, every other name stays. The names each chart
writes are listed in docs/guides/results-charts.md.
"""

import json
import math
import pathlib
import re
import sys

from bnb.runs import provenance

from ..ladder import MODES, mode_of
from ..output import BASELINE, INK, MUTED, PREDICTED, SECOND, SURFACE, TRUE, new_figure

__all__ = ["MODES", "mode_of", "new_figure"]

VALUES = pathlib.Path("docs/results/values.json")
FORMATS = ("png", "pdf")  # the PNG for the docs, the PDF for the paper
# The modes' colours, in this order everywhere (a fixed categorical order).
MODE_COLOURS = {"frontline": PREDICTED, "deathmatch": TRUE, "elimination": SECOND}
# The ratios' ramp: an ordered quantity, so a perceptual sequential map (viridis), sampled
# short of its pale end so the lightest line still reads on the light surface.
RATIO_MAP = "viridis"
RATIO_RANGE = (0.0, 0.82)
DIRECT_LABELS = 4  # at most this many lines are labelled at their ends; more get a legend


def mode_colour(mode: str) -> str:
    """A mode's colour (a preset's mode's), grey for anything else."""
    return MODE_COLOURS.get(mode_of(mode), BASELINE)


def ordered_colours(count: int) -> list:
    """`count` colours from the ratio ramp, lowest first (the same ramp, whatever the
    count: one colour is the ramp's start)."""
    import matplotlib

    ramp = matplotlib.colormaps[RATIO_MAP]
    low, high = RATIO_RANGE
    if count == 1:
        return [ramp(low)]
    return [ramp(low + (high - low) * index / (count - 1)) for index in range(count)]


def label_ends(axis, lines: list[tuple[str, list[float], list[float], object]]) -> bool:
    """Label each line (label, xs, ys, colour) at its last point when there are at most
    `DIRECT_LABELS`, nudging labels apart; else draw a legend. Returns whether it labelled
    directly."""
    lines = [line for line in lines if line[1]]
    if not lines:
        return True
    if len(lines) > DIRECT_LABELS:
        axis.legend(fontsize=7, frameon=False)
        return False
    left, right = axis.get_xlim()
    axis.set_xlim(left, right + 0.18 * (right - left))  # room for the labels
    to_axes = axis.transAxes.inverted()
    gap = 0.08  # in axes fraction, so a log axis spaces them evenly too
    ends = sorted(
        (to_axes.transform(axis.transData.transform((xs[-1], ys[-1])))[1], label, xs[-1], colour)
        for label, xs, ys, colour in lines
    )
    placed: list[float] = []
    for y, *_ in ends:  # upward, a gap apart
        placed.append(max(y, placed[-1] + gap) if placed else y)
    excess = placed[-1] - (1.0 - 0.5 * gap)
    if excess > 0:  # past the top: shift down, keeping the gaps, as far as the bottom allows
        placed = [max(y - excess, 0.5 * gap + index * gap) for index, y in enumerate(placed)]
    for y, (_, label, x, colour) in zip(placed, ends, strict=True):
        axis.annotate(
            label,
            (x, y),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            color=colour,
            fontsize=7,
            va="center",
            annotation_clip=False,
        )
    return True


def title(figure, text: str) -> None:
    """The figure's title, in the house ink."""
    figure.suptitle(text, fontsize=10, color=INK)


def note(axis, text: str) -> None:
    """A muted note in the axis' corner (what is missing, a bound)."""
    axis.text(0.02, 0.02, text, transform=axis.transAxes, fontsize=7, color=MUTED, va="bottom")


def save(
    figure,
    directory: str | pathlib.Path,
    name: str,
    sources: list[str],
    numbers: dict | None = None,
    formats: tuple[str, ...] = FORMATS,
) -> list[pathlib.Path]:
    """Write `figure` as directory/name.<format> for each format: the PNG with its
    provenance (the render's commit, command line and time, the sources read, the numbers
    shown) in its text chunks, a PDF or SVG with the commit and sources in its metadata.
    Closes the figure; returns the paths."""
    import matplotlib.pyplot as plt

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    head, dirty = provenance.commit()
    commit = f"{head}{' (dirty)' if dirty else ''}"
    command = " ".join(["bnb", *sys.argv[1:]])
    written = []
    for kind in formats:
        path = directory / f"{name}.{kind}"
        if kind == "png":
            metadata = {
                "Commit": commit,
                "Command": command,
                "Time": provenance.now(),
                "Sources": json.dumps(sources),
            }
            if numbers:
                metadata["Numbers"] = json.dumps(numbers)
        elif kind == "pdf":
            metadata = {"Creator": f"{command} at {commit}", "Subject": json.dumps(sources)}
        elif kind == "svg":
            metadata = {"Creator": f"{command} at {commit}", "Source": json.dumps(sources)}
        else:
            raise ValueError(f"format {kind!r}: png, pdf or svg")
        figure.savefig(path, dpi=150, facecolor=SURFACE, metadata=metadata)
        written.append(path)
    plt.close(figure)
    return written


def slug(text) -> str:
    """A value name's part: lower case, runs of other characters as one underscore, a
    number without a trailing .0 (1.5 -> 1_5, 2.0 -> 2)."""
    if isinstance(text, float) and text.is_integer():
        text = int(text)
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def write_values(values: dict[str, float | None], path: str | pathlib.Path = VALUES) -> None:
    """Merge `values` into the values file (`path`): each name replaced, the rest kept, the
    names sorted; a value that is not a finite number is written as null (a pending
    number). Written atomically."""
    path = pathlib.Path(path)
    held = json.loads(path.read_text()) if path.exists() else {}
    for name, value in values.items():
        finite = value is not None and math.isfinite(float(value))
        held[name] = float(value) if finite else None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(sorted(held.items())), indent=1) + "\n")
    temporary.replace(path)
