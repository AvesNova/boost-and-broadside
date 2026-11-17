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

from .. import captions, theme
from ..ladder import MODES, mode_of
from ..output import new_figure

__all__ = ["MODES", "mode_of", "new_figure"]

VALUES = pathlib.Path("docs/results/values.json")
FORMATS = ("png", "pdf", "svg")  # the PNG for the docs, the PDF and SVG for the paper
DIRECT_LABELS = 4  # at most this many lines are labelled at their ends; more get a legend


def mode_colour(mode: str) -> str:
    """A mode's colour (a preset's mode's) in the active theme variant, grey for anything
    else (`bnb.figures.theme.mode_colour`)."""
    return theme.mode_colour(mode)


def ordered_colours(count: int) -> list:
    """`count` colours from the theme's sequential ramp (an ordered quantity: the ratios),
    lowest first (`bnb.figures.theme.sequential`)."""
    return theme.sequential(count)


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


def title(figure, name: str, text: str | None = None) -> None:
    """The figure's short title, in the house ink: the captions file's title for the
    figure `name` (`bnb.figures.captions`), or `text` when given. The details go in the
    figure's caption, not the title. The web PNG draws it; the paper's PDF and SVG do not
    (`save`)."""
    figure.suptitle(text or captions.title(name), fontsize=10, color=theme.INK)


def note(axis, text: str) -> None:
    """A muted note in the axis' corner (what is missing, a bound)."""
    axis.text(
        0.02, 0.02, text, transform=axis.transAxes, fontsize=7, color=theme.MUTED, va="bottom"
    )


def save(
    figure,
    directory: str | pathlib.Path,
    name: str,
    sources: list[str],
    numbers: dict | None = None,
    formats: tuple[str, ...] = FORMATS,
) -> list[pathlib.Path]:
    """Write `figure` for each format: the PNG (the web size, as drawn) as
    directory/name<suffix>.png, the suffix the theme variant's (`theme.suffix`: none for
    light, "_dark" for dark), with its provenance (the render's commit, command line and
    time, the sources read, the numbers shown) in its text chunks; in the light variant
    only, the PDF and SVG for the paper, fitted to the page width (`theme.fit`), as
    directory/name.<format> with the commit and sources in their metadata and without the
    drawn title (the paper's caption carries it; the title goes in the metadata). Closes
    the figure; returns the paths."""
    import matplotlib.pyplot as plt

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    head, dirty = provenance.commit()
    commit = f"{head}{' (dirty)' if dirty else ''}"
    command = " ".join(["bnb", *sys.argv[1:]])
    written = []
    unknown = set(formats) - {"png", "pdf", "svg"}
    if unknown:
        raise ValueError(f"format {sorted(unknown)}: png, pdf or svg")
    if "png" in formats:
        metadata = {
            "Commit": commit,
            "Command": command,
            "Time": provenance.now(),
            "Sources": json.dumps(sources),
        }
        if numbers:
            metadata["Numbers"] = json.dumps(numbers)
        path = directory / f"{name}{theme.suffix()}.png"
        figure.savefig(path, dpi=theme.WEB_DPI, facecolor=theme.SURFACE, metadata=metadata)
        written.append(path)
    paper = [kind for kind in formats if kind != "png"]
    if paper and theme.active() == "light":
        heading = figure._suptitle.get_text() if figure._suptitle is not None else ""
        if figure._suptitle is not None:  # the paper's figure environment holds the title
            figure._suptitle.set_text("")
            figure._suptitle.set_in_layout(False)
        theme.fit(figure, "page")
        for kind in paper:
            path = directory / f"{name}.{kind}"
            key = "Subject" if kind == "pdf" else "Source"
            metadata = {"Creator": f"{command} at {commit}", key: json.dumps(sources)}
            if heading:
                metadata["Title"] = heading
            figure.savefig(path, facecolor=theme.SURFACE, metadata=metadata)
            written.append(path)
    plt.close(figure)
    return written


def slug(text) -> str:
    """A value name's part: lower case, runs of other characters as one underscore, a
    number without a trailing .0 (1.5 -> 1_5, 2.0 -> 2)."""
    if isinstance(text, float) and text.is_integer():
        text = int(text)
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def write_values(values: dict[str, float | str | None], path: str | pathlib.Path = VALUES) -> None:
    """Merge `values` into the values file (`path`): each name replaced, the rest kept, the
    names sorted; a string is kept as it is, a value that is not a finite number is
    written as null (a pending number). Written atomically."""
    path = pathlib.Path(path)
    held = json.loads(path.read_text()) if path.exists() else {}
    for name, value in values.items():
        if isinstance(value, str):
            held[name] = value
            continue
        finite = value is not None and math.isfinite(float(value))
        held[name] = float(value) if finite else None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(sorted(held.items())), indent=1) + "\n")
    temporary.replace(path)
