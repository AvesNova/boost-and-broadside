"""Game length per mode: one density curve per mode (a Gaussian kernel estimate, Silverman's
bandwidth, reflected at zero and at the game clock), each mode's median marked. The
labels are long (they carry the clock-out share), so they go in a legend.

A game that runs the clock out ends at the clock, a point mass a kernel would smear: those
games are left out of the curve, which integrates to the share of games decided before the
clock, and their share is printed in the mode's label.

**Input** (`read_lengths`): JSONL files, one game per line {mode or preset, seconds,
clock_seconds?} (the game's length and its clock, in game seconds), or one line per mode
{mode or preset, lengths: [seconds, ...], clock_seconds?}. A game at or past its clock
(less `CLOCK_TOLERANCE`) ran the clock out.
"""

import json
import math
import pathlib

import numpy as np

from .. import theme
from . import common

CLOCK_TOLERANCE = 0.5  # seconds: a game within this of its clock ran it out
GRID_POINTS = 512


def read_lengths(paths: list[str | pathlib.Path]) -> dict[str, dict]:
    """{mode: {"seconds": [...], "clock": the longest clock given or None}}."""
    found: dict[str, dict] = {}
    for path in map(pathlib.Path, paths):
        for text in path.read_text().splitlines():
            if not text.strip():
                continue
            line = json.loads(text)
            mode = common.mode_of(line.get("mode") or line["preset"])
            held = found.setdefault(mode, dict(seconds=[], clock=None))
            lengths = line["lengths"] if "lengths" in line else [line["seconds"]]
            held["seconds"] += [float(value) for value in lengths]
            if line.get("clock_seconds") is not None:
                held["clock"] = max(held["clock"] or 0.0, float(line["clock_seconds"]))
    return found


def density(values: np.ndarray, grid: np.ndarray, upper: float | None) -> np.ndarray:
    """The kernel density of `values` on `grid`, reflected at 0 (and at `upper`)."""
    spread = min(values.std(ddof=1), np.subtract(*np.percentile(values, [75, 25])) / 1.34)
    spread = spread if spread > 0 else max(values.std(ddof=1), 1.0)
    bandwidth = 0.9 * spread * len(values) ** -0.2
    mirrored = [values, -values] + ([2 * upper - values] if upper is not None else [])
    centres = np.concatenate(mirrored)
    kernel = np.exp(-0.5 * ((grid[:, None] - centres[None, :]) / bandwidth) ** 2)
    return kernel.sum(1) / (len(values) * bandwidth * math.sqrt(2 * math.pi))


def draw(
    paths: list[str | pathlib.Path],
    out: str | pathlib.Path,
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "game_length",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the densities of the games in `paths` into `out`; (the files written, the
    values: per mode the median, the quartiles, the clock-out share and the games)."""
    found = read_lengths(paths)
    modes = [m for m in common.MODES if m in found] + sorted(set(found) - set(common.MODES))
    if not modes:
        raise SystemExit(f"no games in {list(map(str, paths))}")
    longest = max(max(found[m]["seconds"]) for m in modes)
    clocks = [found[m]["clock"] for m in modes if found[m]["clock"]]
    right = max([longest, *clocks]) * 1.02
    figure, axes = common.new_figure(1, 1, size=(6.4, 3.6))
    axis = axes[0, 0]
    values: dict[str, float] = {}
    for mode in modes:
        seconds = np.asarray(found[mode]["seconds"])
        clock = found[mode]["clock"]
        ran_out = seconds >= clock - CLOCK_TOLERANCE if clock else np.zeros_like(seconds, bool)
        decided = seconds[~ran_out]
        colour = common.mode_colour(mode)
        share = float(ran_out.mean())
        label = mode + (f" ({share:.0%} ran the clock out)" if clock else "")
        if len(decided) >= 2:
            grid = np.linspace(0.0, clock if clock else right, GRID_POINTS)
            curve = density(decided, grid, clock) * (1.0 - share)
            axis.plot(grid, curve, color=colour, linewidth=1.6, label=label)
            axis.fill_between(grid, curve, color=colour, alpha=0.08, linewidth=0)
            median = float(np.median(decided))
            axis.axvline(median, color=colour, linewidth=0.8, linestyle=":")
        if clock:
            axis.axvline(clock, color=theme.MUTED, linewidth=0.8, linestyle="--")
        key = f"game_length_{common.slug(mode)}"
        quartiles = np.percentile(seconds, [25, 50, 75])
        values[f"{key}_median_seconds"] = float(quartiles[1])
        values[f"{key}_lower_quartile_seconds"] = float(quartiles[0])
        values[f"{key}_upper_quartile_seconds"] = float(quartiles[2])
        values[f"{key}_games"] = float(len(seconds))
        if clock:
            values[f"{key}_clock_out_share"] = share
    axis.set_xlim(0.0, right)
    axis.set_ylim(bottom=0.0)
    axis.legend(fontsize=7, frameon=False, loc="upper right")
    axis.set_xlabel("game length (game seconds); dotted: median, dashed: the clock", fontsize=8)
    axis.set_ylabel("density (per second)", fontsize=8)
    common.title(figure, "Game length per mode")
    figure.tight_layout(rect=(0, 0, 1, 1))
    written = common.save(figure, out, name, list(map(str, paths)), values, formats)
    return written, values
