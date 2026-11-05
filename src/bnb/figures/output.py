"""Writing a figure: the PNG with its provenance in its own text chunks, and the drawing
helpers every figure set shares (the colours, the field map underneath).

The provenance is the capture's (the policy, its environment steps and commit, the
capture's commit and command line) and the render's (the commit and command line that drew
it), as PNG tEXt chunks a viewer or `PIL.Image.open(path).text` shows. There is no manifest.
"""

import json
import pathlib
import sys

import torch

from bnb.contracts.obs import FIELD
from bnb.game import api
from bnb.runs import provenance

# The reference categorical slots (light mode) by role, and the inks.
PREDICTED = "#2a78d6"  # the model
TRUE = "#eb6834"  # the truth
SECOND = "#1baf7a"  # a third series
BASELINE = "#7d7c78"  # a cheap baseline
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# ln n, polarity around 1: a diverging pair with a neutral midpoint.
FIELD_COLOURS = "RdBu_r"


def text_chunks(meta: dict, numbers: dict | None = None) -> dict[str, str]:
    """The PNG text chunks of a figure drawn from a capture with `meta`."""
    head, dirty = provenance.commit()
    chunks = {
        "Source": str(meta.get("policy")),
        "Environment steps": str(meta.get("environment_steps")),
        "Policy commit": str(meta.get("policy_commit")),
        "Capture commit": f"{meta.get('commit')}{' (dirty)' if meta.get('dirty') else ''}",
        "Capture command": str(meta.get("command")),
        "Capture config": json.dumps(meta.get("config")),
        "Commit": f"{head}{' (dirty)' if dirty else ''}",
        "Command": " ".join(["bnb", *sys.argv[1:]]),
        "Time": provenance.now(),
    }
    if numbers:
        chunks["Numbers"] = json.dumps(numbers)
    return chunks


def save(figure, directory: pathlib.Path, name: str, meta: dict, numbers=None) -> pathlib.Path:
    """Write `figure` as directory/name.png with its provenance; closes it."""
    import matplotlib.pyplot as plt

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=130, metadata=text_chunks(meta, numbers), facecolor=SURFACE)
    plt.close(figure)
    return path


def new_figure(rows: int = 1, columns: int = 1, size: tuple[float, float] = (6.0, 4.5), **options):
    """A figure of rows x columns axes in the house style (light surface, recessive grid)."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(size[0] * columns, size[1] * rows),
        squeeze=False,
        facecolor=SURFACE,
        **options,
    )
    for axis in axes.flat:
        style(axis)
    return figure, axes


def style(axis) -> None:
    """Recessive axes: no top or right spine, muted ticks and labels, a light grid."""
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def field_map(
    axis, fields: torch.Tensor, window: tuple[float, float, float, float], points: int = 240
):
    """Draw ln n of the fields (M, len(FIELD_COLUMNS)) over window (x0, x1, y0, y1) m as an
    image under the axis' other marks; returns the image (for a colour bar)."""
    x0, x1, y0, y1 = window
    xs = torch.linspace(x0, x1, points)
    ys = torch.linspace(y0, y1, points)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    fields = fields.float()
    log_index = api.log_index(
        fields[None, :, FIELD["x"]],
        fields[None, :, FIELD["y"]],
        fields[None, :, FIELD["radius"]],
        fields[None, :, FIELD["band_width"]],
        fields[None, :, FIELD["level"]].to(torch.int32),
        grid_x.reshape(1, -1),
        grid_y.reshape(1, -1),
    ).reshape(points, points)
    limit = float(api.LN_SQRT2 * 2.0)
    return axis.imshow(
        log_index.numpy(),
        extent=(x0, x1, y0, y1),
        origin="lower",
        cmap=FIELD_COLOURS,
        vmin=-limit,
        vmax=limit,
        alpha=0.45,
        zorder=0,
        interpolation="bilinear",
    )
