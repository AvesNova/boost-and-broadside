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

from . import theme

# ln n, polarity around 1: a diverging pair with a neutral midpoint.
FIELD_COLOURS = theme.DIVERGING


def __getattr__(name: str):
    """The colour roles (PREDICTED, TRUE, SECOND, BASELINE, INK, MUTED, GRID, SURFACE) in
    the active theme variant (`bnb.figures.theme`), read when used."""
    return getattr(theme, name)


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
    figure.savefig(
        path, dpi=130, metadata=text_chunks(meta, numbers), facecolor=theme.chrome("surface")
    )
    plt.close(figure)
    return path


def save_formats(
    figure,
    directory: pathlib.Path,
    name: str,
    meta: dict,
    numbers=None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 130,
    facecolor: str | None = None,
) -> list[pathlib.Path]:
    """Write `figure` as directory/name.<format> for each format (png, svg, pdf) at `dpi`
    on `facecolor` (None: the theme's surface), the PNG with its provenance in its text
    chunks; closes it."""
    import matplotlib.pyplot as plt

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in formats:
        path = directory / f"{name}.{kind}"
        metadata = text_chunks(meta, numbers) if kind == "png" else None
        figure.savefig(
            path, dpi=dpi, metadata=metadata, facecolor=facecolor or theme.chrome("surface")
        )
        paths.append(path)
    plt.close(figure)
    return paths


def new_figure(rows: int = 1, columns: int = 1, size: tuple[float, float] = (6.0, 4.5), **options):
    """A figure of rows x columns axes in the house style (the theme's surface, a recessive
    grid)."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(size[0] * columns, size[1] * rows),
        squeeze=False,
        facecolor=theme.chrome("surface"),
        **options,
    )
    for axis in axes.flat:
        style(axis)
    return figure, axes


def style(axis) -> None:
    """Recessive axes in the active theme variant (`bnb.figures.theme.style`)."""
    theme.style(axis)


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
        fields[None, :, FIELD["present"]] > 0.5,
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
