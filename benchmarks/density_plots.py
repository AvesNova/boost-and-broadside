"""
SUPERSEDED for the current target. The field panels still render, but the
colour scales, the `log1p units` label and the error metric all assume the
smoothed regression target; the field is now soft ship counts. Re-express
before reading the numbers off it.
Render the hex density head: true field, prediction, error, and where it fails.

Companion to ``density_baselines.py``, which produces the ``.pt`` this reads.
The baselines say *how much* of the field the head explains; these say *what it
gets wrong and where*, which a scalar cannot.

Four figures into ``docs/internal/``:

* ``*-fields.png``   -- per sampled frame, the ally and enemy fields as true /
  predicted / signed error. Drawn as the actual hexagonal cells, not a contour:
  the ring order and cell geometry are part of the target's meaning, and
  interpolating would hide a bad ring or a misbehaving boundary corner.
* ``*-ladder.png``   -- the baseline ladder, per half, as R^2 against the
  constant heat map (the honest denominator: the target's variance, not its
  power).
* ``*-hidden.png``   -- enemy-half error against the number of enemies the
  observer cannot currently see. The sharp test of whether the enemy field is
  belief or bookkeeping.
* ``*-rings.png``    -- error by ring index, centre outward. Says whether the
  boundary cells behave.

    uv run --no-sync python benchmarks/density_plots.py \\
        --samples <path>.pt --out-prefix docs/internal/density-748-sep2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Palette roles, from the data-viz reference instance. Sequential magnitude is
# one hue light->dark; a signed error is diverging with a neutral gray midpoint
# so "no error" reads as nothing. The two categorical slots are validated as a
# pair (adjacent CVD dE 24.7, normal-vision 33.6, both clear).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#ec835a", "#d03b3b", "#8f1f1f"]
SERIES = ("#2a78d6", "#eb6834")  # slot 1 blue, slot 2 orange
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"


def _style(matplotlib):
    """Recessive axes and grid, text in ink tokens rather than series colour."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "grid.color": GRID,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ring_index(rings: int):
    """(C,) ring each cell belongs to, 0 at the centre, in the grid's own order.

    The grid is stored as a flat cell list, so the ring structure has to be
    reconstructed to plot against it. Derived from the ring sizes rather than
    from cell distances, because the two must agree and this is the one the
    target's ordering is actually built from.
    """
    import torch

    index = torch.zeros(1 + 3 * rings * (rings + 1), dtype=torch.long)
    start = 1
    for ring in range(1, rings + 1):
        index[start : start + 6 * ring] = ring
        start += 6 * ring
    return index


def _hex_patches(centres, spacing):
    """One flat-topped hexagon per cell centre, sized to tile the grid."""
    from matplotlib.patches import RegularPolygon

    # Axial spacing between neighbouring centres is `spacing`; a hexagon whose
    # centre-to-vertex radius is spacing/sqrt(3) tiles that lattice exactly.
    return [
        RegularPolygon(
            (float(x), float(y)), numVertices=6, radius=spacing / (3**0.5), orientation=0.0
        )
        for x, y in centres
    ]


def _draw_field(axis, centres, spacing, values, cmap, vmin, vmax, title):
    from matplotlib.collections import PatchCollection

    collection = PatchCollection(_hex_patches(centres, spacing), cmap=cmap, edgecolor="none")
    collection.set_array(values)
    collection.set_clim(vmin, vmax)
    axis.add_collection(collection)
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, color=INK)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap

    from boost_and_broadside.train.rl.hex_density import HEX_DENSITY_RINGS, hex_cell_offsets

    _style(matplotlib)
    sequential = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    diverging = LinearSegmentedColormap.from_list("div", DIVERGING)

    blob = torch.load(args.samples, weights_only=False)
    target, head = blob["target"], blob["head"]
    cells = blob["cells"]
    hidden = blob["hidden_enemies"]
    baselines = blob.get("baselines", {})
    step = blob.get("global_step", 0)
    run = blob.get("run", "?")
    centres = hex_cell_offsets(HEX_DENSITY_RINGS).numpy()
    spacing = 1.0 / HEX_DENSITY_RINGS
    outputs = []

    # ---- 1. fields ----------------------------------------------------------
    # Frames chosen across the hidden-enemy range rather than at random, so the
    # panel shows the easy and the hard case rather than three of whichever is
    # more common.
    order = torch.argsort(hidden)
    picks = [
        order[int(q * (len(order) - 1))] for q in torch.linspace(0.05, 0.95, args.frames).tolist()
    ]
    figure, axes = plt.subplots(2 * args.frames, 3, figsize=(9.5, 6.2 * args.frames))
    for frame, index in enumerate(picks):
        for half, (name, columns) in enumerate(
            (("ally", slice(0, cells)), ("enemy", slice(cells, 2 * cells)))
        ):
            truth = target[index, columns].numpy()
            predicted = head[index, columns].numpy()
            error = predicted - truth
            row = axes[2 * frame + half]
            top = float(max(truth.max(), predicted.max(), 1e-6))
            bound = float(abs(error).max()) or 1e-6
            _draw_field(row[0], centres, spacing, truth, sequential, 0.0, top, f"{name} true")
            _draw_field(
                row[1], centres, spacing, predicted, sequential, 0.0, top, f"{name} predicted"
            )
            mesh = _draw_field(
                row[2],
                centres,
                spacing,
                error,
                diverging,
                -bound,
                bound,
                f"{name} error  (max |e| {bound:.3f})",
            )
            figure.colorbar(mesh, ax=row[2], fraction=0.046, shrink=0.85)
            row[0].set_ylabel(f"{int(hidden[index])} enemies hidden", color=INK_SECONDARY)
    figure.suptitle(
        f"Hex density field, log1p units — {run} @ {step:,} steps",
        color=INK,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    path = args.out_prefix.with_name(args.out_prefix.name + "-fields.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    outputs.append(path)

    # ---- 2. baseline ladder -------------------------------------------------
    if baselines:
        names = ["mean", "front", "ally", "head"]
        figure, axis = plt.subplots(figsize=(7.0, 3.4))
        positions = range(len(names))
        width = 0.38
        for offset, (half, colour) in enumerate(zip(("ally", "enemy"), SERIES, strict=True)):
            values = [baselines[half].get(f"r2_vs_mean/{n}") for n in names]
            spots = [p + (offset - 0.5) * width for p in positions]
            bars = axis.bar(
                [s for s, v in zip(spots, values, strict=True) if v is not None],
                [v for v in values if v is not None],
                width=width - 0.03,
                color=colour,
                label=f"{half} half",
            )
            for bar, value in zip(bars, [v for v in values if v is not None], strict=True):
                axis.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )
        axis.set_xticks(list(positions))
        axis.set_xticklabels(["per-cell\nmean", "+ front", "+ ally field\n(no belief)", "head"])
        axis.set_ylabel("R² vs the constant heat map")
        axis.axhline(0.0, color=GRID, linewidth=1)
        axis.grid(axis="y", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="upper left")
        axis.set_title("What predicts the density field", color=INK, loc="left")
        figure.tight_layout()
        path = args.out_prefix.with_name(args.out_prefix.name + "-ladder.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        outputs.append(path)

    # ---- 3. error against hidden-enemy count --------------------------------
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    levels = sorted({int(v) for v in hidden.tolist()})
    for half, columns, colour in (
        ("ally", slice(0, cells), SERIES[0]),
        ("enemy", slice(cells, 2 * cells), SERIES[1]),
    ):
        xs, ys = [], []
        for level in levels:
            mask = hidden == level
            if int(mask.sum()) < 50:
                continue
            residual = head[mask][:, columns] - target[mask][:, columns]
            xs.append(level)
            ys.append(float(residual.pow(2).mean()))
        axis.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=colour, label=f"{half} half")
        axis.annotate(
            f"{half}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=INK_SECONDARY,
            fontsize=9,
            va="center",
        )
    axis.set_xlabel("enemies the observer cannot see")
    axis.set_ylabel("mean squared error")
    axis.grid(axis="y", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    axis.set_title("Does the error grow as the enemy disappears?", color=INK, loc="left")
    # The direct labels sit outside the last marker; widen so they cannot clip.
    axis.set_xlim(-0.3, max(levels) + 0.75)
    figure.tight_layout()
    path = args.out_prefix.with_name(args.out_prefix.name + "-hidden.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    outputs.append(path)

    # ---- 4. error by ring ----------------------------------------------------
    ring_of = ring_index(HEX_DENSITY_RINGS)
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    for half, columns, colour in (
        ("ally", slice(0, cells), SERIES[0]),
        ("enemy", slice(cells, 2 * cells), SERIES[1]),
    ):
        residual = (head[:, columns] - target[:, columns]).pow(2).mean(0)  # (C,)
        ys = [float(residual[ring_of == ring].mean()) for ring in range(HEX_DENSITY_RINGS + 1)]
        axis.plot(
            range(HEX_DENSITY_RINGS + 1),
            ys,
            marker="o",
            markersize=5,
            linewidth=2,
            color=colour,
            label=f"{half} half",
        )
    axis.set_xlabel("ring index (0 = centre cell, 10 = playable boundary)")
    axis.set_ylabel("mean squared error")
    axis.grid(axis="y", linewidth=0.6)
    axis.set_axisbelow(True)
    # Both series descend from the top left, so that corner is taken.
    axis.legend(frameon=False, loc="lower left")
    axis.set_title("Error from the centre outward", color=INK, loc="left")
    figure.tight_layout()
    path = args.out_prefix.with_name(args.out_prefix.name + "-rings.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    outputs.append(path)

    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
