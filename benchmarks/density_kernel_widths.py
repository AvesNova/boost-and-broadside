"""How wide is one ship on the hex density grid, and at what kernel radius?

The density target smooths every ship with a Gaussian of ``PRESENCE_RADIUS``
before sampling it on the grid. That radius is 500 px against a 2600 px
playable radius and a 260 px ring spacing, so a single ship is not a point on
this field -- it is a broad hill, and the question this answers is how broad.

Three figures, all from the target's own definition. No checkpoint and no
trained head are involved: this is a property of the target, not of anything
that learned it.

* ``*-single.png``  -- one ship at the map centre, at each candidate radius.
  The ask: what does one ship look like.
* ``*-profile.png`` -- the same, as density against distance, with the ring
  spacing and the playable boundary marked. The quantitative view.
* ``*-pairs.png``   -- two ships at four separations, at the production radius.
  Over-smoothing is really a question about resolution -- whether the field can
  tell two ships apart from one pair sitting together -- and a single ship
  cannot answer it.

    uv run --no-sync python benchmarks/density_kernel_widths.py \\
        --out-prefix docs/internal/density-kernel-sep2026
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# One hue, light->dark: radius is an ordered parameter, not an identity, so the
# series take an ordinal ramp rather than categorical hues. Validated as an
# ordinal ramp on the light surface (monotone L, all adjacent gaps >= 0.06).
ORDINAL = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"

#: The 5v5 Frontline geometry the target is defined against.
PLAYABLE_RADIUS = 2600.0
#: Big enough that the torus never wraps within the figure.
WORLD = (65536.0, 65536.0)


def half_max_radius(radius: float) -> float:
    """Distance at which one ship's density falls to half its peak, in pixels.

    The field is ``log1p`` of a Gaussian, so the half-max of the *stored* value
    is not the Gaussian's own half-max. Peak is ``log1p(1) = ln 2``; the
    half-peak weight solves ``log1p(w) = ln(2)/2``.
    """

    weight = math.exp(math.log(2.0) / 2.0) - 1.0
    return radius * math.sqrt(2.0 * math.log(1.0 / weight))


def lit_fraction(values, peak_fraction: float = 0.1) -> float:
    """Fraction of cells carrying at least ``peak_fraction`` of the peak."""

    peak = float(values.max())
    if peak <= 0.0:
        return 0.0
    return float((values >= peak_fraction * peak).float().mean())


def _style(matplotlib):
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "grid.color": GRID,
        "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _field(ships, radius: float, cells):
    """(C,) ally density for ships at ``ships`` (a list of (x, y) in pixels)."""
    import torch

    from boost_and_broadside.train.rl.hex_density import hex_density

    count = len(ships)
    ship_x = torch.tensor([[x for x, _ in ships]], dtype=torch.float32)
    ship_y = torch.tensor([[y for _, y in ships]], dtype=torch.float32)
    world = torch.tensor(cells, dtype=torch.float32) * PLAYABLE_RADIUS  # (C, 2)
    both = hex_density(
        ship_x, ship_y,
        torch.ones(1, count, dtype=torch.bool),
        torch.ones(1, count, dtype=torch.bool),
        world[:, 0].unsqueeze(0), world[:, 1].unsqueeze(0),
        WORLD, radius=radius,
    )
    return both[0, : world.shape[0]]


def _draw(axis, centres, values, cmap, vmax, title, spacing):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import RegularPolygon

    patches = [
        RegularPolygon((float(x), float(y)), numVertices=6, radius=spacing / (3**0.5))
        for x, y in centres
    ]
    collection = PatchCollection(patches, cmap=cmap, edgecolor="none")
    collection.set_array(values)
    collection.set_clim(0.0, vmax)
    axis.add_collection(collection)
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, color=INK, fontsize=9)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=[125.0, 250.0, 500.0, 1000.0, 2000.0],
        help="Kernel radii in pixels; the production value is 500.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap

    from boost_and_broadside.train.rl.features import PRESENCE_RADIUS
    from boost_and_broadside.train.rl.hex_density import HEX_DENSITY_RINGS, hex_cell_offsets

    _style(matplotlib)
    sequential = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    if len(args.radii) > len(ORDINAL):
        parser.error(f"at most {len(ORDINAL)} radii; the ordinal ramp has that many steps")

    cells = hex_cell_offsets(HEX_DENSITY_RINGS)
    spacing = 1.0 / HEX_DENSITY_RINGS
    ring_px = PLAYABLE_RADIUS / HEX_DENSITY_RINGS
    peak = math.log(2.0)  # one ship on its own cell, log1p(1)
    outputs, summary = [], []

    # ---- 1. one ship at the centre, per radius ------------------------------
    figure, axes = plt.subplots(1, len(args.radii), figsize=(2.55 * len(args.radii), 3.5))
    for axis, radius in zip(axes, args.radii, strict=True):
        values = _field([(0.0, 0.0)], radius, cells)
        lit = lit_fraction(values)
        half = half_max_radius(radius)
        summary.append({
            "radius_px": radius,
            "half_max_px": half,
            "half_max_rings": half / ring_px,
            "cells_above_10pct_peak": lit,
            "is_production": radius == PRESENCE_RADIUS,
        })
        label = f"r = {radius:.0f} px" + ("  (production)" if radius == PRESENCE_RADIUS else "")
        _draw(axis, cells.numpy(), values.numpy(), sequential, peak, label, spacing)
        axis.set_xlabel(
            f"half-max {half / ring_px:.1f} rings\n{lit:.0%} of cells lit",
            color=INK_SECONDARY, fontsize=8,
        )
    figure.suptitle(
        "One ship at the map centre — the density target's view of a point",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = args.out_prefix.with_name(args.out_prefix.name + "-single.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    # ---- 2. radial profile ---------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    distance = torch.linspace(0.0, PLAYABLE_RADIUS, 400)
    for radius, colour in zip(args.radii, ORDINAL, strict=False):
        profile = torch.log1p(torch.exp(-(distance**2) / (2.0 * radius**2)))
        width = 2.6 if radius == PRESENCE_RADIUS else 2.0
        axis.plot(distance / ring_px, profile, color=colour, linewidth=width,
                  label=f"r = {radius:.0f} px" + (" (production)" if radius == PRESENCE_RADIUS
                                                  else ""))
    axis.axhline(peak / 2, color=GRID, linewidth=1.2, linestyle="--")
    # Left edge: the right is where the widest kernel is still falling.
    axis.annotate("half peak", (0.15, peak / 2), ha="left", va="bottom",
                  color=INK_SECONDARY, fontsize=8)
    axis.set_xlabel("distance from the ship, in ring spacings (1 ring = 260 px)")
    axis.set_ylabel("density, log1p units")
    axis.set_xlim(0, HEX_DENSITY_RINGS)
    axis.grid(axis="y", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("How far one ship reaches across the grid", color=INK, loc="left")
    figure.tight_layout()
    path = args.out_prefix.with_name(args.out_prefix.name + "-profile.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    # ---- 3. two ships, production radius ------------------------------------
    # Over-smoothing is a resolution question: two equal Gaussians of width r
    # stop reading as two bumps below a separation of about 2r.
    separations = [ring_px, 2 * ring_px, 4 * ring_px, 8 * ring_px]
    figure, axes = plt.subplots(1, len(separations), figsize=(2.55 * len(separations), 3.5))
    for axis, separation in zip(axes, separations, strict=True):
        values = _field([(-separation / 2, 0.0), (separation / 2, 0.0)], PRESENCE_RADIUS, cells)
        centre = float(values[0])
        peaks = float(values.max())
        # A dip at the midpoint is what "two ships" looks like on this field.
        resolved = centre < 0.98 * peaks
        _draw(axis, cells.numpy(), values.numpy(), sequential, float(values.max()),
              f"{separation:.0f} px apart ({separation / ring_px:.0f} rings)", spacing)
        axis.set_xlabel(
            ("two bumps" if resolved else "reads as one") + f"\nmidpoint {centre / peaks:.0%} of peak",
            color=INK_SECONDARY, fontsize=8,
        )
    figure.suptitle(
        f"Two ships at the production radius ({PRESENCE_RADIUS:.0f} px) — when do they separate?",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = args.out_prefix.with_name(args.out_prefix.name + "-pairs.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path)

    result = {
        "production_radius_px": PRESENCE_RADIUS,
        "playable_radius_px": PLAYABLE_RADIUS,
        "rings": HEX_DENSITY_RINGS,
        "ring_spacing_px": ring_px,
        "peak_value": peak,
        "radii": summary,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
