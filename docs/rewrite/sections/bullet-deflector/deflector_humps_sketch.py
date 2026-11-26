"""Design sketch 4 (2026-11-27, not game code): turn by impact parameter with curves that are 0
at the centre (b = 0) and at the bubble's edge (b = F, no seam). Every ray drawn in full, through
the ship (a faint circle); orange: its bent path passes within the 12 m hit radius.

    uv run --no-sync python docs/rewrite/sections/bullet-deflector/deflector_humps_sketch.py
"""

from __future__ import annotations

import math

import deflector_profiles_sketch as prof
import deflector_shapes_sketch as base
import matplotlib.pyplot as plt
import numpy as np

EDGE, SOFT = prof.EDGE, prof.SOFT
CURVES = {
    "soft step x edge taper (1 - x^4)": prof.PROFILES["soft step, edge taper"],
    "soft step x (1 - x^2)": lambda x: x / math.sqrt(x * x + (SOFT / EDGE) ** 2) * (1 - x * x),
    "sine, sin(pi x)": lambda x: math.sin(math.pi * x),
    "parabola, 4 x (1 - x)": lambda x: 4 * x * (1 - x),
    "early hump, 27/4 x (1 - x)^2": lambda x: 6.75 * x * (1 - x) ** 2,
    "sharp hump, x (1 - x)^4 / 0.08192": lambda x: x * (1 - x) ** 4 / 0.08192,
}


def trace_through(p, d, shape, act):
    """The full polyline (never stopped by the ship) and whether the path after the event (or
    the straight one) passes within the hit radius."""
    first = next(((t, n) for t, n in base.crossings(p, d, shape) if t > 0), None)
    if first is None:
        return [p, p + 3 * base.VIEW * d], base.ship_t(p, d) is not None
    t, normal = first
    entry = p + t * d
    new, _ = act(d, normal, entry)
    new = new / np.linalg.norm(new)
    return [p, entry, entry + 3 * base.VIEW * new], base.ship_t(entry, new) is not None


def panel(ax, shape_name, act, aspect, title):
    shape = base.SHAPES[shape_name]
    d = base.direction(aspect)
    across = np.array([-d[1], d[0]])
    for o in np.arange(-40, 40.01, 2.0):
        points, hit = trace_through(o * across - 200 * d, d, shape, act)
        xy = np.array(points)
        ax.plot(xy[:, 0], xy[:, 1], color=base.HIT if hit else base.BENT, lw=0.9)
    for cx, a, b, half in shape:
        lo, hi = {0: (0, 2 * math.pi), 1: (-math.pi / 2, math.pi / 2)}.get(
            half, (math.pi / 2, 3 * math.pi / 2)
        )
        th = np.linspace(lo, hi, 150)
        ax.plot(cx + a * np.cos(th), b * np.sin(th), color="#5e5d59", lw=1, ls="--")
    th = np.linspace(0, 2 * math.pi, 100)
    ax.plot(base.SHIP * np.cos(th), base.SHIP * np.sin(th), color="#5e5d59", lw=1, alpha=0.6)
    ax.annotate(
        "",
        xy=(2.2 * base.SHIP, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.2, alpha=0.6),
    )
    width = base.hit_width(shape, act, aspect)
    ax.set_title(f"{title}\nhit width {width:.1f} m (no field 24 m)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-base.VIEW, base.VIEW)
    ax.set_ylim(-base.VIEW, base.VIEW)
    ax.set_xticks([])
    ax.set_yticks([])


def main(edge: float = EDGE) -> None:
    """The figure and the table for a bubble of half-width `edge` (the egg keeps the 40/18
    proportions: front length 40/18 x edge, rear length = half-width = edge)."""
    global EDGE
    EDGE = prof.EDGE = edge
    front = 40.0 / 18.0 * edge
    circle, egg = f"circle {edge:g}", f"egg {front:.0f}/{edge:g}"
    base.SHAPES[circle] = [(0.0, edge, edge, 0)]
    base.SHAPES[egg] = [(0.0, front, edge, +1), (0.0, edge, edge, -1)]
    theta_max = 45.0
    columns = [(circle, 0), (egg, 0), (egg, 90), (egg, 180)]
    fig, axs = plt.subplots(
        len(CURVES),
        1 + len(columns),
        figsize=(3.2 * (1 + len(columns)), 3.4 * len(CURVES)),
        constrained_layout=True,
    )
    x = np.linspace(0, 1, 300)
    for i, (name, f) in enumerate(CURVES.items()):
        ax = axs[i, 0]
        ax.plot(x * EDGE, [theta_max * f(v) for v in x], color=base.BENT, lw=2)
        ax.set(xlabel="impact parameter b, m", ylabel="turn, deg", ylim=(0, 50))
        ax.set_title(name, fontsize=9)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for j, (shape, aspect) in enumerate(columns):
            panel(
                axs[i, j + 1],
                shape,
                prof.profile_turn(f, theta_max),
                aspect,
                f"{shape}, {aspect} deg",
            )
    fig.suptitle(
        f"Turn by impact parameter, 0 at both ends, max {theta_max:g} deg, bubble half-width "
        f"{edge:g} m, s {SOFT:g} m. Orange: the bent path passes within the 12 m hit radius "
        "(drawn through); blue: misses. Rays 2 m apart.",
        fontsize=10,
    )
    name = "turn_humps.png" if edge == 18.0 else f"turn_humps_{edge:g}.png"
    fig.savefig(base.OUT / name, dpi=95)
    print("wrote", base.OUT / name)
    print(f"hit width, m: curve | max turn | {circle} | {egg} 0 | 90 | 180")
    for cname, f in CURVES.items():
        for tm in (30, 45, 60, 90):
            act = prof.profile_turn(f, tm)
            w = [base.hit_width(base.SHAPES[sh], act, a, step=0.05) for sh, a in columns]
            print(f"{cname:36s} {tm:3d} " + " ".join(f"{v:5.1f}" for v in w))


if __name__ == "__main__":
    import sys

    main(float(sys.argv[1]) if len(sys.argv) > 1 else 18.0)
