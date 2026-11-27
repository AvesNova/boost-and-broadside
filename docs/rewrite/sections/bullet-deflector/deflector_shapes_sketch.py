"""Design sketch 2 (2026-11-27, not game code): deflector shapes and mechanisms at the bubble.

Straight rays in the target's frame (its own motion ignored), the ship a 12 m circle at the
origin, nose along +x. One event per pass, where the bullet first crosses the bubble inward:

- refract: Snell with index n (sin beta = n sin alpha; reflects when n sin alpha > 1);
- kick: a radial kick, the bullet's velocity plus J (as a fraction of its speed) along the
  bubble's outward normal, then rescaled to its old speed (no energy added); a bullet whose
  new path points outward is repelled;
- turn: a fixed angle theta away from the ship's centre, wherever it enters (the side by which
  side of the centre the bullet's line passes; dead centre has no side: drawn turning left).

Shapes: circle (24 m), ellipse shifted forward (40 x 18 m, centre 14 m ahead), egg (front half
40 m long, rear half 20 m, 18 m wide). Bullets keep the bent path after the event (entry only).

    uv run --no-sync python docs/rewrite/sections/bullet-deflector/deflector_shapes_sketch.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "sketch"
SHIP = 12.0
VIEW = 70.0
HIT, BENT, BOUNCE = "#eb6834", "#2a78d6", "#7d56c7"

# A shape: half-ellipse pieces (centre x, semi-axis along x, semi-axis across, which half: +1
# front x >= centre, -1 rear, 0 whole).
SHAPES = {
    "circle": [(0.0, 24.0, 24.0, 0)],
    "shifted ellipse": [(14.0, 40.0, 18.0, 0)],
    "egg": [(0.0, 40.0, 18.0, +1), (0.0, 20.0, 18.0, -1)],
}


def _ellipse_t(p, d, cx, a, b):
    px, py, dx, dy = (p[0] - cx) / a, p[1] / b, d[0] / a, d[1] / b
    qa, qb, qc = dx * dx + dy * dy, 2 * (px * dx + py * dy), px * px + py * py - 1
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return [(-qb - s) / (2 * qa), (-qb + s) / (2 * qa)]


def crossings(p, d, shape):
    """Sorted (t, outward normal) of p + t d on the shape's boundary."""
    out = []
    for cx, a, b, half in shape:
        for t in _ellipse_t(p, d, cx, a, b):
            q = p + t * d
            if half and (q[0] - cx) * half < -1e-9:
                continue
            n = np.array([(q[0] - cx) / a**2, q[1] / b**2])
            out.append((t, n / np.linalg.norm(n)))
    return sorted(out, key=lambda item: item[0])


def ship_t(p, d):
    ts = _ellipse_t(p, d, 0.0, SHIP, SHIP)
    ts = [t for t in ts if t > 1e-9]
    return min(ts) if ts else None


def refract(n):
    def act(d, normal, point):
        cos_i = -float(d @ normal)
        k = 1 - n * n * (1 - cos_i * cos_i)
        if k < 0:
            return d + 2 * cos_i * normal, True
        return n * d + (n * cos_i - math.sqrt(k)) * normal, False

    return act


def kick(j):
    def act(d, normal, point):
        v = d + j * normal
        size = np.linalg.norm(v)
        if size < 1e-9:  # stopped dead: drawn leaving along the normal
            return normal, True
        v = v / size
        return v, float(v @ normal) > 0

    return act


def turn(theta_deg):
    theta = math.radians(theta_deg)

    def act(d, normal, point):
        perp = point - (point @ d) * d  # from the centre's line to the bullet's: the side
        size = np.linalg.norm(perp)
        perp = perp / size if size > 1e-9 else np.array([-d[1], d[0]])
        return math.cos(theta) * d + math.sin(theta) * perp, False

    return act


def trace(p, d, shape, act):
    first = next(((t, n) for t, n in crossings(p, d, shape) if t > 0), None)
    if first is None:
        stop = ship_t(p, d)
        if stop is not None:
            return [p, p + stop * d], "hit"
        return [p, p + 3 * VIEW * d], "straight"
    t, normal = first
    entry = p + t * d
    new, bounced = act(d, normal, entry)
    new = new / np.linalg.norm(new)
    if bounced:
        return [p, entry, entry + 3 * VIEW * new], "bounce"
    stop = ship_t(entry, new)
    if stop is not None:
        return [p, entry, entry + stop * new], "hit"
    return [p, entry, entry + 3 * VIEW * new], "bent"


def direction(aspect_deg):
    a = math.radians(aspect_deg)
    return np.array([-math.cos(a), -math.sin(a)])


def hit_width(shape, act, aspect, step=0.05):
    d = direction(aspect)
    across = np.array([-d[1], d[0]])
    return step * sum(
        trace(o * across - 200 * d, d, shape, act)[1] == "hit" for o in np.arange(-60, 60, step)
    )


def outline(ax, shape):
    for cx, a, b, half in shape:
        lo, hi = {0: (0, 2 * math.pi), 1: (-math.pi / 2, math.pi / 2)}.get(
            half, (math.pi / 2, 3 * math.pi / 2)
        )
        th = np.linspace(lo, hi, 150)
        ax.plot(cx + a * np.cos(th), b * np.sin(th), color="#5e5d59", lw=1, ls="--")
    th = np.linspace(0, 2 * math.pi, 100)
    ax.fill(SHIP * np.cos(th), SHIP * np.sin(th), color="#5e5d59", alpha=0.35)
    ax.annotate(
        "", xy=(2.2 * SHIP, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.4)
    )


def panel(ax, shape_name, act, aspect, title):
    shape = SHAPES[shape_name]
    d = direction(aspect)
    across = np.array([-d[1], d[0]])
    for o in np.arange(-40, 40.01, 2.0):
        points, outcome = trace(o * across - 200 * d, d, shape, act)
        xy = np.array(points)
        ax.plot(xy[:, 0], xy[:, 1], color={"hit": HIT, "bounce": BOUNCE}.get(outcome, BENT), lw=0.9)
    outline(ax, shape)
    width = hit_width(shape, act, aspect)
    ax.set_title(f"{title}\nhit width {width:.1f} m (no field {2 * SHIP:.0f} m)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-VIEW, VIEW)
    ax.set_ylim(-VIEW, VIEW)
    ax.set_xticks([])
    ax.set_yticks([])


def grid(file, title, rows, cols, build):
    fig, axs = plt.subplots(
        len(rows),
        len(cols),
        figsize=(3.2 * len(cols), 3.5 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            build(axs[i, j], row, col)
    fig.suptitle(
        f"{title}. Nose: arrow (+x). Orange: hits the ship; blue: bent past; "
        "purple: reflected or repelled. Rays 2 m apart.",
        fontsize=10,
    )
    fig.savefig(OUT / file, dpi=100)
    plt.close(fig)
    print("wrote", OUT / file)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    aspects = (0, 20, 45, 90, 135, 180)
    for name in ("shifted ellipse", "egg"):
        grid(
            f"refract_{name.replace(' ', '_')}.png",
            f"Refraction, {name}: rows n, columns angle off the nose",
            (1.3, 2.0, 3.0),
            aspects,
            lambda ax, n, a, s=name: panel(ax, s, refract(n), a, f"n = {n:g}, {a} deg"),
        )
    # Kick and turn: circle (any angle looks the same) and the egg by angle.
    columns = [("circle", 0)] + [("egg", a) for a in (0, 45, 90, 180)]
    grid(
        "kick.png",
        "Radial kick at the bubble (J = kick / bullet speed, speed kept)",
        (0.15, 0.3, 0.6, 1.0),
        columns,
        lambda ax, j, c: panel(ax, c[0], kick(j), c[1], f"{c[0]}, J = {j:g}, {c[1]} deg"),
    )
    grid(
        "turn.png",
        "Fixed turn away from the ship at the bubble",
        (10.0, 20.0, 35.0),
        columns,
        lambda ax, th, c: panel(ax, c[0], turn(th), c[1], f"{c[0]}, {th:g} deg turn, {c[1]} deg"),
    )
