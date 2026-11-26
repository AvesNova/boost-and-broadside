"""Design sketch (2026-11-27, not game code): bullets refracting at a deflector bubble.

Straight rays in the target's frame (the target's own motion ignored), the ship a 12 m circle at
the origin with its nose along +x, the bubble a circle or an ellipse around it. At the bubble,
Snell with the field's index n (> 1 bends away from the normal: sin beta = n sin alpha); when
n sin alpha > 1 the bullet reflects. "entry only": the bullet keeps its bent path inside and
beyond; "in and out": it refracts back on leaving (a full lens). A ray that meets the ship circle
is a hit. Each panel title gives the hit width (the band of parallel offsets that hit).

    uv run --no-sync python docs/rewrite/sections/bullet-deflector/refraction_sketch.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "sketch"
SHIP = 12.0  # m, the hit radius
CIRCLE = 24.0  # m, the bubble's radius
ELLIPSE = (40.0, 18.0)  # m, semi-axes along and across the nose
VIEW = 70.0
HIT, BENT, BOUNCE = "#eb6834", "#2a78d6", "#7d56c7"


def circle_shape(radius):
    return (radius, radius)


def crossings(p, d, axes):
    """Parameters t of p + t d on the ellipse with semi-axes `axes` (x along the nose)."""
    a, b = axes
    px, py, dx, dy = p[0] / a, p[1] / b, d[0] / a, d[1] / b
    qa, qb, qc = dx * dx + dy * dy, 2 * (px * dx + py * dy), px * px + py * py - 1
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return None
    s = math.sqrt(disc)
    return (-qb - s) / (2 * qa), (-qb + s) / (2 * qa)


def normal(q, axes):
    a, b = axes
    v = np.array([q[0] / a**2, q[1] / b**2])
    return v / np.linalg.norm(v)


def refract(d, n_out, eta):
    """d the unit direction, n_out the surface normal facing against d, eta = sin out / sin in."""
    cos_i = -float(d @ n_out)
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    if k < 0:
        return d + 2 * cos_i * n_out, True
    return eta * d + (eta * cos_i - math.sqrt(k)) * n_out, False


def ship_hit(p, d):
    t = crossings(p, d, (SHIP, SHIP))
    return t is not None and t[1] > 1e-9 and (t[0] > 1e-9 or t[1] > 1e-9)


def trace(p, d, axes, index, exit_too):
    """Returns the polyline points, the outcome ('hit', 'bent', 'bounce', 'straight')."""
    t = crossings(p, d, axes)
    if t is None or t[0] < 0:
        stop = crossings(p, d, (SHIP, SHIP))
        if stop is not None and stop[0] > 0:
            return [p, p + stop[0] * d], "hit"
        return [p, p + 3 * VIEW * d], "straight"
    entry = p + t[0] * d
    out = normal(entry, axes)
    n = index(entry)
    inside, bounced = refract(d, out, n)
    inside /= np.linalg.norm(inside)
    if bounced:
        return [p, entry, entry + 3 * VIEW * inside], "bounce"
    stop = crossings(entry, inside, (SHIP, SHIP))
    if stop is not None and stop[0] > 1e-9:
        return [p, entry, entry + stop[0] * inside], "hit"
    if not exit_too:
        return [p, entry, entry + 3 * VIEW * inside], "bent"
    chord = crossings(entry, inside, axes)  # None: grazing along the surface (rounding)
    leave = 0.0 if chord is None else max(chord[1], 0.0)
    exit_point = entry + leave * inside
    outward = normal(exit_point, axes)
    after, _ = refract(inside, -outward, 1.0 / index(exit_point))
    after /= np.linalg.norm(after)
    return [p, entry, exit_point, exit_point + 3 * VIEW * after], "bent"


def direction(aspect_deg):
    """A bullet path `aspect` off the target's nose (0 head-on: flying -x at the nose)."""
    a = math.radians(aspect_deg)
    return np.array([-math.cos(a), -math.sin(a)])


def hit_width(axes, index, exit_too, aspect, step=0.05):
    d = direction(aspect)
    across = np.array([-d[1], d[0]])
    offsets = np.arange(-60, 60, step)
    hits = sum(trace(o * across - 200 * d, d, axes, index, exit_too)[1] == "hit" for o in offsets)
    return hits * step


def panel(ax, axes, index, exit_too, aspect, title):
    d = direction(aspect)
    across = np.array([-d[1], d[0]])
    for o in np.arange(-40, 40.01, 2.0):
        points, outcome = trace(o * across - 200 * d, d, axes, index, exit_too)
        colour = {"hit": HIT, "bounce": BOUNCE}.get(outcome, BENT)
        xy = np.array(points)
        ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=0.9)
    theta = np.linspace(0, 2 * math.pi, 200)
    ax.fill(axes[0] * np.cos(theta), axes[1] * np.sin(theta), color="#2a78d6", alpha=0.07)
    ax.plot(axes[0] * np.cos(theta), axes[1] * np.sin(theta), color="#5e5d59", lw=1, ls="--")
    ax.fill(SHIP * np.cos(theta), SHIP * np.sin(theta), color="#5e5d59", alpha=0.35)
    ax.annotate(
        "",
        xy=(2.2 * SHIP, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    width = hit_width(axes, index, exit_too, aspect)
    ax.set_title(f"{title}\nhit width {width:.1f} m (no field {2 * SHIP:.0f} m)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-VIEW, VIEW)
    ax.set_ylim(-VIEW, VIEW)
    ax.set_xticks([])
    ax.set_yticks([])


def constant(n):
    return lambda point: n


def by_angle(front, side, inner=30.0, outer=60.0):
    """n by where the bullet meets the bubble: `front` within `inner` deg of the nose, `side`
    beyond `outer`, a smoothstep between."""

    def index(point):
        off = math.degrees(abs(math.atan2(point[1], point[0])))
        t = min(max((off - inner) / (outer - inner), 0.0), 1.0)
        t = t * t * (3 - 2 * t)
        return front + (side - front) * t

    return index


def grid(name, rows, cols, build):
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
        f"{name}. Nose: arrow (+x). Orange: hits the ship; blue: bent past; purple: reflected. "
        "Rays 2 m apart.",
        fontsize=10,
    )
    path = OUT / f"{name.split(':')[0].replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ns = (1.0, 1.3, 2.0, 3.0)
    aspects = (0, 20, 45, 90, 180)
    circle = circle_shape(CIRCLE)
    grid(
        "Circle: entry only (top) and in and out (bottom), by n",
        (False, True),
        ns,
        lambda ax, ex, n: panel(
            ax, circle, constant(n), ex, 0, f"{'in+out' if ex else 'entry only'}, n = {n:g}"
        ),
    )
    for exit_too, tag in ((False, "entry only"), (True, "in and out")):
        grid(
            f"Ellipse {tag}: {ELLIPSE[0]:g} x {ELLIPSE[1]:g} m, rows n, columns aspect",
            ns[1:],
            aspects,
            lambda ax, n, a, ex=exit_too: panel(
                ax, ELLIPSE, constant(n), ex, a, f"n = {n:g}, {a} deg off the nose"
            ),
        )
        grid(
            f"Circle n by angle {tag}: n by entry point, front -> side",
            ((3.0, 1.3), (6.0, 1.3)),
            aspects,
            lambda ax, fs, a, ex=exit_too: panel(
                ax, circle, by_angle(*fs), ex, a, f"n {fs[0]:g} front, {fs[1]:g} side; {a} deg"
            ),
        )
