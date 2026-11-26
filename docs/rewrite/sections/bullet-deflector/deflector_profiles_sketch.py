"""Design sketch 3 (2026-11-27, not game code): a turn whose angle depends on the bullet's impact
parameter b (the distance of its straight line from the ship's centre), applied where it first
crosses the bubble, away from the ship. theta(b) = theta_max x f(b / F), F the bubble's half-width
across the bullet's path at its widest (the circle's radius; the egg's width). Speed plays no
part. Profiles: linear, quadratic, square root, soft step b / sqrt(b^2 + s^2), and the soft step
tapered to 0 at the edge (no jump where the field starts).

    uv run --no-sync python docs/rewrite/sections/bullet-deflector/deflector_profiles_sketch.py
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import deflector_shapes_sketch as base

base.SHAPES["circle 18"] = [(0.0, 18.0, 18.0, 0)]
base.SHAPES["egg 40/18"] = [(0.0, 40.0, 18.0, +1), (0.0, 18.0, 18.0, -1)]
EDGE = 18.0  # m: F, the bubble's half-width
SOFT = 2.0  # m: s, the soft step's width

PROFILES = {
    "linear": lambda x: x,
    "quadratic": lambda x: x * x,
    "square root": lambda x: math.sqrt(x),
    f"soft step (s {SOFT:g} m)": lambda x: x / math.sqrt(x * x + (SOFT / EDGE) ** 2),
    "soft step, edge taper": lambda x: (
        x / math.sqrt(x * x + (SOFT / EDGE) ** 2) * (1 - min(x, 1.0) ** 4)
    ),
}


def profile_turn(f, theta_max_deg):
    theta_max = math.radians(theta_max_deg)

    def act(d, normal, point):
        perp = point - (point @ d) * d
        b = float(np.linalg.norm(perp))
        side = perp / b if b > 1e-9 else np.array([-d[1], d[0]])
        theta = theta_max * f(min(b / EDGE, 1.0))
        return math.cos(theta) * d + math.sin(theta) * side, False

    return act


if __name__ == "__main__":
    theta_max = 45.0
    columns = [("circle 18", 0), ("egg 40/18", 0), ("egg 40/18", 90), ("egg 40/18", 180)]
    rows = list(PROFILES)
    fig, axs = plt.subplots(
        len(rows),
        1 + len(columns),
        figsize=(3.2 * (1 + len(columns)), 3.4 * len(rows)),
        constrained_layout=True,
    )
    x = np.linspace(0, 1, 200)
    for i, name in enumerate(rows):
        f = PROFILES[name]
        ax = axs[i, 0]
        ax.plot(x * EDGE, [theta_max * f(v) for v in x], color=base.BENT, lw=2)
        ax.set(xlabel="impact parameter b, m", ylabel="turn, deg", title=name, ylim=(0, 50))
        ax.grid(True, color="#e4e3df", lw=0.6)
        for j, (shape, aspect) in enumerate(columns):
            base.panel(
                axs[i, j + 1], shape, profile_turn(f, theta_max), aspect, f"{shape}, {aspect} deg"
            )
    fig.suptitle(
        f"Turn by impact parameter, max {theta_max:g} deg, at the bubble crossing. Orange: hits "
        "the ship (12 m); blue: bent past. Rays 2 m apart.",
        fontsize=10,
    )
    fig.savefig(base.OUT / "turn_profiles.png", dpi=95)
    print("wrote", base.OUT / "turn_profiles.png")
    # Hit widths by max turn.
    print("hit width, m (no field 24): profile | max turn | circle | egg 0 | egg 90 | egg 180")
    for name, f in PROFILES.items():
        for tm in (20, 30, 45, 60, 90):
            act = profile_turn(f, tm)
            w = [base.hit_width(base.SHAPES[s], act, a, step=0.05) for s, a in columns]
            print(f"{name:24s} {tm:3d} " + " ".join(f"{v:5.1f}" for v in w))
