"""Figures for docs/rewrite/doghouse-shape-investigation.md (run through doghouse_shape.py).
The reference palette of docs/rewrite/sections/turns/doghouse.py, light theme."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import doghouse_shape as ds  # noqa: E402
from lean.game import tuning  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1")
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e7e6e2"
DIVERGING = LinearSegmentedColormap.from_list("ps", ["#e34948", "#f0efec", "#2a78d6"])
matplotlib.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.facecolor": SURFACE,
    "legend.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    path = ds.OUT_PNG / f"doghouse-shape-{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def polar(results):
    p, curves = results["polar"], results["polar_curves"]
    cl = np.array(curves["cl"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    ax.plot(cl, curves["parabolic"], color=SERIES[0], lw=2, label="parabolic: C_D0 + k C_L^2")
    ax.plot(cl, curves["walled"], color=SERIES[1], lw=2,
            label=f"game: x (1 + {p['separation']:.2f} (C_L/C_Lmax)^{p['exponent']:.0f})")
    ax.axvline(1.6 * 0.9, color=MUTED, lw=0.8, ls=":")
    ax.text(1.6 * 0.9 - 0.02, 0.2, "0.9 C_Lmax", ha="right", color=INK_2, fontsize=9)
    ax.set_xlabel("lift coefficient C_L")
    ax.set_ylabel("drag coefficient C_D")
    ax.set_title("P-51D card row: drag polar (any speed below the drag rise)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "polar")


def steepness(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    for i, v in enumerate(("60.0", "80.0", "100.0")):
        for name, dash in (("current", "-"), ("point_mass", "--")):
            e = results["steepness"][name][v]
            depth = np.array(e["depth"]) * ds.RATE_RANGE
            ps = np.array(e["ps"])
            keep = depth <= 12.0
            ax.plot(depth[keep], ps[keep], color=SERIES[i], ls=dash, lw=1.8,
                    label=f"{float(v):.0f} m/s, {'game row' if name == 'current' else 'no game terms'}")
    ax.set_xlabel("distance inside the max-lift line (deg/s of turn rate)")
    ax.set_ylabel("Ps (m/s)")
    ax.set_title("Ps just inside the max-lift line (real units)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "steepness")


def doghouse(ax, row, title, numbers=None):
    c = ds.constants(row)
    speeds_real = np.linspace(40.0, 200.0, 241)
    sticks = np.unique(np.concatenate([np.linspace(0, 1, 201), 1 - np.geomspace(1e-5, 0.05, 60)]))
    rate = np.empty((len(sticks), len(speeds_real)))
    ps = np.empty_like(rate)
    for i, s in enumerate(sticks):
        rate[i], ps[i] = ds.steady(c, speeds_real * ds.F, float(s))
    stall = ds.stall_game(row) / ds.F
    rate[:, speeds_real < stall] = np.nan
    axis = np.linspace(0.0, 36.0, 361)
    field = np.full((len(axis), len(speeds_real)), np.nan)
    for j in range(len(speeds_real)):
        if not np.isfinite(rate[-1, j]):
            continue
        order = np.argsort(rate[:, j])
        inside = axis <= rate[order, j][-1]
        field[inside, j] = np.interp(axis[inside], rate[order, j], ps[order, j])
    norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=30.0)
    levels = np.arange(-100.0, 31.0, 10.0)
    ax.contourf(speeds_real, axis, field, levels=levels, cmap=DIVERGING, norm=norm, extend="both")
    ax.contour(speeds_real, axis, field, levels=[lv for lv in levels if lv != 0], colors=INK_2,
               linewidths=0.6, linestyles="solid")
    ax.contour(speeds_real, axis, field, levels=[0.0], colors=INK, linewidths=1.6,
               linestyles="--")
    ax.plot(speeds_real, rate[-1], color=INK, lw=2.0)
    if numbers:
        ax.plot(numbers["sustained_speed"], numbers["sustained_rate"], "s", color=SERIES[0],
                ms=6, mec=SURFACE)
        tight_rate = math.degrees(numbers["min_radius_speed"] / numbers["min_radius"])
        ax.plot(numbers["min_radius_speed"], tight_rate, "^", color=SERIES[1], ms=6, mec=SURFACE)
    ax.set_xlim(40, 200)
    ax.set_ylim(0, 36)
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    return norm, levels


def variants(results, rows, name, suptitle):
    n = len(rows)
    cols = 2
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.0, 3.1 * rows_n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, row) in zip(axes, rows):
        numbers = ds.readout(row)
        doghouse(ax, row, title, numbers)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("turn rate (deg/s)", fontsize=8)
    for ax in axes[-cols:]:
        ax.set_xlabel("speed (m/s)", fontsize=8)
    fig.suptitle(suptitle, fontsize=10, x=0.02, ha="left")
    fig.text(0.02, -0.01, "Shading and thin lines: Ps every 10 m/s (blue gain, red loss); dashed: Ps = 0;"
             " thick: max-lift line; square: best sustained rate; triangle: tightest sustained.",
             fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, name)


def excess(results):
    pr = results["prop"]
    v = np.array(pr["v"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style(ax)
    w = tuning.P51D.mass * tuning.REAL_GRAVITY
    ax.plot(v, np.array(pr["drag_1g"]) / w, color=INK, lw=2, label="drag at 1 g")
    shown = [n for n in pr["curves"] if not n.startswith("game")]
    for color, name in zip(SERIES, shown):
        ax.plot(v, np.array(pr["curves"][name]["thrust"]) / w, color=color, lw=1.8,
                label=f"thrust: {name}")
    a = tuning.P51D
    imp = tuning.implied(tuning.row_of("p51d"), a)
    k = 1.0 / (math.pi * imp["oswald"] * a.aspect_ratio)
    q = 0.5 * tuning.AIR_DENSITY * v**2 * a.wing_area
    ax.plot(v, q * (imp["zero_lift_drag_coefficient"] + k * 1.6**2) / w, color=MUTED, lw=1.2,
            ls="--", label="drag at C_Lmax (parabolic)")
    ax.set_ylim(0, 0.8)
    ax.set_xlim(40, 175)
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("force / weight")
    ax.set_title("Thrust against drag (P-51D point mass, e from the card row)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "excess")


def main(results):
    polar(results)
    steepness(results)
    excess(results)
    current = tuning.row_of("p51d")
    start = current
    rows = [("now: s 0.41, exponent 16 (e 0.97)", current),
            ("no separation drag (e 0.79)", ds.solve(
                "separation_closed", start=start, separation_drag=0.0))]
    for m in (8.0, 4.0, 2.0):
        rows.append((f"exponent {m:.0f}, gap 0.85 solves s", ds.solve("documents", separation_exponent=m)))
    variants(results, rows[:5], "exponents",
             "P-51D doghouse, real units: separation exponent lowered, other targets held")
    extra = getattr(ds, "RECOMMENDED_ROWS", None)
    if extra:
        variants(results, extra(), "recommended", "P-51D doghouse, real units: candidate models")
