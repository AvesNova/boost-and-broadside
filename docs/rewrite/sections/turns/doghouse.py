"""Turn-performance (doghouse) charts of lean's flight model, for every card type, in game units
and at 1x real time (speeds, turn rates and Ps halved; radii and loads in g unchanged), so a
real aircraft's chart reads against published energy-manoeuvrability diagrams.

For every speed and stick fraction, the quasi-steady turn from the flight code's own
function (`lean.game.tuning.envelope`, which calls `core.steady_rates`): the nose settled on the stick's
angle of attack, turn rate = (C_L(angle) v^2 + T sin angle) / v, and the rates of change of
speed and of the energy store. No fields, no guns. Power: coast.

Energy rate (Ps) is the total specific energy rate, d(v^2/2 + store)/dt, shown as an
altitude-equivalent rate in m/s at the game's gravity (39.24 m/s^2), as energy-manoeuvrability
diagrams show Ps. Positive = gaining energy.

    for theme in light dark; do THEME=$theme OMP_NUM_THREADS=1 PYTHONPATH=.:src \
        uv run --no-sync python docs/rewrite/sections/turns/doghouse.py; done

Writes doghouse-<type>-core[-real][-dark].png and doghouse-comparison-core[-dark].png into
docs/rewrite/.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from sim import CONFIG, MAX_ENERGY, SPAWN, SUFFIX, THEME, TYPES  # noqa: E402

from lean.game import card, tuning  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "..", "..")
DEGREES = 180.0 / math.pi
G = card.GRAVITY
NAMES = tuple({"p51d": "P-51D"}.get(name, name) for name in card.TYPE_NAMES)
TIME_FACTOR = 2.0  # the game against real time (lean.game.tuning.TIME_FACTOR)
# Reference palette (dataviz skill, references/palette.md): categorical slots 1-3, diverging
# blue <-> red through a grey midpoint, chart surface and inks.
if THEME == "dark":  # the reference palette's dark steps (validated against the dark surface)
    SERIES = ("#3987e5", "#d95926", "#199e70")
    SURFACE, INK, INK_2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2e2e2c"
    DIVERGING = LinearSegmentedColormap.from_list("ps", ["#e66767", "#383835", "#3987e5"])
else:
    SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
    SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e7e6e2"
    DIVERGING = LinearSegmentedColormap.from_list("ps", ["#e34948", "#f0efec", "#2a78d6"])
matplotlib.rcParams.update({"text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED,
                            "xtick.color": INK_2, "ytick.color": INK_2, "legend.facecolor": SURFACE,
                            "legend.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
SPAWN_PERCENT = 100.0 * SPAWN[0] / MAX_ENERGY[0]
# One panel: coasting does not depend on the store; only dive and climb fade inside the outer
# 5% at each edge (the footnote says so).
STORES = ((f"any store outside the edge bands (shown at {SPAWN_PERCENT:.0f}%)", None),)

speeds = torch.linspace(20.0, 470.0, 226)
sticks = torch.linspace(0.0, 1.0, 201)


def evaluate(ship_type: int, store_fraction, power: int = 0):
    """(stick, speed) grids: turn rate deg/s, radius m, Ps m/s, speed rate, store rate; from
    lean.game.tuning.envelope, the one place the envelope is computed."""
    row = tuning.row_of(card.TYPE_NAMES[ship_type])
    grid = tuning.envelope(row, speeds, sticks, power, store_fraction, plain=False)
    return {**grid, "rate": grid["rate"] * DEGREES, "sustained_rate": grid["sustained_rate"] * DEGREES}


def sustained(grid):
    """Per speed: the largest turn rate with Ps >= 0 (nan if none), and its radius."""
    return grid["sustained_rate"], grid["sustained_radius"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


DOGHOUSE_FOOT = (
    "1 g of weight always on the wing, the turn gets the rest of the lift and g limit; the engine drives\n"
    "speed only. Thick: full stick. Thin: the game's sustained and fine stick. Dash-dot: full stick while\n"
    "diving or climbing (diving lowers the weight on the wing; climbing gets only the lift the turn leaves).\n"
    + ("White" if THEME == "dark" else "Black") + " dashed: Ps = 0 (sustained). Grey dotted: turn radius. Grey dashed: load in\n"
    "game g (39.24 m/s^2). Shaded below 100 m/s: quasi-steady maths only, no best-sustained mark there.\n"
    "Coasting, turn rate and Ps do not depend on the store. Inside an edge band (the outer 5% at the top and\n"
    "bottom) dive or climb fades to coast, in the trade and in the wing's load; outside it, as drawn."
)
TYPE_NOTES = {
    "p51d": "P-51D: the real airframe at 2x time with the game terms on (drag rise, separation drag, soft limits,\n"
    "automatic flaps below 1.6 x stall), solved to fly its point mass's top speed, climb and best sustained\n"
    "rate (lean.game.tuning.game_row, docs/rewrite/p51d-model.md).",
}


def foot(ship_type: int, real: bool) -> str:
    """The footnote, with the type's note and real units where drawn."""
    text = DOGHOUSE_FOOT
    note = TYPE_NOTES.get(card.TYPE_NAMES[ship_type])
    if note:
        text += "\n" + note
    if real:
        text = text.replace("load in\ngame g (39.24 m/s^2)", "load in\ng (9.81 m/s^2)").replace("below 100 m/s", "below 50 m/s")
        text += "\nReal units: the game at 2x time shown at real time (speeds, turn rates and Ps halved; radii unchanged)."
    return text


def draw_panel(figure, ax, ship_type: int, label: str, fraction, real: bool, top=None, limit=None):
    """One doghouse panel on ax: Ps shading, Ps = 0, the stick levels, dive and climb, corner and
    best sustained. top (deg/s) and limit (the Ps colour range) default to the panel's own;
    returns both, so panels can share them."""
    f = TIME_FACTOR if real else 1.0
    game_v = speeds.numpy()
    v = game_v / f
    grid = evaluate(ship_type, fraction)
    grid["rate"], grid["ps"] = grid["rate"] / f, grid["ps"] / f
    grid["sustained_rate"] = grid["sustained_rate"] / f
    style(ax)
    envelope = grid["rate"][-1]
    top = float(np.nanmax(envelope)) * 1.15 if top is None else top
    rate_axis = np.linspace(0.0, top, 300)
    # Ps on a regular (speed, rate) grid: for each speed, interpolate over the stick sweep.
    ps_field = tuning.ps_over_rate(grid, rate_axis)
    shared = limit is not None  # a given range: fixed levels, so panels' colours match
    limit = np.nanpercentile(np.abs(ps_field), 98) if limit is None else limit
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    levels = np.linspace(-limit, limit, 21) if shared else 21
    filled = ax.contourf(v, rate_axis, ps_field, levels=levels, cmap=DIVERGING, norm=norm,
                         extend="both" if shared else "neither")
    ax.contour(v, rate_axis, ps_field, levels=[0.0], colors=INK, linewidths=2.0, linestyles=[(0, (6, 3))])
    # Radius and load lines: geometry only.
    for radius in (150, 200, 300, 500, 1000, 2000):
        line = v / radius * DEGREES
        ax.plot(v, line, color=MUTED, linewidth=0.7, linestyle=":")
        j = np.searchsorted(line, top * 0.92)
        if 0 < j < len(v):
            ax.text(v[j], line[j], f"{radius} m", color=MUTED, fontsize=8, ha="right", va="bottom")
    for load in (2, 4, 6, 8, 9):
        line = load * G / game_v * DEGREES / f
        ax.plot(v, line, color=MUTED, linewidth=0.7, linestyle="--")
        j = np.searchsorted(-line, -top * 0.95)
        if 0 < j < len(v) - 1:
            ax.text(v[j], line[j], f"{load} g", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.plot(v, envelope, color=INK, linewidth=2.0)
    label_at = int(np.searchsorted(game_v, 175.0))
    ax.annotate("full stick", (v[label_at], envelope[label_at]), xytext=(-4, 6), textcoords="offset points",
                color=INK, fontsize=8, ha="right")
    # The game's other stick levels (the card's fine and sustained values for this type).
    for name, key in (("sustained stick", "sustained_stick"), ("fine stick", "fine_stick")):
        row = int(round(card.CARD[key][ship_type] * (len(sticks) - 1)))
        level = grid["rate"][row]
        ax.plot(v, level, color=INK, linewidth=1.1)
        at = int(np.searchsorted(game_v, 215.0))  # right of the steep low-speed part of the full-stick line
        ax.annotate(name, (v[at], level[at]), xytext=(4, -12), textcoords="offset points",
                    color=INK, fontsize=8, ha="left")
    # Full stick while diving and while climbing (power 1 and 2), same store.
    offsets = ((4, 7), (4, -24))
    for (power, name), offset in zip(((1, "full stick, diving"), (2, "full stick, climbing")), offsets):
        moved = evaluate(ship_type, fraction, power)["rate"][-1] / f
        ax.plot(v, moved, color=INK_2, linewidth=1.1, linestyle="-.")
        here = int(np.searchsorted(game_v, 380.0))
        ax.annotate(name, (v[here], moved[here]), xytext=offset, textcoords="offset points",
                    color=INK_2, fontsize=8)
    corner_v, corner_rate = grid["corner_speed"] / f, grid["corner_rate"] * DEGREES / f
    ax.plot(corner_v, corner_rate, "o", color=INK, markersize=8)
    ax.annotate(f"corner {corner_v:.0f} m/s, {corner_rate:.0f} deg/s", (corner_v, corner_rate),
                xytext=(8, 6), textcoords="offset points", color=INK, fontsize=9)
    if grid["best_sustained_speed"] is not None:  # from 100 m/s (game) up: below it the turn cannot be held
        best_v, best_rate = grid["best_sustained_speed"] / f, grid["best_sustained_rate"] * DEGREES / f
        ax.plot(best_v, best_rate, "s", color=INK, markersize=8)
        ax.annotate(f"best sustained {best_v:.0f} m/s, {best_rate:.0f} deg/s", (best_v, best_rate),
                    xytext=(8, -14), textcoords="offset points", color=INK, fontsize=9)
        tight_v = grid["tightest_sustained_speed"] / f
        tight_rate = tight_v / grid["tightest_sustained_radius"] * DEGREES
        ax.plot(tight_v, tight_rate, "^", color=INK, markersize=8)
        ax.annotate(f"tightest sustained {tight_v:.0f} m/s, {grid['tightest_sustained_radius']:.0f} m",
                    (tight_v, tight_rate), xytext=(14, -56), textcoords="offset points", color=INK,
                    fontsize=9, ha="left", arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel("speed (m/s, real)" if real else "speed (m/s)", color=INK_2)
    ax.set_ylabel("turn rate (deg/s)", color=INK_2)
    ax.axvspan(v[0], 100.0 / f, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title(f"{NAMES[ship_type]}, coast" + (", real units" if real else "") + f"\n{label}",
                 color=INK, fontsize=10, loc="left")
    bar = figure.colorbar(filled, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("energy rate Ps, m/s altitude-equivalent (blue gain, red loss)", color=INK_2, fontsize=8)
    bar.ax.tick_params(labelsize=8, colors=INK_2)
    return top, limit


def doghouse(ship_type: int, real: bool = False):
    """real: in real units at the game's 2x time (speeds, turn rates and Ps halved; radii and
    loads unchanged)."""
    panels = len(STORES)
    figure, axes = plt.subplots(panels, 1, figsize=(7.5, 5.4 * panels + 2.0), facecolor=SURFACE, squeeze=False)
    axes = axes[:, 0]
    for ax, (label, fraction) in zip(axes, STORES):
        draw_panel(figure, ax, ship_type, label, fraction, real)
    figure.text(0.01, 0.004, foot(ship_type, real), color=INK_2, fontsize=8)
    figure.tight_layout(rect=(0, 0.21, 1, 1))
    suffix = "-core" + ("-real" if real else "") + ("-dark" if THEME == "dark" else "")
    path = os.path.join(OUT, f"doghouse-{card.TYPE_NAMES[ship_type].replace('_', '-')}{suffix}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


def comparison():
    v = speeds.numpy()
    grids = [evaluate(t, None) for t in TYPES]
    keep = v >= 100.0
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 13), facecolor=SURFACE)
    panels = (
        ("turn rate, full stick and sustained (deg/s)", lambda g: (g["rate"][-1], sustained(g)[0])),
        ("turn radius, full stick and sustained (m)", lambda g: (g["radius"][-1], sustained(g)[1])),
        ("time for a sustained 360 (s)", lambda g: (None, 360.0 / sustained(g)[0])),
    )
    for ax, (title, pick) in zip(axes, panels):
        style(ax)
        for t, grid in enumerate(grids):
            full, held = pick(grid)
            if full is not None:
                ax.plot(v[keep], full[keep], color=SERIES[t], linewidth=2.0, label=f"{NAMES[t]} full stick")
            ax.plot(v[keep], held[keep], color=SERIES[t], linewidth=2.0, linestyle="--" if full is not None else "-",
                    label=f"{NAMES[t]} sustained")
            series = np.where(keep, held if full is None else full, np.nan)
            # The curves converge past their peaks: label inside the gaps.
            if "rate" in title:
                j, offset, align = int(np.searchsorted(v, 200.0)), (6, -14), "left"
            elif "radius" in title:
                j, offset, align = int(np.searchsorted(v, 200.0)), (0, -14 if t % 2 else 4), "left"
            else:
                j, offset, align = int(np.nanargmin(series)), (0, -16), "center"
            ax.annotate(NAMES[t], (v[j], series[j]), xytext=offset, textcoords="offset points", color=INK,
                        fontsize=9, ha=align)
        if "radius" in title:
            ax.set_ylim(0, 800)
        if "360" in title:
            ax.set_ylim(0, 20)
        ax.set_xlim(100.0, v[-1])
        ax.set_xlabel("speed (m/s)", color=INK_2)
        ax.set_title(title + ", coast, spawn store", color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, ncol=2)
    foot = ("Solid: full stick (instantaneous). Dashed: sustained (Ps >= 0, total energy; the engine never\n"
            f"feeds the store, so at coast this is speed held). Spawn store = {SPAWN_PERCENT:.0f}% of each ship's store.")
    figure.text(0.01, 0.005, foot, color=INK_2, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(OUT, f"doghouse-comparison{SUFFIX}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


if __name__ == "__main__":
    for ship_type in TYPES:
        for real in (False, True):
            print(doghouse(ship_type, real=real))
    print(comparison())
