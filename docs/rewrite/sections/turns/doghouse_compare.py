"""The P-51D against Vizia under the core flight model: the two doghouses side by side on shared
axes and Ps colours, and both envelopes overlaid, in game units and at 1x real time.

At 1x real time speeds, turn rates and Ps are the game's divided by the time factor (2); radii
and loads in g are unchanged, so the P-51D's chart reads directly against published real P-51D
energy-manoeuvrability diagrams (a top axis gives mph).

    for theme in light dark; do THEME=$theme OMP_NUM_THREADS=1 PYTHONPATH=.:src \\
        python docs/rewrite/sections/turns/doghouse_compare.py; done

Writes doghouse-p51d-vizia-core[-real][-dark].png next to this script.
"""

import os
import sys

os.environ["FLIGHT_MODEL"] = "core"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
from doghouse import (  # noqa: E402
    DEGREES,
    G,
    GRID,
    INK,
    INK_2,
    MUTED,
    NAMES,
    SERIES,
    STORES,
    SURFACE,
    THEME,
    TIME_FACTOR,
    draw_panel,
    evaluate,
    plt,
    speeds,
    style,
    sustained,
)

from lean.game import card  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MPH = 0.44704  # m/s
TYPES = (card.TYPE_NAMES.index("p51d"), card.TYPE_NAMES.index("vizia"))


def compare(real: bool) -> str:
    f = TIME_FACTOR if real else 1.0
    game_v = speeds.numpy()
    v = game_v / f
    label, fraction = STORES[0]
    grids = {t: evaluate(t, fraction) for t in TYPES}
    top = max(float(np.nanmax(g["rate"][-1])) for g in grids.values()) / f * 1.15
    figure = plt.figure(figsize=(14.0, 12.5), facecolor=SURFACE)
    layout = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
    # One Ps colour range for both panels: the smaller of the two ships' 98th percentiles (the
    # colour bar's end arrows mark Ps beyond it: Vizia's drag rise at high speed).
    limit = min(float(np.nanpercentile(np.abs(g["ps"]), 98)) for g in grids.values()) / f
    for column, ship_type in enumerate(TYPES):
        ax = figure.add_subplot(layout[0, column])
        draw_panel(figure, ax, ship_type, label, fraction, real, top=top, limit=limit)
        if real:
            ax.secondary_xaxis("top", functions=(lambda x: x / MPH, lambda x: x * MPH)).set_xlabel(
                "mph", color=INK_2, fontsize=8)
    ax = figure.add_subplot(layout[1, :])
    style(ax)
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
    keep = game_v >= 100.0  # below 100 m/s game the turn cannot be held in flight
    for colour, ship_type in zip(SERIES, TYPES):
        grid = grids[ship_type]
        full = grid["rate"][-1] / f
        held = np.where(keep, sustained(grid)[0] / f, np.nan)
        name = NAMES[ship_type]
        ax.plot(v, full, color=colour, linewidth=2.0, label=f"{name}: full stick (instantaneous)")
        ax.plot(v, held, color=colour, linewidth=2.0, linestyle="--", label=f"{name}: sustained (Ps = 0)")
        corner = int(np.nanargmax(full))
        best = int(np.nanargmax(held))
        ax.plot(v[corner], full[corner], "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.plot(v[best], held[best], "s", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        # Labels on opposite sides: the two ships' points sit close together.
        right = full[corner] >= max(float(np.nanmax(g["rate"][-1])) for g in grids.values()) / f
        ax.annotate(f"{name} corner {v[corner]:.0f} m/s, {full[corner]:.1f} deg/s", (v[corner], full[corner]),
                    xytext=(10, 6) if right else (-10, -16), textcoords="offset points", color=INK, fontsize=9,
                    ha="left" if right else "right")
        right = held[best] < max(float(np.nanmax(sustained(g)[0][keep])) for g in grids.values()) / f
        ax.annotate(f"{name} best sustained {v[best]:.0f} m/s, {held[best]:.1f} deg/s", (v[best], held[best]),
                    xytext=(10, -16) if right else (-60, 45), textcoords="offset points", color=INK, fontsize=9,
                    ha="left" if right else "right",
                    arrowprops=None if right else dict(arrowstyle="-", color=INK_2, linewidth=0.8))
    ax.axvspan(v[0], 100.0 / f, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel("speed (m/s, 1x real time)" if real else "speed (m/s, game)", color=INK_2)
    ax.set_ylabel("turn rate (deg/s)", color=INK_2)
    if real:
        ax.secondary_xaxis("top", functions=(lambda x: x / MPH, lambda x: x * MPH)).set_xlabel(
            "mph", color=INK_2, fontsize=8)
    ax.set_title(f"{NAMES[TYPES[0]]} against {NAMES[TYPES[1]]}, coast, core model"
                 + (", 1x real time" if real else ", game units (2x real time)"), color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.42))
    units = ("1x real time: the game's speeds, turn rates and Ps divided by 2; radii and loads in g (9.81 m/s^2) unchanged."
             if real else "Game units (2x real time): speeds and turn rates x2, Ps x2, g = 39.24 m/s^2; radii and loads unchanged.")
    foot = ("Core model, coasting, any store outside the edge bands. Top: each ship's doghouse (shared axes and Ps colours). "
            "Thick: full stick; thin: the game's\nsustained and fine stick; dash-dot: full stick diving and climbing; "
            "black dashed: Ps = 0. Bottom: both envelopes; circle corner, square best sustained.\n"
            "P-51D: the real airframe (lean.game.tuning, docs/rewrite/p51d-model.md) with the game terms off: its point mass, "
            "the October fit. Vizia: v2's card with its game terms.\n" + units
            + (" Shaded below 50 m/s: quasi-steady maths only." if real else " Shaded below 100 m/s: quasi-steady maths only."))
    if THEME == "dark":
        foot = foot.replace("black dashed", "white dashed")
    figure.text(0.01, 0.005, foot, color=INK_2, fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    suffix = ("-real" if real else "") + ("-dark" if THEME == "dark" else "")
    path = os.path.join(HERE, f"doghouse-p51d-vizia-core{suffix}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


if __name__ == "__main__":
    for real in (False, True):
        print(compare(real))
