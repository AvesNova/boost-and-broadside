"""Turn-performance (doghouse) charts of lean's flight model, which is v2's mathematics.

For every speed and stick fraction, the quasi-steady turn from the flight code's own
functions (`core._setting`, `_lift_coefficient`, `_rates`): the nose settled on the stick's
angle of attack, turn rate = (C_L(angle) v^2 + T sin angle) / v, and the rates of change of
speed and of the energy store. No fields, no guns. Power: coast.

Energy rate (Ps) is the total specific energy rate, d(v^2/2 + store)/dt, shown as an
altitude-equivalent rate in m/s at the game's gravity (39.24 m/s^2), as energy-manoeuvrability
diagrams show Ps. Positive = gaining energy.

    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python docs/rewrite/sections/turns/doghouse.py

FLIGHT_MODEL=core draws the core flight model (GameConfig.core()); the default is v2. The core
model's files get a "-core" suffix.

Writes PNGs next to docs/rewrite/turn-performance.md.
"""

import math
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from sim import CONFIG, MAX_ENERGY, MODEL, SPAWN, SUFFIX  # noqa: E402

from lean.game import card, core  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "..", "..")
DEGREES = 180.0 / math.pi
G = card.GRAVITY
NAMES = ("Yongma", "Jujak", "Vizia")
# Reference palette (dataviz skill, references/palette.md): categorical slots 1-3, diverging
# blue <-> red through a grey midpoint, chart surface and inks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
DIVERGING = LinearSegmentedColormap.from_list("ps", ["#e34948", "#f0efec", "#2a78d6"])
SPAWN_PERCENT = 100.0 * SPAWN[0] / MAX_ENERGY[0]
if MODEL == "core":
    STORES = (("store 97.5%: inside the top edge band", 0.975), (f"spawn store ({SPAWN_PERCENT:.0f}%)", None),
              ("store 2.5%: inside the bottom edge band", 0.025))
    VIZIA_ONLY = "The core game currently flies Vizia only; Yongma and Jujak are the card's other two types under the core physics."
else:
    STORES = (("high store (90%): engine at its full-store power", 0.9),
              ("spawn store (40%): engine surplus charging the store", None),
              ("low store (1%): weight back on the wing", 0.01))

speeds = torch.linspace(20.0, 470.0, 226)
sticks = torch.linspace(0.0, 1.0, 201)


def evaluate(ship_type: int, store_fraction, power: int = 0):
    """(stick, speed) grids: turn rate deg/s, radius m, Ps m/s, speed rate, store rate."""
    v = speeds.repeat(len(sticks))
    s = sticks.repeat_interleave(len(speeds))
    b = v.shape[0]
    p = SimpleNamespace(**core.ship_constants(CONFIG, torch.full((b,), ship_type)))
    store = SPAWN[ship_type] if store_fraction is None else store_fraction * MAX_ENERGY[ship_type]
    energy = torch.full((b,), float(store))
    control = SimpleNamespace(
        model=MODEL, stick_size=s, stick_sign=torch.ones(b),
        commanded=p.commanded_table[:, power], demand=p.demand_table[:, power],
    )
    control.inverse_demand = 1.0 / control.demand
    with torch.no_grad():
        setting = core._setting(p, control, v, energy)
        lateral, _, _, thrust, _ = setting
        reach = (lateral.abs() / (v * v) * p.inverse_lift).clamp(0.0, 1.0)
        angle = p.stall * (1.0 - torch.sqrt(1.0 - reach))
        lift_coefficient = core._lift_coefficient(p, angle)
        rate = (lift_coefficient * v * v + thrust * torch.sin(angle)) / v
        speed_rate, store_rate = core._rates(p, setting, torch.ones(b), v, lift_coefficient, torch.cos(angle), 1.0 / (v * v))
    shape = (len(sticks), len(speeds))
    rate = rate.reshape(shape).numpy()
    return {
        "rate": rate * DEGREES,
        "radius": (v.reshape(shape).numpy() / np.maximum(rate, 1e-9)),
        "ps": ((v * speed_rate + store_rate) / G).reshape(shape).numpy(),
        "speed_rate": speed_rate.reshape(shape).numpy(),
        "store_rate": store_rate.reshape(shape).numpy(),
    }


def sustained(grid):
    """Per speed: the largest turn rate with Ps >= 0 (nan if none), and its radius."""
    rate, ps, radius = grid["rate"], grid["ps"], grid["radius"]
    best_rate = np.full(rate.shape[1], np.nan)
    best_radius = np.full(rate.shape[1], np.nan)
    for j in range(rate.shape[1]):
        ok = np.where(ps[:, j] >= 0.0)[0]
        if len(ok):
            i = ok[np.argmax(rate[ok, j])]
            best_rate[j], best_radius[j] = rate[i, j], radius[i, j]
    return best_rate, best_radius


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color="#e7e6e2", linewidth=0.6)
    ax.set_axisbelow(True)


DOGHOUSE_FOOT = {
    "v2": "Thick: full stick (lift and g limit). Thin: the game's sustained and fine stick.\n"
    "Dash-dot: full stick while diving or climbing (on the full-stick line whenever the store is\n"
    "above 2%: today diving and climbing do not change the turn). Black dashed: Ps = 0 (sustained).\n"
    "Grey dotted: turn radius. Grey dashed: load in game g (39.24 m/s^2). Shaded below 100 m/s:\n"
    "quasi-steady maths only; hard turns there cannot be held in flight.",
    "core": "Core model: 1 g of weight always on the wing, the turn gets the rest of the lift and g limit;\n"
    "the engine drives speed only. Thick: full stick. Thin: the game's sustained and fine stick.\n"
    "Dash-dot: full stick while diving or climbing (diving lowers the weight on the wing, climbing\n"
    "raises it). Black dashed: Ps = 0 (sustained). Grey dotted: turn radius. Grey dashed: load in\n"
    "game g (39.24 m/s^2). Shaded below 100 m/s: quasi-steady maths only, no best-sustained mark there.\n"
    "Coasting, turn rate and Ps do not depend on the store. Inside an edge band (the outer 5%) dive or climb\n"
    "fades to coast, in the trade and in the wing's load: at 97.5% and 2.5% the band is half faded.\n"
    + "The core game currently flies Vizia only; Yongma and Jujak are the card's other types under core physics.",
}


def doghouse(ship_type: int):
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 16), facecolor=SURFACE)
    v = speeds.numpy()
    for ax, (label, fraction) in zip(axes, STORES):
        grid = evaluate(ship_type, fraction)
        style(ax)
        envelope = grid["rate"][-1]
        top = float(np.nanmax(envelope)) * 1.15
        rate_axis = np.linspace(0.0, top, 300)
        # Ps on a regular (speed, rate) grid: for each speed, interpolate over the stick sweep.
        ps_field = np.full((len(rate_axis), len(v)), np.nan)
        for j in range(len(v)):
            column_rate, column_ps = grid["rate"][:, j], grid["ps"][:, j]
            order = np.argsort(column_rate)
            inside = rate_axis <= column_rate[order][-1]
            ps_field[inside, j] = np.interp(rate_axis[inside], column_rate[order], column_ps[order])
        limit = np.nanpercentile(np.abs(ps_field), 98)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        filled = ax.contourf(v, rate_axis, ps_field, levels=21, cmap=DIVERGING, norm=norm)
        ax.contour(v, rate_axis, ps_field, levels=[0.0], colors=INK, linewidths=2.0, linestyles=[(0, (6, 3))])
        # Radius and load lines: geometry only.
        for radius in (150, 200, 300, 500, 1000, 2000):
            line = v / radius * DEGREES
            ax.plot(v, line, color=MUTED, linewidth=0.7, linestyle=":")
            j = np.searchsorted(line, top * 0.92)
            if 0 < j < len(v):
                ax.text(v[j], line[j], f"{radius} m", color=MUTED, fontsize=8, ha="right", va="bottom")
        for load in (2, 4, 6, 8, 9):
            line = load * G / v * DEGREES
            ax.plot(v, line, color=MUTED, linewidth=0.7, linestyle="--")
            j = np.searchsorted(-line, -top * 0.95)
            if 0 < j < len(v) - 1:
                ax.text(v[j], line[j], f"{load} g", color=MUTED, fontsize=8, ha="left", va="bottom")
        ax.plot(v, envelope, color=INK, linewidth=2.0)
        label_at = int(np.searchsorted(v, 135.0))
        ax.annotate("full stick", (v[label_at], envelope[label_at]), xytext=(-4, 6), textcoords="offset points",
                    color=INK, fontsize=8, ha="right")
        # The game's other stick levels (the card's fine and sustained values for this type).
        for name, key in (("sustained stick", "sustained_stick"), ("fine stick", "fine_stick")):
            row = int(round(card.CARD[key][ship_type] * (len(sticks) - 1)))
            level = grid["rate"][row]
            ax.plot(v, level, color=INK, linewidth=1.1)
            if MODEL == "core":  # right of the steep low-speed part of the full-stick line
                at = int(np.searchsorted(v, 215.0))
                ax.annotate(name, (v[at], level[at]), xytext=(4, -12), textcoords="offset points",
                            color=INK, fontsize=8, ha="left")
            else:
                ax.annotate(name, (v[label_at], level[label_at]), xytext=(-4, 4), textcoords="offset points",
                            color=INK, fontsize=8, ha="right")
        # Full stick while diving and while climbing (power 1 and 2), same store.
        offsets = ((4, 7), (4, -24)) if MODEL == "core" else ((4, 4), (4, -12))
        for (power, name), offset in zip(((1, "full stick, diving"), (2, "full stick, climbing")), offsets):
            moved = evaluate(ship_type, fraction, power)["rate"][-1]
            ax.plot(v, moved, color=INK_2, linewidth=1.1, linestyle="-.")
            here = int(np.searchsorted(v, 380.0))
            ax.annotate(name, (v[here], moved[here]), xytext=offset, textcoords="offset points",
                        color=INK_2, fontsize=8)
        corner = int(np.nanargmax(envelope))
        ax.plot(v[corner], envelope[corner], "o", color=INK, markersize=8)
        ax.annotate(f"corner {v[corner]:.0f} m/s, {envelope[corner]:.0f} deg/s",
                    (v[corner], envelope[corner]), xytext=(8, 6), textcoords="offset points",
                    color=INK, fontsize=9)
        best, _ = sustained(grid)
        best = np.where(v >= 100.0, best, np.nan)  # below 100 m/s the turn cannot be held in flight
        if np.any(np.isfinite(best)):
            k = int(np.nanargmax(best))
            ax.plot(v[k], best[k], "s", color=INK, markersize=8)
            ax.annotate(f"best sustained {v[k]:.0f} m/s, {best[k]:.0f} deg/s", (v[k], best[k]),
                        xytext=(8, -14), textcoords="offset points", color=INK, fontsize=9)
        ax.set_xlim(v[0], v[-1])
        ax.set_ylim(0, top)
        ax.set_xlabel("speed (m/s)", color=INK_2)
        ax.set_ylabel("turn rate (deg/s)", color=INK_2)
        ax.axvspan(v[0], 100.0, color="#e7e6e2", alpha=0.5, linewidth=0)
        ax.set_title(f"{NAMES[ship_type]}, coast, {label}" + (", core model" if MODEL == "core" else ""),
                     color=INK, fontsize=10, loc="left")
        bar = figure.colorbar(filled, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("energy rate Ps, m/s altitude-equivalent (blue gain, red loss)", color=INK_2, fontsize=8)
        bar.ax.tick_params(labelsize=8, colors=INK_2)
    figure.text(0.01, 0.004, DOGHOUSE_FOOT[MODEL], color=INK_2, fontsize=8)
    figure.tight_layout(rect=(0, 0.055 if MODEL == "v2" else 0.08, 1, 1))
    path = os.path.join(OUT, f"doghouse-{NAMES[ship_type].lower()}{SUFFIX}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


def comparison():
    v = speeds.numpy()
    grids = [evaluate(t, None) for t in range(3)]
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
            if MODEL == "core":  # the core curves converge past their peaks: label inside the gaps
                if "rate" in title:
                    j, offset, align = int(np.searchsorted(v, 200.0)), (6, -14), "left"
                elif "radius" in title:
                    j, offset, align = int(np.searchsorted(v, 200.0)), (0, (4, -14, 4)[t]), "left"
                else:
                    j, offset, align = int(np.nanargmin(series)), (0, -16), "center"
                ax.annotate(NAMES[t], (v[j], series[j]), xytext=offset, textcoords="offset points", color=INK,
                            fontsize=9, ha=align)
                continue
            if "rate" in title:
                j = int(np.nanargmax(series))
                offset = (4, 4)
            else:
                j = int(np.searchsorted(v, 200.0 if "radius" in title else 130.0))
                offset = (0, 4) if "radius" in title else (0, (6, -14, -4)[t])
            ax.annotate(NAMES[t], (v[j], series[j]), xytext=offset, textcoords="offset points", color=INK, fontsize=9)
        if "radius" in title:
            ax.set_ylim(0, 800)
        if "360" in title:
            ax.set_ylim(0, 20)
        ax.set_xlim(100.0, v[-1])
        ax.set_xlabel("speed (m/s)", color=INK_2)
        ax.set_title(title + ", coast, spawn store" + (", core" if MODEL == "core" else ""), color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, ncol=2)
    if MODEL == "core":
        foot = ("Core model. Solid: full stick (instantaneous). Dashed: sustained (Ps >= 0, total energy; the engine never\n"
                f"feeds the store, so at coast this is speed held). Spawn store = {SPAWN_PERCENT:.0f}% of each ship's store.\n"
                + VIZIA_ONLY)
    else:
        foot = ("Solid: full stick (instantaneous). Dashed: sustained (Ps >= 0, total energy, so it includes "
                "the engine surplus charging the store at 40%).")
    figure.text(0.01, 0.005, foot, color=INK_2, fontsize=8)
    figure.tight_layout(rect=(0, 0.015 if MODEL == "v2" else 0.04, 1, 1))
    path = os.path.join(OUT, f"doghouse-comparison{SUFFIX}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


if __name__ == "__main__":
    for ship_type in range(3):
        print(doghouse(ship_type))
    print(comparison())
