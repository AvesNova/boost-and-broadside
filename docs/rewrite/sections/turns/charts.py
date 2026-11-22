"""Five flight-model charts: who wins the turn, acceleration, dive and climb, bleed, circles.

Quasi-steady values come from the flight code's own functions through doghouse.evaluate
(`core._setting`, `_lift_coefficient`, `_rates`, nose settled on the stick's angle of attack);
time histories step `lean.game.core.flight_tick` (no fields, no guns).

    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python docs/rewrite/sections/turns/charts.py

Writes chart-<name>.png into docs/rewrite/.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from doghouse import (  # noqa: E402
    DEGREES,
    INK,
    INK_2,
    MUTED,
    NAMES,
    OUT,
    SERIES,
    SURFACE,
    evaluate,
    speeds,
    style,
    sustained,
)
from sim import H, make, track  # noqa: E402

from lean.game import card, core  # noqa: E402

torch.set_num_threads(1)
plt.rcParams.update({"font.size": 10})
V = speeds.numpy()
MAX_ENERGY = card.CARD["max_energy"]
SPAWN = card.CARD["spawn_energy"]
SPAWN_PERCENT = 100.0 * SPAWN[0] / MAX_ENERGY[0]
FULL_STICK, STRAIGHT = 6, 0
COAST, DIVE, CLIMB = 0, 1, 2
FOOT = 8.5
TIE = 0.05  # deg/s: closer than this counts as equal


def save(figure, name, footnote, foot_height):
    figure.text(0.01, 0.004, footnote, color=INK_2, fontsize=FOOT, va="bottom")
    figure.tight_layout(rect=(0, foot_height, 1, 1))
    path = os.path.abspath(os.path.join(OUT, f"chart-{name}.png"))
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


def title(ax, text):
    ax.set_title(text, color=INK, fontsize=11, loc="left")


def axes_labels(ax, x, y):
    ax.set_xlabel(x, color=INK_2)
    ax.set_ylabel(y, color=INK_2)


def legend(ax, **kwargs):
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK_2, **kwargs)


def zero_crossing(x, y):
    """The x of the last downward zero crossing of y (linear interpolation), or nan."""
    sign = np.sign(y)
    down = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if not len(down):
        return math.nan
    i = down[-1]
    return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))


def simulate(ship_type, speed, energy, stick, power, ticks):
    """Step flight_tick for one ship; per tick: time, speed, store, turn rate deg/s, heading
    turned (deg, velocity direction, cumulative), x, y. Row 0 is the start."""
    constants, s = make(ship_type, torch.tensor([float(speed)]), float(energy))
    stick_t = torch.full((1, 1), stick, dtype=torch.long)
    power_t = torch.full((1, 1), power, dtype=torch.long)
    rows = [(0.0, speed, energy, 0.0, 0.0, 0.0, 0.0)]
    previous, turned = track(s), 0.0
    with torch.no_grad():
        for k in range(ticks):
            s = core.flight_tick(constants, H, s, stick_t, power_t)
            now = track(s)
            step = float(torch.remainder(now - previous + math.pi, 2 * math.pi) - math.pi)
            previous = now
            turned += abs(step)
            rows.append(
                (
                    (k + 1) * H,
                    float(s.speed),
                    float(s.energy),
                    abs(step) / H * DEGREES,
                    turned * DEGREES,
                    float(s.x),
                    float(s.y),
                )
            )
    names = ("time", "speed", "store", "rate", "turned", "x", "y")
    return {n: np.array(c) for n, c in zip(names, zip(*rows))}


def corner_speed(ship_type, fraction=None):
    envelope = evaluate(ship_type, fraction)["rate"][-1]
    return float(V[int(np.nanargmax(envelope))]), float(np.nanmax(envelope))


# --------------------------------------------------------------------------- 1. who wins


def segments(v, winner):
    """Runs of equal winner: list of (winner, v_start, v_end)."""
    runs, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or winner[i] != winner[start]:
            runs.append((winner[start], v[start], v[i - 1]))
            start = i
    return runs


def who_wins():
    grids = [evaluate(t, None) for t in range(3)]
    keep = V >= 100.0
    v = V[keep]
    pairs = ((1, 0), (2, 0), (1, 2))
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 16.5), facecolor=SURFACE)
    for ax, (a, b) in zip(axes, pairs):
        style(ax)
        full = {t: grids[t]["rate"][-1][keep] for t in (a, b)}
        held = {t: sustained(grids[t])[0][keep] for t in (a, b)}
        higher = np.where(full[a] >= full[b], a, b)
        for t in (a, b):
            ax.fill_between(
                v,
                full[a],
                full[b],
                where=(higher == t) & (np.abs(full[a] - full[b]) >= TIE),
                color=SERIES[t],
                alpha=0.18,
                linewidth=0,
                interpolate=True,
            )
        for t in (a, b):
            ax.plot(v, full[t], color=SERIES[t], linewidth=2.0, label=f"{NAMES[t]} full stick")
            ax.plot(v, held[t], color=SERIES[t], linewidth=2.0, linestyle="--", label=f"{NAMES[t]} sustained (Ps >= 0)")
            j = int(np.nanargmax(full[t]))
            ax.plot(v[j], full[t][j], "o", color=SERIES[t], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(
                f"{NAMES[t]} {full[t][j]:.0f} deg/s at {v[j]:.0f}",
                (v[j], full[t][j]),
                xytext=(-6, 8) if t == a else (40, 14),
                ha="right" if t == a else "left",
                textcoords="offset points",
                color=INK,
                fontsize=9,
                arrowprops=None if t == a else dict(arrowstyle="-", color=MUTED, linewidth=0.6),
            )
            k = int(np.nanargmax(held[t]))
            ax.plot(v[k], held[t][k], "s", color=SERIES[t], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        runs = segments(v, np.where(np.abs(full[a] - full[b]) < TIE, -1, higher))
        lines = []
        for w, lo, hi in runs:
            if w < 0:
                lines.append(f"equal {lo:.0f} to {hi:.0f} m/s (both at the g limit)")
            else:
                gap = np.max(np.abs(full[a] - full[b])[(v >= lo) & (v <= hi)])
                lines.append(f"{NAMES[w]} {lo:.0f} to {hi:.0f} m/s (by up to {gap:.1f} deg/s)")
        text = "Faster full-stick turn:\n" + "\n".join(lines) + "\nBest sustained (squares):"
        for t in (a, b):
            k = int(np.nanargmax(held[t]))
            text += f"\n{NAMES[t]} {held[t][k]:.0f} deg/s at {v[k]:.0f} m/s"
        ax.text(
            0.98,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="#e7e6e2", boxstyle="round,pad=0.4"),
        )
        ax.set_xlim(100.0, V[-1])
        ax.set_ylim(0, max(np.nanmax(full[a]), np.nanmax(full[b])) * 1.45)
        axes_labels(ax, "speed (m/s)", "turn rate (deg/s)")
        title(ax, f"{NAMES[a]} vs {NAMES[b]}")
        legend(ax, loc="upper left", ncol=1)
    foot = (
        "Coast, spawn store (40%), no fields; quasi-steady (nose settled on the stick's angle of attack, from\n"
        "core._setting/_lift_coefficient/_rates). Solid: full stick (instantaneous). Dashed: the highest rate with\n"
        "total energy rate Ps >= 0 (includes the engine surplus charging the store). Shading: the gap between the\n"
        "two full-stick curves, in the colour of the ship that turns faster there (equal: within 0.05 deg/s)."
    )
    return save(figure, "who-wins", foot, 0.035)


# --------------------------------------------------------------------------- 2. acceleration


def straight(ship_type, fraction, power):
    grid = evaluate(ship_type, fraction, power)
    speed_rate, store_rate = grid["speed_rate"][0], grid["store_rate"][0]
    return speed_rate, store_rate, speed_rate + store_rate / V  # last: dE/dt over v


def acceleration():
    stores = (("store 90%", 0.9), (f"spawn store ({SPAWN_PERCENT:.0f}%)", None), ("store 1%", 0.01))
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 16), facecolor=SURFACE)
    tops = {}
    curves = {(fraction, t): straight(t, fraction, COAST) for _, fraction in stores for t in range(3)}
    high = 1.12 * max(np.nanmax(c[0]) for c in curves.values())
    for ax, (label, fraction) in zip(axes, stores):
        style(ax)
        marks = []
        for t in range(3):
            speed_rate, _, energy_rate = curves[(fraction, t)]
            ax.plot(V, speed_rate, color=SERIES[t], linewidth=2.0, label=NAMES[t])
            ax.plot(V, energy_rate, color=SERIES[t], linewidth=1.6, linestyle="--")
            top = zero_crossing(V, speed_rate)
            tops[(label, t)] = top
            ax.plot(top, 0.0, "o", color=SERIES[t], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            marks.append(f"{NAMES[t]} {top:.0f} m/s")
        ax.axhline(0.0, color=MUTED, linewidth=0.9)
        ax.text(
            0.47,
            0.06,
            "Top level speed (dots):\n" + "\n".join(marks),
            transform=ax.transAxes,
            fontsize=9,
            color=INK,
            va="bottom",
            ha="left",
        )
        ax.plot([], [], color=MUTED, linewidth=2.0, label="solid: dv/dt")
        ax.plot([], [], color=MUTED, linewidth=1.6, linestyle="--", label="dashed: total energy rate / v")
        ax.set_ylim(-60.0, high)
        ax.set_xlim(V[0], V[-1])
        axes_labels(ax, "speed (m/s)", "rate (m/s per s)")
        title(ax, f"Straight flight, coast, {label}")
        legend(ax, loc="lower left", ncol=1)
    foot = (
        "Straight flight (no turn), coast, no fields; quasi-steady from core._setting/_rates. Solid: the speed's "
        "rate of\nchange. Dashed: the total energy rate d(v^2/2 + store)/dt divided by v, i.e. the acceleration "
        "if all of it went\ninto speed (the gap is the engine surplus charging the store). Lines leave the frame at "
        "the top (low speed)\nand the bottom (above 450 m/s); the frame is the same on all three panels."
    )
    return save(figure, "acceleration", foot, 0.04), tops


# --------------------------------------------------------------------------- 3. dive and climb


def dive_climb():
    figure, axes = plt.subplots(4, 1, figsize=(7.5, 21), facecolor=SURFACE)
    ax_speed, ax_store, ax_time_speed, ax_time_store = axes
    for ax in axes:
        style(ax)
    results = {}
    for t in range(3):
        for power, dash, word in ((DIVE, "-", "dive"), (CLIMB, "--", "climb")):
            speed_rate, store_rate, _ = straight(t, None, power)
            ax_speed.plot(V, speed_rate, color=SERIES[t], linestyle=dash, linewidth=2.0, label=f"{NAMES[t]} {word}")
            ax_store.plot(
                V,
                store_rate / MAX_ENERGY[t] * 100.0,
                color=SERIES[t],
                linestyle=dash,
                linewidth=2.0,
                label=f"{NAMES[t]} {word}",
            )
    for ax in (ax_speed, ax_store):
        ax.axhline(0.0, color=MUTED, linewidth=0.9)
        ax.set_xlim(V[0], V[-1])
        legend(ax, ncol=3, loc="upper right" if ax is ax_speed else "upper left")
    axes_labels(ax_speed, "speed (m/s)", "dv/dt (m/s per s)")
    title(ax_speed, f"(a) Speed rate, straight, spawn store ({SPAWN_PERCENT:.0f}%)")
    axes_labels(ax_store, "speed (m/s)", "store rate (% of max per s)")
    title(ax_store, f"(a) Store rate, straight, spawn store ({SPAWN_PERCENT:.0f}%)")
    ticks = int(round(30.0 / H))
    second = int(round(1.0 / H))
    lines = []
    for t in range(3):
        start = zero_crossing(V, straight(t, None, COAST)[0])
        climb_floor = zero_crossing(V[::-1], -straight(t, None, CLIMB)[0][::-1])  # climb's dv/dt = 0 from below
        dive = simulate(t, start, SPAWN[t], STRAIGHT, DIVE, ticks)
        band = card.STORE_FADE_BAND * MAX_ENERGY[t]
        emptied = dive["store"] <= band
        stop = int(np.argmax(emptied)) if emptied.any() else ticks
        climb = simulate(t, start, SPAWN[t], STRAIGHT, CLIMB, ticks)
        settled = np.zeros(len(climb["speed"]), bool)
        settled[second:] = np.abs(climb["speed"][second:] - climb["speed"][:-second]) < 0.1
        done = (climb["speed"] <= 100.0) | (climb["store"] >= MAX_ENERGY[t] * 0.999) | settled
        climb_stop = int(np.argmax(done)) if done.any() else ticks
        results[t] = dict(
            start=start,
            dive_time=dive["time"][stop],
            dive_speed=dive["speed"][stop],
            dive_peak=float(dive["speed"][: stop + 1].max()),
            dive_end_30s=dive["speed"][-1],
            dive_store_30s=dive["store"][-1] / MAX_ENERGY[t],
            climb_time=climb["time"][climb_stop],
            climb_speed=climb["speed"][climb_stop],
            climb_store=climb["store"][climb_stop] / MAX_ENERGY[t],
            climb_floor=climb_floor,
            climb_30s_speed=climb["speed"][-1],
            climb_30s_store=climb["store"][-1] / MAX_ENERGY[t],
        )
        for run, last, dash, word in ((dive, stop, "-", "dive"), (climb, climb_stop, "--", "climb")):
            sl = slice(0, last + 1)
            ax_time_speed.plot(
                run["time"][sl],
                run["speed"][sl],
                color=SERIES[t],
                linestyle=dash,
                linewidth=2.0,
                label=f"{NAMES[t]} {word}",
            )
            ax_time_store.plot(
                run["time"][sl],
                run["store"][sl] / MAX_ENERGY[t] * 100.0,
                color=SERIES[t],
                linestyle=dash,
                linewidth=2.0,
                label=f"{NAMES[t]} {word}",
            )
            for ax, key, scale in ((ax_time_speed, "speed", 1.0), (ax_time_store, "store", 100.0 / MAX_ENERGY[t])):
                ax.plot(
                    run["time"][last:],
                    run[key][last:] * scale,
                    color=SERIES[t],
                    linestyle=dash,
                    linewidth=1.0,
                    alpha=0.35,
                )
                ax.plot(
                    run["time"][last],
                    run[key][last] * scale,
                    "o",
                    color=SERIES[t],
                    markersize=8,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    zorder=5,
                )
        why = "settles" if settled[climb_stop] and climb["speed"][climb_stop] > 100.0 else "reaches"
        lines.append(
            f"{NAMES[t]}: start {start:.0f}; dive peak {results[t]['dive_peak']:.0f}, store in the "
            f"{100 * card.STORE_FADE_BAND:.0f}% band at {dive['time'][stop]:.1f} s;\n    climb {why} "
            f"{climb['speed'][climb_stop]:.0f} m/s at {climb['time'][climb_stop]:.1f} s"
        )
    ax_time_speed.text(
        0.99,
        0.56,
        "\n".join(lines),
        transform=ax_time_speed.transAxes,
        ha="right",
        va="center",
        multialignment="left",
        fontsize=8.5,
        color=INK,
        bbox=dict(facecolor=SURFACE, edgecolor="#e7e6e2", boxstyle="round,pad=0.4"),
    )
    axes_labels(ax_time_speed, "time (s)", "speed (m/s)")
    title(ax_time_speed, "(b) Simulated from coast top speed, spawn store (speeds in m/s)")
    axes_labels(ax_time_store, "time (s)", "store (% of max)")
    title(ax_time_store, "(b) Store over the same runs")
    ax_time_store.set_ylim(0, 105)
    for ax in (ax_time_speed, ax_time_store):
        ax.set_xlim(0, 30)
        legend(ax, ncol=3, loc="lower center" if ax is ax_time_speed else "upper center")
    ylow, yhigh = ax_time_speed.get_ylim()
    ax_time_speed.set_ylim(ylow - 0.25 * (yhigh - ylow), yhigh)
    foot = (
        "Straight flight, no fields. (a) quasi-steady from core._setting/_rates at the spawn store; dive = power 1 "
        "('boost',\nstore into speed), climb = power 2 ('reverse', speed into store). (b) flight_tick at 30 Hz from "
        "each ship's coast\ntop speed at spawn store. Dive held until the store enters the 2% fade band, where the "
        "code fades the exchange\nout (it never reaches zero); climb held until speed reaches 100 m/s, the store is "
        "full, or speed settles (< 0.1 m/s\nchange in 1 s). Dots: the stop; faint lines: the same hold continued to 30 s."
    )
    return save(figure, "dive-climb", foot, 0.04), results


# --------------------------------------------------------------------------- 4. bleed


def bleed():
    ticks = int(round(20.0 / H))
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 16), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    results, notes, differences = {}, [], []
    for t in range(3):
        start, peak = corner_speed(t)
        runs = {}
        for fraction, dash, width in ((None, "-", 2.4), (0.9, (0, (3, 3)), 1.4)):
            energy = SPAWN[t] if fraction is None else fraction * MAX_ENERGY[t]
            run = simulate(t, start, energy, FULL_STICK, COAST, ticks)
            tag = "spawn store" if fraction is None else "store 90%"
            runs[tag] = run
            colour = SERIES[t] if fraction is None else SURFACE
            label = NAMES[t] if fraction is None else None
            axes[0].plot(run["time"], run["speed"], color=colour, linestyle=dash, linewidth=width, label=label)
            axes[1].plot(run["time"][1:], run["rate"][1:], color=colour, linestyle=dash, linewidth=width, label=label)
            axes[2].plot(run["time"], run["turned"], color=colour, linestyle=dash, linewidth=width, label=label)
            five = int(round(5.0 / H))
            results[(t, tag)] = dict(
                start=start,
                peak=peak,
                speed5=run["speed"][five],
                end_speed=run["speed"][-1],
                end_rate=run["rate"][-1],
                max_rate=run["rate"][1:].max(),
                turned=run["turned"][-1],
                first360=run["time"][np.argmax(run["turned"] >= 360)],
                store_end=run["store"][-1] / MAX_ENERGY[t],
            )
        r = results[(t, "spawn store")]
        differences.append(float(np.abs(runs["spawn store"]["speed"] - runs["store 90%"]["speed"]).max()))
        notes.append(
            (
                f"{NAMES[t]}: {start:.0f} to {r['end_speed']:.0f} m/s",
                f"{NAMES[t]}: peak {r['max_rate']:.0f}, held {r['end_rate']:.0f} deg/s",
                f"{NAMES[t]}: {r['turned']:.0f} deg, first 360 at {r['first360']:.1f} s",
            )
        )
    for k, ax in enumerate(axes):
        ax.text(
            0.98,
            0.62 if k < 2 else 0.04,
            "\n".join(n[k] for n in notes),
            transform=ax.transAxes,
            ha="right",
            va="center" if k < 2 else "bottom",
            fontsize=9,
            color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="#e7e6e2", boxstyle="round,pad=0.4"),
        )
        ax.plot([], [], color=MUTED, linewidth=2.4, label="solid: spawn store (40%)")
        ax.plot([], [], color=MUTED, linewidth=1.4, linestyle=(0, (3, 3)), label="light dashes on top: store 90%")
        ax.set_xlim(0, 20)
        legend(ax, ncol=2, loc="upper right" if k < 2 else "upper left")
    top = max(r["turned"] for r in results.values()) * 1.3
    for k in range(1, int(top // 360) + 1):
        axes[2].axhline(360.0 * k, color=MUTED, linewidth=0.6, linestyle=":")
    axes_labels(axes[0], "time (s)", "speed (m/s)")
    title(axes[0], "Speed, full stick from corner speed, coast")
    axes_labels(axes[1], "time (s)", "turn rate (deg/s)")
    title(axes[1], "Turn rate (velocity direction)")
    axes_labels(axes[2], "time (s)", "degrees turned")
    title(axes[2], "Total turned (dotted: each full 360)")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.25 * (hi - lo))
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.3)
    axes[2].set_ylim(0, top)
    foot = (
        "flight_tick at 30 Hz, no fields, coast, full stick held 20 s from straight flight at each ship's corner "
        "speed (the peak\nof the quasi-steady full-stick envelope at spawn store, doghouse.evaluate). The store 90% "
        f"runs lie exactly on the spawn-store\nruns (largest speed difference {max(differences):.3g} m/s): at coast "
        "the store changes only the engine's charge, not thrust or lift."
    )
    return save(figure, "bleed", foot, 0.03), results


# --------------------------------------------------------------------------- 5. circles


def circle(ship_type, speed):
    run = simulate(ship_type, speed, SPAWN[ship_type], FULL_STICK, COAST, int(round(30.0 / H)))
    end = int(np.argmax(run["turned"] >= 360.0))
    return {k: v[: end + 1] for k, v in run.items()}


def circles():
    reach = card.CARD["muzzle_speed"][0] * card.CARD["bullet_life"][0]
    hit = card.CARD["hit_radius"][0]
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 15), facecolor=SURFACE)
    results = {}
    for ax, case in zip(axes, ("corner", 130.0)):
        style(ax)
        extent = []
        for t in range(3):
            start = corner_speed(t)[0] if case == "corner" else case
            run = circle(t, start)
            x, y = run["x"], -run["y"]  # flip so the turn draws counter-clockwise from the start
            if t == 0:
                sign = 1.0 if y[len(y) // 4] > 0 else -1.0
            y = y * sign
            ax.plot(x, y, color=SERIES[t], linewidth=2.0, label=f"{NAMES[t]}, start {start:.0f} m/s")
            ax.plot(x[-1], y[-1], "o", color=SERIES[t], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            width = float(np.ptp(y))
            results[(t, case)] = dict(
                start=start,
                time=run["time"][-1],
                width=width,
                end_speed=run["speed"][-1],
                reach_x=float(x.max()),
                drift=float(math.hypot(x[-1], y[-1])),
            )
            top = int(np.argmax(y))
            ax.annotate(
                f"{NAMES[t]}: {run['time'][-1]:.1f} s, {width:.0f} m across",
                (x[top], y[top]),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=9,
            )
            extent += [x.min(), x.max(), y.min(), y.max()]
        ax.plot(0, 0, "o", color=INK, markersize=5)
        ax.annotate("start, heading +x", (0, 0), xytext=(6, -14), textcoords="offset points", color=INK_2, fontsize=8.5)
        low_x, high_x, high_y = min(extent[0::4]), max(extent[1::4]), max(extent[3::4])
        bar_y = -0.12 * high_y
        ax.plot([low_x, low_x + reach], [bar_y, bar_y], color=INK, linewidth=3, solid_capstyle="butt")
        ax.annotate(
            f"gun reach {reach:.0f} m (muzzle {card.CARD['muzzle_speed'][0]:.0f} m/s x life "
            f"{card.CARD['bullet_life'][0]:.3f} s)",
            (low_x, bar_y),
            xytext=(0, -14),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
        )
        ax.add_patch(plt.Circle((0, 0), hit, color=INK, fill=False, linewidth=1))
        ax.set_aspect("equal")
        ax.set_xlim(min(low_x, -50) - 60, max(high_x, reach + low_x) + 60)
        ax.set_ylim(bar_y - 0.12 * high_y, high_y * 1.35)
        axes_labels(ax, "x (m)", "y (m)")
        title(
            ax,
            "(a) One 360 of heading from corner speed"
            if case == "corner"
            else f"(b) One 360 of heading from {case:.0f} m/s",
        )
        legend(ax, loc="upper right" if case == "corner" else "upper right")
    foot = (
        "flight_tick at 30 Hz, coast, spawn store (40%), no fields, full stick from straight flight, until the "
        "velocity\ndirection has turned 360 deg. All ships start at the origin heading +x; equal axes, to scale. "
        f"The small ring at\nthe start is the {hit:.0f} m hit radius to scale (a ship is a {2 * hit:.0f} m target). "
        "Dots: end of the 360. Gun reach\nignores bullet drag and the shooter's own velocity."
    )
    return save(figure, "circles", foot, 0.04), results, reach


if __name__ == "__main__":
    print(who_wins())
    path, tops = acceleration()
    print(path)
    for k, v in tops.items():
        print("  top", k, round(v, 1))
    path, results = dive_climb()
    print(path)
    for k, v in results.items():
        print("  ", NAMES[k], {a: round(float(b), 3) for a, b in v.items()})
    path, results = bleed()
    print(path)
    for k, v in results.items():
        print("  ", NAMES[k[0]], k[1], {a: round(float(b), 3) for a, b in v.items()})
    path, results, reach = circles()
    print(path, "reach", reach)
    for k, v in results.items():
        print("  ", NAMES[k[0]], k[1], {a: round(float(b), 2) for a, b in v.items()})
