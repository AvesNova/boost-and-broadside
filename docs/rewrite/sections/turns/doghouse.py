"""Turn-performance (doghouse) charts of lean's flight model, in game units and at 1x real time
(speeds, turn rates and Ps divided by the time factor; radii and loads in g unchanged), so a real
aircraft's chart reads against published energy-manoeuvrability diagrams. The web tool
(lean/tool) draws its charts with this module's functions: one drawing, matplotlib.

For every speed and stick fraction, the quasi-steady turn from the flight code's own function
(`lean.game.tuning.envelope`, which calls `core.steady_rates`): the nose settled on the stick's
angle of attack, turn rate = (C_L(angle) v^2 + T sin angle) / v, and the rates of change of
speed and of the energy store. No fields, no guns. Coast unless the line says dive or climb.

Only what the game flies is drawn: each line starts at the lowest speed at which a ship flown
in the game (lean.tool.flown.achievable_speeds: the game's flight tick from the spawn state)
actually flies that condition, its turn within 5% of the line. The Ps shading stops where no
stick level is flown. The corner, best sustained and tightest sustained points are flown too
(lean.tool.flown.critical_points) and drawn as small hollow markers beside the quasi-steady ones.

Energy rate (Ps) is the total specific energy rate, d(v^2/2 + store)/dt, shown as an
altitude-equivalent rate in m/s at the game's gravity, as energy-manoeuvrability diagrams show
Ps. Positive = gaining energy.

    for theme in light dark; do THEME=$theme OMP_NUM_THREADS=4 PYTHONPATH=.:src \
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
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from sim import THEME  # noqa: E402

from lean.game import card, tuning  # noqa: E402
from lean.tool import flown, units  # noqa: E402

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), "..", "..")
DEGREES = 180.0 / math.pi
TIME_FACTOR = tuning.TIME_FACTOR  # the game against real time
DISPLAY_NAMES = {"p51d": "P-51D"}
SPEED_TOP = (
    470.0  # game m/s at the game's time factor: the charts' right edge, scaled with the factor
)
SPEED_POINTS, STICKS = 226, np.linspace(0.0, 1.0, 201)
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
FLOWN = SERIES[1]  # the flown checks' markers
matplotlib.rcParams.update(
    {
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": GRID,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)
# Line styles, shared by the lines and the legend's samples.
FULL_STYLE = dict(color=INK, linewidth=2.2)
STICK_STYLE = dict(color=INK, linewidth=1.0)
TRADE_STYLE = dict(color=INK_2, linewidth=1.1, linestyle="-.")
PS_ZERO_STYLE = dict(color=INK, linewidth=1.8, linestyle=(0, (6, 3)))
RADIUS_STYLE = dict(color=MUTED, linewidth=0.7, linestyle=":")
LOAD_STYLE = dict(color=MUTED, linewidth=0.7, linestyle="--")
MARKERS = {"corner": "o", "best_sustained": "s", "tightest_sustained": "^"}


def _row() -> dict[str, float]:
    """The card's type 0 (the design inside lean.game.tuning.flying)."""
    return {name: values[0] for name, values in card.CARD.items()}


def speed_axis(lowest: dict[str, float], factor: float = TIME_FACTOR) -> np.ndarray:
    """Game speeds from the lowest flown speed of any line to the chart's right edge."""
    return np.linspace(min(lowest.values()), SPEED_TOP * factor / TIME_FACTOR, SPEED_POINTS)


def lowest_speeds() -> dict[str, float]:
    """Per line (lean.tool.flown.CONDITIONS), the lowest speed the game flies it (game m/s), for
    the card's type 0. Flies about a thousand ticks: cache it."""
    return flown.achievable_speeds()


def key_points(lowest: dict[str, float], fraction=None) -> dict:
    """The quasi-steady corner, best sustained and tightest sustained points (game units; the
    sustained ones from the lowest full-stick speed up), from tuning.envelope."""
    grid = tuning.envelope(
        _row(), speed_axis(lowest), STICKS, 0, fraction, minimum_speed=lowest["full"]
    )
    return {
        name: grid[name]
        for name in (
            "corner_speed",
            "corner_rate",
            "best_sustained_speed",
            "best_sustained_rate",
            "tightest_sustained_speed",
            "tightest_sustained_radius",
        )
    }


def flown_points(lowest: dict[str, float]) -> dict:
    """The key points flown in the game (lean.tool.flown.critical_points), game units."""
    key = key_points(lowest)
    return flown.critical_points(
        key["corner_speed"], key["best_sustained_speed"], key["tightest_sustained_speed"]
    )


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _from(values: np.ndarray, speeds: np.ndarray, lowest: float) -> np.ndarray:
    """values with nan below the lowest flown speed: no line where the game does not fly it."""
    return np.where(speeds >= lowest, values, np.nan)


def draw_panel(
    figure,
    ax,
    title: str,
    lowest: dict[str, float],
    real: bool = False,
    factor: float = TIME_FACTOR,
    flown_points: dict | None = None,
    fraction=None,
    top=None,
    limit=None,
    system: str | None = None,
):
    """One doghouse panel on ax for the card's type 0 at `factor` times real time (inside
    lean.game.tuning.flying): Ps shading where some stick level is flown, Ps = 0, the stick
    levels, dive and climb, the corner, best sustained and tightest sustained points, and the
    flown points (game units, as flown.critical_points returns them) if given. `lowest`: game
    m/s per line (lowest_speeds). real: divide speeds, rates and Ps by `factor` (1x real time);
    system (lean.tool.units: game, real_metric, real_imperial) overrides it and sets every axis,
    label and guide in that system's units.
    top (deg/s) and limit (the Ps colour range) default to the panel's own; returns both."""
    system = _system(real, system)
    sv, sr, sl, sc = (
        units.factor(kind, system, factor) for kind in ("speed", "angular_rate", "length", "climb")
    )  # game values times these are displayed
    speed_unit, length_unit = units.label("speed", system), units.label("length", system)
    row = _row()
    game_v = speed_axis(lowest, factor)
    v = game_v * sv
    grid = tuning.envelope(row, game_v, STICKS, 0, fraction, minimum_speed=lowest["full"])
    rate, ps = grid["rate"] * sr, grid["ps"] * sc
    style(ax)
    full = _from(rate[-1], game_v, lowest["full"])
    top = float(np.nanmax(full)) * 1.15 if top is None else top
    rate_axis = np.linspace(0.0, top, 300)
    field = tuning.ps_over_rate(grid, rate_axis, rate=rate, ps=ps)
    # Shade only up to the largest stick flown at each speed: the stick levels' lowest speeds,
    # linear between them, none below the fine stick's.
    stick = tuning.stick_levels(row)
    levels = sorted(
        (
            (stick["fine"], lowest["fine"]),
            (stick["sustained"], lowest["sustained"]),
            (1.0, lowest["full"]),
        )
    )
    level_sticks = [stick for stick, _ in levels]
    level_speeds = np.maximum.accumulate([speed for _, speed in levels])
    reach = np.interp(game_v, level_speeds, level_sticks, left=np.nan)
    reach_rate = np.array(
        [np.interp(s, STICKS, rate[:, j]) if np.isfinite(s) else -1.0 for j, s in enumerate(reach)]
    )
    field[rate_axis[:, None] > reach_rate[None, :]] = np.nan
    shared = limit is not None  # a given range: fixed levels, so panels' colours match
    limit = float(np.nanpercentile(np.abs(field), 98)) if limit is None else limit
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    filled = ax.contourf(
        v,
        rate_axis,
        field,
        levels=np.linspace(-limit, limit, 21) if shared else 21,
        cmap=DIVERGING,
        norm=norm,
        extend="both" if shared else "neither",
    )
    ax.contour(
        v,
        rate_axis,
        field,
        levels=[0.0],
        colors=PS_ZERO_STYLE["color"],
        linewidths=PS_ZERO_STYLE["linewidth"],
        linestyles=[PS_ZERO_STYLE["linestyle"]],
    )
    # Radius and load guides: geometry only, across the axes.
    left = math.floor(0.8 * min(lowest.values()) * sv / 10.0) * 10.0
    guide_v = np.linspace(max(left, 1.0), v[-1], 200)
    for radius in units.RADIUS_GUIDES[system]:
        line = guide_v / sv / (radius / sl) * sr
        ax.plot(guide_v, line, **RADIUS_STYLE)
        j = np.searchsorted(line, top * 0.92)
        if 0 < j < len(guide_v):
            ax.text(
                guide_v[j],
                line[j],
                f"{radius} {length_unit}",
                color=MUTED,
                fontsize=8,
                ha="right",
                va="bottom",
            )
    for load in (2, 4, 6, 8, 9):
        line = load * card.GRAVITY * sv / guide_v * sr
        ax.plot(guide_v, line, **LOAD_STYLE)
        j = np.searchsorted(-line, -top * 0.95)
        if 0 < j < len(guide_v) - 1:
            ax.text(
                guide_v[j], line[j], f"{load} g", color=MUTED, fontsize=8, ha="left", va="bottom"
            )
    ax.plot(v, full, gid="full", **FULL_STYLE)
    span = v[-1] - v[0]
    for name, key in (("sustained stick", "sustained"), ("fine stick", "fine")):
        level = _from(rate[int(round(stick[key] * (len(STICKS) - 1)))], game_v, lowest[key])
        ax.plot(v, level, gid=key, **STICK_STYLE)
        at = int(np.searchsorted(v, v[0] + 0.33 * span))
        ax.annotate(
            name,
            (v[at], level[at]),
            xytext=(4, -12),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    for power, name, offset in ((1, "dive", (-4, 2)), (2, "climb", (-4, -10))):
        moved = tuning.envelope(row, game_v, np.array([1.0]), power, fraction)["rate"][-1] * sr
        moved = _from(moved, game_v, lowest[name])
        ax.plot(v, moved, gid=name, **TRADE_STYLE)
        start = int(
            np.argmax(np.isfinite(moved))
        )  # labelled where it starts (they meet the full stick higher up)
        ax.annotate(
            name,
            (v[start], moved[start]),
            xytext=offset,
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            ha="right",
        )
    # The key points, quasi-steady (filled) and flown (hollow, beside them).
    points = {"corner": (grid["corner_speed"], grid["corner_rate"])}
    if grid["best_sustained_speed"] is not None:
        points["best_sustained"] = (grid["best_sustained_speed"], grid["best_sustained_rate"])
        tight = grid["tightest_sustained_speed"]
        points["tightest_sustained"] = (tight, tight / grid["tightest_sustained_radius"])
    offsets = {"corner": (8, 6), "best_sustained": (8, -14), "tightest_sustained": (14, -50)}
    for name, (speed, turn) in points.items():
        x, y = speed * sv, turn * sr
        ax.plot(x, y, MARKERS[name], color=INK, markersize=7)
        if name == "tightest_sustained":
            text = f"tightest sustained {x:.0f} {speed_unit}, {speed / turn * sl:.0f} {length_unit}"
        else:
            text = f"{name.replace('_', ' ')} {x:.0f} {speed_unit}, {y:.0f} deg/s"
        check = (flown_points or {}).get(name)
        if check is not None:
            fx, fy = check["speed"] * sv, check["rate"] * sr
            ax.plot(
                fx,
                fy,
                MARKERS[name],
                markerfacecolor="none",
                markeredgecolor=FLOWN,
                markersize=5,
                markeredgewidth=1.3,
            )
            text += (
                f" (flown {fx:.0f}, {check['radius'] * sl:.0f} {length_unit})"
                if name == "tightest_sustained"
                else f" (flown {fx:.0f}, {fy:.0f})"
            )
        arrow = (
            dict(arrowstyle="-", color=INK, linewidth=0.8) if name == "tightest_sustained" else None
        )
        ax.annotate(
            text,
            (x, y),
            xytext=offsets[name],
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
            arrowprops=arrow,
        )
    ax.set_xlim(left, v[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel(f"speed ({speed_unit}{_real_suffix(system)})", color=INK_2)
    ax.set_ylabel("turn rate (deg/s)", color=INK_2)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    bar = figure.colorbar(filled, ax=ax, location="bottom", fraction=0.04, pad=0.0, aspect=40)
    bar.set_label(
        f"Ps, {units.label('climb', system)} altitude-equivalent (blue gain, red loss)",
        color=INK_2,
        fontsize=8,
    )
    bar.ax.tick_params(labelsize=8, colors=INK_2)
    return top, limit


def legend(figure, note: str, flown_shown: bool, length_unit: str = "m"):
    """The legend below the plot: each line's own style, the markers, and a short note."""

    def sample(label, **style):
        return Line2D([], [], label=label, **style)

    def point(label, marker, **style):
        return Line2D([], [], linestyle="none", marker=marker, label=label, markersize=6, **style)

    handles = [
        sample("full stick", **FULL_STYLE),
        sample("sustained and fine stick", **STICK_STYLE),
        sample("full stick, diving and climbing", **TRADE_STYLE),
        sample("Ps = 0 (sustained)", **PS_ZERO_STYLE),
        sample(f"turn radius ({length_unit})", **RADIUS_STYLE),
        sample("load (g)", **LOAD_STYLE),
        point("corner", "o", color=INK),
        point("best sustained (rate)", "s", color=INK),
        point("tightest sustained (radius)", "^", color=INK),
    ]
    if flown_shown:
        handles.append(
            point(
                "flown in the game",
                "o",
                markerfacecolor="none",
                markeredgecolor=FLOWN,
                markeredgewidth=1.3,
            )
        )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        title=note,
        title_fontsize=8,
        alignment="left",
        labelcolor=INK_2,
        handlelength=3.2,
        columnspacing=1.2,
    )


def _system(real: bool, system: str | None) -> str:
    """The units system: `system` if given, else real_metric for real, else game."""
    return units.system_of(system if system is not None else ("real_metric" if real else "game"))


def _real_suffix(system: str) -> str:
    return ", real" if system == "real_metric" else ""


def note(real: bool, factor: float, lowest_source: str = "flown", system: str | None = None) -> str:
    unit_note = units.describe(_system(real, system), factor)
    source = {
        "flown": "Lines start at the lowest speed the game flies them.",
        "estimate": "Lines start at the preset's flown lowest speeds x the stall speed ratio.",
    }[lowest_source]
    head = "Coast, spawn store, no fields. Ps: total energy rate as a climb rate."
    return f"{head}\n{unit_note}\n{source}"


def doghouse_figure(
    title: str,
    lowest: dict[str, float],
    real: bool = False,
    factor: float = TIME_FACTOR,
    flown_points: dict | None = None,
    lowest_source: str = "flown",
    system: str | None = None,
):
    """The doghouse of the card's type 0 (inside tuning.flying) as a figure: the panel, the
    legend and the note (units: real, or `system`)."""
    figure = plt.figure(figsize=(6.4, 8.4), layout="constrained", facecolor=SURFACE)
    ax = figure.add_subplot()
    draw_panel(figure, ax, title, lowest, real, factor, flown_points, system=system)
    legend(
        figure,
        note(real, factor, lowest_source, system),
        flown_points is not None,
        units.label("length", _system(real, system)),
    )
    return figure


def sustained_curve(lowest: dict[str, float], factor: float = TIME_FACTOR) -> dict:
    """The card's type 0's sustained turn (Ps >= 0, the largest turn) against speed, from the
    lowest flown full-stick speed (game units, rad/s, m): {speeds, rate, radius}."""
    game_v = speed_axis(lowest, factor)
    grid = tuning.envelope(_row(), game_v, STICKS, 0, None, minimum_speed=lowest["full"])
    # Per speed, the turn where Ps crosses zero along the stick sweep, interpolated between the
    # grid's sticks (the grid's largest held rate alone steps with the stick grid).
    rate = np.full(len(game_v), np.nan)
    for j in range(len(game_v)):
        ps, turn = grid["ps"][:, j], grid["rate"][:, j]
        if ps[-1] >= 0.0:
            rate[j] = turn[-1]
            continue
        held = np.nonzero(ps >= 0.0)[0]
        if len(held):
            i = held[-1]
            share = ps[i] / (ps[i] - ps[i + 1])
            rate[j] = turn[i] + share * (turn[i + 1] - turn[i])
    rate = np.where(game_v >= lowest["full"], rate, np.nan)
    return dict(speeds=game_v, rate=rate, radius=game_v / rate)


def sustained_figure(
    curves,
    real: bool = False,
    factor: float = TIME_FACTOR,
    note_text: str = "",
    system: str | None = None,
):
    """The sustained turn: rate and radius against speed, one line per curve. curves: a list of
    (label, sustained_curve(...), reference: bool); the reference dotted and muted."""
    system = _system(real, system)
    sv, sr, sl = (units.factor(k, system, factor) for k in ("speed", "angular_rate", "length"))
    speed_unit, length_unit = units.label("speed", system), units.label("length", system)
    figure, axes = plt.subplots(
        2, 1, figsize=(6.4, 6.0), sharex=True, layout="constrained", facecolor=SURFACE
    )
    for ax, (key, unit) in zip(
        axes,
        (("rate", "sustained turn rate (deg/s)"), ("radius", f"sustained radius ({length_unit})")),
    ):
        style(ax)
        for label, curve, reference in curves:
            values = curve[key] * (sr if key == "rate" else sl)
            ax.plot(
                curve["speeds"] * sv,
                values,
                label=label,
                **(
                    dict(color=MUTED, linewidth=1.4, linestyle=":")
                    if reference
                    else dict(color=SERIES[0], linewidth=2.0)
                ),
            )
        ax.set_ylabel(unit, color=INK_2)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(
        0,
        min(2000.0, float(np.nanmax([np.nanmax(c["radius"]) for _, c, _ in curves])) * 1.05) * sl,
    )
    axes[1].set_xlabel(f"speed ({speed_unit}{_real_suffix(system)})", color=INK_2)
    axes[0].set_title(
        "Sustained turn (Ps = 0, the largest turn held) against speed",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        labelcolor=INK_2,
        title=note_text,
        title_fontsize=8,
        alignment="left",
    )
    return figure


def _name(index: int) -> str:
    return DISPLAY_NAMES.get(card.TYPE_NAMES[index], card.TYPE_NAMES[index])


def doghouse(ship_type: int, lowest, checks, real: bool = False):
    """The committed chart of a card type (inside tuning.flying of its row)."""
    title = f"{_name(ship_type)}, doghouse" + (", real units" if real else "")
    figure = doghouse_figure(title, lowest, real, TIME_FACTOR, checks)
    suffix = "-core" + ("-real" if real else "") + ("-dark" if THEME == "dark" else "")
    path = os.path.join(OUT, f"doghouse-{card.TYPE_NAMES[ship_type].replace('_', '-')}{suffix}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


def comparison(curves):
    """Every type's full-stick and sustained turn against speed (from each line's lowest flown
    speed). curves: per type (name, speeds, full-stick rate, sustained curve)."""
    figure, axes = plt.subplots(3, 1, figsize=(6.4, 11), layout="constrained", facecolor=SURFACE)
    panels = (
        (
            "turn rate, full stick and sustained (deg/s)",
            lambda c: (c["full_rate"] * DEGREES, c["held"]["rate"] * DEGREES),
        ),
        (
            "turn radius, full stick and sustained (m)",
            lambda c: (c["speeds"] / c["full_rate"], c["held"]["radius"]),
        ),
        ("time for a sustained 360 (s)", lambda c: (None, 2 * math.pi / c["held"]["rate"])),
    )
    for ax, (title, pick) in zip(axes, panels):
        style(ax)
        for t, c in enumerate(curves):
            full, held = pick(c)
            if full is not None:
                ax.plot(
                    c["speeds"],
                    full,
                    color=SERIES[t],
                    linewidth=2.0,
                    label=f"{c['name']} full stick",
                )
            ax.plot(
                c["speeds"],
                held,
                color=SERIES[t],
                linewidth=2.0,
                linestyle="--" if full is not None else "-",
                label=f"{c['name']} sustained",
            )
        if "radius" in title:
            ax.set_ylim(0, 800)
        if "360" in title:
            ax.set_ylim(0, 20)
        ax.set_xlabel("speed (m/s)", color=INK_2)
        ax.set_xlim(min(c["speeds"][0] for c in curves), max(c["speeds"][-1] for c in curves))
        ax.set_title(title + ", coast, spawn store", color=INK, fontsize=10.5, loc="left")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, ncol=2)
    figure.supxlabel(
        "Solid: full stick (instantaneous). Dashed: sustained (Ps >= 0). From each line's lowest\n"
        "speed flown in the game.",
        color=INK_2,
        fontsize=8,
        x=0.02,
        ha="left",
    )
    path = os.path.join(OUT, f"doghouse-comparison-core{'-dark' if THEME == 'dark' else ''}.png")
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path


if __name__ == "__main__":
    curves = []
    for ship_type, name in enumerate(card.TYPE_NAMES):
        with tuning.flying(tuning.row_of(name)):
            lowest = lowest_speeds()
            checks = flown_points(lowest)
            print(
                name, "lowest flown speeds (game m/s):", {k: round(v, 1) for k, v in lowest.items()}
            )
            print(name, "flown points:", checks)
            for real in (False, True):
                print(doghouse(ship_type, lowest, checks, real=real))
            speeds = speed_axis(lowest)
            full = tuning.envelope(_row(), speeds, np.array([1.0]))["rate"][-1]
            curves.append(
                dict(
                    name=_name(ship_type),
                    speeds=speeds,
                    full_rate=_from(full, speeds, lowest["full"]),
                    held=sustained_curve(lowest),
                )
            )
    print(comparison(curves))
