"""The trail stills: one game, or a window of it, as one picture. Each ship's path is a
long trail in its team's colour, older stretches faded, so a whole manoeuvre reads at a
glance; an X marks where a ship died; light ticks mark the shots it fired (a short stroke
along its nose) and the hits it took (a small ring). With the fog overlay, a hidden
enemy's true path is drawn faint and dashed, with the view's ghost of it (the belief's
mean and its sigma circle, or Imugi's dead-reckoned memory) at intervals, joined to where
the ship really was.

It reads a match record (`bnb.figures.match`) only. The colours: each team one hue, its
ships stepped light to dark along that hue's ramp (blue and orange, the reference
palette's first two slots: told apart under colour-vision deficiency), on a light surface
or a dark one. Sizes: "web" (1024 px wide), "column" (3.4 in at 300 dpi, a paper column)
and "page" (7 in at 300 dpi).
"""

import dataclasses
import math
import pathlib

import numpy as np

from . import output

# Ramp steps light to dark, (light mode, dark mode). Blue is the reference palette's
# sequential ramp; orange is the same lightness steps at slot 2's hue (OKLCH 41 degrees).
RAMPS = {
    0: (
        ("#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
        ("#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"),
    ),
    1: (
        ("#fe743e", "#e15102", "#b43f03", "#8a2e00", "#611e01"),
        ("#ffbfa9", "#fe9068", "#f55d16", "#cb4801", "#9e3601"),
    ),
}
THEMES = {
    "light": dict(
        surface=output.SURFACE,
        ink=output.INK,
        muted=output.MUTED,
        grid=output.GRID,
        field="#8a8984",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ecebe8", muted="#a3a29e", grid="#383835", field="#8f8e89"
    ),
}
SIZES = {  # width (in), dpi, the line width (pt) and the font size (pt)
    "web": (6.4, 160, 1.0, 8.0),
    "column": (3.4, 300, 0.6, 5.5),
    "page": (7.0, 300, 0.9, 7.5),
}
OLDEST_ALPHA = 0.18  # a trail's oldest stretch, against 1 for its newest
TELEPORT = 300.0  # m in one tick: a respawn, not flight
MARGIN = 0.06  # of the cropped view's extent, on every side
TEAM_NAMES = ("blue team", "orange team")  # team 0's and team 1's, by their colour
GHOST_CROWD = 2  # hidden enemies at once the fog window favours


@dataclasses.dataclass(frozen=True)
class StillConfig:
    """start, end: the window, seconds from the game's start (None: its first or last
    tick); crop: fit the view to the action's bounding box (else the whole arena); busy:
    draw the shot and hit ticks; fog: the fog overlay, the hidden enemies as `fog_team`'s
    view saw them, with a ghost every `ghost_seconds`; theme: "light" or "dark"; size:
    one of `SIZES`; formats: the files written (png, svg, pdf); title: the caption line
    above the view."""

    start: float | None = None
    end: float | None = None
    crop: bool = True
    busy: bool = True
    fog: bool = False
    fog_team: int = 0
    ghost_seconds: float = 3.0
    theme: str = "light"
    size: str = "web"
    formats: tuple[str, ...] = ("png",)
    title: bool = True


def ship_colours(team: np.ndarray, theme: str) -> list[str]:
    """Each ship's colour: its team's ramp, its slot's step (light to dark, spread over the
    ramp for any fleet size; one ship takes the middle step)."""
    dark = theme == "dark"
    colours = []
    for ship, side in enumerate(team.tolist()):
        ramp = RAMPS[int(side)][int(dark)]
        slots = int((team == side).sum())
        slot = int((team[:ship] == side).sum())
        place = 0.5 if slots == 1 else slot / (slots - 1)
        colours.append(ramp[round(place * (len(ramp) - 1))])
    return colours


def window_ticks(match: dict, config: StillConfig) -> tuple[int, int]:
    """The window's first and last tick (inclusive)."""
    tick_length = float(match["clock"][1] - match["clock"][0]) if len(match["clock"]) > 1 else 1.0
    last = len(match["clock"]) - 1
    first = 0 if config.start is None else max(0, round(config.start / tick_length))
    final = last if config.end is None else min(last, round(config.end / tick_length))
    if final <= first:
        raise ValueError(f"an empty window: ticks {first} to {final} of {last + 1}")
    return first, final


def busiest_window(match: dict, tick_length: float, seconds: float, team: int | None = None):
    """(start, end) s of the window of `seconds` with the most hits and deaths (deaths
    weighted 5); with `team`, also the enemies hidden from that team's view while it holds
    a ghost of them (up to GHOST_CROWD at once: more would bury the fight in circles)."""
    activity = match["hit"].numpy().sum(-1) + 5.0 * match["died"].numpy().sum(-1)
    if team is not None:
        hidden = hidden_enemies(match, team)
        held = ~np.isnan(match["ghost_x"].numpy()[:, team])
        ghosts = np.minimum((hidden & held).sum(-1), GHOST_CROWD)
        activity = activity + 0.5 * ghosts * _ghost_ticks(match)
    length = max(1, round(seconds / tick_length))
    if len(activity) <= length:
        return 0.0, (len(activity) - 1) * tick_length
    summed = np.convolve(activity, np.ones(length), mode="valid")
    first = int(summed.argmax())
    return first * tick_length, (first + length - 1) * tick_length


def _ghost_ticks(match: dict) -> int:
    """Ticks between the record's ghost samples (1 when it has none)."""
    held = np.flatnonzero(~np.isnan(match["ghost_x"].numpy()).all((1, 2)))
    return int(np.diff(held).min()) if len(held) > 1 else 1


def hidden_enemies(match: dict, team: int) -> np.ndarray:
    """(T, N) the living enemies of `team` its view does not see."""
    enemy = match["team"].numpy() != team
    return match["alive"].numpy() & ~match["visible"].numpy()[:, team] & enemy[None]


def _lives(alive: np.ndarray, x: np.ndarray, y: np.ndarray, first: int, final: int):
    """The (start, end) inclusive tick spans of one ship's continuous flight in the window:
    alive throughout, with no teleport (a respawn) inside."""
    spans, start = [], None
    for tick in range(first, final + 1):
        jump = (
            start is not None
            and math.hypot(x[tick] - x[tick - 1], y[tick] - y[tick - 1]) > TELEPORT
        )
        if alive[tick] and start is not None and not jump:
            continue
        if start is not None:
            spans.append((start, tick - 1))
            start = None
        if alive[tick]:
            start = tick
    if start is not None:
        spans.append((start, final))
    return [span for span in spans if span[1] > span[0]]


def _faded(colour: str, ages: np.ndarray):
    """RGBA per segment: the colour, its alpha from OLDEST_ALPHA (age 0) to 1 (age 1)."""
    from matplotlib.colors import to_rgba

    rgba = np.tile(np.array(to_rgba(colour)), (len(ages), 1))
    rgba[:, 3] = OLDEST_ALPHA + (1.0 - OLDEST_ALPHA) * ages**1.5
    return rgba


def _halo(width: float, surface: str):
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=width, foreground=surface)]


def _scale_bar(axis, view: tuple[float, float, float, float], theme: dict, font: float):
    """A 1-2-5 length near a fifth of the view's width, bottom left."""
    x0, x1, y0, y1 = view
    target = (x1 - x0) / 5.0
    power = 10.0 ** math.floor(math.log10(target))
    length = max(step * power for step in (1.0, 2.0, 5.0) if step * power <= target)
    left, bottom = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    axis.plot([left, left + length], [bottom, bottom], color=theme["ink"], linewidth=1.2)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    axis.annotate(
        label,
        (left + length / 2, bottom),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=font,
        color=theme["ink"],
    )


def _view(points: np.ndarray, arena: float, crop: bool) -> tuple[float, float, float, float]:
    """The drawn extent: the points' bounding box with a margin (crop), else the arena."""
    if not crop or len(points) == 0:
        return (-arena * 1.02, arena * 1.02, -arena * 1.02, arena * 1.02)
    low, high = points.min(0), points.max(0)
    span = np.maximum(high - low, 400.0)
    centre = (low + high) / 2
    # no thinner than 0.6 of its width, nor taller than 1.25
    span = np.array([max(span[0], span[1] / 1.25), max(span[1], 0.6 * span[0])])
    half = span * (0.5 + MARGIN)
    return (centre[0] - half[0], centre[0] + half[0], centre[1] - half[1], centre[1] + half[1])


def short_label(label: str) -> str:
    """A side's label for the title: a training checkpoint's as its run and steps."""
    run, _, step = label.partition("/step_")
    if step.isdigit():
        return f"{run} at {int(step) / 1e6:.0f}M steps"
    return label


def _title(about: dict, first: int, final: int, tick_length: float, characters: int) -> str:
    """The caption: the sides, the game and the window, wrapped to `characters`."""
    import textwrap

    blue, orange = (short_label(label) for label in about["labels"])
    sides = f"{blue}, self-play" if blue == orange else f"{blue} (blue) vs {orange} (orange)"
    size = about["ships_per_team"]
    mode = about["preset"].replace("_", " ")
    text = (
        f"{sides}, {mode} {size}v{size}: {first * tick_length:.0f} to {final * tick_length:.0f} s"
    )
    return "\n".join(textwrap.wrap(text, characters))


def draw(artifact: dict, config: StillConfig):
    """The still's figure and its numbers (what it shows)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    match = artifact["match"]
    about = match["about"]
    theme = THEMES[config.theme]
    width, dpi, line, font = SIZES[config.size]
    tick_length = float(about["tick_length"])
    first, final = window_ticks(match, config)
    x, y = match["x"].numpy(), match["y"].numpy()
    heading, alive = match["heading"].numpy(), match["alive"].numpy()
    team = match["team"].numpy()
    colours = ship_colours(team, config.theme)
    count = len(team)
    fog = config.fog and bool(match["fogged"])
    hidden = hidden_enemies(match, config.fog_team) if fog else np.zeros_like(alive)

    figure = plt.figure(figsize=(width, width), dpi=dpi, facecolor=theme["surface"])
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_facecolor(theme["surface"])
    axis.set_axis_off()
    points, numbers = [], dict(deaths=[0, 0], hits=[0, 0], shots=[0, 0], ghosts=0)
    span_ticks = max(final - first, 1)

    for ship in range(count):
        colour = colours[ship]
        for start, end in _lives(alive[:, ship], x[:, ship], y[:, ship], first, final):
            path = np.stack([x[start : end + 1, ship], y[start : end + 1, ship]], -1)
            points.append(path)
            segments = np.stack([path[:-1], path[1:]], 1)
            ages = (np.arange(start, end) - first) / span_ticks
            shown = ~hidden[start:end, ship] if fog else np.ones(len(ages), bool)
            axis.add_collection(
                LineCollection(
                    segments[shown],
                    colors=_faded(colour, ages[shown]),
                    linewidths=line,
                    capstyle="butt",
                    zorder=3,
                )
            )
            if fog and (~shown).any():
                unseen = np.where(~shown, 1.0, np.nan)[:, None]
                faint = np.concatenate([path[:-1] * unseen, path[-1:] * unseen[-1:]])
                axis.plot(
                    *faint.T,
                    color=colour,
                    linewidth=0.8 * line,
                    linestyle=(0, (2, 2)),
                    alpha=0.45,
                    zorder=2,
                )
            axis.plot(
                *path[0],
                "o",
                markersize=2.4 * line + 1.4,
                markerfacecolor=colour,
                markeredgecolor=theme["surface"],
                markeredgewidth=0.5 * line,
                zorder=5,
            )
        if alive[final, ship]:
            angle = math.degrees(float(heading[final, ship])) - 90.0
            axis.plot(
                x[final, ship],
                y[final, ship],
                marker=(3, 0, angle),
                markersize=4.5 * line + 2.0,
                color=colour,
                markeredgecolor=theme["surface"],
                markeredgewidth=0.5,
                zorder=6,
            )

    died = match["died"].numpy()[first : final + 1]
    for tick, ship in np.argwhere(died):
        tick += first
        numbers["deaths"][int(team[ship])] += 1
        axis.plot(
            x[tick, ship],
            y[tick, ship],
            "x",
            markersize=5.0 * line + 2.5,
            markeredgewidth=1.6 * line,
            color=colours[ship],
            path_effects=_halo(3.2 * line, theme["surface"]),
            zorder=7,
        )
    if config.busy:
        size = (
            0.006 * math.sqrt(max(np.ptp(np.concatenate(points), 0).prod(), 1.0)) if points else 0
        )
        fired = match["fired"].numpy()[first : final + 1]
        for ship in range(count):
            ticks = np.flatnonzero(fired[:, ship]) + first
            ticks = ticks[alive[ticks, ship] & ~hidden[ticks, ship]]
            numbers["shots"][int(team[ship])] += len(ticks)
            if len(ticks) == 0:
                continue
            centre = np.stack([x[ticks, ship], y[ticks, ship]], -1)
            # across the nose, so the tick stands off the trail it lies on
            side = np.stack([-np.sin(heading[ticks, ship]), np.cos(heading[ticks, ship])], -1)
            strokes = np.stack([centre - size * side, centre + size * side], 1)
            axis.add_collection(
                LineCollection(
                    strokes, colors=colours[ship], linewidths=0.5 * line, alpha=0.6, zorder=4
                )
            )
        hit = match["hit"].numpy()[first : final + 1]
        for tick, ship in np.argwhere(hit):
            tick += first
            numbers["hits"][int(team[ship])] += 1
            axis.plot(
                x[tick, ship],
                y[tick, ship],
                "o",
                markersize=1.6 * line + 1.0,
                markerfacecolor="none",
                markeredgecolor=theme["ink"],
                markeredgewidth=0.4 * line,
                alpha=0.5,
                zorder=4,
            )

    ghost_legend = False
    if fog:
        ghost_x = match["ghost_x"].numpy()[:, config.fog_team]
        ghost_y = match["ghost_y"].numpy()[:, config.fog_team]
        sigma = match["ghost_sigma"].numpy()[:, config.fog_team]
        every = max(1, round(config.ghost_seconds / tick_length))
        held = np.flatnonzero(~np.isnan(ghost_x).all(-1))
        held = held[(held >= first) & (held <= final)]
        last_drawn = {}
        for tick in held:
            for ship in np.flatnonzero(hidden[tick] & ~np.isnan(ghost_x[tick])):
                if tick - last_drawn.get(ship, -every) < every:
                    continue
                last_drawn[ship] = tick
                ghost = np.array([ghost_x[tick, ship], ghost_y[tick, ship]])
                truth = np.array([x[tick, ship], y[tick, ship]])
                reach = math.sqrt(2.0) * float(np.nan_to_num(sigma[tick, ship]))
                points.append(ghost[None] + reach * np.array([[0, 0], [1, 1], [-1, -1]]))
                numbers["ghosts"] += 1
                ghost_legend = True
                axis.plot(
                    [truth[0], ghost[0]],
                    [truth[1], ghost[1]],
                    color=theme["muted"],
                    linewidth=0.5 * line,
                    alpha=0.7,
                    zorder=3,
                )
                axis.plot(
                    *ghost,
                    "+",
                    color=colours[ship],
                    markersize=3.0 * line + 2,
                    markeredgewidth=0.9 * line,
                    zorder=5,
                )
                if sigma[tick, ship] > 0:
                    axis.add_patch(
                        Circle(
                            tuple(ghost),
                            math.sqrt(2.0) * float(sigma[tick, ship]),
                            fill=False,
                            edgecolor=colours[ship],
                            linewidth=0.6 * line,
                            linestyle=(0, (3, 2)),
                            alpha=0.8,
                            zorder=3,
                        )
                    )

    every_point = np.concatenate(points) if points else np.zeros((0, 2))
    view = _view(every_point, float(match["arena_radius"]), config.crop)
    _fields(axis, match["fields"], theme, line)
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            float(match["arena_radius"]),
            fill=False,
            edgecolor=theme["muted"],
            linewidth=0.8 * line,
            zorder=1,
        )
    )
    boundary = float(match["boundary_radius"][final])
    if boundary < float(match["arena_radius"]) - 1.0:
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                max(boundary, 0.0),
                fill=False,
                edgecolor=theme["muted"],
                linewidth=0.8 * line,
                linestyle=(0, (4, 3)),
                zorder=1,
            )
        )
    axis.set_xlim(view[0], view[1])
    axis.set_ylim(view[2], view[3])
    axis.set_aspect("equal")
    _scale_bar(axis, view, theme, font)

    # the figure's height follows the view; the legend and the title sit outside it
    aspect = (view[3] - view[2]) / (view[1] - view[0])
    legend_rows = 3 if config.size == "column" else 2
    legend_height = (font * 1.9 * legend_rows + 6) / 72.0
    title = _title(about, first, final, tick_length, int(width * 72.0 / (0.6 * font)))
    title_height = (font * 1.4 * (title.count("\n") + 1) + 0.6 * font) / 72.0
    title_height = title_height if config.title else 0.0
    height = width * aspect + legend_height + title_height
    figure.set_size_inches(width, height)
    axis.set_position((0.0, legend_height / height, 1.0, width * aspect / height))
    if config.title:
        figure.text(
            0.5,
            1.0 - 0.3 * font / 72.0 / height,
            title,
            ha="center",
            va="top",
            fontsize=font + 0.5,
            color=theme["ink"],
        )
    middle = [ship_colours(np.array([side]), config.theme)[0] for side in (0, 1)]
    handles = [
        Line2D([], [], color=middle[side], linewidth=1.5 * line, label=TEAM_NAMES[side])
        for side in (0, 1)
    ]
    handles += [
        Line2D(
            [], [], marker="o", color=theme["ink"], linestyle="none", markersize=3.5,
            label="start",
        ),
        Line2D(
            [], [], marker=(3, 0, -90), color=theme["ink"], linestyle="none", markersize=4.5,
            label="end, heading",
        ),
        Line2D(
            [], [], marker="x", color=theme["ink"], linestyle="none", markersize=5, label="killed"
        ),
    ]  # fmt: skip
    if config.busy:
        handles += [
            Line2D(
                [], [], marker="o", markerfacecolor="none", markeredgecolor=theme["ink"],
                alpha=0.6, linestyle="none", markersize=3, label="hit taken",
            ),
            Line2D(
                [], [], marker="|", color=theme["muted"], linestyle="none", markersize=5,
                label="shot fired",
            ),
        ]  # fmt: skip
    if ghost_legend:
        handles += [
            Line2D(
                [], [], color=theme["muted"], linestyle=(0, (2, 2)), label="hidden, true path"
            ),
            Line2D(
                [], [], marker="+", color=theme["ink"], linestyle="none", markersize=6,
                label=f"{TEAM_NAMES[config.fog_team]}'s ghost, 1 sigma",
            ),
        ]  # fmt: skip
    columns = math.ceil(len(handles) / legend_rows)
    legend = figure.legend(
        handles=handles,
        loc="lower center",
        ncol=columns,
        frameon=False,
        fontsize=font,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.3,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    numbers.update(
        window_seconds=[round(first * tick_length, 2), round(final * tick_length, 2)],
        view_metres=[round(float(value), 1) for value in view],
    )
    return figure, numbers


def _fields(axis, fields: dict, theme: dict, line: float) -> None:
    """The fields as neutral discs: thick air (a positive level) filled, darker per level;
    thin air (negative) an open dashed ring."""
    from matplotlib.patches import Circle

    for index in range(len(fields["x"])):
        centre = (float(fields["x"][index]), float(fields["y"][index]))
        radius, level = float(fields["radius"][index]), int(fields["level"][index])
        if level > 0:
            patch = Circle(
                centre,
                radius,
                facecolor=theme["field"],
                alpha=min(0.10 * level, 0.3),
                edgecolor="none",
                zorder=0,
            )
        else:
            patch = Circle(
                centre,
                radius,
                fill=False,
                edgecolor=theme["field"],
                linewidth=0.7 * line,
                linestyle=(0, (1, 2)),
                alpha=0.8,
                zorder=0,
            )
        axis.add_patch(patch)


def still(artifact: dict, directory, name: str, config: StillConfig) -> tuple[list, dict]:
    """Draw one still and write it in each of `config.formats` as directory/name.<format>
    (the PNG with its provenance); returns (the files, its numbers)."""
    figure, numbers = draw(artifact, config)
    paths = output.save_formats(
        figure,
        pathlib.Path(directory),
        name,
        artifact["meta"],
        numbers,
        formats=config.formats,
        dpi=SIZES[config.size][1],
        facecolor=THEMES[config.theme]["surface"],
    )
    return paths, numbers


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The trail stills: trails_game (the whole record), trails_game_dark, trails_busiest
    (the busiest 20 s, cropped) and, with fog and ghosts, trails_fog (the busiest 30 s of
    hidden enemies as team 0 saw them)."""
    match = artifact["match"]
    tick_length = float(match["about"]["tick_length"])
    written, numbers = [], {}
    stills = {
        "trails_game": StillConfig(),
        "trails_game_dark": StillConfig(theme="dark"),
    }
    start, end = busiest_window(match, tick_length, 20.0)
    stills["trails_busiest"] = StillConfig(start=start, end=end)
    has_ghosts = bool((~np.isnan(match["ghost_x"].numpy()[:, 0])).any())
    if bool(match["fogged"]) and has_ghosts:
        start, end = busiest_window(match, tick_length, 30.0, team=0)
        stills["trails_fog"] = StillConfig(start=start, end=end, fog=True)
    for name, config in stills.items():
        paths, numbers[name] = still(artifact, directory, name, config)
        written += paths
    return written, numbers
