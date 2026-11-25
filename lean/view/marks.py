"""Lean's own rules drawn over v2's battlefield (adaption layer): each ship's sight ellipse
outline (the game's `sight_pairs`: the observer at one focus, reach p / (1 - e) ahead, p to the
side, p / (1 + e) behind), the tracking circle of reach p / (1 - e) about each observer while
its team holds an enemy with it, and marks where ships died (a cross) and collided (a ring),
fading over MARK_SECONDS. Drawn with v2's arcs and lines programs (lean.view.play.render)."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

from .play.render.arcs import CIRCLE, arcs
from .play.render.lines import segments
from .play.themes import Theme

if TYPE_CHECKING:
    from .snapshot import Snapshot

ELLIPSE_SEGMENTS = 96
OUTLINE_PIXELS = 1.2
OUTLINE_OPACITY = 0.55
TRACKING_OPACITY = 0.35
MARK_SECONDS = 1.5
DEATH_CROSS_PIXELS = 9.0  # half the cross's arm
DEATH_LINE_PIXELS = 2.5
COLLISION_RING_PIXELS = 20.0
COLLISION_LINE_PIXELS = 2.5


@dataclasses.dataclass(frozen=True)
class Mark:
    """A death (kind "death": team the dead ship's) or a collision (kind "collision": damage
    the two ships took, first and second), at game time `time`."""

    kind: str
    time: float
    x: float
    y: float
    team: int = -1
    damage: tuple[float, float] = (0.0, 0.0)


def marks_of(frame: Snapshot) -> list[Mark]:
    """The marks a frame's events leave."""
    events, ships = frame.events, frame.ships
    found = [
        Mark("death", frame.clock, float(ships.x[i]), float(ships.y[i]), int(ships.team[i]))
        for i in np.flatnonzero(events.died)
    ]
    for k in range(len(events.collision_x)):
        damage = (float(events.collision_damage_first[k]), float(events.collision_damage_second[k]))
        found.append(
            Mark(
                "collision",
                frame.clock,
                float(events.collision_x[k]),
                float(events.collision_y[k]),
                damage=damage,
            )
        )
    return found


def marks_at(frames: list[Snapshot], index: int) -> list[Mark]:
    """The marks still showing at frames[index] (those of the last MARK_SECONDS)."""
    now = frames[index].clock
    found = []
    for frame in frames[: index + 1][::-1]:
        if now - frame.clock > MARK_SECONDS:
            break
        found.extend(marks_of(frame))
    return found


def fade(mark: Mark, now: float) -> float:
    """A mark's opacity at game time `now`: full, then fading out over its last third."""
    left = (MARK_SECONDS - (now - mark.time)) / MARK_SECONDS
    return float(np.clip(3.0 * left, 0.0, 1.0))


def ellipse_table(theme: Theme, ships, shown: np.ndarray) -> np.ndarray:
    """The sight ellipses' outlines of the shown ships, as line segments in their team
    colour."""
    index = np.flatnonzero(shown)
    angles = np.linspace(0.0, 2.0 * math.pi, ELLIPSE_SEGMENTS + 1)
    p = ships.vision_range[index, None]
    e = ships.vision_eccentricity[index, None]
    reach = p / (1.0 - e * np.cos(angles)[None])
    turned = ships.heading[index, None] + angles[None]
    x = ships.x[index, None] + reach * np.cos(turned)
    y = ships.y[index, None] + reach * np.sin(turned)
    table = segments(len(index) * ELLIPSE_SEGMENTS).reshape(len(index), ELLIPSE_SEGMENTS, -1)
    table[..., 0], table[..., 1] = x[:, :-1], y[:, :-1]
    table[..., 2], table[..., 3] = x[:, 1:], y[:, 1:]
    table[..., 4] = OUTLINE_PIXELS
    colours = np.array([colour.floats() for colour in theme.team], np.float32)[ships.team[index]]
    table[..., 5:9] = colours[:, None]
    table[..., 8] *= OUTLINE_OPACITY
    return table.reshape(-1, table.shape[-1])


def tracking_table(theme: Theme, ships, shown: np.ndarray) -> np.ndarray:
    """The tracking circles (reach p / (1 - e)) of the shown ships whose team holds some enemy
    with it (Ships.tracked), where the circle reaches past the ellipse (e > 0)."""
    holding = ships.tracked.any(-1)  # (2,): the team holds an enemy
    index = np.flatnonzero(shown & holding[ships.team] & (ships.vision_eccentricity > 0.0))
    table = arcs(len(index))
    table[:, 0], table[:, 1] = ships.x[index], ships.y[index]
    table[:, 2] = ships.vision_range[index] / (1.0 - ships.vision_eccentricity[index])
    table[:, 4] = OUTLINE_PIXELS
    colours = np.array([colour.floats() for colour in theme.team], np.float32)
    table[:, 9:13] = colours[ships.team[index]]
    table[:, 12] *= TRACKING_OPACITY
    table[:, 14] = CIRCLE
    return table


def death_table(theme: Theme, marks: list[Mark], now: float, pixel: float) -> np.ndarray:
    """A cross per death mark in the dead ship's team colour."""
    deaths = [mark for mark in marks if mark.kind == "death"]
    table = segments(2 * len(deaths))
    arm = DEATH_CROSS_PIXELS * pixel
    for k, mark in enumerate(deaths):
        colour = np.array(theme.team[mark.team].floats(), np.float32)
        colour[3] *= fade(mark, now)
        for row, sign in ((2 * k, 1.0), (2 * k + 1, -1.0)):
            table[row, :4] = (mark.x - arm, mark.y - sign * arm, mark.x + arm, mark.y + sign * arm)
            table[row, 4] = DEATH_LINE_PIXELS
            table[row, 5:9] = colour
    return table


def collision_table(theme: Theme, marks: list[Mark], now: float, pixel: float) -> np.ndarray:
    """A ring per collision mark in the outline colour."""
    collisions = [mark for mark in marks if mark.kind == "collision"]
    table = arcs(len(collisions))
    for k, mark in enumerate(collisions):
        table[k, :3] = (mark.x, mark.y, COLLISION_RING_PIXELS * pixel)
        table[k, 4] = COLLISION_LINE_PIXELS
        table[k, 9:13] = theme.outline.floats()
        table[k, 12] *= fade(mark, now)
        table[k, 14] = CIRCLE
    return table


def draw(
    renderer,
    camera,
    theme: Theme,
    frame: Snapshot,
    marks: list[Mark],
    view: int,
    draw_vision: bool,
    target,
) -> None:
    """Draw lean's extras over the frame v2's renderer resolved onto `target` (the bound
    framebuffer, no multisampling)."""
    uniforms, pixel = camera.uniforms(), camera.metres_per_pixel
    ships = frame.ships
    target.use()
    if draw_vision:
        observers = ships.alive & ((ships.team == view) if view >= 0 else True)
        renderer.lines.draw(uniforms, ellipse_table(theme, ships, observers))
        if frame.tracking:
            renderer.arcs.draw(uniforms, tracking_table(theme, ships, observers))
    renderer.lines.draw(uniforms, death_table(theme, marks, frame.clock, pixel))
    renderer.arcs.draw(uniforms, collision_table(theme, marks, frame.clock, pixel))
