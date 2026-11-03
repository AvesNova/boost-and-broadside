"""What the player has selected, and the hit test that picks it.

The hit test runs on the CPU on the drawn snapshot, in world coordinates: the nearest
ship within its drawn ring radius plus a few pixels, else the zone under the point, else
the field (the one whose centre is nearest, where fields overlap). In a team view only
what that team sees can be picked: its own ships and the enemies its observation shows.
Fields and zones are terrain, which every view shows.

A selection is by sim index, so it survives a ship's death and respawn.
"""

import dataclasses
from typing import Literal

import numpy as np

from .render.ships import ring_radius
from .snapshot import Snapshot

SLACK_PIXELS = 4.0  # beyond a ship's drawn ring radius


@dataclasses.dataclass(frozen=True)
class Selection:
    """A ship, field or zone, by its index (sim order for ships)."""

    kind: Literal["ship", "field", "zone"]
    index: int


def shown_ships(frame: Snapshot, view: int) -> np.ndarray:
    """(N,) bool: the living ships the view draws (view -1 is the spectator's)."""
    ships = frame.ships
    if view < 0:
        return ships.alive.copy()
    return ships.alive & ships.visible[view]


def hit_test(
    frame: Snapshot, view: int, x: float, y: float, metres_per_pixel: float
) -> Selection | None:
    """The object under the world point (x, y) as `view` (-1 spectator, or a team) shows
    it, at this zoom; None for empty sky."""
    ships = frame.ships
    shown = shown_ships(frame, view)
    reach = ring_radius(ships.hit_radius, metres_per_pixel) + SLACK_PIXELS * metres_per_pixel
    distance = np.hypot(ships.x - x, ships.y - y)
    near = shown & (distance <= reach)
    if near.any():
        return Selection("ship", int(np.flatnonzero(near)[np.argmin(distance[near])]))
    zones = frame.zones
    inside = zones.present & (np.hypot(zones.x - x, zones.y - y) <= zones.radius)
    if inside.any():
        return Selection("zone", int(np.flatnonzero(inside)[0]))
    fields = frame.fields
    from_centre = np.hypot(fields.x - x, fields.y - y)
    inside = from_centre <= fields.radius
    if inside.any():
        return Selection("field", int(np.flatnonzero(inside)[np.argmin(from_centre[inside])]))
    return None


def selectable(frame: Snapshot, view: int, selection: Selection) -> bool:
    """Whether the view shows the selected object now (a ship the view cannot see, or a
    dead one, is still selected but not drawn)."""
    if selection.kind != "ship":
        return True
    return bool(shown_ships(frame, view)[selection.index])
