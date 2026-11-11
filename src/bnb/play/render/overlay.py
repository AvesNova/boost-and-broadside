"""The selection's outline: a ring around the selected ship, field or zone, drawn above
everything (fog and bullets included) on the battlefield and on the minimap."""

from typing import TYPE_CHECKING

import numpy as np

from ..themes import Theme
from .arcs import CIRCLE, arcs
from .ships import ring_radius

if TYPE_CHECKING:  # rendering never imports torch, which the snapshot module does
    from ..snapshot import Snapshot

SHIP_GAP_PIXELS = 5.0  # between the ship's ring and the outline
AREA_GAP_PIXELS = 3.0  # outside a field's or a zone's edge
LINE_PIXELS = 1.6
HALO_PIXELS = 4.0  # a faint wider line under the outline, so it reads on any colour
HALO_OPACITY = 0.35


def outline_table(
    theme: Theme,
    frame: Snapshot,
    kind: str,
    index: int,
    metres_per_pixel: float,
    ship_pixels: float | None = None,
    true_size: bool = False,
) -> np.ndarray:
    """The outline's arcs (a halo and the line) around the selected object; empty for a
    ship that is not alive. ship_pixels: a ship's drawn radius in pixels where it is not
    its ring (the minimap's dots); true_size: the ships are drawn at true size (`ships`)."""
    if kind == "ship":
        ships = frame.ships
        if not ships.alive[index]:
            return arcs(0)
        radius = ring_radius(ships.hit_radius[index : index + 1], metres_per_pixel, true_size)[0]
        if ship_pixels is not None:
            radius = ship_pixels * metres_per_pixel
        centre = ships.x[index], ships.y[index]
        radius += SHIP_GAP_PIXELS * metres_per_pixel
    else:
        area = frame.fields if kind == "field" else frame.zones
        centre = area.x[index], area.y[index]
        radius = area.radius[index] + AREA_GAP_PIXELS * metres_per_pixel
    table = arcs(2)
    for row, (width, opacity, colour) in enumerate(
        ((HALO_PIXELS, HALO_OPACITY, theme.sky), (LINE_PIXELS, 1.0, theme.selection))
    ):
        table[row, 0:3] = centre[0], centre[1], radius
        table[row, 4] = width
        table[row, 9:13] = colour.floats()
        table[row, 12] *= opacity
        table[row, 14] = CIRCLE
    return table
