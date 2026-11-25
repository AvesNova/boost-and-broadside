"""The distance contours: faint lines at equal distances from the map's shapes, inside and
outside them, as a topographic map draws heights, so movement and position read against
them where nothing else is near. They draw over the fields' fill (the fields' own
contours, of the refractive index, crowd at their edges, these run across their middles)
and under everything else: zones, trails, ships, bullets. The dot grid, under the fields,
can be on with them.

The shapes are circles: the fields (their radius), Frontline's zones, and the map edge,
whose lines run parallel to the boundary. The distance drawn is to the nearest edge of
any shape, from either side, so a shape inside another (a field within a field, a zone
over a field) keeps its own rings inside the larger one's. It is a soft minimum of the
edges' distances ((sum d^-p)^(-1/p), `edge_distance` in the shader): 0 on every edge, and
where two edges are about equally far it bends below the plain minimum, so the crease where
two shapes' rings meet is rounded and nearby shapes' rings outside them merge into one
rounded outline, as hills' contours do. The rounding is relative to the distance, so it
does not change with the zoom.

Lines sit at whole multiples of a spacing that steps by 2 with the zoom, the finer lines
fading out across a step, like the dot grid (`grid.zoom_level`); every fifth line of each
level, an index contour, is drawn more strongly. One full-screen triangle draws it,
`contours.frag.glsl`.
"""

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..themes import Theme
from .grid import zoom_level
from .programs import program, set_uniforms

if TYPE_CHECKING:  # rendering never imports torch, which the snapshot module does
    from ..snapshot import Fields, Zones

# Each shape set: whether it holds the fields, the map edge and the zones.
SHAPE_SETS = {
    "fields": (True, False, False),
    "edge": (False, True, False),
    "fields+edge": (True, True, False),
    "fields+edge+zones": (True, True, True),
    "zones": (False, False, True),
}
CONTOUR_FACTOR = 2.0  # between spacing levels
CONTOUR_PIXELS = 64.0  # the fine level's spacing on screen just as the coarser level takes over


def contour_level(metres_per_pixel: float) -> tuple[float, float]:
    """The fine level's spacing (m) at this zoom and how far its own lines have faded
    (`grid.zoom_level`)."""
    return zoom_level(CONTOUR_FACTOR, CONTOUR_PIXELS, metres_per_pixel)


class ContoursPass:
    """The distance contours, one full-screen triangle over a table of the map's circles."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.program = program(context, "contours", vertex="screen")
        self.vertex_array = context.vertex_array(self.program, [])
        self.circles = None
        self.field_count = self.zone_count = 0

    def set_map(self, fields: Fields, zones: Zones) -> None:
        """A new map: the fields' circles, then the present zones', as a texture."""
        if self.circles is not None:
            self.circles.release()
        present = zones.present
        self.field_count, self.zone_count = len(fields.x), int(present.sum())
        count = self.field_count + self.zone_count
        table = np.zeros((max(count, 1), 4), np.float32)  # a texture has at least one texel
        table[: self.field_count, :3] = np.stack([fields.x, fields.y, fields.radius], -1)
        table[self.field_count : count, :3] = np.stack(
            [zones.x[present], zones.y[present], zones.radius[present]], -1
        )
        self.circles = self.context.texture((len(table), 1), 4, table.tobytes(), dtype="f4")
        self.circles.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def draw(self, camera: dict, theme: Theme, arena_radius: float, shapes: str) -> None:
        """Draw the contours of the shape set `shapes` (a key of SHAPE_SETS); nothing for a
        set of shapes this map does not have."""
        with_fields, with_edge, with_zones = SHAPE_SETS[shapes]
        first = 0 if with_fields else self.field_count
        last = self.field_count + (self.zone_count if with_zones else 0)
        if first == last and not with_edge:
            return
        spacing, fade = contour_level(camera["pixel"])
        self.circles.use(0)
        set_uniforms(
            self.program,
            **camera,
            circles=0,
            first_circle=first,
            last_circle=last,
            edge=with_edge,
            arena_radius=arena_radius,
            ink=theme.contours.floats(),
            spacing=spacing,
            factor=CONTOUR_FACTOR,
            fade=fade,
        )
        self.vertex_array.render(moderngl.TRIANGLES, vertices=3)
