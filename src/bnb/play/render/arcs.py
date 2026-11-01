"""Circles and arcs, one instance each: the fields' core edges, the zones and the ships'
shield and energy rings all draw through this one program."""

import math

import moderngl
import numpy as np

from .programs import program, set_uniforms

# Per instance: centre (2), radius, width in metres and in pixels, fill (4), line (4), arc
# middle and half span.
FORMAT = "2f 1f 2f 4f 4f 2f/i"
ATTRIBUTES = ("centre", "radius", "width", "fill", "line", "arc")
FLOATS = 15
CIRCLE = math.pi  # the half span of a whole circle


def arcs(count: int) -> np.ndarray:
    """A zeroed instance table (count, FLOATS) to fill: columns 0-1 centre, 2 radius, 3-4
    width (m, px), 5-8 fill, 9-12 line, 13-14 arc."""
    return np.zeros((count, FLOATS), np.float32)


class ArcProgram:
    def __init__(self, context: moderngl.Context, capacity: int = 1024) -> None:
        self.context = context
        self.program = program(context, "arcs")
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.corners = context.buffer(corners.tobytes())
        self.capacity = 0
        self._reserve(capacity)

    def _reserve(self, count: int) -> None:
        if count <= self.capacity:
            return
        self.capacity = max(count, 2 * self.capacity)
        self.instances = self.context.buffer(reserve=self.capacity * FLOATS * 4, dynamic=True)
        self.vertex_array = self.context.vertex_array(
            self.program, [(self.corners, "2f", "corner"), (self.instances, FORMAT, *ATTRIBUTES)])

    def draw(self, camera: dict, table: np.ndarray) -> None:
        if len(table) == 0:
            return
        self._reserve(len(table))
        self.instances.write(np.ascontiguousarray(table, np.float32).tobytes())
        set_uniforms(self.program, **camera)
        self.vertex_array.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(table))
