"""Line segments, one instance each, a fixed width on screen with round ends: the
prediction overlays' paths and connectors draw through this one program."""

import moderngl
import numpy as np

from .programs import program, set_uniforms

# Per instance: start (2), end (2), width in pixels, colour (4).
FORMAT = "2f 2f 1f 4f/i"
ATTRIBUTES = ("start", "end", "width_pixels", "tint")
FLOATS = 9


def segments(count: int) -> np.ndarray:
    """A zeroed instance table (count, FLOATS) to fill: columns 0-1 start, 2-3 end (m), 4
    width (px), 5-8 colour."""
    return np.zeros((count, FLOATS), np.float32)


class LinesPass:
    """The one program for line segments: a quad per instance, from an instance table
    (`segments`)."""

    def __init__(self, context: moderngl.Context, capacity: int = 1024) -> None:
        self.context = context
        self.program = program(context, "lines")
        corners = np.array([0, -1, 1, -1, 0, 1, 1, 1], np.float32)
        self.corners = context.buffer(corners.tobytes())
        self.capacity = 0
        self._reserve(capacity)

    def _reserve(self, count: int) -> None:
        if count <= self.capacity:
            return
        if self.capacity:
            self.vertex_array.release()
            self.instances.release()
        self.capacity = max(count, 2 * self.capacity)
        self.instances = self.context.buffer(reserve=self.capacity * FLOATS * 4, dynamic=True)
        self.vertex_array = self.context.vertex_array(
            self.program, [(self.corners, "2f", "corner"), (self.instances, FORMAT, *ATTRIBUTES)]
        )

    def draw(self, camera: dict, table: np.ndarray) -> None:
        """Draw the instance table's segments with the camera's uniforms."""
        if len(table) == 0:
            return
        self._reserve(len(table))
        self.instances.orphan()  # fresh storage: no wait on the last frame's draw
        self.instances.write(np.ascontiguousarray(table, np.float32).tobytes())
        set_uniforms(self.program, **camera)
        self.vertex_array.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(table))
