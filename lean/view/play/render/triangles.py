"""Filled triangles in world coordinates with a colour per vertex: the density overlay's
hex cells draw through this one program."""

import moderngl
import numpy as np

from .programs import program, set_uniforms

FLOATS = 6  # per vertex: position (2), colour (4)


class TrianglesPass:
    """Triangles from a vertex table (vertices, FLOATS), three rows per triangle."""

    def __init__(self, context: moderngl.Context, capacity: int = 4096) -> None:
        self.context = context
        self.program = program(context, "triangles")
        self.capacity = 0
        self._reserve(capacity)

    def _reserve(self, count: int) -> None:
        if count <= self.capacity:
            return
        if self.capacity:
            self.vertex_array.release()
            self.vertices.release()
        self.capacity = max(count, 2 * self.capacity)
        self.vertices = self.context.buffer(reserve=self.capacity * FLOATS * 4, dynamic=True)
        self.vertex_array = self.context.vertex_array(
            self.program, [(self.vertices, "2f 4f", "position", "tint")]
        )

    def draw(self, camera: dict, table: np.ndarray) -> None:
        """Draw the vertex table's triangles with the camera's uniforms."""
        if len(table) == 0:
            return
        self._reserve(len(table))
        self.vertices.orphan()  # fresh storage: no wait on the last frame's draw
        self.vertices.write(np.ascontiguousarray(table, np.float32).tobytes())
        set_uniforms(self.program, **camera)
        self.vertex_array.render(moderngl.TRIANGLES, vertices=len(table))
