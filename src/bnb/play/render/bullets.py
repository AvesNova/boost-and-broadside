"""Pass 8: bullets, short streaks along their direction in the shooter's team colour. Every
bullet draws in every view."""

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms

LENGTH_PIXELS = 10.0
WIDTH_PIXELS = 2.5
FLOATS = 8


class BulletsPass:
    def __init__(self, context: moderngl.Context, capacity: int = 1024) -> None:
        self.context = context
        self.program = program(context, "bullets")
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
            self.program,
            [
                (self.corners, "2f", "corner"),
                (self.instances, "2f 2f 4f/i", "tip", "direction", "tint"),
            ],
        )

    def draw(
        self,
        camera: dict,
        theme: Theme,
        x: np.ndarray,
        y: np.ndarray,
        direction_x: np.ndarray,
        direction_y: np.ndarray,
        team: np.ndarray,
    ) -> None:
        """The listed bullets (arrays of one length)."""
        if len(x) == 0:
            return
        self._reserve(len(x))
        colours = np.array([c.floats() for c in theme.team], np.float32)[team]
        table = np.concatenate(
            [np.stack([x, y, direction_x, direction_y], -1), colours], -1
        ).astype(np.float32)
        self.instances.orphan()  # fresh storage: no wait on the last frame's draw
        self.instances.write(table.tobytes())
        set_uniforms(self.program, **camera, length_pixels=LENGTH_PIXELS, width_pixels=WIDTH_PIXELS)
        self.vertex_array.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(x))
