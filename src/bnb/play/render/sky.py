"""The arena's ground (pass 1: the sky inside the disc, the outside colour beyond) and its
edge (over the fields, contours and zones: the hatched boundary damage zone beyond the
edge as a veil, and the edge line), one full-screen triangle each."""

import moderngl

from ..themes import Theme
from .programs import program, set_uniforms


class SkyPass:
    """Pass 1: the sky, one full-screen triangle."""

    def __init__(self, context: moderngl.Context) -> None:
        self.program = program(context, "sky", vertex="screen")
        self.vertex_array = context.vertex_array(self.program, [])

    def draw(self, camera: dict, theme: Theme, arena_radius: float, edge: bool = False) -> None:
        """Draw the ground (the sky inside the arena, the outside colour beyond), or with
        `edge` the edge layer over it: the hatched boundary zone beyond as a veil, and the
        edge line."""
        set_uniforms(
            self.program,
            **camera,
            arena_radius=arena_radius,
            sky=theme.sky.floats(),
            outside=theme.outside.floats(),
            boundary=theme.boundary.floats(),
            edge=theme.thick_core.floats(),
            layer=int(edge),
        )
        self.vertex_array.render(moderngl.TRIANGLES, vertices=3)
