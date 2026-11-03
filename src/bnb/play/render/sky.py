"""Pass 1: the sky inside the arena disc, the hatched boundary damage zone beyond it."""

import moderngl

from ..themes import Theme
from .programs import program, set_uniforms


class SkyPass:
    """Pass 1: the sky, one full-screen triangle."""

    def __init__(self, context: moderngl.Context) -> None:
        self.program = program(context, "sky", vertex="screen")
        self.vertex_array = context.vertex_array(self.program, [])

    def draw(self, camera: dict, theme: Theme, arena_radius: float) -> None:
        """Draw the sky inside the arena and the hatched boundary zone beyond it."""
        set_uniforms(
            self.program,
            **camera,
            arena_radius=arena_radius,
            sky=theme.sky.floats(),
            outside=theme.outside.floats(),
            boundary=theme.boundary.floats(),
            edge=theme.thick_core.floats(),
        )
        self.vertex_array.render(moderngl.TRIANGLES, vertices=3)
