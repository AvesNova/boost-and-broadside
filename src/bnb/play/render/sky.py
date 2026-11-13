"""The arena's ground (pass 1: the sky inside the disc, the outside colour beyond) and its
edge (over the fields, contours and zones: the hatched boundary damage zone beyond the
boundary as a veil, and the edge line on the boundary; where the boundary has shrunk inside
the arena, elimination's, a faint ring on the arena's edge), one full-screen triangle each."""

import moderngl

from ..themes import Theme
from .programs import program, set_uniforms


class SkyPass:
    """Pass 1: the sky, one full-screen triangle."""

    def __init__(self, context: moderngl.Context) -> None:
        self.program = program(context, "sky", vertex="screen")
        self.vertex_array = context.vertex_array(self.program, [])

    def draw(
        self,
        camera: dict,
        theme: Theme,
        arena_radius: float,
        edge: bool = False,
        boundary: float | None = None,
    ) -> None:
        """Draw the ground (the sky inside the arena, the outside colour beyond), or with
        `edge` the edge layer over it: the hatched boundary zone beyond the boundary (radius
        `boundary`, m; None: the arena's) as a veil, the edge line on it, and a faint ring on
        the arena's edge where the boundary lies inside it."""
        set_uniforms(
            self.program,
            **camera,
            arena_radius=arena_radius,
            boundary_radius=arena_radius if boundary is None else boundary,
            sky=theme.sky.floats(),
            outside=theme.outside.floats(),
            boundary=theme.boundary.floats(),
            edge=theme.thick_core.floats(),
            layer=int(edge),
        )
        self.vertex_array.render(moderngl.TRIANGLES, vertices=3)
