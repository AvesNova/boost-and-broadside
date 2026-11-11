"""The dot grid, drawn just after the sky: faint dots fixed to the world inside the arena,
so movement reads against them where nothing else is near.

Three patterns: square (a dot every `spacing` along both axes), triangle (the triangular
lattice: rows `spacing` apart along x, each row offset by half a spacing, rows
spacing·√3/2 apart) and hex (the honeycomb: dots at the corners of hexagons whose sides are
`spacing` long). Each lattice holds the next coarser level's dots: square and triangle at
twice the spacing, hex at three times. The spacing follows the zoom in those steps, as map
apps do, so the dots stay about as dense on screen: the fine level's dots fade out as the
zoom widens toward the next level, while the coarse level's (also fine-level dots) stay.
One full-screen triangle draws it, each fragment finding its nearest dot (`grid.frag.glsl`).
"""

import math

import moderngl

from ..themes import Theme
from .programs import program, set_uniforms

# Each pattern: its index in the shader and the factor between its spacing levels.
PATTERNS = {"square": (0, 2.0), "triangle": (1, 2.0), "hex": (2, 3.0)}
GRID_PIXELS = 48.0  # the fine level's spacing on screen just as a coarser level takes over


def grid_level(pattern: str, metres_per_pixel: float) -> tuple[float, float]:
    """The fine level's spacing (m) at this zoom, and how far its own dots have faded, in
    [0, 1]: the fine spacing is the largest power of the pattern's factor (in metres) at
    most GRID_PIXELS on screen; the fade runs smoothly from 0, just after a level change,
    to 1, when the fine spacing has shrunk by the factor and the coarse level takes over."""
    factor = PATTERNS[pattern][1]
    level = math.log(GRID_PIXELS * metres_per_pixel, factor)
    floor = math.floor(level)
    within = level - floor
    return factor**floor, within * within * (3.0 - 2.0 * within)


class GridPass:
    """The dot grid, one full-screen triangle."""

    def __init__(self, context: moderngl.Context) -> None:
        self.program = program(context, "grid", vertex="screen")
        self.vertex_array = context.vertex_array(self.program, [])

    def draw(self, camera: dict, theme: Theme, arena_radius: float, pattern: str) -> None:
        """Draw the grid in `pattern` (a key of PATTERNS) inside the arena."""
        spacing, fade = grid_level(pattern, camera["pixel"])
        set_uniforms(
            self.program,
            **camera,
            arena_radius=arena_radius,
            dot=theme.grid.floats(),
            pattern=PATTERNS[pattern][0],
            spacing=spacing,
            factor=PATTERNS[pattern][1],
            fade=fade,
        )
        self.vertex_array.render(moderngl.TRIANGLES, vertices=3)
