"""Pass 2: the fields as a topographic map of the true refractive index.

The fill and the thin contours come from the index texture (`bnb.play.index_map`, the
game's `medium` on a grid); the bold line on each core edge is drawn from the field
geometry, where the game's sight test puts the core.
"""

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..themes import Theme
from .arcs import CIRCLE, ArcProgram, arcs
from .programs import program, set_uniforms

if TYPE_CHECKING:  # rendering never imports torch, which these modules do
    from ..index_map import IndexGrid, TileUpdate
    from ..snapshot import Fields

CORE_LINE_PIXELS = 2.4
CONTOURS_PER_LEVEL = 3


class FieldsPass:
    """Pass 2: the fields' fill and contours from the index texture, and their core edges."""

    def __init__(self, context: moderngl.Context, arc_program: ArcProgram) -> None:
        self.context = context
        self.program = program(context, "fields")
        self.arcs = arc_program
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.corners = context.buffer(corners.tobytes())
        self.resources = ()
        self.vertex_array = None
        self.count = 0

    def set_map(self, fields: Fields, grid: IndexGrid, values: np.ndarray | None = None) -> None:
        """A new map: the field geometry, and the index texture (zero until tiles arrive)."""
        for resource in self.resources:
            resource.release()
        context = self.context
        self.grid, self.fields, self.count = grid, fields, len(fields.x)
        if values is None:
            values = np.zeros((grid.size, grid.size), np.float32)
        self.index_texture = context.texture(
            (grid.size, grid.size),
            1,
            np.ascontiguousarray(values, np.float32).tobytes(),
            dtype="f4",
        )
        self.index_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.resources, self.vertex_array = (self.index_texture,), None
        if self.count == 0:  # a map without fields: the uniform medium, nothing to draw
            return
        outer = fields.radius + 0.5 * fields.band_width
        discs = np.stack([fields.x, fields.y, outer, np.zeros_like(outer)], -1).astype(np.float32)
        self.outer_discs = context.texture((self.count, 1), 4, discs.tobytes(), dtype="f4")
        self.outer_discs.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.instances = context.buffer(discs[:, :3].tobytes())
        self.vertex_array = context.vertex_array(
            self.program, [(self.corners, "2f", "corner"), (self.instances, "3f/i", "field")]
        )
        self.resources = (self.index_texture, self.outer_discs, self.instances, self.vertex_array)

    def write(self, updates: list[TileUpdate]) -> None:
        """Tiles of the index texture as the builder finishes them."""
        for update in updates:
            self.index_texture.write(
                np.ascontiguousarray(update.values, np.float32).tobytes(),
                viewport=(
                    update.column,
                    update.row,
                    update.values.shape[1],
                    update.values.shape[0],
                ),
            )

    def core_edges(self, theme: Theme) -> np.ndarray:
        """The core edges' circles as an arcs table, in each field's thin or thick core colour."""
        fields = self.fields
        cored = fields.core_radius > 0.0
        table = arcs(int(cored.sum()))
        table[:, 0], table[:, 1] = fields.x[cored], fields.y[cored]
        table[:, 2] = fields.core_radius[cored]
        table[:, 4] = CORE_LINE_PIXELS
        thick = fields.level[cored] > 0
        table[:, 9:13] = np.where(
            thick[:, None], theme.thick_core.floats(), theme.thin_core.floats()
        )
        table[:, 14] = CIRCLE
        return table

    def draw(self, camera: dict, theme: Theme, log_index_per_level: float) -> None:
        """Draw the fields' fill and contours, then their core edges."""
        if self.vertex_array is None:
            return
        self.index_texture.use(0)
        self.outer_discs.use(1)
        set_uniforms(
            self.program,
            **camera,
            index_map=0,
            outer_discs=1,
            index_grid=(self.grid.origin, self.grid.texel),
            sky=theme.sky.floats(),
            thin_air=theme.thin_air.floats(),
            thick_air=theme.thick_air.floats(),
            thin_contour=theme.thin_contour.floats(),
            thick_contour=theme.thick_contour.floats(),
            fill_strength=theme.fill_strength,
            contour_step=log_index_per_level / CONTOURS_PER_LEVEL,
        )
        self.vertex_array.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.count)
        self.arcs.draw(camera, self.core_edges(theme))
