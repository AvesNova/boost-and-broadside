"""Pass 7: fog, in a team's view. The visibility mask is drawn first, in screen space at
MASK_SCALE of the target's resolution, from the team's living ships and the field cores;
the fog pass reads it and multiplies what the team cannot see by the theme's
fog colour (brightness down, saturation kept). Terrain stays drawn under the fog."""

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms

MASK_SCALE = 0.5


class FogPass:
    """Pass 7: the team's visibility mask and the fog it applies."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.sight = program(context, "sight", vertex="screen")
        self.apply = program(context, "fog", vertex="screen")
        self.sight_array = context.vertex_array(self.sight, [])
        self.apply_array = context.vertex_array(self.apply, [])
        self.size = None
        self.observers = self.cores = None

    def _resize(self, width: int, height: int) -> None:
        size = (max(int(width * MASK_SCALE), 1), max(int(height * MASK_SCALE), 1))
        if size == self.size:
            return
        if self.size is not None:
            self.framebuffer.release()
            self.mask.release()
        self.size = size
        self.mask = self.context.texture(size, 1, dtype="f1")
        self.mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.mask.repeat_x = self.mask.repeat_y = False
        self.framebuffer = self.context.framebuffer(color_attachments=[self.mask])

    @staticmethod
    def _table(
        context: moderngl.Context, texture: moderngl.Texture | None, rows: np.ndarray
    ) -> moderngl.Texture:
        """A (len(rows), 1) RGBA float texture holding rows (k, 4), grown as needed."""
        count = max(len(rows), 1)
        if texture is None or texture.width < count:
            if texture is not None:
                texture.release()
            texture = context.texture((max(count, 64), 1), 4, dtype="f4")
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        if len(rows):
            texture.write(
                np.ascontiguousarray(rows, np.float32).tobytes(), viewport=(0, 0, len(rows), 1)
            )
        return texture

    def draw_mask(self, camera: dict, observers: np.ndarray, cores: np.ndarray) -> None:
        """Draw the mask into its own framebuffer (the caller binds its target again).
        observers (O, 4): x, y, vision range, unused; cores (M, 4): x, y, radius squared,
        unused."""
        width, height = (int(value) for value in camera["viewport"])
        self._resize(width, height)
        self.observers = self._table(self.context, self.observers, observers)
        self.cores = self._table(self.context, self.cores, cores)
        self.framebuffer.use()
        self.context.disable(moderngl.BLEND)  # the mask's one channel has no alpha to blend
        self.observers.use(0)
        self.cores.use(1)
        mask_viewport = tuple(float(value) for value in self.size)
        set_uniforms(
            self.sight,
            **{**camera, "viewport": mask_viewport},
            observers=0,
            observer_count=len(observers),
            cores=1,
            core_count=len(cores),
        )
        self.sight_array.render(moderngl.TRIANGLES, vertices=3)

    def draw(self, camera: dict, theme: Theme) -> None:
        """Multiply the bound framebuffer by the fog colour where the mask says the team
        cannot see."""
        context = self.context
        self.mask.use(0)
        set_uniforms(self.apply, **camera, fog_mask=0, fog=theme.fog.floats())
        context.blend_func = moderngl.DST_COLOR, moderngl.ZERO
        self.apply_array.render(moderngl.TRIANGLES, vertices=3)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
