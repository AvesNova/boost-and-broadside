"""The minimap: the whole arena drawn into its own small texture, which the interface
shows in its lower right card. The sky, the fields' fill and contours and the zones come
from the battlefield's passes at the minimap's zoom; each ship is a dot in its team's
colour; a team's view of a fogged match fogs what that team cannot see (its own mask at
the minimap's size), and draws only the ships it sees, as the battlefield does."""

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..camera import Camera
from ..themes import Theme
from .arcs import CIRCLE, ArcProgram, arcs
from .fields import FieldsPass
from .fog import FogPass
from .sky import SkyPass
from .zones import zone_table

if TYPE_CHECKING:
    from ..snapshot import Snapshot

DOT_PIXELS = 3.2  # a ship dot's radius
DOT_OUTLINE_PIXELS = 1.0
SAMPLES = 4


def dot_table(theme: Theme, frame: Snapshot, shown: np.ndarray, pixel: float) -> np.ndarray:
    """A filled, outlined dot per shown ship, in its team's colour."""
    ships = frame.ships
    index = np.flatnonzero(shown)
    table = arcs(len(index))
    team = np.array([colour.floats() for colour in theme.team], np.float32)
    table[:, 0], table[:, 1] = ships.x[index], ships.y[index]
    table[:, 2] = DOT_PIXELS * pixel
    table[:, 4] = DOT_OUTLINE_PIXELS
    table[:, 5:9] = team[ships.team[index]]
    table[:, 9:13] = theme.outline.floats()
    table[:, 14] = CIRCLE
    return table


class MinimapPass:
    """The minimap's own multisampled scene, the texture it resolves into, and its fog."""

    def __init__(
        self, context: moderngl.Context, sky: SkyPass, fields: FieldsPass, arcs: ArcProgram
    ) -> None:
        self.context, self.sky, self.fields, self.arcs = context, sky, fields, arcs
        self.fog = FogPass(context)
        self.size = None
        self.texture: moderngl.Texture | None = None

    def _resize(self, size: int) -> None:
        if self.size == size:
            return
        if self.texture is not None:
            for resource in (self.scene, self.scene_colour, self.target, self.texture):
                resource.release()
        self.size = size
        self.scene_colour = self.context.renderbuffer((size, size), 4, samples=SAMPLES)
        self.scene = self.context.framebuffer(color_attachments=[self.scene_colour])
        self.texture = self.context.texture((size, size), 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self.target = self.context.framebuffer(color_attachments=[self.texture])

    def draw(
        self,
        frame: Snapshot,
        view: int,
        camera: Camera,
        theme: Theme,
        cores: np.ndarray,
        shown: np.ndarray,
        overlay: np.ndarray,
    ) -> moderngl.Texture:
        """Draw the minimap (camera: the minimap's square view) and return its texture.
        shown (N,): the ships the view draws; overlay: the selection's outline."""
        self._resize(camera.width)
        context, uniforms = self.context, camera.uniforms()
        ships, pixel = frame.ships, camera.metres_per_pixel
        fogged = view >= 0 and frame.fogged
        if fogged:
            team = ships.alive & (ships.team == view)
            observers = np.stack(
                [ships.x[team], ships.y[team], ships.vision_range[team], np.zeros(int(team.sum()))],
                -1,
            ).astype(np.float32)
            self.fog.draw_mask(uniforms, observers, cores)
        self.scene.use()
        context.viewport = (0, 0, self.size, self.size)
        context.enable(moderngl.BLEND)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.sky.draw(uniforms, theme, frame.arena_radius)
        self.fields.draw(uniforms, theme, frame.log_index_per_level)
        self.arcs.draw(uniforms, zone_table(theme, frame.zones, pixel))
        if fogged:
            self.fog.draw(uniforms, theme)
        self.arcs.draw(uniforms, dot_table(theme, frame, shown, pixel))
        self.arcs.draw(uniforms, overlay)
        context.copy_framebuffer(self.target, self.scene)
        return self.texture
