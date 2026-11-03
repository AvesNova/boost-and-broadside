"""Pass 4: trails. Each ship's trail is a strip that tapers and fades along its length; in
hard turns (the angle of attack beyond the theme's vortex threshold, as a fraction of the
stall angle) a faint vortex line trails from the inside wingtip.

Only the samples the view saw draw: a segment needs both ends seen (`Trails.seen`), so a
team's view never shows the stretch of an enemy's path it did not see.

The trail store goes to the GPU as three small float textures each frame, and the vertex
shader builds the strips from them, so the CPU does no per-vertex work."""

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms
from .silhouettes import WINGTIPS, silhouette_of

if TYPE_CHECKING:
    from ..trails import Trails

NEWEST_PIXELS = 2.5  # the trail's width at the ship
OLDEST_PIXELS = 0.5
VORTEX_PIXELS = 1.2
VORTEX_RAMP = 4.0  # the vortex reaches full strength this many thresholds' worth past it
NEVER = 1e30  # the age of an empty sample


class TrailsPass:
    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.program = program(context, "trails", vertex="trails")
        self.vertex_array = context.vertex_array(self.program, [])
        self.shape = None

    def _resize(self, ships: int, samples: int) -> None:
        if self.shape == (ships, samples):
            return
        if self.shape is not None:
            for texture in (self.samples, self.turns, self.ships):
                texture.release()
        self.shape = (ships, samples)

        def table(width: int, height: int, components: int) -> moderngl.Texture:
            texture = self.context.texture((width, height), components, dtype="f4")
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            return texture

        self.samples, self.turns = table(samples, ships, 4), table(samples, ships, 2)
        self.ships = table(1, ships, 3)

    def draw(
        self,
        camera: dict,
        theme: Theme,
        trails: Trails,
        now: float,
        seen: np.ndarray,
        team: np.ndarray,
        ship_type: np.ndarray,
        size: np.ndarray,
    ) -> None:
        """Every ship's trail at game time `now` (s). seen (N, SAMPLES): the samples the
        view saw; team, ship_type and size (the ring radius, m) are per ship."""
        ships, samples = trails.x.shape
        self._resize(ships, samples)
        age = np.where(np.isfinite(trails.time), now - trails.time, NEVER)
        self.samples.write(
            np.stack([trails.x, trails.y, age, trails.heading], -1).astype(np.float32).tobytes()
        )
        self.turns.write(np.stack([trails.turn, seen], -1).astype(np.float32).tobytes())
        wingtip = np.array([WINGTIPS[key] for key in silhouette_of(ship_type)], np.float32)
        self.ships.write(np.stack([team, size, wingtip], -1).astype(np.float32).tobytes())
        self.samples.use(0)
        self.turns.use(1)
        self.ships.use(2)
        colours = np.array([c.floats() for c in theme.trail], np.float32)
        set_uniforms(
            self.program,
            **camera,
            samples=0,
            turns=1,
            ships=2,
            sample_count=samples,
            seconds=trails.seconds,
            trail_colours=colours,
            vortex_colour=theme.vortex.floats(),
            vortex_threshold=theme.vortex_threshold,
            vortex_ramp=VORTEX_RAMP,
            trail_pixels=(OLDEST_PIXELS, NEWEST_PIXELS),
            vortex_pixels=VORTEX_PIXELS,
        )
        vertices = ships * (samples - 1) * 6
        for vortex in (True, False):
            set_uniforms(self.program, vortex=vortex)
            self.vertex_array.render(moderngl.TRIANGLES, vertices=vertices)
