"""The renderer: the passes in draw order, into a multisampled scene resolved onto the
target, then the ImGui layer (HUD and panels) on top.

Draw order: sky and arena, fields, zones, trails, bullets, ships, rings, fog. A spectator
sees everything; a team's view draws only the ships that team sees (its observation's
`visible`) and the parts of their trails it saw, and in a fogged match darkens what it
cannot see. Every bullet is drawn in every view: agents never see bullets, so this is an
advantage human players have. In a match without fog every ship is visible to both teams,
so a team's view draws no fog. Rendering reads snapshots (numpy) and never imports torch.
"""

import contextlib
import time
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..camera import Camera
from ..themes import Theme
from .arcs import ArcProgram
from .bullets import BulletsPass
from .fields import FieldsPass
from .fog import FogPass
from .imgui_renderer import ImguiRenderer
from .rings import ring_table
from .ships import ShipsPass, ring_radius
from .sky import SkyPass
from .trails import TrailsPass
from .zones import zone_table

if TYPE_CHECKING:
    from ..index_map import IndexGrid, TileUpdate
    from ..snapshot import Snapshot
    from ..trails import Trails

SPECTATOR = -1
SAMPLES = 4  # multisampling


class PassTimer:
    """GPU time per pass (ms) from timer queries, and CPU time per pass for the work done
    in Python before each draw (geometry tables). Reading the queries waits for the GPU,
    so it is for measurement runs only."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.queries: dict[str, moderngl.Query] = {}
        self.cpu: dict[str, float] = {}

    @contextlib.contextmanager
    def measure(self, name: str):
        query = self.queries.get(name)
        if query is None:
            query = self.queries[name] = self.context.query(time=True)
        started = time.perf_counter()
        with query:
            yield
        self.cpu[name] = time.perf_counter() - started

    def gpu(self) -> dict[str, float]:
        return {name: query.elapsed * 1e-6 for name, query in self.queries.items()}


class Renderer:
    def __init__(self, context: moderngl.Context, width: int, height: int) -> None:
        self.context = context
        self.arcs = ArcProgram(context)
        self.sky = SkyPass(context)
        self.fields = FieldsPass(context, self.arcs)
        self.trails = TrailsPass(context)
        self.bullets = BulletsPass(context)
        self.ships = ShipsPass(context)
        self.fog = FogPass(context)
        self.imgui = ImguiRenderer(context)
        self.size = None
        self.resize(width, height)
        self.timer: PassTimer | None = None

    def resize(self, width: int, height: int) -> None:
        size = (max(width, 1), max(height, 1))
        if size == self.size:
            return
        if self.size is not None:
            self.scene.release()
            self.scene_colour.release()
        self.size = size
        self.scene_colour = self.context.renderbuffer(size, 4, samples=SAMPLES)
        self.scene = self.context.framebuffer(color_attachments=[self.scene_colour])

    def set_map(self, frame: Snapshot, grid: IndexGrid, values: np.ndarray | None = None):
        """A new map: the fields' geometry and index texture (built whole, or zero until
        `write_index` brings its tiles)."""
        self.fields.set_map(frame.fields, grid, values)
        fields = frame.fields
        self.cores = np.stack([fields.x, fields.y, fields.core_radius ** 2,
                               np.zeros_like(fields.x)], -1).astype(np.float32)

    def write_index(self, updates: list[TileUpdate]) -> None:
        self.fields.write(updates)

    def _measure(self, name: str):
        return self.timer.measure(name) if self.timer else contextlib.nullcontext()

    def draw(self, frame: Snapshot, view: int, camera: Camera, theme: Theme,
             trails: Trails, now: float, target: moderngl.Framebuffer) -> None:
        """Draw the frame as `view` (SPECTATOR or a team) sees it onto `target`; `now`:
        game time (s), for the trails."""
        context, uniforms = self.context, camera.uniforms()
        ships, pixel = frame.ships, camera.metres_per_pixel
        team_view = view != SPECTATOR
        fogged = team_view and frame.fogged
        shown = ships.alive & ships.visible[view] if team_view else ships.alive.copy()
        size = ring_radius(ships.hit_radius, pixel)
        if fogged:
            with self._measure("fog mask"):
                team = ships.alive & (ships.team == view)
                observers = np.stack([ships.x[team], ships.y[team], ships.vision_range[team],
                                      np.zeros(int(team.sum()), np.float32)], -1)
                self.fog.draw_mask(uniforms, observers, self.cores)
        self.scene.use()
        context.viewport = (0, 0, *self.size)
        context.enable(moderngl.BLEND)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        with self._measure("sky"):
            self.sky.draw(uniforms, theme, frame.arena_radius)
        with self._measure("fields"):
            self.fields.draw(uniforms, theme, frame.log_index_per_level)
        with self._measure("zones"):
            self.arcs.draw(uniforms, zone_table(theme, frame.zones, pixel))
        with self._measure("trails"):
            seen = trails.seen[view] if team_view else np.ones_like(trails.seen[0])
            self.trails.draw(uniforms, theme, trails, now, seen, ships.team, ships.ship_type,
                             size)
        with self._measure("bullets"):
            bullets = frame.bullets
            active = bullets.active
            self.bullets.draw(uniforms, theme, bullets.x[active], bullets.y[active],
                              bullets.direction_x[active], bullets.direction_y[active],
                              bullets.team[active])
        with self._measure("ships"):
            self.ships.draw(uniforms, theme, ships.x[shown], ships.y[shown],
                            ships.heading[shown], size[shown], ships.team[shown],
                            ships.ship_type[shown])
        with self._measure("rings"):
            self.arcs.draw(uniforms, ring_table(theme, ships.x[shown], ships.y[shown],
                                                ships.heading[shown], size[shown],
                                                ships.team[shown], ships.shield[shown],
                                                ships.energy[shown]))
        if fogged:
            with self._measure("fog"):
                self.fog.draw(uniforms, theme)
        with self._measure("resolve"):
            context.copy_framebuffer(target, self.scene)
        target.use()

    def draw_imgui(self, draw_data) -> None:
        with self._measure("imgui"):
            self.imgui.render(draw_data)
