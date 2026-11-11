"""The renderer: the passes in draw order, into a multisampled scene resolved onto the
target, then the ImGui layer (HUD and panels) on top.

Draw order: sky and arena, the dot grid (`grid.py`), fields, zones, trails, ships, rings,
fog, the density overlay, bullets, the other prediction overlays (`predictions.py`), the
selection's outline; then the minimap into its own texture (`minimap.py`). A spectator
sees everything; a team's view draws only the ships that team sees (its observation's
`visible`) and the parts of their trails it saw, and in a fogged match darkens what it
cannot see. Every bullet is drawn in every view, above the fog at full brightness: agents
never see bullets, so this is an advantage human players have. In a match without fog
every ship is visible to both teams, so a team's view draws no fog. Rendering reads
snapshots (numpy) and never imports torch.
"""

import contextlib
import time
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..camera import Camera
from ..themes import Theme
from .arcs import ArcProgram
from .bullets import TRAIL_SECONDS, BulletsPass, tracer_lengths
from .fields import FieldsPass
from .fog import FogPass
from .grid import GridPass
from .imgui_renderer import ImguiRenderer
from .lines import LinesPass
from .minimap import MinimapPass
from .rings import ring_table
from .ships import ShipsPass, ring_radius, silhouette_size
from .sky import SkyPass
from .trails import TrailsPass
from .triangles import TrianglesPass
from .zones import zone_table

if TYPE_CHECKING:
    from ..index_map import IndexGrid, TileUpdate
    from ..snapshot import Snapshot
    from ..trails import Trails
    from .predictions import OverlayDrawing

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
        """Time the block as the pass `name`: a GPU timer query around it and its CPU time."""
        query = self.queries.get(name)
        if query is None:
            query = self.queries[name] = self.context.query(time=True)
        started = time.perf_counter()
        with query:
            yield
        self.cpu[name] = time.perf_counter() - started

    def gpu(self) -> dict[str, float]:
        """The last measured GPU time of each pass, in milliseconds (waits for the GPU)."""
        return {name: query.elapsed * 1e-6 for name, query in self.queries.items()}


class Renderer:
    """Every pass of the battlefield, the multisampled scene they draw into, and the ImGui layer."""

    def __init__(self, context: moderngl.Context, width: int, height: int) -> None:
        self.context = context
        self.arcs = ArcProgram(context)
        self.sky = SkyPass(context)
        self.grid = GridPass(context)
        self.fields = FieldsPass(context, self.arcs)
        self.trails = TrailsPass(context)
        self.bullets = BulletsPass(context)
        self.ships = ShipsPass(context)
        self.fog = FogPass(context)
        self.lines = LinesPass(context)
        self.triangles = TrianglesPass(context)
        self.imgui = ImguiRenderer(context)
        self.minimap = MinimapPass(context, self.sky, self.fields, self.arcs)
        self.minimap_id: int | None = None
        self.size = None
        self.resize(width, height)
        self.timer: PassTimer | None = None

    def resize(self, width: int, height: int) -> None:
        """Reallocate the multisampled scene for a target of this size."""
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
        self.cores = np.stack(
            [fields.x, fields.y, fields.core_radius**2, np.zeros_like(fields.x)], -1
        ).astype(np.float32)

    def write_index(self, updates: list[TileUpdate]) -> None:
        """Write built tiles into the index texture."""
        self.fields.write(updates)

    def _measure(self, name: str):
        return self.timer.measure(name) if self.timer else contextlib.nullcontext()

    def draw(
        self,
        frame: Snapshot,
        view: int,
        camera: Camera,
        theme: Theme,
        trails: Trails,
        now: float,
        target: moderngl.Framebuffer,
        overlay: np.ndarray | None = None,
        predictions: OverlayDrawing | None = None,
        true_size: bool = False,
        playback: float = 1.0,
        grid: str = "off",
    ) -> None:
        """Draw the frame as `view` (SPECTATOR or a team) sees it onto `target`; `now`:
        game time (s), for the trails; overlay: an arcs table drawn last (the selection's
        outline); predictions: the prediction overlays' tables (None: none on); true_size:
        the ships at their size in metres, with no minimum on screen (`ships`); playback:
        the playback speed (seconds of play per second shown), for the bullets' tracers; grid:
        the dot grid's pattern, or "off"."""
        context, uniforms = self.context, camera.uniforms()
        ships, pixel = frame.ships, camera.metres_per_pixel
        team_view = view != SPECTATOR
        fogged = team_view and frame.fogged
        shown = ships.alive & ships.visible[view] if team_view else ships.alive.copy()
        size = ring_radius(ships.hit_radius, pixel, true_size)
        body = silhouette_size(size, ships.ship_type, true_size)
        if fogged:
            with self._measure("fog mask"):
                team = ships.alive & (ships.team == view)
                observers = np.stack(
                    [
                        ships.x[team],
                        ships.y[team],
                        ships.vision_range[team],
                        np.zeros(int(team.sum()), np.float32),
                    ],
                    -1,
                )
                self.fog.draw_mask(uniforms, observers, self.cores)
        self.scene.use()
        context.viewport = (0, 0, *self.size)
        context.enable(moderngl.BLEND)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        with self._measure("sky"):
            self.sky.draw(uniforms, theme, frame.arena_radius)
        if grid != "off":
            with self._measure("grid"):
                self.grid.draw(uniforms, theme, frame.arena_radius, grid)
        with self._measure("fields"):
            self.fields.draw(uniforms, theme, frame.log_index_per_level)
        with self._measure("zones"):
            self.arcs.draw(uniforms, zone_table(theme, frame.zones, pixel))
        with self._measure("trails"):
            seen = trails.seen[view] if team_view else np.ones_like(trails.seen[0])
            self.trails.draw(uniforms, theme, trails, now, seen, ships.team, ships.ship_type, body)
        with self._measure("ships"):
            self.ships.draw(
                uniforms,
                theme,
                ships.x[shown],
                ships.y[shown],
                ships.heading[shown],
                body[shown],
                ships.team[shown],
                ships.ship_type[shown],
            )
        with self._measure("rings"):
            self.arcs.draw(
                uniforms,
                ring_table(
                    theme,
                    ships.x[shown],
                    ships.y[shown],
                    ships.heading[shown],
                    size[shown],
                    ships.team[shown],
                    ships.shield[shown],
                    ships.energy[shown],
                ),
            )
        if fogged:
            with self._measure("fog"):
                self.fog.draw(uniforms, theme)
        if predictions is not None:
            with self._measure("density"):
                self.triangles.draw(uniforms, predictions.density)
        with self._measure("bullets"):
            bullets = frame.bullets
            active = bullets.active
            trail = tracer_lengths(
                bullets.speed[active], bullets.age[active], TRAIL_SECONDS * playback
            )
            self.bullets.draw(
                uniforms,
                theme,
                bullets.x[active],
                bullets.y[active],
                bullets.direction_x[active],
                bullets.direction_y[active],
                trail,
                bullets.team[active],
            )
        if predictions is not None:
            with self._measure("predictions"):
                self.lines.draw(uniforms, predictions.segments)
                ghosts = predictions.ghosts
                self.ships.draw(
                    uniforms,
                    theme,
                    ghosts["x"],
                    ghosts["y"],
                    ghosts["heading"],
                    silhouette_size(ghosts["size"], ghosts["ship_type"], true_size),
                    ghosts["team"].astype(np.int64),
                    ghosts["ship_type"].astype(np.int64),
                    ghosts["opacity"],
                )
                self.arcs.draw(uniforms, predictions.circles)
        if overlay is not None:
            self.arcs.draw(uniforms, overlay)
        with self._measure("resolve"):
            context.copy_framebuffer(target, self.scene)
        target.use()

    def draw_minimap(
        self,
        frame: Snapshot,
        view: int,
        camera: Camera,
        theme: Theme,
        shown: np.ndarray,
        overlay: np.ndarray,
    ) -> int:
        """Draw the minimap with its own square camera; returns its texture's id for ImGui."""
        with self._measure("minimap"):
            texture = self.minimap.draw(frame, view, camera, theme, self.cores, shown, overlay)
        if self.minimap_id is not None and self.minimap_id != texture.glo:
            self.imgui.textures.pop(self.minimap_id, None)  # a resize released the old one
        self.imgui.textures[texture.glo] = texture
        self.minimap_id = texture.glo
        return texture.glo

    def draw_imgui(self, draw_data) -> None:
        """Draw this frame's ImGui output over the bound target."""
        with self._measure("imgui"):
            self.imgui.render(draw_data)
