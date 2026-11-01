"""The shell: a session (the match, the camera, the view, the theme and the panels), the
window that drives it from the keyboard and mouse, and the offscreen runner that renders
frames to PNG files without a window.

The session runs everything on one thread: each frame it advances the match loop, builds
any pending part of a new map's index texture within a time budget, records the trails,
draws, and then draws the ImGui layer (HUD and theme editor).
"""

import dataclasses
import pathlib
import time
from typing import Literal

import moderngl
import numpy as np
from imgui_bundle import imgui

from bnb.contracts import GameConfig
from bnb.game.modes.rules import PRESETS

from . import themes
from .camera import Camera
from .index_map import IndexBuilder, build
from .loop import MatchLoop
from .png import write_png
from .render.hud import HudState, draw_hud
from .render.renderer import SPECTATOR, Renderer
from .theme_editor import ThemeEditor
from .trails import Trails

VIEWS = ("spectator", "team0", "team1")
VIEW_LABELS = {"spectator": "Spectator", "team0": "Team 0's view", "team1": "Team 1's view"}
INDEX_BUDGET = 0.004  # seconds per frame spent building a new map's index texture
FRAME_SECONDS = 1.0 / 60.0  # the offscreen runner's clock
UI_FONT_SIZE = 16.0


@dataclasses.dataclass
class Options:
    """Play or watch a scripted match. The game comes from a preset, or from the sizes
    (which override the preset's)."""

    preset: Literal["deathmatch", "deathmatch_wide", "frontline"] = "deathmatch"
    ships_per_team: int | None = None
    fields: int | None = None
    arena_radius: float | None = None  # m
    fog: bool = False  # the match has fog of war (what the chasers can see)
    view: Literal["spectator", "team0", "team1"] = "spectator"
    theme: str = themes.DEFAULT
    seed: int = 0
    fly: bool = True  # fly team 0's first ship from the keyboard
    offscreen: pathlib.Path | None = None  # render --frames frames to PNG files here
    frames: int = 60
    width: int = 1920
    height: int = 1080
    vsync: bool = True
    compile: bool = True  # compile the tick (startup takes longer, each tick less)
    threads: int = 1  # torch's CPU threads for the sim

    def game(self) -> GameConfig:
        preset = PRESETS[self.preset]
        sizes = dict(ships_per_team=self.ships_per_team, num_fields=self.fields,
                     arena_radius=self.arena_radius)
        return dataclasses.replace(preset, **{k: v for k, v in sizes.items() if v is not None})


class Session:
    """One match being watched or played, independent of any window."""

    def __init__(self, options: Options, context: moderngl.Context, width: int,
                 height: int, theme_directory: pathlib.Path = themes.THEMES) -> None:
        self.loop = MatchLoop(options.game(), fogged=options.fog, compile=options.compile,
                              pilot=0 if options.fly else None)
        self.context = context
        imgui.create_context()
        io = imgui.get_io()
        io.set_ini_filename("")  # no imgui.ini next to wherever the shell was started
        io.display_size = (width, height)
        self.font = io.fonts.add_font_default_vector()
        imgui.get_style().font_size_base = UI_FONT_SIZE
        self.renderer = Renderer(context, width, height)
        self.camera = Camera(width, height)
        self.view = VIEWS.index(options.view)
        self.theme_directory = theme_directory
        self.theme_names = themes.available(theme_directory)
        self.theme = themes.load(options.theme, theme_directory)
        self.editor = ThemeEditor(theme_directory)
        self.trails = Trails(self.loop.rules.num_ships, self.loop.current.stall_angle)
        self.follow = False
        self.help = False
        self.builder: IndexBuilder | None = None
        self.map_fields = None
        self.timings: dict[str, float] = {}
        self.original = self.theme
        self._new_map(whole=True)
        self.camera.fit(self.loop.current.arena_radius)

    # The map ----------------------------------------------------------------------------

    def _map_changed(self) -> bool:
        fields = self.loop.current.fields
        return not all(np.array_equal(getattr(fields, k), getattr(self.map_fields, k))
                       for k in ("x", "y", "radius", "band_width", "level"))

    def _new_map(self, whole: bool) -> None:
        """Start the static layers of the current map: the whole index texture now, or
        tile by tile over the next frames."""
        frame = self.loop.current
        self.map_fields = frame.fields
        if whole:
            grid, values = build(frame.fields, frame.arena_radius)
            self.renderer.set_map(frame, grid, values)
            self.builder = None
        else:
            self.builder = IndexBuilder(frame.fields, frame.arena_radius)
            self.renderer.set_map(frame, self.builder.grid)
        self.trails.clear()

    # Input --------------------------------------------------------------------------------

    @property
    def view_name(self) -> str:
        return VIEWS[self.view]

    def cycle_view(self) -> None:
        self.view = (self.view + 1) % len(VIEWS)

    def cycle_theme(self) -> None:
        index = self.theme_names.index(self.theme.name) if self.theme.name in \
            self.theme_names else -1
        self.theme = themes.load(self.theme_names[(index + 1) % len(self.theme_names)],
                                 self.theme_directory)

    def resize(self, width: int, height: int) -> None:
        self.renderer.resize(width, height)
        self.camera.resize(width, height)
        imgui.get_io().display_size = (width, height)

    # A frame ----------------------------------------------------------------------------------

    def update(self, elapsed: float) -> None:
        """Advance the match by `elapsed` seconds of display time and keep the static
        layers and trails current."""
        self.elapsed = elapsed
        started = time.perf_counter()
        ticks = self.loop.advance(elapsed)
        self.timings["sim"] = time.perf_counter() - started
        if ticks and self._map_changed():
            self._new_map(whole=False)
        if self.builder is not None:
            started = time.perf_counter()
            self.renderer.write_index(self.builder.run(INDEX_BUDGET))
            if self.builder.done:
                self.builder = None
            self.timings["index"] = time.perf_counter() - started
        started = time.perf_counter()
        frame = self.loop.frame()
        self.timings["interpolate"] = time.perf_counter() - started
        self.trails.record(frame, self.loop.game_time, self.loop.restarted)
        self.loop.restarted[:] = False
        pilot = self.loop.pilot
        if self.follow and pilot is not None and frame.ships.alive[pilot]:
            self.camera.follow(float(frame.ships.x[pilot]), float(frame.ships.y[pilot]))
        self.frame = frame

    def draw(self, target: moderngl.Framebuffer) -> None:
        """Draw the last updated frame and the ImGui layer onto `target`."""
        view = SPECTATOR if self.view == 0 else self.view - 1
        self.renderer.draw(self.frame, view, self.camera, self.theme, self.trails,
                           self.loop.game_time, target)
        imgui.get_io().delta_time = max(self.elapsed, 1e-4)
        imgui.new_frame()
        loop = self.loop
        draw_hud(self.frame, HudState(view=VIEW_LABELS[self.view_name], theme=self.theme.name,
                                      time_scale=loop.time_scale, paused=loop.paused,
                                      pilot=loop.pilot, follow=self.follow, help=self.help),
                 self.theme, self.font)
        if self.original.name != self.theme.name:
            self.original = self._file_theme()
        self.theme, saved = self.editor.draw(self.theme, self.original)
        if saved is not None:
            self.theme_names = themes.available(self.theme_directory)
        imgui.render()
        self.renderer.draw_imgui(imgui.get_draw_data())

    def _file_theme(self) -> themes.Theme:
        """The current theme as its file holds it (what "Revert" restores)."""
        try:
            return themes.load(self.theme.name, self.theme_directory)
        except FileNotFoundError:
            return self.theme


def offscreen_context() -> moderngl.Context:
    """An OpenGL 3.3 context without a window (EGL)."""
    return moderngl.create_context(standalone=True, backend="egl", require=330)


def read_pixels(framebuffer: moderngl.Framebuffer) -> np.ndarray:
    """(height, width, 3) uint8, top row first."""
    width, height = framebuffer.size
    data = np.frombuffer(framebuffer.read(components=3), np.uint8)
    return data.reshape(height, width, 3)[::-1]


def run_offscreen(options: Options) -> list[pathlib.Path]:
    """Render options.frames frames at 60 frames per second of game display time; write
    each to options.offscreen/frame_NNNN.png."""
    context = offscreen_context()
    target = context.framebuffer(color_attachments=[context.texture((options.width,
                                                                       options.height), 4)])
    session = Session(options, context, options.width, options.height)
    options.offscreen.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(options.frames):
        session.update(FRAME_SECONDS)
        session.draw(target)
        path = options.offscreen / f"frame_{index:04d}.png"
        write_png(path, read_pixels(target))
        paths.append(path)
    return paths
