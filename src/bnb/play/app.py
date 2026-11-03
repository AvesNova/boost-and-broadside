"""The shell: a session (the match or replay, the camera, the view, the selection, the
theme and the interface), the window that drives it, and the offscreen runner that renders
frames to PNG files without a window.

The session runs everything on one thread: each frame it advances the match loop, builds
any pending part of a new map's index texture within a time budget, records the trails,
draws the battlefield and the minimap, and then draws the interface (`bnb.play.interface`).
All input reaches the session through ImGui's input state, which the window (or a test)
fills, so the same code serves a window and an offscreen test. The interface's controls
call the session's methods (`toggle_pause`, `select`, `take_control`, `new_match`, ...).
"""

import dataclasses
import pathlib
import time
from typing import Literal

import moderngl
import numpy as np
from imgui_bundle import imgui

from bnb.contracts import GameConfig
from bnb.game.api import PRESETS

from . import replay as replays
from . import themes
from .camera import Camera
from .index_map import IndexBuilder, build
from .interface.interface import Interface
from .interface.kill_feed import FEED_SECONDS
from .interface.style import load_fonts
from .interface.text import TEAM_NAMES
from .loop import TIME_SCALES, MatchEvent, MatchLoop, MatchSetup, SideSetup
from .png import write_png
from .render.minimap import DOT_PIXELS
from .render.overlay import outline_table
from .render.renderer import SPECTATOR, Renderer
from .selection import Selection, shown_ships
from .sides import AgentError, Scripted, load_agent
from .theme_editor import ThemeEditor
from .trails import Trails

VIEWS = ("spectator", "team0", "team1")
VIEW_LABELS = ("Spectator", "Team 0", "Team 1")
INDEX_BUDGET = 0.004  # seconds per frame spent building a new map's index texture
FRAME_SECONDS = 1.0 / 60.0  # the offscreen runner's clock
NOTICE_SECONDS = 6.0  # a notice stays this long (display time)
SEEK_BEFORE_MARKER = 3.0  # seconds before an event a marker seeks to


@dataclasses.dataclass
class Options:
    """Play or watch a match. The game comes from a preset, or from the sizes (which
    override the preset's)."""

    preset: Literal["deathmatch", "deathmatch_wide", "frontline"] = "deathmatch"
    ships_per_team: int | None = None
    fields: int | None = None
    arena_radius: float | None = None  # m
    fog: bool = True  # the match has fog of war (team views show what the team sees)
    view: Literal["spectator", "team0", "team1"] = "spectator"
    theme: str = themes.DEFAULT
    seed: int = 0
    fly: bool = False  # start flying team 0's first ship
    agent0: pathlib.Path | None = None  # a policy file for team 0 (scripted without one)
    agent1: pathlib.Path | None = None  # a policy file for team 1
    temperature: float = 1.0  # the agents' sampling temperature
    replay: pathlib.Path | None = None  # open this replay instead of a match
    replays: pathlib.Path = pathlib.Path("replays")  # where matches are recorded
    runs: pathlib.Path = pathlib.Path("runs")  # where the agent list looks for policies
    record: bool = True  # record every live match into --replays
    scale: float = 1.0  # the interface scale
    offscreen: pathlib.Path | None = None  # render --frames frames to PNG files here
    frames: int = 60
    width: int = 1920
    height: int = 1080
    vsync: bool = True
    compile: bool = True  # compile the tick (in the background, running uncompiled meanwhile)
    threads: int = 1  # torch's CPU threads for the sim

    def game(self) -> GameConfig:
        """The game configuration: the preset with any sizes given overriding it."""
        preset = PRESETS[self.preset]
        sizes = dict(
            ships_per_team=self.ships_per_team,
            num_fields=self.fields,
            arena_radius=self.arena_radius,
        )
        changed = {name: value for name, value in sizes.items() if value is not None}
        if "ships_per_team" in changed and preset.spawn_groups:
            changed["spawn_groups"] = ()
        return dataclasses.replace(preset, **changed)

    def setup(self) -> MatchSetup:
        """The first match's setup."""
        sides = tuple(
            SideSetup() if policy is None else SideSetup("agent", str(policy), self.temperature)
            for policy in (self.agent0, self.agent1)
        )
        return MatchSetup(self.game(), fogged=self.fog, seed=self.seed, sides=sides)


class Session:
    """One match or replay being watched or played, independent of any window."""

    def __init__(
        self,
        options: Options,
        context: moderngl.Context,
        width: int,
        height: int,
        library: themes.Library | None = None,
    ) -> None:
        self.options = options
        self.context = context
        self.imgui_context = imgui.create_context()
        imgui.set_current_context(self.imgui_context)
        io = imgui.get_io()
        io.set_ini_filename("")  # no imgui.ini next to wherever the shell was started
        io.display_size = (width, height)
        self.fonts = load_fonts()
        self.renderer = Renderer(context, width, height)
        self.camera = Camera(width, height)
        self.minimap_camera = Camera(1, 1)
        self.view = VIEWS.index(options.view)
        self.library = themes.Library() if library is None else library
        self.theme_names = self.library.names()
        self.theme = self.library.load(options.theme)
        self.original = self.theme  # what "Revert" restores
        self.editor = ThemeEditor(self.library)
        self.scale = options.scale
        self.fog_default = options.fog
        self.interface = Interface(self)
        self.selection: Selection | None = None
        self.following: int | None = None
        self.show_keys = False
        self.show_interface = True  # False draws the battlefield alone (tests, screenshots)
        self.quit_requested = False
        self.notice, self.notice_until = "", 0.0
        self.clock = 0.0  # display seconds since the session started
        self.feed: list[tuple[float, MatchEvent]] = []
        self.last_shot = np.full(1, -np.inf)
        self.recorder: replays.Recorder | None = None
        self.builder: IndexBuilder | None = None
        self.map_fields = None
        self.timings: dict[str, float] = {}
        self.elapsed = 0.0
        self.loop: MatchLoop | None = None
        if options.replay is not None:
            self.open_replay(options.replay, whole=True)
        else:
            pilot = 0 if options.fly else None
            if not self.start(options.setup(), pilot=pilot, whole=True):
                raise AgentError(self.notice)
        self.interface.restyle()

    # Matches and replays ------------------------------------------------------------------

    def _adopt(self, loop: MatchLoop, whole: bool, fit: bool = True) -> None:
        """Make `loop` the session's match: a new map, trails, feed, selection."""
        self.loop = loop
        self.loop.on_tick = self._on_tick
        self.trails = Trails(loop.rules.num_ships, loop.current.stall_angle)
        self.frame = loop.frame()
        self.feed = []
        self.last_shot = np.full(loop.rules.num_ships, -np.inf)
        selection = self.selection
        if selection is not None and selection.kind == "ship":
            if selection.index >= loop.rules.num_ships:
                self.selection = None
        self._new_map(whole)
        if fit:
            self.camera.fit(loop.current.arena_radius)

    def start(
        self,
        setup: MatchSetup,
        pilot: int | None = None,
        whole: bool = False,
        realtime=None,
        sides: list | None = None,
    ) -> bool:
        """Start a live match (its tick compiles in the background when its graph is new);
        returns False, with a notice, when a side's agent cannot play it. realtime and
        sides: continue this match with these controllers (a replay forked)."""
        try:
            loop = MatchLoop(
                setup,
                sides=sides,
                compiled=self.options.compile,
                background=True,
                pilot=pilot,
                realtime=realtime,
            )
        except AgentError as error:
            self.say(str(error))
            return False
        if self.loop is not None:
            self._finish_recording()
        self.following = pilot
        self._adopt(loop, whole, fit=realtime is None)
        if self.options.record:
            self.recorder = replays.Recorder(setup, loop.realtime, self._match_name())
        return True

    def new_match(self, setup: MatchSetup) -> bool:
        """Start a new match from the menu's setup; the menu closes when it starts."""
        started = self.start(setup)
        if started:
            self.selection = None
            self.interface.close_menu()
        return started

    def restart(self) -> None:
        """Start the current match's setup again (a replay's: as a live match)."""
        self.new_match(self.loop.setup)

    def open_replay(self, directory: str | pathlib.Path, whole: bool = False) -> bool:
        """Play back a saved replay; returns False, with a notice, when it cannot be read."""
        try:
            replay = replays.Replay.load(directory)
        except (replays.ReplayError, ValueError, OSError) as error:
            self.say(f"cannot open the replay: {error}")
            return False
        if self.loop is not None:
            self._finish_recording()
        loop = replays.ReplayLoop(replay, compiled=self.options.compile)
        self.selection, self.following = None, None
        self._adopt(loop, whole)
        self.interface.close_menu()
        return True

    def save_replay(self) -> pathlib.Path | None:
        """Save the live match recorded so far (recording goes on); returns its directory."""
        if self.recorder is None or self.recorder.ticks == 0:
            self.say("nothing recorded yet")
            return None
        path = replays.save(self.recorder, self.options.replays)
        self.say(f"saved the replay {path.name}")
        return path

    def _finish_recording(self) -> None:
        """Save the live match's recording, if it has any ticks."""
        if self.recorder is not None and self.recorder.ticks > 0:
            replays.save(self.recorder, self.options.replays)
        self.recorder = None

    def _match_name(self) -> str:
        return "-vs-".join(side.label.replace("/", "-") for side in self.loop.sides)

    def _on_tick(self, actions, realtime, events: list[MatchEvent]) -> None:
        loop = self.loop
        now = loop.ticks * loop.tick_length
        self.last_shot[loop.current.ships.fired] = now
        self.feed += [(now, event) for event in events if event.kind == "kill"]
        if self.recorder is not None:
            self.recorder.record(actions, realtime, events)
            if self.recorder.ended:
                replays.save(self.recorder, self.options.replays)
                self.recorder = replays.Recorder(loop.setup, realtime, self._match_name())

    # The sides ------------------------------------------------------------------------------

    def set_side(self, team: int, side: SideSetup) -> bool:
        """Change a side's controller in the live match; returns False, with a notice, when
        the agent cannot play it."""
        loop = self.loop
        if loop.replaying:
            return False
        if side.controller == "agent":
            try:
                controller = load_agent(
                    side.policy,
                    loop.setup.config,
                    loop.rules,
                    loop.types,
                    side.temperature,
                    loop.setup.seed + team,
                )
            except AgentError as error:
                self.say(str(error))
                return False
        else:
            controller = Scripted(loop.rules.ships_per_team)
        loop.sides[team] = controller
        sides = list(loop.setup.sides)
        sides[team] = side
        loop.setup = dataclasses.replace(loop.setup, sides=tuple(sides))
        return True

    def set_temperature(self, team: int, temperature: float) -> None:
        """An agent side's sampling temperature (from the next tick)."""
        controller = self.loop.sides[team]
        if hasattr(controller, "temperature"):
            controller.temperature = temperature
            sides = list(self.loop.setup.sides)
            sides[team] = dataclasses.replace(sides[team], temperature=temperature)
            self.loop.setup = dataclasses.replace(self.loop.setup, sides=tuple(sides))

    # Commands (the controls and their shortcuts) ---------------------------------------------

    def say(self, text: str) -> None:
        """Show a notice under the top bar for a few seconds."""
        self.notice, self.notice_until = text, self.clock + NOTICE_SECONDS

    @property
    def view_name(self) -> str:
        """The current view's name ("spectator", "team0" or "team1")."""
        return VIEWS[self.view]

    @property
    def view_team(self) -> int:
        """The view as the renderer takes it: SPECTATOR or a team."""
        return SPECTATOR if self.view == 0 else self.view - 1

    def set_view(self, view: int) -> None:
        """Show the battlefield as the spectator (0) or a team (1, 2) sees it."""
        self.view = view

    def cycle_view(self) -> None:
        """Switch to the next view: spectator, team 0, team 1."""
        self.view = (self.view + 1) % len(VIEWS)

    def cycle_theme(self) -> None:
        """Switch to the next theme in the library."""
        names = self.theme_names
        index = names.index(self.theme.name) if self.theme.name in names else -1
        self.set_theme(names[(index + 1) % len(names)])

    def set_theme(self, name: str) -> None:
        """Use the named theme."""
        self.theme = self.library.load(name)

    def set_scale(self, scale: float) -> None:
        """The interface scale (font and sizes)."""
        self.scale = scale
        self.interface.restyle()

    def toggle_editor(self) -> None:
        """Show or hide the theme editor."""
        self.editor.open = not self.editor.open

    def toggle_keys(self) -> None:
        """Show or hide the key list."""
        self.show_keys = not self.show_keys

    def toggle_menu(self) -> None:
        """Open the menu (pausing a live match), or close it."""
        self.interface.toggle_menu()

    def toggle_pause(self) -> None:
        """Pause or play (a replay paused at its end plays from its start)."""
        loop = self.loop
        if loop.replaying and loop.paused and loop.position >= loop.replay.last:
            self.seek(0)
        loop.paused = not loop.paused

    def single_step(self) -> None:
        """One tick, while paused."""
        self.loop.single_step()

    def step_back(self) -> None:
        """In a replay, one tick back, while paused."""
        if self.loop.replaying and self.loop.paused:
            self.seek(self.loop.position - 1)

    def faster(self) -> None:
        """The next larger time scale."""
        self.loop.faster()

    def slower(self) -> None:
        """The next smaller time scale."""
        self.loop.slower()

    def set_time_scale(self, scale: float) -> None:
        """Play at this time scale (one of TIME_SCALES)."""
        if scale in TIME_SCALES:
            self.loop.time_scale = scale

    def fit(self) -> None:
        """Fit the arena in the view (and stop following)."""
        self.following = None
        self.camera.fit(self.frame.arena_radius)

    def select(self, selection: Selection | None) -> None:
        """Select a ship, field or zone (None clears the selection)."""
        self.selection = selection
        if selection is not None:
            self.interface.reveal("selection")

    def selected_ship(self) -> int | None:
        """The selected ship's sim index, if a ship is selected."""
        selection = self.selection
        return selection.index if selection is not None and selection.kind == "ship" else None

    def _fork_sides(self, setup: MatchSetup) -> tuple[list, MatchSetup]:
        """Controllers for a live match forked from a replay: its agents where they load,
        scripted otherwise (with a notice)."""
        sides, chosen = [], []
        for team, side in enumerate(setup.sides):
            if side.controller == "agent":
                try:
                    sides.append(
                        load_agent(
                            side.policy,
                            setup.config,
                            self.loop.rules,
                            self.loop.types,
                            side.temperature,
                            setup.seed + team,
                        )
                    )
                    chosen.append(side)
                    continue
                except AgentError as error:
                    self.say(f"{TEAM_NAMES[team]} plays scripted: {error}")
            sides.append(Scripted(self.loop.rules.ships_per_team))
            chosen.append(SideSetup())
        return sides, dataclasses.replace(setup, sides=tuple(chosen))

    def take_control(self, ship: int) -> None:
        """Fly this ship (its side keeps controlling the others); in a replay, fork a live
        match from the current tick and fly it there (the replay is untouched)."""
        loop = self.loop
        if loop.replaying:
            sides, setup = self._fork_sides(loop.setup)
            self.start(setup, pilot=ship, realtime=loop.realtime, sides=sides)
            self.selection = Selection("ship", ship)
        self.loop.pilot = ship
        self.following = ship
        self.interface.collapse_for_flying()

    def release(self) -> None:
        """Stop flying: the ship returns to its side's controller."""
        self.loop.pilot = None
        self.following = None

    def toggle_control(self) -> None:
        """Take control of the selected ship, or release the flown one."""
        ship = self.selected_ship()
        if self.loop.pilot is not None and (ship is None or ship == self.loop.pilot):
            self.release()
        elif ship is not None:
            self.take_control(ship)

    def follow(self, ship: int | None) -> None:
        """Keep the camera on this ship (None: stop)."""
        self.following = ship

    def toggle_follow(self) -> None:
        """Follow the selected ship, or stop following."""
        ship = self.selected_ship()
        self.following = None if self.following is not None or ship is None else ship

    def seek(self, tick: int) -> None:
        """In a replay, show this tick."""
        if self.loop.replaying:
            self.loop.seek(tick)
            self.feed = []
            self.trails.clear()
            self.frame = self.loop.frame()
            if self._map_changed():
                self._new_map(whole=False)

    def seek_marker(self, event: MatchEvent) -> None:
        """In a replay, show the moments before an event."""
        self.seek(event.tick - round(SEEK_BEFORE_MARKER / self.loop.tick_length))

    def quit(self) -> None:
        """Ask the window to close (the live match's recording is saved on close)."""
        self.quit_requested = True

    # The map ----------------------------------------------------------------------------

    def _map_changed(self) -> bool:
        fields = self.loop.current.fields
        return not all(
            np.array_equal(getattr(fields, name), getattr(self.map_fields, name))
            for name in ("x", "y", "radius", "band_width", "level")
        )

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

    def resize(self, width: int, height: int) -> None:
        """Take the target's new size in pixels."""
        self.renderer.resize(width, height)
        self.camera.resize(width, height)
        imgui.set_current_context(self.imgui_context)
        imgui.get_io().display_size = (width, height)

    # A frame ----------------------------------------------------------------------------------

    def update(self, elapsed: float) -> None:
        """Advance the match by `elapsed` seconds of display time and keep the static
        layers and trails current."""
        self.elapsed = elapsed
        self.clock += elapsed
        started = time.perf_counter()
        ticks = self.loop.advance(elapsed)
        self.timings["advance"] = time.perf_counter() - started
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
        following = self.following
        if following is not None and frame.ships.alive[following]:
            self.camera.follow(float(frame.ships.x[following]), float(frame.ships.y[following]))
        now = self.loop.ticks * self.loop.tick_length
        self.feed = [(when, event) for when, event in self.feed if now - when < FEED_SECONDS]
        self.frame = frame

    def outline(
        self, metres_per_pixel: float, ship_pixels: float | None = None
    ) -> np.ndarray | None:
        """The selection's outline (an arcs table) at this zoom, or None when nothing
        shown is selected (ship_pixels: see `outline_table`)."""
        selection = self.selection
        if selection is None:
            return None
        if (
            selection.kind == "ship"
            and not shown_ships(self.frame, self.view_team)[selection.index]
        ):
            return None
        return outline_table(
            self.theme, self.frame, selection.kind, selection.index, metres_per_pixel, ship_pixels
        )

    def draw(self, target: moderngl.Framebuffer) -> None:
        """Draw the last updated frame, the minimap and the interface onto `target`."""
        outline = self.outline(self.camera.metres_per_pixel)
        self.renderer.draw(
            self.frame,
            self.view_team,
            self.camera,
            self.theme,
            self.trails,
            self.loop.game_time,
            target,
            outline,
        )
        imgui.set_current_context(self.imgui_context)
        imgui.get_io().delta_time = max(self.elapsed, 1e-4)
        started = time.perf_counter()
        imgui.new_frame()
        if self.show_interface:
            self.interface.draw()
        imgui.render()
        self.timings["interface"] = time.perf_counter() - started
        target.use()
        self.context.viewport = (0, 0, *target.size)
        self.renderer.draw_imgui(imgui.get_draw_data())

    def draw_minimap(self, side: int) -> int:
        """Draw the minimap at `side` pixels square; returns its texture's id for ImGui."""
        camera = self.minimap_camera
        camera.resize(side, side)
        camera.fit(self.frame.arena_radius)
        outline = self.outline(camera.metres_per_pixel, DOT_PIXELS)
        return self.renderer.draw_minimap(
            self.frame,
            self.view_team,
            camera,
            self.theme,
            shown_ships(self.frame, self.view_team),
            outline if outline is not None else np.zeros((0, 15), np.float32),
        )

    def close(self) -> None:
        """Save the live match's recording and release the session's ImGui context."""
        self._finish_recording()
        imgui.destroy_context(self.imgui_context)

    def file_theme(self) -> themes.Theme:
        """The current theme as its file holds it (what "Revert" restores)."""
        try:
            return self.library.load(self.theme.name)
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
    size = (options.width, options.height)
    target = context.framebuffer(color_attachments=[context.texture(size, 4)])
    session = Session(options, context, options.width, options.height)
    options.offscreen.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(options.frames):
        session.update(FRAME_SECONDS)
        session.draw(target)
        path = options.offscreen / f"frame_{index:04d}.png"
        write_png(path, read_pixels(target))
        paths.append(path)
    session.close()
    return paths
