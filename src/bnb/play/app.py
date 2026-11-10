"""The shell: a session (the match or replay, the camera, the view, the selection, the
theme and the interface), the window that drives it, and the offscreen runner that renders
frames to PNG files without a window.

The session runs everything on one thread: each frame it advances the match loop, builds
any pending part of a new map's index texture within a time budget, records the trails,
makes the prediction overlays' arrays when one is on and the match moved on
(`bnb.play.predictions`), draws the battlefield and the minimap, and then draws the
interface (`bnb.play.interface`).
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
from bnb.game import api
from bnb.game.api import PRESETS

from . import predictions, themes
from . import replay as replays
from .camera import Camera
from .index_map import IndexBuilder, build
from .interface.interface import Interface
from .interface.kill_feed import FEED_SECONDS
from .interface.style import load_fonts
from .interface.text import TEAM_NAMES, show_ship_set
from .loop import TIME_SCALES, MatchEvent, MatchLoop, MatchSetup, SideSetup
from .overlays import NAMES as OVERLAY_NAMES
from .overlays import OverlaySettings, Predictions, offered, offered_by, provider
from .png import write_png
from .render.minimap import DOT_PIXELS
from .render.overlay import outline_table
from .render.predictions import OverlayDrawing, overlay_drawing
from .render.renderer import SPECTATOR, Renderer
from .selection import Selection, shown_ships
from .sides import AgentError, Scripted, controller_of
from .theme_editor import ThemeEditor
from .trails import Trails
from .tuning import Tuner, Tuning, base_rules
from .tuning_panel import TuningPanel

VIEWS = ("spectator", "team0", "team1")
VIEW_LABELS = ("Spectator", "Team 0", "Team 1")
INDEX_BUDGET = 0.004  # seconds per frame spent building a new map's index texture
FRAME_SECONDS = 1.0 / 60.0  # the offscreen runner's clock
NOTICE_SECONDS = 6.0  # a notice stays this long (display time)
FLYING_ZOOM = 3.0  # taking control zooms in to at most this multiple of the fitted view
SEEK_BEFORE_MARKER = 3.0  # seconds before an event a marker seeks to
SOLVED_UNDER = ("flight.", "rules.bullet_life")  # the rules a card's handles depend on


PresetName = Literal[tuple(PRESETS)]  # the game presets' names (`api.PRESETS`)


@dataclasses.dataclass
class Options:
    """Play or watch a match. The game comes from a preset, or from the sizes (which
    override the preset's)."""

    preset: PresetName = "deathmatch"
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
    scripted0: Literal["scripted", "scripted_agent"] = "scripted"  # team 0 without an agent
    scripted1: Literal["scripted", "scripted_agent"] = "scripted"  # the chaser or the agent
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
    overlays: tuple[str, ...] = ()  # prediction overlays on at the start (bnb.play.overlays)
    overlay_side: int | None = None  # whose predictions they draw (None: the default)
    truth: bool = False  # draw the truth beside the predictions

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
            SideSetup(scripted)
            if policy is None
            else SideSetup("agent", str(policy), self.temperature)
            for policy, scripted in ((self.agent0, self.scripted0), (self.agent1, self.scripted1))
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
        self.tuning_panel = TuningPanel()
        self.tuner: Tuner | None = None
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
        unknown = set(options.overlays) - set(OVERLAY_NAMES)
        if unknown:
            raise ValueError(f"no overlays {sorted(unknown)}: {OVERLAY_NAMES}")
        self.overlays = OverlaySettings(
            set(options.overlays), side=options.overlay_side, truth=options.truth
        )
        self.predicted: dict[int, Predictions] = {}  # by team, at predicted_at
        self.bullet_paths = None
        self.predicted_at: int | None = None  # the loop's tick count the arrays are of
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
        if self.tuner is not None:
            self.tuner.close()
            self.tuner = None
        self.loop = loop
        show_ship_set(loop.setup.config.ship_set)
        self.loop.on_tick = self._on_tick
        self.loop.on_retune = self._on_retune
        if self.tuning_panel.open:
            self._ensure_tuner()
        self.predicted, self.bullet_paths, self.predicted_at = {}, None, None
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
        self._warn(loop.sides)
        if self.options.record and not setup.tuning:  # a tuned match is not recorded
            self.recorder = replays.Recorder(setup, loop.realtime, self._match_name())
        return True

    def new_match(self, setup: MatchSetup) -> bool:
        """Start a new match from the menu's setup, tuned as the current match is when it
        plays the same ship set (untuned otherwise); the menu closes when it starts."""
        same_ships = setup.config.ship_set == self.loop.setup.config.ship_set
        tuning = self.current_tuning() if same_ships else Tuning()  # a tuning is of one set
        started = self.start(dataclasses.replace(setup, tuning=tuning))
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
        if self.recorder is None and self.current_tuning():
            self.say("a tuned match is not recorded")
            return None
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

    def _warn(self, sides) -> None:
        """Say what an agent was not trained with (another ship card or lineup)."""
        warnings = [side.warning for side in sides if getattr(side, "warning", None)]
        if warnings:
            self.say("; ".join(warnings))

    # Live tuning ------------------------------------------------------------------------

    def current_tuning(self) -> Tuning:
        """What the current match is tuned to (a new match starts from it)."""
        loop = self.loop
        config = loop.setup.config
        return Tuning.of(
            loop.rules, loop.types, base_rules(config, loop.setup.lineup), config.ship_set
        )

    def _ensure_tuner(self) -> None:
        loop = self.loop
        if self.tuner is None and not loop.replaying:
            self.tuner = Tuner(loop, base_rules(loop.setup.config, loop.setup.lineup))
            self.tuner.show(self.tuning_panel.ship)  # measured first

    def toggle_tuning(self) -> None:
        """Show or hide the tuning panel (a live match's tuner starts when it first shows)."""
        self.tuning_panel.open = not self.tuning_panel.open
        if self.tuning_panel.open:
            self._ensure_tuner()

    def retune_to(self, tuning: Tuning) -> None:
        """Play on with a whole tuning (a preset, or the game's numbers)."""
        self._ensure_tuner()
        if self.tuner is not None:
            self.tuner.load(tuning)

    def time_scale(self, factor: float) -> str:
        """Make the game `factor` times faster at fixed map lengths (`Tuner.time_scale`);
        returns what it did."""
        self._ensure_tuner()
        return "" if self.tuner is None else self.tuner.time_scale(factor)

    def make_speed_normal(self) -> str:
        """Make the current playback speed the game's own: the game becomes that many times
        faster (`time_scale`) and playback goes back to x1, so it looks the same; returns
        what it did."""
        factor = self.loop.time_scale
        if factor == 1.0 or self.loop.replaying:
            return ""
        done = self.time_scale(factor)
        self.loop.time_scale = 1.0
        return done

    def save_ships(self, ships: list, moved: dict[int, int | None] | None = None) -> None:
        """Write the ship files of the match's ship set (`api.ships`, tracked in git), and
        the tuned rule values the cards fly under (`SOLVED_UNDER`: the flight constants and the
        bullet life) into the set's rules.toml, and play on with them. The
        current card becomes the files' default, so the match plays on as it is; when the
        types change (`moved`: each old type's new index, None for a removed one), a new
        match starts with the lineup remapped (a removed type's slots take the first type)."""
        ship_set = self.loop.setup.config.ship_set
        api.ships.write(ships, ship_set)
        # The rules the cards fly under go with them (the set's rules.toml).
        solved_under = {
            key: value
            for key, value in self.current_tuning().rules.items()
            if key.startswith(SOLVED_UNDER)
        }
        if solved_under:
            api.ships.write_rules(dict(api.ships.rules(ship_set), **solved_under), ship_set)
        api.reload_ships()
        if moved is None:
            return
        setup = self.loop.setup
        lineup = setup.lineup or self.loop.rules.team_types
        lineup = tuple(moved.get(kind) or 0 for kind in lineup)
        tuning = Tuning(rules=self.current_tuning().rules)  # the card is the files' now
        self.start(dataclasses.replace(setup, lineup=lineup, tuning=tuning))

    def _on_retune(self, realtime) -> None:
        """A live change: the match is tuned from here, so its recording is dropped (a
        replay re-simulates with the numbers it started from)."""
        self.recorder = None

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
        try:
            controller = controller_of(side, team, loop.setup, loop.rules, loop.types)
        except AgentError as error:
            self.say(str(error))
            return False
        loop.sides[team] = controller
        self._warn([controller])
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
            try:
                sides.append(controller_of(side, team, setup, self.loop.rules, self.loop.types))
                chosen.append(side)
                continue
            except AgentError as error:
                self.say(f"{TEAM_NAMES[team]} plays scripted: {error}")
            sides.append(Scripted(self.loop.rules, self.loop.types))
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
        self.follow(ship)
        self.camera.zoom_in_to(FLYING_ZOOM)
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
        """Keep the camera on this ship (None: stop); it moves there now."""
        self.following = ship
        if ship is not None and self.frame.ships.alive[ship]:
            self.camera.follow(float(self.frame.ships.x[ship]), float(self.frame.ships.y[ship]))

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

    # Prediction overlays ---------------------------------------------------------------------

    def offered_overlays(self) -> tuple[str, ...]:
        """The overlays this match offers (`bnb.play.overlays.offered`)."""
        return offered(self.loop.sides)

    def toggle_overlay(self, name: str) -> None:
        """Turn an overlay on or off."""
        self.overlays.enabled ^= {name}
        self.predicted_at = None

    def set_overlay_side(self, team: int | None) -> None:
        """Draw this team's predictions (None: the default, `bnb.play.overlays.provider`)."""
        self.overlays.side = team
        self.predicted_at = None

    def overlay_provider(self, name: str) -> int | None:
        """The team whose predictions overlay `name` draws now, or None."""
        ship = self.selected_ship()
        selected_team = None if ship is None else int(self.frame.ships.team[ship])
        return provider(name, self.loop.sides, self.overlays.side, selected_team, self.view_team)

    def active_overlays(self) -> tuple[str, ...]:
        """The overlays on that this match offers."""
        return tuple(name for name in self.offered_overlays() if name in self.overlays.enabled)

    def _watch(self, active: tuple[str, ...]) -> None:
        """Mark each agent watched while one of its overlays is on (it then steps eagerly,
        keeping its decision's inputs)."""
        for side in self.loop.sides:
            if hasattr(side, "watched"):
                side.watched = bool(set(active) & set(offered_by(side)))

    def predict(self) -> None:
        """Make the arrays of the overlays on, when the match has moved on since they were
        made (or the overlays changed): the bullet paths, and each providing team's
        predictions."""
        active = self.active_overlays()
        self._watch(active)
        if not active:
            self.predicted, self.bullet_paths, self.predicted_at = {}, None, None
            return
        loop = self.loop
        teams = {self.overlay_provider(name) for name in active} - {None}
        if self.predicted_at == loop.ticks and teams <= self.predicted.keys():
            return
        started = time.perf_counter()
        if self.predicted_at != loop.ticks:
            self.predicted = {}
        self.bullet_paths = None
        if "bullet_paths" in active:
            self.bullet_paths = predictions.bullet_paths(
                loop.rules, loop.types, loop.realtime.state, loop.current
            )
        for team in teams - self.predicted.keys():
            made = predictions.side_predictions(loop.sides[team], team, loop.previous, loop.current)
            if made is not None:
                self.predicted[team] = made
        self.predicted_at = loop.ticks
        self.timings["predictions"] = time.perf_counter() - started

    def overlay_arrays(self) -> dict:
        """Each overlay on that has arrays this frame: its name to (the providing team, or
        -1 for physics, and its arrays)."""
        drawn = {}
        for name in self.active_overlays():
            if name == "bullet_paths":
                if self.bullet_paths is not None:
                    drawn[name] = (-1, self.bullet_paths)
                continue
            team = self.overlay_provider(name)
            made = self.predicted.get(team)
            if made is not None and getattr(made, name) is not None:
                drawn[name] = (team, getattr(made, name))
        return drawn

    def overlay_drawing(self) -> OverlayDrawing | None:
        """This frame's overlay tables, or None when no overlay draws on the battlefield."""
        drawn = self.overlay_arrays()
        drawn.pop("enemy_action", None)  # the readouts live in the selection card
        drawn.pop("value", None)
        if not drawn:
            return None
        return overlay_drawing(
            self.theme,
            self.overlays,
            drawn,
            self.frame,
            self.selected_ship(),
            self.camera.metres_per_pixel,
        )

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
        if self.tuner is not None:
            self.tuner.update()
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
        self.predict()

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
            self.overlay_drawing(),
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
        """Save the live match's recording, stop the tuner and release the session's ImGui
        context."""
        if self.tuner is not None:
            self.tuner.close()
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
