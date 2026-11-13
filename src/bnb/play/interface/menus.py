"""The menu (from the top bar's menu button, or Escape), in the middle of the screen:

- the main page (the pause menu when a live match runs, which it pauses): resume,
  restart, new match, maps, replays, save the replay, tune the game, settings, quit;
- New match: mode, map (one per game preset's map, then the saved maps: `maps`; choosing
  one takes its fleet size, fields and lineup, and a saved map's mode and tuning), fleet
  size, fields, lineup (each slot's ship type, a dropdown of the map's ship set, both
  teams: `lineups.lineup_row`), fog, seed, and each side's controller; "Start";
- Maps: make, edit, save, rename and delete maps (`map_editor`);
- Replays: the saved replays, newest first; "Open";
- Settings: theme (and the theme editor), the dot grid, the distance contours, the
  look-ahead camera, the bullets' look (tracer or trail only, the head's size, the trail's
  widths and length), fog in new matches, the frame-rate readout, interface scale, the key
  list.
"""

import dataclasses
import time
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from bnb.contracts import GameConfig
from bnb.contracts.config import LAYOUT_MODES
from bnb.game import api
from bnb.game.api import PRESETS

from .. import replay as replays
from ..loop import MatchSetup, SideSetup
from ..maps import MAP_PRESETS, MODES, Map, map_of
from ..render.bullets import STYLES as BULLET_STYLES
from ..sides import policy_choices
from ..tuning import Tuning
from .controls import button, tooltip
from .fleets import controller_combo
from .lineups import default_lineup, lineup_row, resized_lineup
from .map_editor import MapEditor
from .shortcuts import shortcut
from .style import SCALES
from .text import TEAM_NAMES
from .top_bar import CONTOUR_HELP, CONTOUR_LABELS, GRID_LABELS

if TYPE_CHECKING:
    from ..app import Session
    from .interface import Interface

MODE_LABELS = dict(MODES)
POLICY_REFRESH_SECONDS = 5.0
WIDTH = 460.0  # the menu's width at scale 1
LOOK_AHEAD_LABELS = {"off": "Off", "flying": "Flying", "always": "Always"}
# The bullets' look's numbers (`render.bullets.BulletLook`): field, label, low, high, format,
# tooltip.
BULLET_SLIDERS = (
    ("head_pixels", "Head size", 0.5, 8.0, "%.1f px", "The head's radius (tracer style)"),
    ("trail_head_pixels", "Trail at head", 0.25, 16.0, "%.2f px", "The trail's width at the head"),
    ("trail_tail_pixels", "Trail at tail", 0.0, 16.0, "%.2f px", "The trail's width at its tail"),
    (
        "trail_seconds",
        "Trail length",
        0.0,
        0.5,
        "%.3f s",
        "The trail's length: the bullet's last seconds of flight as the screen shows them",
    ),
)


@dataclasses.dataclass(frozen=True)
class GameMap:
    """A map the New match page offers: its key (a game preset's name without its mode
    word, "standard" for none; a saved map's name), label, ship set, arena radius (m), and
    its first preset's fleet size and fields; `saved`: the saved map it is (which plays its
    own mode), None for a game preset's."""

    key: str
    label: str
    ship_set: str
    arena_radius: float
    ships_per_team: int
    num_fields: int
    saved: Map | None = None


def maps(saved: list[Map] | tuple[Map, ...] = ()) -> tuple[GameMap, ...]:
    """The maps of the game presets (`MAP_PRESETS`), in the presets' order, then the saved
    maps given."""
    found: dict[str, GameMap] = {}
    for name, config in MAP_PRESETS.items():
        key = map_of(name)
        if key not in found:
            label = key.replace("_", " ").capitalize()
            found[key] = GameMap(
                key,
                label,
                config.ship_set,
                config.arena_radius,
                config.ships_per_team,
                config.num_fields,
            )
    for game_map in saved:
        config = game_map.config
        found[game_map.name] = GameMap(
            game_map.name,
            game_map.name.replace("-", " ").capitalize(),
            config.ship_set,
            config.arena_radius,
            config.ships_per_team,
            config.num_fields,
            game_map,
        )
    return tuple(found.values())


def preset_of(key: str, mode: str) -> str:
    """The preset the map `key` plays in `mode`: its preset of that mode, else its first."""
    of_map = [name for name in MAP_PRESETS if map_of(name) == key]
    exact = [name for name in of_map if PRESETS[name].mode == mode]
    return (exact or of_map)[0]


def map_lineup(key: str, mode: str) -> list[int]:
    """The lineup a game preset's map starts with: its preset's own (`api.preset_lineup`),
    else the play shell's default for its fleet size and ship set."""
    preset = preset_of(key, mode)
    own = api.preset_lineup(preset)
    if own is not None:
        return list(own)
    return default_lineup(PRESETS[preset].ships_per_team, PRESETS[preset].ship_set)


DEFAULT_MAP = "real_scale"  # the New match page's map at the start


def _ship_set_of(key: str) -> str:
    """The ship set of the game presets' map `key`."""
    return {game_map.key: game_map for game_map in maps()}[key].ship_set


@dataclasses.dataclass
class Draft:
    """The New match page's choices; `offered`: the maps it offers (the game presets' and
    the saved ones when the page opened)."""

    mode: str = "team_deathmatch"
    map: str = DEFAULT_MAP
    ships_per_team: int = 5
    fields: int = 10
    lineup: list[int] = dataclasses.field(
        default_factory=lambda: default_lineup(5, _ship_set_of(DEFAULT_MAP))
    )
    fogged: bool = True
    seed: int = 0
    sides: list[SideSetup] = dataclasses.field(default_factory=lambda: [SideSetup(), SideSetup()])
    offered: tuple[GameMap, ...] = dataclasses.field(default_factory=maps)

    @property
    def chosen(self) -> GameMap:
        """The chosen map."""
        return {game_map.key: game_map for game_map in self.offered}[self.map]

    @property
    def ship_set(self) -> str:
        """The chosen map's ship set."""
        return self.chosen.ship_set

    @property
    def tuned(self) -> bool:
        """Whether the chosen map brings a tuning of its own (a saved map's), which a new
        match plays instead of the current match's."""
        saved = self.chosen.saved
        return saved is not None and bool(saved.tuning)

    def config(self) -> GameConfig:
        """The game of these choices: the saved map's, or the preset of this mode and map
        (for a pair no preset has, a preset of the map in this mode), with the chosen sizes
        (its spawn groups kept when they fit the fleet, and its fixed layout when it fits
        the sizes and the mode: else the map's arena draws its fields and spawns)."""
        saved = self.chosen.saved
        preset = saved.config if saved is not None else PRESETS[preset_of(self.map, self.mode)]
        groups = preset.spawn_groups if self.mode == "team_deathmatch" else ()
        if sum(groups) != self.ships_per_team:
            groups = ()
        layout = preset.layout
        if layout is not None and (
            self.mode not in LAYOUT_MODES
            or (layout.fields and 2 * len(layout.fields) != self.fields)
            or (layout.spawns and len(layout.spawns) != self.ships_per_team)
        ):
            layout = None
        return dataclasses.replace(
            preset,
            ships_per_team=self.ships_per_team,
            num_fields=self.fields,
            mode=self.mode,
            spawn_groups=groups,
            # elimination has no respawn (a preset of the map in another mode may have one)
            respawn_seconds=None if self.mode == "elimination" else preset.respawn_seconds,
            layout=layout,
        )

    def setup(self) -> MatchSetup:
        """The match of these choices (the game's own lineup, `Rules.team_types`, is
        passed as None), with a saved map's tuning."""
        config = self.config()
        lineup = tuple(self.lineup)
        own = lineup == api.rules_of(config).team_types
        saved = self.chosen.saved
        tuning = saved.tuning if saved is not None else Tuning()
        return MatchSetup(
            config, None if own else lineup, self.fogged, self.seed, tuple(self.sides), tuning
        )

    @classmethod
    def of(cls, setup: MatchSetup, offered: tuple[GameMap, ...] | None = None) -> Draft:
        """The choices that give this setup: the map whose game it is at these sizes,
        else the game presets' map of its ship set nearest its arena."""
        offered = maps() if offered is None else offered
        config = setup.config
        size = config.ships_per_team
        lineup = list(setup.lineup if setup.lineup is not None else api.rules_of(config).team_types)
        mode = config.mode if config.mode in dict(MODES) else "team_deathmatch"
        draft = cls(
            mode,
            offered[0].key,
            size,
            config.num_fields,
            lineup,
            setup.fogged,
            setup.seed,
            list(setup.sides),
            offered,
        )
        for game_map in offered:
            draft.map = game_map.key
            if game_map.saved is not None and game_map.saved.config.mode != mode:
                continue
            if draft.config() == config:
                return draft
        candidates = [m for m in offered if m.saved is None and m.ship_set == config.ship_set] or [
            m for m in offered if m.saved is None
        ]
        draft.map = min(candidates, key=lambda m: abs(m.arena_radius - config.arena_radius)).key
        return draft

    def choose_map(self, key: str) -> None:
        """Choose a map, with its fleet size, fields and lineup: a saved map's own (and its
        mode), a game preset's map's in this mode (`map_lineup`)."""
        self.map = key
        saved = self.chosen.saved
        if saved is not None:
            self.mode = saved.config.mode
            self.ships_per_team, self.fields = saved.config.ships_per_team, saved.config.num_fields
            own = saved.lineup_types()
            size = self.ships_per_team
            self.lineup = (
                list(own) if own is not None else list(api.rules_of(saved.config).team_types)
            )
            self.lineup = resized_lineup(self.lineup, size, saved.config.ship_set)
            return
        preset = PRESETS[preset_of(key, self.mode)]
        self.ships_per_team, self.fields = preset.ships_per_team, preset.num_fields
        self.lineup = map_lineup(key, self.mode)


def switch(controls, name: str, value: bool, help: str) -> bool:
    """An On / Off pair of buttons (recorded as `name/on` and `name/off`); returns the
    value after this frame's click."""
    if button(controls, f"{name}/on", "On", help, active=value):
        value = True
    imgui.same_line(0.0, 2.0)
    if button(controls, f"{name}/off", "Off", help, active=not value):
        value = False
    return value


class Menu:
    """The menu's state: whether it is open, its page, the New match draft, and whether
    opening it paused the match (closing resumes it)."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.page: str | None = None
        self.draft = Draft()
        self.editor = MapEditor(session)
        self.paused_by_menu = False
        self.chosen_replay = 0
        self._policies: list = []
        self._policies_read = -1e9
        self.message = ""

    @property
    def is_open(self) -> bool:
        """Whether the menu shows."""
        return self.page is not None

    def show(self, page: str) -> None:
        """Open the menu at a page (pausing a live match that plays)."""
        session = self.session
        if self.page is None and not session.loop.replaying and not session.loop.paused:
            session.loop.paused = True
            self.paused_by_menu = True
        if page == "new match":
            self.draft = Draft.of(session.loop.setup, maps(session.maps.saved()))
            self.draft.fogged = session.fog_default  # Settings' fog in new matches
        if page == "maps":
            self.editor.open()
        self.page, self.message = page, ""

    def close(self) -> None:
        """Close the menu (resuming the match it paused)."""
        if self.paused_by_menu:
            self.session.loop.paused = False
        self.page, self.paused_by_menu = None, False

    def policies(self) -> list:
        """The local runs' policy files (re-read every few seconds)."""
        now = time.monotonic()
        if now - self._policies_read > POLICY_REFRESH_SECONDS:
            self._policies = policy_choices(self.session.options.runs)
            self._policies_read = now
        return self._policies

    # Drawing --------------------------------------------------------------------------------

    def draw(self, interface: Interface) -> None:
        """The menu, when open, in the middle of the screen."""
        if self.page is None:
            return
        session = self.session
        width, height = imgui.get_io().display_size
        imgui.set_next_window_pos(
            imgui.ImVec2(0.5 * width, 0.5 * height), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5)
        )
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(WIDTH * session.scale, 0.0),
            imgui.ImVec2(WIDTH * session.scale, 0.9 * height),
        )
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_collapse
        )
        imgui.push_style_color(imgui.Col_.window_bg, imgui.get_style().color_(imgui.Col_.popup_bg))
        imgui.begin("###menu", None, flags)
        imgui.pop_style_color()
        if imgui.is_window_appearing():
            imgui.set_window_focus()
        pages = {
            "main": self._main,
            "new match": self._new_match,
            "replays": self._replays,
            "settings": self._settings,
            "maps": lambda interface: self.editor.draw(interface, self),
        }
        page = self.page
        pages[page](interface)
        if self.message:
            imgui.spacing()
            imgui.text_wrapped(self.message)
        imgui.end()

    def _title(self, title: str) -> None:
        imgui.push_font(self.session.fonts.bold, imgui.get_font_size() * 1.3)
        imgui.text(title)
        imgui.pop_font()
        imgui.separator()
        imgui.spacing()

    def _back(self, interface: Interface) -> None:
        imgui.spacing()
        if button(interface.controls, "menu/back", "Back"):
            self.page = "main"
            self.message = ""

    def _main(self, interface: Interface) -> None:
        session, controls = self.session, interface.controls
        live = not session.loop.replaying
        self._title("Paused" if live and self.paused_by_menu else "Menu")
        full = imgui.ImVec2(-1.0, 0.0)
        entries = [
            ("resume", "Resume", "Back to the match"),
            (
                "restart",
                "Restart" if live else "Play this match live",
                "Start this match's setup again",
            ),
            ("new match", "New match", "Choose the mode, map, fleets and sides"),
            ("maps", "Maps", "Make, edit, rename and delete your maps"),
            ("replays", "Replays", "Watch a saved match"),
        ]
        if live:
            entries.append(("save replay", "Save replay", "Save this match so far as a replay"))
            entries.append(
                ("tune", "Tune the game", "Change the ships' handles and the rules as it plays")
            )
        entries += [
            ("settings", "Settings", "Theme, dot grid, contours, look ahead, fog, interface scale"),
            ("quit", "Quit", "Close the game (a live match's replay is saved)"),
        ]
        for key, label, help in entries:
            if button(controls, f"menu/{key}", label, help, size=(full.x, full.y)):
                if key == "resume":
                    self.close()
                elif key == "restart":
                    session.restart()
                elif key in ("new match", "maps", "replays", "settings"):
                    self.show(key)
                elif key == "save replay":
                    path = session.save_replay()
                    self.message = f"saved {path.name}" if path else session.notice
                elif key == "tune":
                    self.close()
                    if not session.tuning_panel.open:
                        session.toggle_tuning()
                else:
                    session.quit()

    def _new_match(self, interface: Interface) -> None:
        session, controls, draft = self.session, interface.controls, self.draft
        scale = session.scale
        self._title("New match")
        label_width = 120.0 * scale

        def label(text: str) -> None:
            imgui.align_text_to_frame_padding()
            imgui.text(text)
            imgui.same_line(label_width)

        label("Mode")
        saved = draft.chosen.saved
        for key, name in MODES:
            imgui.begin_disabled(saved is not None and key != saved.config.mode)
            if button(controls, f"new match/mode/{key}", name, active=draft.mode == key):
                draft.mode = key
            imgui.end_disabled()
            if saved is not None and key != saved.config.mode:
                tooltip("A saved map plays its own mode (the Maps page changes it)")
            imgui.same_line()
        imgui.new_line()
        label("Map")
        imgui.begin_group()
        room = imgui.get_content_region_avail().x
        used = 0.0
        for game_map in draft.offered:
            width = imgui.calc_text_size(game_map.label).x + 2.0 * imgui.get_style().frame_padding.x
            if used and used + width > room:
                used = 0.0
            elif used:
                imgui.same_line()
            used += width + imgui.get_style().item_spacing.x
            own = f"; {MODE_LABELS[game_map.saved.config.mode]}, saved" if game_map.saved else ""
            if button(
                controls,
                f"new match/map/{game_map.key}",
                game_map.label,
                f"An arena of radius {game_map.arena_radius:.0f} m, "
                f"{game_map.ships_per_team} ships a side and {game_map.num_fields} fields; "
                f"the {game_map.ship_set} ships{own}",
                active=draft.map == game_map.key,
            ):
                try:
                    draft.choose_map(game_map.key)
                except ValueError as error:
                    self.message = str(error)
        imgui.end_group()
        label("Ships a side")
        imgui.set_next_item_width(-1.0)
        changed, size = imgui.slider_int("##ships", draft.ships_per_team, 1, 50)
        controls.record("new match/ships")
        if changed:
            draft.ships_per_team = size
            draft.lineup = resized_lineup(draft.lineup, size, draft.ship_set)
        label("Fields")
        imgui.set_next_item_width(-1.0)
        changed, fields = imgui.slider_int("##fields", draft.fields, 0, 64)
        controls.record("new match/fields")
        if changed:
            draft.fields = fields
        label("Lineup")
        lineup_row(
            interface, "new match/lineup", draft.lineup, draft.ship_set, draft.ships_per_team
        )
        label("Fog of war")
        draft.fogged = switch(
            controls,
            "new match/fog",
            draft.fogged,
            "Each team sees only what its ships see (the spectator sees everything)",
        )
        label("Seed")
        imgui.set_next_item_width(120.0 * scale)
        _, draft.seed = imgui.input_int("##seed", draft.seed)
        controls.record("new match/seed")
        imgui.spacing()
        for team in range(2):
            label(TEAM_NAMES[team])
            chosen = controller_combo(
                interface, team, f"new match/side{team}", draft.sides[team], -1.0
            )
            if chosen is not None:
                draft.sides[team] = chosen
        imgui.spacing()
        imgui.separator()
        if button(controls, "new match/start", "Start", "Start this match"):
            try:
                setup = draft.setup()
            except ValueError as error:
                self.message = str(error)
            else:
                if not session.new_match(setup, tuned=draft.tuned):
                    self.message = session.notice
        imgui.same_line()
        if button(controls, "menu/back", "Back"):
            self.page = "main"

    def _replays(self, interface: Interface) -> None:
        session, controls = self.session, interface.controls
        self._title("Replays")
        found = replays.replays(session.options.replays)
        if not found:
            imgui.text_disabled(f"no replays in {session.options.replays} yet")
        imgui.begin_child(
            "replay list",
            imgui.ImVec2(0.0, min(len(found), 10) * imgui.get_frame_height_with_spacing() + 8.0),
            imgui.ChildFlags_.borders,
        )
        for index, path in enumerate(found):
            if imgui.selectable(
                path.name, index == self.chosen_replay, imgui.SelectableFlags_.allow_double_click
            )[0]:
                self.chosen_replay = index
                if imgui.is_mouse_double_clicked(0):
                    self._open(found[index])
            controls.record(f"replays/{index}")
        imgui.end_child()
        if found and button(controls, "replays/open", "Open", "Watch the chosen replay"):
            self._open(found[min(self.chosen_replay, len(found) - 1)])
        imgui.same_line()
        if button(controls, "menu/back", "Back"):
            self.page = "main"

    def _open(self, path) -> None:
        if not self.session.open_replay(path):
            self.message = self.session.notice

    def _choices(
        self,
        interface: Interface,
        title: str,
        name: str,
        labels: dict[str, str],
        chosen: str,
        choose,
        help: str,
        key: str | None = None,
    ) -> None:
        """A settings row: a button per value (labels: value to label), the chosen one
        active; `choose` takes the value clicked."""
        scale = self.session.scale
        imgui.align_text_to_frame_padding()
        imgui.text(title)
        imgui.same_line(140.0 * scale)
        for index, (value, label) in enumerate(labels.items()):
            if index:
                imgui.same_line(0.0, 3.0 * scale)
            on = chosen == value
            labelled = f"{label}###{name}/{value}"  # two rows may share a label (Off)
            if button(interface.controls, f"{name}/{value}", labelled, help, key, active=on):
                choose(value)

    def _bullets(self, interface: Interface) -> None:
        """The bullets' look: the style, then a slider per number (the head's size only for
        the tracer)."""
        session, controls = self.session, interface.controls
        look = session.bullet_look
        self._choices(
            interface,
            "Bullets",
            "settings/bullets",
            BULLET_STYLES,
            look.style,
            lambda style: session.set_bullet_look(style=style),
            "A round head and its fading trail, or the fading trail alone",
        )
        for field, label, low, high, form, help in BULLET_SLIDERS:
            if field == "head_pixels" and look.style != "tracer":
                continue
            imgui.align_text_to_frame_padding()
            imgui.text(f"  {label}")
            imgui.same_line(140.0 * session.scale)
            imgui.set_next_item_width(220.0 * session.scale)
            changed, value = imgui.slider_float(
                f"##bullets {field}", getattr(look, field), low, high, form
            )
            controls.record(f"settings/bullets/{field}")
            tooltip(help)
            if changed:
                session.set_bullet_look(**{field: value})

    def _settings(self, interface: Interface) -> None:
        session, controls = self.session, interface.controls
        self._title("Settings")
        imgui.align_text_to_frame_padding()
        imgui.text("Theme")
        imgui.same_line(140.0 * session.scale)
        imgui.set_next_item_width(180.0 * session.scale)
        if imgui.begin_combo("##theme", session.theme.name):
            for name in session.theme_names:
                if imgui.selectable(name, name == session.theme.name)[0]:
                    session.set_theme(name)
                controls.record(f"settings/theme/{name}")
            imgui.end_combo()
        controls.record("settings/theme")
        imgui.same_line()
        if button(controls, "settings/editor", "Edit", "The theme editor", "E"):
            session.toggle_editor()
        self._choices(
            interface,
            "Dot grid",
            "settings/grid",
            GRID_LABELS,
            session.grid,
            session.set_grid,
            "Faint dots fixed to the map, so movement shows where nothing else is near",
            shortcut("cycle_grid"),
        )
        self._choices(
            interface,
            "Contours",
            "settings/contours",
            CONTOUR_LABELS,
            session.contours,
            session.set_contours,
            CONTOUR_HELP,
            shortcut("cycle_contours"),
        )
        self._choices(
            interface,
            "Look ahead",
            "settings/look ahead",
            LOOK_AHEAD_LABELS,
            session.look_ahead,
            session.set_look_ahead,
            "Lead a followed ship's view in its direction of travel: never, while flying it, "
            "or always",
        )
        self._bullets(interface)
        imgui.align_text_to_frame_padding()
        imgui.text("Fog of war")
        imgui.same_line(140.0 * session.scale)
        session.fog_default = switch(
            controls, "settings/fog", session.fog_default, "Fog of war in new matches"
        )
        imgui.align_text_to_frame_padding()
        imgui.text("Frame rate")
        imgui.same_line(140.0 * session.scale)
        session.show_fps = switch(
            controls,
            "settings/fps",
            session.show_fps,
            "The frame rate in the top bar: frames per second and the frame time (smoothed), "
            "and the longest frame of the last second (F4)",
        )
        imgui.align_text_to_frame_padding()
        imgui.text("Interface scale")
        imgui.same_line(140.0 * session.scale)
        for value in SCALES:
            if button(
                controls, f"settings/scale/{value:g}", f"{value:g}", active=session.scale == value
            ):
                session.set_scale(value)
            imgui.same_line(0.0, 3.0 * session.scale)
        imgui.new_line()
        imgui.align_text_to_frame_padding()
        imgui.text("Keys")
        imgui.same_line(140.0 * session.scale)
        if button(controls, "settings/keys", "Show the key list", "Every shortcut", "F1"):
            session.toggle_keys()
        self._back(interface)
