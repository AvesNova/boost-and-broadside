"""The menu (from the top bar's menu button, or Escape), in the middle of the screen:

- the main page (the pause menu when a live match runs, which it pauses): resume,
  restart, new match, replays, save the replay, settings, quit;
- New match: mode, map, fleet size, fields, lineup (each slot's ship type, both teams),
  fog, seed, and each side's controller; "Start";
- Replays: the saved replays, newest first; "Open";
- Settings: theme (and the theme editor), fog in new matches, interface scale, the key
  list.
"""

import dataclasses
import time
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.api import PRESETS

from .. import replay as replays
from ..loop import MatchSetup, SideSetup
from ..sides import policy_choices
from .controls import button
from .fleets import controller_combo
from .style import SCALES
from .text import TEAM_NAMES, type_name

if TYPE_CHECKING:
    from ..app import Session
    from .interface import Interface

MODES = (("team_deathmatch", "Team deathmatch"), ("frontline", "Frontline"))
MAPS = (("standard", "Standard", 2600.0), ("wide", "Wide", 8200.0))
POLICY_REFRESH_SECONDS = 5.0
WIDTH = 460.0  # the menu's width at scale 1


@dataclasses.dataclass
class Draft:
    """The New match page's choices."""

    mode: str = "team_deathmatch"
    map: str = "standard"
    ships_per_team: int = 5
    fields: int = 10
    lineup: list[int] = dataclasses.field(default_factory=lambda: [i % 2 for i in range(5)])
    fogged: bool = True
    seed: int = 0
    sides: list[SideSetup] = dataclasses.field(default_factory=lambda: [SideSetup(), SideSetup()])

    def config(self) -> GameConfig:
        """The game of these choices."""
        radius = dict((key, value) for key, _, value in MAPS)[self.map]
        groups = ()
        if self.mode == "team_deathmatch" and self.map == "wide":
            wide = PRESETS["deathmatch_wide"].spawn_groups
            groups = wide if sum(wide) == self.ships_per_team else ()
        return GameConfig(
            self.ships_per_team, self.fields, radius, mode=self.mode, spawn_groups=groups
        )

    def setup(self) -> MatchSetup:
        """The match of these choices (the default lineup is passed as None)."""
        default = [i % 2 for i in range(self.ships_per_team)]
        lineup = None if self.lineup == default else tuple(self.lineup)
        return MatchSetup(self.config(), lineup, self.fogged, self.seed, tuple(self.sides))

    @classmethod
    def of(cls, setup: MatchSetup) -> Draft:
        """The choices that give this setup."""
        config = setup.config
        size = config.ships_per_team
        lineup = list(setup.lineup) if setup.lineup is not None else [i % 2 for i in range(size)]
        game_map = "wide" if config.arena_radius > 5000.0 else "standard"
        return cls(
            config.mode if config.mode in dict(MODES) else "team_deathmatch",
            game_map,
            size,
            config.num_fields,
            lineup,
            setup.fogged,
            setup.seed,
            list(setup.sides),
        )


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
            self.draft = Draft.of(session.loop.setup)
            self.draft.fogged = session.fog_default  # Settings' fog in new matches
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
            ("replays", "Replays", "Watch a saved match"),
        ]
        if live:
            entries.append(("save replay", "Save replay", "Save this match so far as a replay"))
            entries.append(
                ("tune", "Tune the game", "Change the ships' handles and the rules as it plays")
            )
        entries += [
            ("settings", "Settings", "Theme, fog, interface scale"),
            ("quit", "Quit", "Close the game (a live match's replay is saved)"),
        ]
        for key, label, help in entries:
            if button(controls, f"menu/{key}", label, help, size=(full.x, full.y)):
                if key == "resume":
                    self.close()
                elif key == "restart":
                    session.restart()
                elif key in ("new match", "replays", "settings"):
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
        for key, name in MODES:
            if button(controls, f"new match/mode/{key}", name, active=draft.mode == key):
                draft.mode = key
            imgui.same_line()
        imgui.new_line()
        label("Map")
        for key, name, radius in MAPS:
            if button(
                controls,
                f"new match/map/{key}",
                name,
                f"An arena {radius:.0f} m across the radius",
                active=draft.map == key,
            ):
                draft.map = key
            imgui.same_line()
        imgui.new_line()
        label("Ships a side")
        imgui.set_next_item_width(-1.0)
        changed, size = imgui.slider_int("##ships", draft.ships_per_team, 1, 50)
        controls.record("new match/ships")
        if changed:
            draft.ships_per_team = size
            draft.lineup = (draft.lineup + [i % 2 for i in range(size)])[:size]
            draft.lineup = [
                draft.lineup[i] if i < len(draft.lineup) else i % 2 for i in range(size)
            ]
        label("Fields")
        imgui.set_next_item_width(-1.0)
        changed, fields = imgui.slider_int("##fields", draft.fields, 1, 64)
        controls.record("new match/fields")
        if changed:
            draft.fields = fields
        label("Lineup")
        imgui.begin_group()
        per_row = 10
        for slot, ship_type in enumerate(draft.lineup):
            if slot % per_row:
                imgui.same_line(0.0, 3.0 * scale)
            letter = type_name(ship_type)[0]
            if button(
                controls,
                f"new match/lineup/{slot}",
                f"{letter}##slot{slot}",
                f"Slot {slot + 1}: {type_name(ship_type)} (click to change)",
            ):
                draft.lineup[slot] = (ship_type + 1) % len(api.type_names())
        imgui.end_group()
        imgui.same_line()
        if button(
            controls,
            "new match/lineup/default",
            "Default",
            "Fighters and heavies alternating, as the game sets them",
        ):
            draft.lineup = [i % 2 for i in range(draft.ships_per_team)]
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
                if not session.new_match(setup):
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
        imgui.align_text_to_frame_padding()
        imgui.text("Fog of war")
        imgui.same_line(140.0 * session.scale)
        session.fog_default = switch(
            controls, "settings/fog", session.fog_default, "Fog of war in new matches"
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
