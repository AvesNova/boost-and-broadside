"""The menu's Maps page: make, edit, save, rename, delete and play maps (`bnb.play.maps`).

A dropdown chooses what the form starts from: this match (its game, lineup and tuning,
the tuning panel's changes included), a game preset, or a saved map. The form holds the
whole game: the mode, the ship set, the ships a side, the fields, the arena radius, the
match length, the respawn wait (none in elimination), team deathmatch's spawn groups and
kills to win, and the lineup; a tuning that comes with it is kept (or dropped), and so is
a fixed layout (Clover's fields and spawn points, `GameConfig.layout`), which the form
does not edit: it plays at the layout's own sizes, in team deathmatch or elimination.
Below it, the name and where the map is saved (yours, in the user's configuration
directory; or the project's, tracked in git), and **Save** (a new map, or the chosen saved
map again when the name is its own: editing it, and moving it when the place changed),
**Rename**, **Delete** (after a popup asks) and **Play** (a new match of the form, saved
or not). A saved map shows on the New match page and starts with
`bnb play --preset <name>`.
"""

import dataclasses
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from bnb.contracts import GameConfig
from bnb.contracts.config import ELIMINATION, FRONTLINE, TEAM_DEATHMATCH, MapLayout
from bnb.game import api

from ..loop import MatchSetup
from ..maps import MAP_PRESETS, MODES, PLACE_LABELS, PLACES, Map, preset_map
from ..tuning import Tuning
from .controls import button, tooltip
from .lineups import default_lineup, lineup_row, resized_lineup

if TYPE_CHECKING:
    from .interface import Interface
    from .menus import Menu

THIS_MATCH = "this match"  # the dropdown's first entry
DELETE = "Delete the map?###delete map"


@dataclasses.dataclass
class MapForm:
    """The Maps page's form: a whole game (a `Map` without its name and place), the clocks
    as the seconds they play at x1 (the mode's own when the map leaves them out), the spawn
    groups as typed ("2, 2, 1"; empty: uniform); layout: the map's fixed layout, carried
    unedited (None: a random map)."""

    mode: str
    ship_set: str
    ships_per_team: int
    num_fields: int
    arena_radius: float
    episode_seconds: float
    respawn_seconds: float
    spawn_groups: str
    kills_per_ship: int
    lineup: list[int]
    tuning: Tuning = Tuning()
    layout: MapLayout | None = None

    @classmethod
    def of(cls, game_map: Map) -> MapForm:
        """The form of a map."""
        config = game_map.config
        pool = api.rules_of(config).pools[0]
        tick = api.rules_of(config).tick_length
        lineup = game_map.lineup_types() or api.rules_of(config).team_types
        return cls(
            config.mode,
            config.ship_set,
            config.ships_per_team,
            config.num_fields,
            config.arena_radius,
            pool.episode_ticks * tick,
            pool.respawn_ticks * tick,
            ", ".join(str(group) for group in config.spawn_groups),
            config.kills_per_ship,
            list(lineup),
            game_map.tuning,
            config.layout,
        )

    def config(self) -> GameConfig:
        """The form's game; fails saying what is wrong (spawn groups that do not sum to the
        fleet, ...). The clocks are held to whole ticks, and one equal to the mode's is left
        out, so the map follows its mode's."""
        try:
            groups = tuple(int(part) for part in self.spawn_groups.replace(",", " ").split())
        except ValueError:
            message = f"spawn groups {self.spawn_groups!r}: whole numbers, as 2, 2, 1"
            raise ValueError(message) from None
        if self.mode != TEAM_DEATHMATCH:
            groups = ()
        config = GameConfig(
            self.ships_per_team,
            self.num_fields,
            self.arena_radius,
            mode=self.mode,
            spawn_groups=groups,
            kills_per_ship=self.kills_per_ship,
            ship_set=self.ship_set,
            layout=self.layout,
        )
        own = api.rules_of(config).pools[0]
        tick_rate = round(1.0 / api.rules_of(config).tick_length)
        clocks = {}
        for field, seconds, ticks in (
            ("episode_seconds", self.episode_seconds, own.episode_ticks),
            ("respawn_seconds", self.respawn_seconds, own.respawn_ticks),
        ):
            whole = max(round(seconds * tick_rate), 0)  # the box holds float32: whole ticks
            clocks[field] = None if whole == ticks else whole / tick_rate
        if self.mode == ELIMINATION:  # no respawn: the wait follows the match length
            clocks["respawn_seconds"] = None
        return dataclasses.replace(config, **clocks)

    def lineup_names(self) -> tuple[str, ...] | None:
        """The lineup by type name, None when it is the game's own."""
        config = self.config()
        if tuple(self.lineup) == api.rules_of(config).team_types:
            return None
        names = api.type_names(self.ship_set)
        return tuple(names[kind] for kind in self.lineup)

    def map(self, name: str, place: str) -> Map:
        """The form as a map of this name and place."""
        return Map(name, self.config(), self.lineup_names(), self.tuning, place)

    def setup(self, fogged: bool, seed: int, sides) -> MatchSetup:
        """A match of the form."""
        config = self.config()
        lineup = tuple(self.lineup)
        own = lineup == api.rules_of(config).team_types
        return MatchSetup(config, None if own else lineup, fogged, seed, sides, self.tuning)


def tuning_summary(tuning: Tuning) -> str:
    """A tuning in a few words ("2 rule settings, 1 card field, x2")."""
    parts = []
    if tuning.rules:
        parts.append(f"{len(tuning.rules)} rule setting{'s' * (len(tuning.rules) != 1)}")
    if tuning.card:
        parts.append(f"{len(tuning.card)} card field{'s' * (len(tuning.card) != 1)}")
    if tuning.time_factor != 1.0:
        parts.append(f"x{tuning.time_factor:g}")
    return ", ".join(parts)


class MapEditor:
    """The Maps page's state: what the form started from (`chosen`: THIS_MATCH, a game
    preset's name or a saved map's), the form, the name and the place a save writes to."""

    def __init__(self, session) -> None:
        self.session = session
        self.chosen = THIS_MATCH
        self.form: MapForm | None = None
        self.name = ""
        self.place = "user"
        self.message = ""
        self._saved: list[Map] = []

    def open(self) -> None:
        """Start from this match (the menu opens the page)."""
        self._saved = self.session.maps.saved()
        self.choose(THIS_MATCH)
        problems = self.session.maps.problems()
        self.message = "; ".join(f"unreadable: {problem}" for problem in problems)

    def this_match(self) -> Map:
        """The current match as a map (unnamed), its tuning included."""
        session = self.session
        setup = session.loop.setup
        return Map.of_match("", setup.config, setup.lineup, session.current_tuning())

    def choose(self, name: str) -> None:
        """Start the form from this match, a game preset or a saved map."""
        if name == THIS_MATCH:
            game_map = self.this_match()
        elif name in MAP_PRESETS:
            game_map = preset_map(name)
        else:
            game_map = self.session.maps.find(name)
        self.chosen, self.form, self.message = name, MapForm.of(game_map), ""
        self.name = game_map.name if game_map.saved else ""
        self.place = game_map.place if game_map.saved else "user"

    @property
    def chosen_saved(self) -> bool:
        """Whether the form started from a saved map."""
        return any(game_map.name == self.chosen for game_map in self._saved)

    # Actions --------------------------------------------------------------------------

    def save(self) -> None:
        """Save the form under the name, to the place: the chosen saved map again when the
        name is its own (moved when the place changed); a new map otherwise."""
        maps = self.session.maps
        name = self.name.strip()
        replacing = self.chosen if self.chosen_saved and name == self.chosen else None
        try:
            path = maps.save(self.form.map(name, self.place), replacing=replacing)
        except (ValueError, OSError) as error:
            self.message = str(error)
            return
        self._saved = maps.saved()
        self.chosen = name
        self.message = f"saved {path}"

    def rename(self) -> None:
        """Give the chosen saved map the typed name."""
        maps = self.session.maps
        name = self.name.strip()
        try:
            path = maps.rename(self.chosen, name)
        except (ValueError, OSError) as error:
            self.message = str(error)
            return
        self._saved = maps.saved()
        self.message = f"renamed {self.chosen} to {name}: {path}"
        self.chosen = name

    def delete(self) -> None:
        """Delete the chosen saved map; the form stays, as an unsaved map."""
        try:
            self.session.maps.delete(self.chosen)
        except (ValueError, OSError) as error:
            self.message = str(error)
            return
        self._saved = self.session.maps.saved()
        self.message = f"deleted {self.chosen}"
        self.chosen = THIS_MATCH

    def play(self, menu: Menu) -> None:
        """A new match of the form (with its tuning, when it brings one)."""
        session = self.session
        try:
            setup = self.form.setup(session.fog_default, 0, session.loop.setup.sides)
        except ValueError as error:
            self.message = str(error)
            return
        if not session.new_match(setup, tuned=bool(self.form.tuning)):
            self.message = session.notice

    # Drawing --------------------------------------------------------------------------

    def draw(self, interface: Interface, menu: Menu) -> None:
        """The page."""
        session, controls = self.session, interface.controls
        scale = session.scale
        menu._title("Maps")
        label_width = 120.0 * scale

        def label(text: str) -> None:
            imgui.align_text_to_frame_padding()
            imgui.text(text)
            imgui.same_line(label_width)

        label("Start from")
        self._choice(interface)
        form = self.form
        imgui.separator()
        label("Mode")
        for key, title in MODES:
            if button(controls, f"maps/mode/{key}", title, active=form.mode == key):
                form.mode = key
            imgui.same_line()
        imgui.new_line()
        label("Ship set")
        imgui.set_next_item_width(-1.0)
        if imgui.begin_combo("##ship set", form.ship_set):
            for ship_set in api.ships.sets():
                if imgui.selectable(ship_set, ship_set == form.ship_set)[0]:
                    form.ship_set = ship_set
                    form.lineup = default_lineup(form.ships_per_team, ship_set)
                controls.record(f"maps/ship set/{ship_set}")
            imgui.end_combo()
        controls.record("maps/ship set")
        label("Ships a side")
        imgui.set_next_item_width(-1.0)
        changed, size = imgui.slider_int("##map ships", form.ships_per_team, 1, 50)
        controls.record("maps/ships")
        if changed:
            form.ships_per_team = size
            form.lineup = resized_lineup(form.lineup, size, form.ship_set)
        label("Fields")
        imgui.set_next_item_width(-1.0)
        _, form.num_fields = imgui.slider_int("##map fields", form.num_fields, 0, 64)
        controls.record("maps/fields")
        boxes = (
            ("Arena radius", "arena_radius", "m", "The arena's radius"),
            ("Match length", "episode_seconds", "s", "Seconds at x1, whole ticks (1/30 s)"),
            (
                "Respawn minimum" if form.mode == FRONTLINE else "Respawn wait",
                "respawn_seconds",
                "s",
                "Seconds at x1, whole ticks (1/30 s)"
                + (", then the team's next respawn wave" if form.mode == FRONTLINE else ""),
            ),
        )
        for title, field, unit, help in boxes[:2] if form.mode == ELIMINATION else boxes:
            label(title)
            imgui.set_next_item_width(120.0 * scale)
            _, value = imgui.input_float(f"##{field}", getattr(form, field), 0.0, 0.0, "%.4g")
            controls.record(f"maps/{field}")
            tooltip(help)
            setattr(form, field, value)
            imgui.same_line()
            imgui.text(unit)
        if form.mode == TEAM_DEATHMATCH:
            label("Spawn groups")
            imgui.set_next_item_width(120.0 * scale)
            _, form.spawn_groups = imgui.input_text_with_hint(
                "##spawn groups", "uniform", form.spawn_groups
            )
            controls.record("maps/spawn groups")
            tooltip("Ships a side in each spawn group, as 2, 2, 1 (summing to the ships a side)")
            label("Kills to win")
            imgui.set_next_item_width(120.0 * scale)
            _, form.kills_per_ship = imgui.input_int("##kills", form.kills_per_ship)
            controls.record("maps/kills")
            tooltip("Kills per ship a side: the winning score is this times the ships a side")
        label("Lineup")
        lineup_row(interface, "maps/lineup", form.lineup, form.ship_set, form.ships_per_team)
        if form.tuning:
            label("Tuning")
            imgui.text(tuning_summary(form.tuning))
            imgui.same_line()
            if button(controls, "maps/drop tuning", "Drop", "Play the game's own numbers"):
                form.tuning = Tuning()
        imgui.separator()
        label("Name")
        imgui.set_next_item_width(-1.0)
        _, self.name = imgui.input_text_with_hint("##map name", "lowercase-with-hyphens", self.name)
        controls.record("maps/name")
        label("Saved")
        for place in PLACES:
            title = "Yours" if place == "user" else "In the project"
            where = session.maps.directories[place]
            if button(
                controls, f"maps/place/{place}", title, str(where), active=self.place == place
            ):
                self.place = place
            imgui.same_line()
        imgui.new_line()
        imgui.spacing()
        if button(controls, "maps/save", "Save", "Save the form under the name"):
            self.save()
        imgui.same_line()
        imgui.begin_disabled(not self.chosen_saved)
        if button(controls, "maps/rename", "Rename", "Give the chosen saved map the name"):
            self.rename()
        imgui.same_line()
        if button(controls, "maps/delete", "Delete", "Delete the chosen saved map"):
            imgui.open_popup(DELETE)
        imgui.end_disabled()
        self._confirm_delete(interface)
        imgui.same_line()
        if button(controls, "maps/play", "Play", "A new match of the form, saved or not"):
            self.play(menu)
        imgui.same_line()
        if button(controls, "menu/back", "Back"):
            menu.page = "main"
        if self.message:
            imgui.spacing()
            imgui.text_wrapped(self.message)

    def _choice(self, interface: Interface) -> None:
        """The dropdown of what the form starts from, recorded as `maps/start from` and
        `maps/start from/<name>`."""
        controls = interface.controls
        imgui.set_next_item_width(-1.0)
        entries = [(THIS_MATCH, "This match")]
        entries += [(name, f"{name} (game preset)") for name in MAP_PRESETS]
        entries += [
            (game_map.name, f"{game_map.name} ({PLACE_LABELS[game_map.place]})")
            for game_map in self._saved
        ]
        shown = dict(entries).get(self.chosen, self.chosen)
        if imgui.begin_combo("##start from", shown, imgui.ComboFlags_.height_largest):
            for name, title in entries:
                if imgui.selectable(title, name == self.chosen)[0]:
                    try:
                        self.choose(name)
                    except ValueError as error:
                        self.message = str(error)
                controls.record(f"maps/start from/{name}")
            imgui.end_combo()
        controls.record("maps/start from")

    def _confirm_delete(self, interface: Interface) -> None:
        """The modal that asks before **Delete** deletes a saved map's file."""
        if not imgui.begin_popup_modal(DELETE, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        controls = interface.controls
        imgui.text(f"Delete the map {self.chosen}?")
        if button(controls, "maps/delete/confirm", "Delete##confirm"):
            self.delete()
            imgui.close_current_popup()
        imgui.same_line()
        if button(controls, "maps/delete/cancel", "Cancel"):
            imgui.close_current_popup()
        imgui.end_popup()
