"""The Map / mode panel: everything that makes the map and the game's mode, apart from the
ship types (the "Ships and flight" panel, `tuning_panel`).

From the top: **New map** (the same game on a freshly drawn map, now), **Restart match**
(the same map, a fresh match) and **Save map** (the menu's Maps page, starting from this
match); the game mode; then the rule groups that belong to the map: arena and clocks, the
random fields' knobs, spawns, and the mode's own (tickets to win, a frontline's zone).
A setting marked * applies from the next episode, so **New map** and **Restart match**
are what show it. Opens with F3 or the top bar's **Map / mode**.
"""

import dataclasses
import math

from imgui_bundle import imgui

from bnb.game import api

from .interface.layout import MARGIN, TOP_BAR, Rect
from .maps import MODES
from .tuning_panel import DIM, ERROR, MAP_GROUPS, PANEL_WIDTH, draw_rules

BUTTON_HELP = dict(
    new_map="A new match on a freshly drawn map: the fields, the spawn axis and a frontline's "
    "zone ring are drawn again, with the rules as tuned now",
    restart="The same map and setup again (the same seed), a fresh match",
    save="Save this game, its lineup and its tuning as a named map (the menu's Maps page)",
)


class MapPanel:
    """open: whether the window shows; `window`: where it was last drawn."""

    def __init__(self) -> None:
        self.open = False
        self.window: Rect | None = None
        self.message = ""

    def draw(self, session) -> None:
        """The window, in the current ImGui frame."""
        if not self.open:
            return
        scale = session.scale
        width, height = imgui.get_io().display_size
        panel = min(PANEL_WIDTH * scale, width - 2.0 * MARGIN * scale)
        top = (TOP_BAR + 2.0 * MARGIN) * scale
        # Left of the ship panel's first place, so the two sit side by side.
        imgui.set_next_window_pos(
            (width - 2.0 * panel - 2.0 * MARGIN * scale, top), imgui.Cond_.first_use_ever
        )
        imgui.set_next_window_size(
            (panel, max(height - top - MARGIN * scale, 0.0)), imgui.Cond_.first_use_ever
        )
        imgui.set_next_window_size_constraints((0.0, 0.0), (math.inf, max(height - top, 0.0)))
        expanded, self.open = imgui.begin("Map / mode###map panel", self.open)
        position, size = imgui.get_window_pos(), imgui.get_window_size()
        self.window = Rect(position.x, position.y, size.x, size.y)
        if expanded:
            tuner = session.tuner
            if session.loop.replaying or tuner is None:
                imgui.text_wrapped(
                    "The map and mode panel changes a live match; a replay plays as recorded."
                )
            else:
                self._buttons(session)
                self._mode(session)
                imgui.separator()
                draw_rules(tuner, scale, lambda group: group in MAP_GROUPS, open_groups=True)
                imgui.text_colored(DIM, self.radius_line(tuner))
        imgui.end()

    @staticmethod
    def radius_line(tuner) -> str:
        """The field radii the current settings draw, at map scale 1."""
        names = (
            "field_radius_min",
            "field_radius_max",
            "field_radius_distribution",
            "field_radius_median",
            "field_radius_log_sigma",
        )
        changes = {name: tuner.values[f"rules.{name}"] for name in names}
        changes["field_radius_distribution"] = int(changes["field_radius_distribution"])
        mean, low, high = api.field_radius_summary(dataclasses.replace(tuner.base, **changes))
        return f"field radius: mean {mean:.0f} m, 10th {low:.0f} m, 90th {high:.0f} m (map scale 1)"

    def _buttons(self, session) -> None:
        controls = session.interface.controls
        if imgui.button("New map"):
            session.new_map()
        controls.record("map/new map")
        self._tip(BUTTON_HELP["new_map"])
        imgui.same_line()
        if imgui.button("Restart match"):
            session.restart()
        controls.record("map/restart")
        self._tip(BUTTON_HELP["restart"])
        imgui.same_line()
        if imgui.button("Save map"):
            session.interface.menu.show("maps")
        controls.record("map/save")
        self._tip(BUTTON_HELP["save"])
        if self.message:
            imgui.text_colored(ERROR, self.message)
        imgui.text_colored(DIM, "* applies from the next episode: New map or Restart shows it")
        imgui.separator()

    def _mode(self, session) -> None:
        controls = session.interface.controls
        current = session.loop.setup.config.mode
        imgui.align_text_to_frame_padding()
        imgui.text("Mode")
        for key, name in MODES:
            imgui.same_line()
            active = key == current
            if active:
                imgui.push_style_color(
                    imgui.Col_.button, imgui.get_style().color_(imgui.Col_.button_active)
                )
            if imgui.button(f"{name}##mode") and not active:
                self.message = "" if session.set_mode(key) else session.notice
            if active:
                imgui.pop_style_color()
            controls.record(f"map/mode/{key}")
            self._tip("Start a new match of this game in this mode")

    @staticmethod
    def _tip(text: str) -> None:
        if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
            imgui.set_tooltip(text)
