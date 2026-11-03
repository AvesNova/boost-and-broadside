"""The interface: the cards, the menus and the battlefield's own input, in one ImGui frame.

Each frame draws, in order: the top bar, the fleets card, the selection card, the
minimap, the kill feed, the scrubber (replays), the flying HUD (while flying), the menu,
the theme editor and the key list; then it reads the input nothing else took: a click on
the battlefield selects (the hit test), a drag pans, the wheel zooms, the keys fly the
flown ship, and the shortcuts call their session methods. Which cards are open is
remembered for the session.
"""

from typing import TYPE_CHECKING

import torch
from imgui_bundle import imgui

from bnb.controllers.human import INPUT, INPUTS

from ..selection import hit_test
from . import fleets, flying, inspector, kill_feed, menus, minimap, scrubber, top_bar
from .controls import Controls
from .layout import CARDS, Layout
from .shortcuts import FLIGHT_KEYS, KEYS, SHORTCUTS
from .style import apply_style

if TYPE_CHECKING:
    from ..app import Session

DRAG_THRESHOLD = 3.0  # pixels a press moves before it pans instead of clicking
WHEEL_ZOOM = 1.15  # per wheel step


class Interface:
    """The interface of one session: which cards are open, the menu's state, the named
    controls of the last frame, and the battlefield's drag."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.open = set(CARDS)
        self.opened_while_flying: set[str] = set()
        self.controls = Controls()
        self.menu = menus.Menu(session)
        self.pressed_on_battlefield = False
        self.dragging = False
        self.scrubbing = False  # the scrubber's timeline is being dragged
        self.typed_policy = ""  # a policy directory being typed into a controller combo
        self.styled = None  # the theme and scale the style was last set from

    # Cards ------------------------------------------------------------------------------------

    def restyle(self) -> None:
        """Set ImGui's style from the session's theme and scale."""
        imgui.set_current_context(self.session.imgui_context)
        apply_style(self.session.theme, self.session.scale)
        self.styled = (self.session.theme, self.session.scale)

    def toggle(self, card: str) -> None:
        """Open or collapse a card."""
        if card in self.open:
            self.open.discard(card)
        else:
            self.open.add(card)
            if self.session.loop.pilot is not None:
                self.opened_while_flying.add(card)

    def reveal(self, card: str) -> None:
        """Open a card the player asked for (a selection opens its card)."""
        if card not in self.open and self.session.loop.pilot is None:
            self.open.add(card)

    def collapse_for_flying(self) -> None:
        """While the player flies, the cards stay collapsed unless opened."""
        self.open -= {"fleets", "selection"}

    def layout(self) -> Layout:
        """This frame's layout."""
        session = self.session
        width, height = imgui.get_io().display_size
        return Layout(
            width,
            height,
            session.scale,
            frozenset(self.open),
            replay=session.loop.replaying,
            flying=session.loop.pilot is not None,
            selected=session.selection is not None,
        )

    # Menus ------------------------------------------------------------------------------------

    def toggle_menu(self) -> None:
        """Open the menu (pausing a live match), or close it."""
        if self.menu.is_open:
            self.menu.close()
        else:
            self.menu.show("main")

    def close_menu(self) -> None:
        """Close the menu (resuming a match it paused)."""
        self.menu.close()

    # The frame ----------------------------------------------------------------------------

    def draw(self) -> None:
        """The whole interface for this frame (inside an ImGui frame)."""
        session = self.session
        if self.styled != (session.theme, session.scale):
            self.restyle()
        self.controls.begin_frame()
        layout = self.layout()
        top_bar.draw(self, layout)
        fleets.draw(self, layout)
        inspector.draw(self, layout)
        minimap.draw(self, layout)
        kill_feed.draw(self, layout)
        # A control above may have changed the match this frame: the layout decides.
        if layout.replay and session.loop.replaying:
            scrubber.draw(self, layout)
        if layout.flying and session.loop.pilot is not None:
            flying.draw(self, layout)
        self.menu.draw(self)
        self._editor()
        self._keys()
        self._notice(layout)
        self._battlefield()
        self._keyboard()

    def _editor(self) -> None:
        session = self.session
        if session.original.name != session.theme.name:
            session.original = session.file_theme()
        session.theme, saved = session.editor.draw(session.theme, session.original)
        if saved is not None:
            session.theme_names = session.library.names()

    def _keys(self) -> None:
        if not self.session.show_keys:
            return
        width, height = imgui.get_io().display_size
        imgui.set_next_window_pos(
            imgui.ImVec2(0.5 * width, 0.5 * height), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5)
        )
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings
        shown, keep = imgui.begin("Keys", True, flags | imgui.WindowFlags_.no_collapse)
        if shown and imgui.begin_table("keys", 2):
            for key, meaning in KEYS:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(key)
                imgui.table_next_column()
                imgui.text(meaning)
            imgui.end_table()
        imgui.end()
        if not keep:
            self.session.show_keys = False

    def _notice(self, layout: Layout) -> None:
        """A passing notice (an agent that cannot load, a saved replay), inside its
        rectangle under the top bar."""
        session = self.session
        if not session.notice or session.clock > session.notice_until:
            return
        rect = layout.notice
        draw = imgui.get_foreground_draw_list()
        draw.push_clip_rect(imgui.ImVec2(rect.x, rect.y), imgui.ImVec2(rect.right, rect.bottom))
        padding = 8.0 * session.scale
        size = imgui.calc_text_size(session.notice)
        width = min(size.x + 2 * padding, rect.width)
        left = 0.5 * (rect.x + rect.right - width)
        bottom = rect.y + size.y + 2 * padding
        rounding = 5.0 * session.scale
        draw.add_rect_filled(
            imgui.ImVec2(left, rect.y),
            imgui.ImVec2(left + width, bottom),
            imgui.get_color_u32(imgui.Col_.popup_bg),
            rounding,
        )
        draw.add_rect(
            imgui.ImVec2(left, rect.y),
            imgui.ImVec2(left + width, bottom),
            imgui.get_color_u32(imgui.Col_.border),
            rounding,
        )
        draw.add_text(
            imgui.ImVec2(left + padding, rect.y + padding),
            imgui.get_color_u32(imgui.Col_.text),
            session.notice,
        )
        draw.pop_clip_rect()

    # Input the cards did not take -------------------------------------------------------------

    def _over_interface(self) -> bool:
        return imgui.is_window_hovered(imgui.HoveredFlags_.any_window) or imgui.is_any_item_active()

    def _battlefield(self) -> None:
        """Click to select, drag to pan, wheel to zoom, where no card is."""
        session, io = self.session, imgui.get_io()
        camera = session.camera
        if imgui.is_mouse_clicked(0):
            self.pressed_on_battlefield = not self._over_interface()
            self.dragging = False
        if self.pressed_on_battlefield and imgui.is_mouse_down(0):
            drag = imgui.get_mouse_drag_delta(0, 0.0)
            if self.dragging or max(abs(drag.x), abs(drag.y)) > DRAG_THRESHOLD:
                if not self.dragging:
                    camera.pan(drag.x, drag.y)  # the movement before it counted as a drag
                else:
                    camera.pan(io.mouse_delta.x, io.mouse_delta.y)
                self.dragging = True
                session.following = None
        if self.pressed_on_battlefield and imgui.is_mouse_released(0):
            if not self.dragging:
                x, y = camera.world(*io.mouse_pos)
                session.select(
                    hit_test(session.frame, session.view_team, x, y, camera.metres_per_pixel)
                )
            self.pressed_on_battlefield = self.dragging = False
        if io.mouse_wheel != 0.0 and not self._over_interface():
            camera.zoom(WHEEL_ZOOM**io.mouse_wheel, io.mouse_pos.x, io.mouse_pos.y)

    def _keyboard(self) -> None:
        """The flight keys into the pilot's inputs; the shortcuts (unless typing)."""
        session, io = self.session, imgui.get_io()
        inputs = torch.zeros(len(INPUTS), dtype=torch.bool)
        if not io.want_text_input:
            for name, keys in FLIGHT_KEYS.items():
                inputs[INPUT[name]] = any(imgui.is_key_down(key) for key in keys)
            for key, _, method, _ in SHORTCUTS:
                if imgui.is_key_pressed(key, False):
                    getattr(session, method)()
        session.loop.inputs = inputs
