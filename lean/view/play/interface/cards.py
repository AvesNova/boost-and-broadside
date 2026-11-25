"""What the cards share: a window placed exactly on its layout rectangle, the card's
header (its title and the collapse control), and the slim tab a collapsed card leaves on
its edge."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from .controls import collapse_icon, icon_button, right_edge

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Rect

FLAGS = (
    imgui.WindowFlags_.no_decoration
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_saved_settings
    | imgui.WindowFlags_.no_bring_to_front_on_focus
    | imgui.WindowFlags_.no_focus_on_appearing
    | imgui.WindowFlags_.no_nav
)
TITLES = {"fleets": "Fleets", "selection": "Selection", "minimap": "Map"}
LEFT_CARDS = ("fleets",)


def window(name: str, rect: Rect, flags: int = FLAGS, fit_height: bool = False) -> bool:
    """Begin a window at `rect` (the caller ends it). fit_height: as tall as its content,
    up to the rectangle's height."""
    imgui.set_next_window_pos(imgui.ImVec2(rect.x, rect.y), imgui.Cond_.always)
    if fit_height:
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(rect.width, 0.0), imgui.ImVec2(rect.width, rect.height)
        )
        flags |= imgui.WindowFlags_.always_auto_resize
    else:
        imgui.set_next_window_size(imgui.ImVec2(rect.width, rect.height), imgui.Cond_.always)
    shown, _ = imgui.begin(name, None, flags)
    return shown


def header(interface: Interface, card: str, title: str) -> None:
    """A card's first row: its title in bold and, on the right, its collapse control."""
    session = interface.session
    right = right_edge()
    imgui.push_font(session.fonts.bold, 0.0)
    imgui.align_text_to_frame_padding()
    imgui.text(title)
    imgui.pop_font()
    side = imgui.get_frame_height()
    imgui.same_line(right - side)
    left = card in LEFT_CARDS
    if icon_button(
        interface.controls,
        f"{card}/collapse",
        collapse_icon(left),
        f"Collapse the {TITLES[card].lower()} card",
    ):
        interface.toggle(card)
    imgui.separator()


def tab(interface: Interface, card: str, rect: Rect) -> None:
    """A collapsed card's tab: a button on its edge that opens it."""
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
    window(f"###{card} tab", rect)
    imgui.pop_style_var()
    arrow = "› " if card in LEFT_CARDS else "‹ "
    label = arrow + TITLES[card] if card in LEFT_CARDS else TITLES[card] + " ‹"
    if card not in LEFT_CARDS:
        label = "‹ " + TITLES[card]
    if imgui.button(f"{label}###{card} open", imgui.ImVec2(rect.width, rect.height)):
        interface.toggle(card)
    interface.controls.record(f"{card}/open")
    if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
        imgui.set_tooltip(f"Open the {TITLES[card].lower()} card")
    imgui.end()
