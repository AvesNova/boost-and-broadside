"""Controls by name, and the small widgets the cards share.

Every control records its rectangle under a stable name each frame (`Controls.record`),
so a test, or anything else driving the interface with synthetic input, can click it by
name. Tooltips name the control's keyboard shortcut, when it has one.
"""

import dataclasses
from collections.abc import Callable

from imgui_bundle import imgui

from ..themes import Colour
from .layout import Rect


@dataclasses.dataclass
class Controls:
    """The rectangles of this frame's controls by name (`last` keeps the previous
    frame's, which is what a click on the screen meets)."""

    current: dict[str, Rect] = dataclasses.field(default_factory=dict)
    last: dict[str, Rect] = dataclasses.field(default_factory=dict)

    def begin_frame(self) -> None:
        """Start a frame: this frame's controls replace the last's as they are drawn."""
        self.last, self.current = self.current, {}

    def record(self, name: str) -> None:
        """Record the last ImGui item's rectangle under `name`."""
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.current[name] = Rect(low.x, low.y, high.x - low.x, high.y - low.y)

    def rect(self, name: str) -> Rect:
        """A control's rectangle as last drawn (this frame's if it is drawn already)."""
        if name in self.current:
            return self.current[name]
        if name in self.last:
            return self.last[name]
        raise KeyError(f"no control {name!r} is on screen; shown: {sorted(self.last)}")

    def shown(self, name: str) -> bool:
        """Whether a control of this name was drawn in the last frame or this one."""
        return name in self.current or name in self.last


def right_edge() -> float:
    """The x (window coordinates) of the right edge of the current line's content."""
    return imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x


def tooltip(text: str, shortcut: str | None = None) -> None:
    """A tooltip for the last item, naming its shortcut key."""
    if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
        imgui.set_tooltip(text if shortcut is None else f"{text}  ({shortcut})")


def u32(colour: Colour, alpha: float = 1.0) -> int:
    """A theme colour as ImGui's packed colour, its alpha times `alpha`."""
    red, green, blue, own = colour.floats()
    return imgui.get_color_u32(imgui.ImVec4(red, green, blue, own * alpha))


def button(
    controls: Controls,
    name: str,
    label: str,
    help: str | None = None,
    shortcut: str | None = None,
    size: tuple[float, float] | None = None,
    active: bool = False,
) -> bool:
    """A text button, recorded as `name`; `active` draws it chosen (a toggle that is on)."""
    if active:
        colour = imgui.get_style().color_(imgui.Col_.button_active)
        imgui.push_style_color(imgui.Col_.button, colour)
    pressed = imgui.button(label, None if size is None else imgui.ImVec2(*size))
    if active:
        imgui.pop_style_color()
    controls.record(name)
    if help is not None:
        tooltip(help, shortcut)
    return pressed


def icon_button(
    controls: Controls,
    name: str,
    draw: Callable[[imgui.ImDrawList, float, float, float, int], None],
    help: str,
    shortcut: str | None = None,
    active: bool = False,
    enabled: bool = True,
) -> bool:
    """A square button with an icon drawn by `draw(draw_list, x, y, size, colour)` (the
    icon's square's top left corner and side), recorded as `name`."""
    side = imgui.get_frame_height()
    imgui.begin_disabled(not enabled)
    if active:
        imgui.push_style_color(
            imgui.Col_.button, imgui.get_style().color_(imgui.Col_.button_active)
        )
    pressed = imgui.button(f"##{name}", imgui.ImVec2(side, side))
    if active:
        imgui.pop_style_color()
    controls.record(name)
    low = imgui.get_item_rect_min()
    colour = imgui.get_color_u32(imgui.Col_.text_disabled if not enabled else imgui.Col_.text)
    inset = 0.27 * side
    draw(imgui.get_window_draw_list(), low.x + inset, low.y + inset, side - 2 * inset, colour)
    imgui.end_disabled()
    tooltip(help, shortcut)
    return pressed and enabled


# Icons: each draws into the square (x, y, side) -------------------------------------------


def play_icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
    """A triangle pointing right."""
    draw.add_triangle_filled(
        imgui.ImVec2(x + 0.1 * side, y),
        imgui.ImVec2(x + side, y + 0.5 * side),
        imgui.ImVec2(x + 0.1 * side, y + side),
        colour,
    )


def pause_icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
    """Two bars."""
    bar = 0.32 * side
    draw.add_rect_filled(
        imgui.ImVec2(x + 0.05 * side, y), imgui.ImVec2(x + 0.05 * side + bar, y + side), colour, 1.0
    )
    draw.add_rect_filled(
        imgui.ImVec2(x + 0.95 * side - bar, y), imgui.ImVec2(x + 0.95 * side, y + side), colour, 1.0
    )


def step_icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
    """A triangle and a bar: one tick on."""
    draw.add_triangle_filled(
        imgui.ImVec2(x, y),
        imgui.ImVec2(x + 0.68 * side, y + 0.5 * side),
        imgui.ImVec2(x, y + side),
        colour,
    )
    draw.add_rect_filled(
        imgui.ImVec2(x + 0.76 * side, y), imgui.ImVec2(x + side, y + side), colour, 1.0
    )


def back_icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
    """A bar and a triangle: one tick back."""
    draw.add_rect_filled(imgui.ImVec2(x, y), imgui.ImVec2(x + 0.24 * side, y + side), colour, 1.0)
    draw.add_triangle_filled(
        imgui.ImVec2(x + side, y),
        imgui.ImVec2(x + 0.32 * side, y + 0.5 * side),
        imgui.ImVec2(x + side, y + side),
        colour,
    )


def menu_icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
    """Three lines."""
    for row in range(3):
        top = y + (0.08 + 0.38 * row) * side
        draw.add_rect_filled(
            imgui.ImVec2(x, top), imgui.ImVec2(x + side, top + 0.14 * side), colour, 1.0
        )


def collapse_icon(left: bool) -> Callable:
    """A chevron pointing toward the card's edge (left or right)."""

    def icon(draw: imgui.ImDrawList, x: float, y: float, side: float, colour: int) -> None:
        tip, base = (x + 0.25 * side, x + 0.7 * side) if left else (x + 0.75 * side, x + 0.3 * side)
        points = [
            imgui.ImVec2(base, y + 0.05 * side),
            imgui.ImVec2(tip, y + 0.5 * side),
            imgui.ImVec2(base, y + 0.95 * side),
        ]
        draw.add_polyline(points, colour, max(1.5, 0.16 * side), 0)

    return icon


def bar(
    draw: imgui.ImDrawList,
    x: float,
    y: float,
    width: float,
    height: float,
    fraction: float,
    colour: int,
    track: int,
) -> None:
    """A flat meter: a track and its filled part, rounded."""
    rounding = 0.5 * height
    draw.add_rect_filled(imgui.ImVec2(x, y), imgui.ImVec2(x + width, y + height), track, rounding)
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction > 0.0:
        filled = max(width * fraction, height)
        draw.add_rect_filled(
            imgui.ImVec2(x, y), imgui.ImVec2(x + filled, y + height), colour, rounding
        )
