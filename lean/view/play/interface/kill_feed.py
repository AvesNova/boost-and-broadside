"""The kill feed (upper right, under the top bar): the last few kills, newest on top,
killer and victim in their team colours, each fading out FEED_SECONDS after it happened
(game time, so a paused match keeps its feed)."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from .controls import u32
from .text import ship_name

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

LINES = 5
FADE_SECONDS = 1.0  # the last second of a line's life fades it out
FEED_SECONDS = 6.0


def lines(session) -> list[tuple[float, list[tuple[str, int]]]]:
    """The feed's lines, newest first: (opacity, [(text, team or -1)])."""
    frame, loop = session.frame, session.loop
    ships = frame.ships
    now = loop.ticks * loop.tick_length
    found = []
    for when, event in reversed(session.feed[-LINES:]):
        age = now - when
        opacity = min(max((FEED_SECONDS - age) / FADE_SECONDS, 0.0), 1.0)
        victim = (ship_name(ships.team, ships.ship_type, event.ship), int(ships.team[event.ship]))
        if event.other >= 0:
            killer = (
                ship_name(ships.team, ships.ship_type, event.other),
                int(ships.team[event.other]),
            )
            parts = [killer, ("  →  ", -1), victim]
        else:
            parts = [("the boundary  →  ", -1), victim]
        found.append((opacity, parts))
    return found


def draw(interface: Interface, layout: Layout) -> None:
    """The feed, right-aligned in its rectangle, on the background draw list."""
    session = interface.session
    theme, scale = session.theme, session.scale
    rect = layout.kill_feed
    draw_list = imgui.get_background_draw_list()
    line_height = imgui.get_text_line_height() + 8.0 * scale
    y = rect.y
    for opacity, parts in lines(session):
        width = sum(imgui.calc_text_size(text).x for text, _ in parts)
        padding = 7.0 * scale
        x = rect.right - width - padding
        draw_list.add_rect_filled(
            imgui.ImVec2(x - padding, y),
            imgui.ImVec2(rect.right, y + line_height - 2.0 * scale),
            u32(theme.panel, opacity),
            4.0 * scale,
        )
        for text, team in parts:
            colour = theme.text if team < 0 else theme.team[team]
            draw_list.add_text(imgui.ImVec2(x, y + 3.0 * scale), u32(colour, opacity), text)
            x += imgui.calc_text_size(text).x
        y += line_height
        if y + line_height > rect.bottom:
            break
