"""The flying HUD (bottom centre, while the player flies a ship): minimal and in the
game's style, with no card behind it. The ship's name, its shield and energy as bars in
the ring's colours, the power (reverse, coast, boost) and the stick (three levels each
way) the keys are giving, the respawn wait while it is dead, and "Release"."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from bnb.controllers.human import human_action

from . import cards
from .controls import bar, button, u32
from .shortcuts import shortcut
from .text import ship_name

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

POWER_LABELS = ("reverse", "coast", "boost")
POWER_ORDER = (2, 0, 1)  # the action's power index of each label


def draw(interface: Interface, layout: Layout) -> None:
    """The flying HUD, in the current ImGui frame."""
    session, controls = interface.session, interface.controls
    loop, frame, theme, scale = session.loop, session.frame, session.theme, session.scale
    ship = loop.pilot
    rect = layout.flying_hud
    flags = cards.FLAGS | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_scrollbar
    cards.window("###flying", rect, flags)
    draw_list = imgui.get_window_draw_list()
    ships = frame.ships
    team = int(ships.team[ship])
    team_colour = u32(theme.team[team])
    text, muted, track = u32(theme.text), u32(theme.text, 0.45), u32(theme.panel_line)
    origin = imgui.get_cursor_screen_pos()
    line = imgui.get_text_line_height()
    name = "Flying " + ship_name(ships.team, ships.ship_type, ship)
    draw_list.add_text(origin, team_colour, name)
    y = origin.y + line + 6.0 * scale
    left = origin.x
    bar_width, bar_height = 150.0 * scale, 7.0 * scale
    if ships.alive[ship]:
        for label, value, colour in (
            ("shield", ships.shield[ship], theme.shield[team]),
            ("energy", ships.energy[ship], theme.energy),
        ):
            draw_list.add_text(imgui.ImVec2(left, y), muted, label)
            bar(
                draw_list,
                left + 52.0 * scale,
                y + 0.5 * (line - bar_height),
                bar_width,
                bar_height,
                float(value),
                u32(colour),
                track,
            )
            y += line + 4.0 * scale
    else:
        draw_list.add_text(
            imgui.ImVec2(left, y), text, f"respawn in {ships.respawn_wait[ship]:.1f} s"
        )

    # The power and the stick the keys give now.
    action = human_action(loop.inputs)
    power, stick = int(action.power), int(action.stick)
    x = left + 52.0 * scale + bar_width + 26.0 * scale
    y = origin.y + line + 6.0 * scale
    for label, index in zip(POWER_LABELS, POWER_ORDER, strict=True):
        on = power == index
        draw_list.add_text(imgui.ImVec2(x, y), text if on else muted, label)
        if on:
            width = imgui.calc_text_size(label).x
            draw_list.add_line(
                imgui.ImVec2(x, y + line + 1.0),
                imgui.ImVec2(x + width, y + line + 1.0),
                team_colour,
                2.0 * scale,
            )
        x += imgui.calc_text_size(label).x + 10.0 * scale
    x = left + 52.0 * scale + bar_width + 26.0 * scale
    y += line + 6.0 * scale
    level = (stick + 1) // 2
    signed = 0 if stick == 0 else (level if stick % 2 else -level)  # right positive
    cell, gap = 11.0 * scale, 3.0 * scale
    for position in range(-3, 4):
        on = position != 0 and (0 < position <= signed or signed <= position < 0)
        corner = imgui.ImVec2(x + (position + 3) * (cell + gap), y + 0.5 * (line - cell))
        colour = team_colour if on else (text if position == 0 else track)
        size = cell if position else 0.45 * cell
        offset = 0.5 * (cell - size)
        draw_list.add_rect_filled(
            imgui.ImVec2(corner.x + offset, corner.y + offset),
            imgui.ImVec2(corner.x + offset + size, corner.y + offset + size),
            colour,
            2.0 * scale,
        )
    draw_list.add_text(imgui.ImVec2(x + 7 * (cell + gap) + 6.0 * scale, y), muted, "stick")

    release = imgui.calc_text_size("Release").x + 2 * imgui.get_style().frame_padding.x
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(
            rect.right - release - 4.0 * scale,
            rect.y + 0.5 * (rect.height - imgui.get_frame_height()),
        )
    )
    if button(
        controls,
        "flying/release",
        "Release",
        "Hand the ship back to its side",
        shortcut("toggle_control"),
    ):
        session.release()
    imgui.end()
