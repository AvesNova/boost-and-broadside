"""The top bar, always shown, one row: the menu button, the view selector, the overlays
button (its submenu, `overlays`), the true-size switch, the dot grid's pattern and the
distance contours' shapes on the left;
the score or the mode's objective and the clock in the middle; the speed controls on the
right (pause or play, a tick while paused, x1 to x8).

Team deathmatch shows the kill score in team colours and the winning score; Frontline
shows the front as a tug of war (a cell per capture up to the threshold, filled toward
the side being pushed back) and each zone as a dot in its owner's colour with its meter.
A zone's dot selects the zone.
"""

import math
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..loop import TIME_SCALE_LABELS, TIME_SCALES
from ..selection import Selection
from . import overlays
from .cards import window
from .controls import button, icon_button, menu_icon, pause_icon, play_icon, step_icon, tooltip, u32
from .inspector import ROLE_NAMES
from .shortcuts import shortcut
from .text import TEAM_NAMES, clock

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

LUMA = (0.2126, 0.7152, 0.0722)
VIEW_BUTTONS = ("Spectator", "Team 0", "Team 1")
GRID_LABELS = {"off": "Off", "square": "Square", "triangle": "Triangle", "hex": "Hex"}
CONTOUR_LABELS = {
    "off": "Off",
    "fields": "Fields",
    "edge": "Edge",
    "fields+edge": "Fields + edge",
    "fields+edge+zones": "All",
    "zones": "Zones",
}
CONTOUR_HELP = (
    "Faint lines at equal distances from the map's shapes, inside and outside them: the "
    "fields, the map edge, both, all three with the zones, or the zones"
)


def draw(interface: Interface, layout: Layout) -> None:
    """The top bar, in the current ImGui frame."""
    session, controls = interface.session, interface.controls
    loop, theme, scale = session.loop, session.theme, session.scale
    rect = layout.top_bar
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
    window("###top bar", rect)
    imgui.pop_style_var(2)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_line(
        imgui.ImVec2(rect.x, rect.bottom - 1),
        imgui.ImVec2(rect.right, rect.bottom - 1),
        u32(theme.panel_line),
    )
    row = imgui.get_frame_height()
    top = 0.5 * (rect.height - row)

    # Left: the menu and the view selector.
    imgui.set_cursor_pos(imgui.ImVec2(8.0 * scale, top))
    if icon_button(controls, "menu", menu_icon, "Menu", shortcut("toggle_menu")):
        session.toggle_menu()
    imgui.same_line(0.0, 14.0 * scale)
    for view, label in enumerate(VIEW_BUTTONS):
        if view:
            imgui.same_line(0.0, 2.0 * scale)
        chosen = session.view == view
        if view:
            colour = theme.team[view - 1]
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour.floats()))
        if button(
            controls,
            f"view/{VIEW_BUTTONS[view].lower().replace(' ', '')}",
            label,
            "Show the battlefield as "
            + ("the spectator sees it" if view == 0 else f"{label} sees it"),
            shortcut("cycle_view"),
            active=chosen,
        ):
            session.set_view(view)
        if view:
            imgui.pop_style_color()
    imgui.same_line(0.0, 14.0 * scale)
    overlays.draw_button(interface)
    imgui.same_line(0.0, 2.0 * scale)
    if button(
        controls,
        "true size",
        "True size",
        "Draw the ships at their size in metres, the ring at the hitbox and the wingspan "
        "the real one (off: never smaller than a minimum on screen)",
        shortcut("toggle_true_size"),
        active=session.true_size,
    ):
        session.toggle_true_size()
    imgui.same_line(0.0, 2.0 * scale)
    if button(
        controls,
        "grid",
        f"Grid: {GRID_LABELS[session.grid]}###grid",  # one ID whatever the pattern
        "Faint dots fixed to the map, so movement shows where nothing else is near: "
        "off, square, triangle or hex (click for the next)",
        shortcut("cycle_grid"),
        active=session.grid != "off",
    ):
        session.cycle_grid()
    imgui.same_line(0.0, 2.0 * scale)
    if button(
        controls,
        "contours",
        f"Contours: {CONTOUR_LABELS[session.contours]}###contours",
        CONTOUR_HELP + " (click for the next)",
        shortcut("cycle_contours"),
        active=session.contours != "off",
    ):
        session.cycle_contours()
    imgui.same_line(0.0, 14.0 * scale)
    if button(
        controls,
        "map panel",
        "Map / mode",
        "The map and mode panel: game mode, arena, field randomisation, zones, clocks, "
        "tickets, spawns, New map, Restart and Save map",
        shortcut("toggle_map_panel"),
        active=session.map_panel.open,
    ):
        session.toggle_map_panel()
    imgui.same_line(0.0, 2.0 * scale)
    if button(
        controls,
        "ship panel",
        "Ships",
        "Ship editing: the ship types' handles, bullets and flight constants",
        shortcut("toggle_tuning"),
        active=session.tuning_panel.open,
    ):
        session.toggle_tuning()
    if not loop.replaying:
        imgui.same_line(0.0, 14.0 * scale)
        if button(
            controls,
            "new map",
            "New map",
            "A new match on a freshly drawn map (fields, spawn axis, zone ring), as it is tuned",
        ):
            session.new_map()
        imgui.same_line(0.0, 2.0 * scale)
        if button(controls, "restart", "Restart", "The same map and setup, a fresh match"):
            session.restart()
    left_end = imgui.get_item_rect_max().x

    # Right: the speed controls.
    spacing = imgui.get_style().item_spacing.x
    labels = [TIME_SCALE_LABELS.get(value, f"×{value:g}") for value in TIME_SCALES]
    widths = [
        imgui.calc_text_size(label).x + 2 * imgui.get_style().frame_padding.x for label in labels
    ]
    right_width = 2 * row + spacing + 14.0 * scale + sum(widths) + 2.0 * scale * (len(widths) - 1)
    if loop.replaying:
        right_width += imgui.calc_text_size("REPLAY").x + 14.0 * scale
    imgui.set_cursor_pos(imgui.ImVec2(rect.width - right_width - 10.0 * scale, top))
    right_start = imgui.get_cursor_screen_pos().x
    if loop.replaying:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("REPLAY")
        imgui.same_line(0.0, 14.0 * scale)
    if loop.paused:
        if icon_button(controls, "play", play_icon, "Play", shortcut("toggle_pause")):
            session.toggle_pause()
    elif icon_button(controls, "pause", pause_icon, "Pause", shortcut("toggle_pause")):
        session.toggle_pause()
    imgui.same_line()
    if icon_button(
        controls,
        "step",
        step_icon,
        "One tick (while paused)",
        shortcut("single_step"),
        enabled=loop.paused,
    ):
        session.single_step()
    imgui.same_line(0.0, 14.0 * scale)
    for index, (value, label) in enumerate(zip(TIME_SCALES, labels, strict=True)):
        if index:
            imgui.same_line(0.0, 2.0 * scale)
        if button(
            controls,
            f"speed/{value:g}",
            label,
            f"Play at {value:g} times real time",
            "+ / -",
            active=loop.time_scale == value,
        ):
            session.set_time_scale(value)

    _objective(interface, rect, left_end, right_start)
    imgui.end()


def _objective(interface: Interface, rect, left: float, right: float) -> None:
    """The score or the front and zones, and the clock, centred between the two groups."""
    session = interface.session
    frame, theme, scale = session.frame, session.theme, session.scale
    draw = imgui.get_window_draw_list()
    fonts = session.fonts
    text = u32(theme.text)
    muted = imgui.get_color_u32(imgui.Col_.text_disabled)
    team = [u32(colour) for colour in theme.team]
    middle_y = rect.y + 0.5 * rect.height
    big = imgui.get_font_size() * 1.35
    small = imgui.get_font_size()

    def width_of(line: str, size: float, font=None) -> float:
        imgui.push_font(font, size)
        width = imgui.calc_text_size(line).x
        imgui.pop_font()
        return width

    parts = []  # (kind, value): ("text", (string, colour, size, font)) or ("gap", pixels)
    if frame.mode == "team_deathmatch":
        parts += [
            ("text", (f"{int(frame.score[0])}", team[0], big, fonts.bold)),
            ("text", ("  :  ", text, big, fonts.bold)),
            ("text", (f"{int(frame.score[1])}", team[1], big, fonts.bold)),
            ("gap", 10.0 * scale),
            ("text", (f"first to {frame.score_target}", muted, small, None)),
        ]
    elif frame.mode == "elimination":  # the ships each team has left
        alive = [int(frame.ships.alive[frame.ships.team == side].sum()) for side in range(2)]
        parts += [
            ("text", (f"{alive[0]}", team[0], big, fonts.bold)),
            ("text", ("  :  ", text, big, fonts.bold)),
            ("text", (f"{alive[1]}", team[1], big, fonts.bold)),
            ("gap", 10.0 * scale),
            ("text", ("ships left", muted, small, None)),
        ]
    elif frame.mode == "frontline":
        parts += [
            ("text", ("Front", muted, small, None)),
            ("gap", 6.0 * scale),
            ("front", None),
            ("gap", 16.0 * scale),
            ("text", ("Zones", muted, small, None)),
            ("gap", 6.0 * scale),
            ("zones", None),
        ]
    else:
        parts += [("text", ("Sandbox", text, small, None))]
    parts += [
        ("gap", 22.0 * scale),
        ("text", (clock(frame.clock), text, big, fonts.bold)),
        ("text", (f"  / {clock(frame.episode_length)}", muted, small, None)),
    ]

    cell, cell_gap = 12.0 * scale, 3.0 * scale
    threshold = max(frame.front_threshold, 1)
    zone_radius, zone_gap = 10.0 * scale, 6.0 * scale
    zones = [index for index in range(len(frame.zones.x)) if frame.zones.present[index]]
    lock_gap = 3.0 * scale
    lock_labels = {zone: _lock_label(frame.zones, zone) for zone in zones}

    def zone_width(zone: int) -> float:
        """A zone's slot: its dot, and the capture lock's countdown beside it when locked."""
        label = lock_labels[zone]
        return 2 * zone_radius + (lock_gap + width_of(label[0], small) if label else 0.0)

    def size_of(kind, value) -> float:
        if kind == "text":
            return width_of(value[0], value[2], value[3])
        if kind == "gap":
            return value
        if kind == "front":
            return (2 * threshold) * (cell + cell_gap) + 4.0 * scale
        return sum(zone_width(zone) for zone in zones) + zone_gap * max(len(zones) - 1, 0)

    total = sum(size_of(kind, value) for kind, value in parts)
    x = max(0.5 * (rect.x + rect.right - total), left + 16.0 * scale)
    if x + total > right - 16.0 * scale:
        x = 0.5 * (left + right - total)
    for kind, value in parts:
        if kind == "text":
            string, colour, size, font = value
            imgui.push_font(font, size)
            height = imgui.get_text_line_height()
            draw.add_text(imgui.ImVec2(x, middle_y - 0.5 * height), colour, string)
            imgui.pop_font()
        elif kind == "front":
            _front(draw, x, middle_y, frame, threshold, cell, cell_gap, team, theme, scale)
            imgui.set_cursor_screen_pos(imgui.ImVec2(x, middle_y - 0.5 * cell))
            imgui.invisible_button("front", imgui.ImVec2(size_of(kind, value), cell))
            tooltip(
                f"The front, a tug of war over captures. Cells fill from the middle in the "
                f"colour of the side that is ahead (team 0 pushes right, team 1 left); a side "
                f"wins {frame.front_threshold} captures ahead. Now {frame.front:+d} "
                f"(positive: team 0)."
            )
        elif kind == "zones":
            start = x
            for slot, zone in enumerate(zones):
                centre_x = start + zone_radius
                _zone_dot(
                    draw, centre_x, middle_y, zone_radius, frame.zones, zone, slot + 1, theme, scale
                )
                label = lock_labels[zone]
                if label:  # the lock's countdown, in the held team's colour
                    imgui.push_font(None, small)
                    height = imgui.get_text_line_height()
                    draw.add_text(
                        imgui.ImVec2(centre_x + zone_radius + lock_gap, middle_y - 0.5 * height),
                        team[label[1]],
                        label[0],
                    )
                    imgui.pop_font()
                start += zone_width(zone) + zone_gap
                imgui.set_cursor_screen_pos(
                    imgui.ImVec2(centre_x - zone_radius, middle_y - zone_radius)
                )
                if imgui.invisible_button(
                    f"zone {zone}", imgui.ImVec2(2 * zone_radius, 2 * zone_radius)
                ):
                    session.select(Selection("zone", zone))
                interface.controls.record(f"top/zone/{zone}")
                tooltip(_zone_tooltip(frame.zones, zone, slot + 1))
        x += size_of(kind, value)


def _front(draw, x, middle_y, frame, threshold, cell, gap, team, theme, scale) -> None:
    """The tug of war: `threshold` cells a side; captures ahead fill toward the side being
    pushed back, in the leader's colour."""
    track = u32(theme.panel_line)
    top = middle_y - 0.5 * cell
    middle = x + threshold * (cell + gap)
    draw.add_line(
        imgui.ImVec2(middle, top - 2.0 * scale),
        imgui.ImVec2(middle, top + cell + 2.0 * scale),
        u32(theme.text),
        1.5 * scale,
    )
    for index in range(2 * threshold):
        side = 1 if index >= threshold else -1  # cells right of the middle: team 0 ahead
        reach = index - threshold + 1 if side > 0 else threshold - index
        filled = (frame.front * side) >= reach
        left = x + index * (cell + gap) + (4.0 * scale if side > 0 else 0.0)
        colour = (team[0] if side > 0 else team[1]) if filled else track
        draw.add_rect_filled(
            imgui.ImVec2(left, top), imgui.ImVec2(left + cell, top + cell), colour, 2.0 * scale
        )


def _lock_label(zones, zone: int) -> tuple[str, int] | None:
    """The capture lock's countdown beside a zone's dot ("45s") and the held team, or None
    when no team is held there."""
    held = [team for team in range(2) if zones.lock[team, zone] > 0.0]
    if not held:
        return None
    team = max(held, key=lambda side: zones.lock[side, zone])
    return f"{math.ceil(float(zones.lock[team, zone]))}s", team


def _zone_tooltip(zones, zone: int, number: int) -> str:
    """What a zone's dot shows: its role, owner, capture progress and capture lock."""
    role, owner, attacker = int(zones.role[zone]), int(zones.owner[zone]), int(zones.attacker[zone])
    text = f"Zone {number}: {ROLE_NAMES[role] if 0 <= role < len(ROLE_NAMES) else role}, "
    text += f"owned by {TEAM_NAMES[owner]}" if owner >= 0 else "owned by nobody"
    text += ". The dot is the owner's colour (grey: nobody; pale: a spawn zone)"
    if attacker >= 0 and zones.meter[zone] > 0.0:
        text += (
            f"; the arc is {TEAM_NAMES[attacker]}'s capture, {100 * float(zones.meter[zone]):.0f}%"
        )
    for team in range(2):
        if zones.lock[team, zone] > 0.0:
            text += (
                f". Locked for {TEAM_NAMES[team]}, who lost it: no capture progress for "
                f"{float(zones.lock[team, zone]):.1f} s more (the number beside the dot)"
            )
    return text + ". Click to inspect."


def _zone_dot(draw, x, y, radius, zones, zone, number, theme, scale) -> None:
    """A zone: a dot in its owner's colour (grey for the neutral zone), lighter for a
    spawn zone, labelled with its number, and a defense zone's meter as an arc in the
    attacker's colour."""
    owner, attacker = int(zones.owner[zone]), int(zones.attacker[zone])
    colour = theme.team[owner] if owner >= 0 else theme.neutral
    alpha = 1.0 if attacker >= 0 or owner < 0 else 0.35
    draw.add_circle_filled(imgui.ImVec2(x, y), radius, u32(colour, alpha))
    draw.add_circle(imgui.ImVec2(x, y), radius, u32(colour), 0, 1.2 * scale)
    if attacker >= 0 and zones.meter[zone] > 0.0:
        start = -0.5 * math.pi
        end = start + 2.0 * math.pi * min(float(zones.meter[zone]), 1.0)
        draw.path_arc_to(imgui.ImVec2(x, y), radius + 3.0 * scale, start, end, 0)
        draw.path_stroke(u32(theme.team[attacker]), 2.5 * scale)
    label = str(number)
    shown = 1.0 if attacker >= 0 or owner < 0 else 0.35  # the fill's opacity over the panel
    luminance = sum(
        weight * (shown * value + (1.0 - shown)) for weight, value in zip(LUMA, colour.floats())
    )
    imgui.push_font(None, imgui.get_font_size())
    size = imgui.calc_text_size(label)
    draw.add_text(
        imgui.ImVec2(x - 0.5 * size.x, y - 0.5 * size.y),
        u32(theme.text) if luminance > 0.5 else imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 1)),
        label,
    )
    imgui.pop_font()
