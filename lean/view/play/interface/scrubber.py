"""The scrubber (the bottom strip, in replays): a tick back, play or pause, a tick on, the
time, and the timeline: drag or click to seek, with a marker for every kill (in the
killer's colour), capture (in the capturing team's) and the match's end. Clicking a
marker seeks a few seconds before its event."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import cards
from .controls import back_icon, icon_button, pause_icon, play_icon, step_icon, tooltip, u32
from .layout import Rect
from .shortcuts import shortcut
from .text import clock

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

MARKER_REACH = 5.0  # pixels either side of a marker that click it


def marker_x(track: Rect, tick: int, last: int) -> float:
    """The x of a tick on the timeline."""
    return track.x + track.width * min(max(tick / max(last, 1), 0.0), 1.0)


def draw(interface: Interface, layout: Layout) -> None:
    """The scrubber, in the current ImGui frame."""
    session, controls = interface.session, interface.controls
    loop, theme, scale = session.loop, session.theme, session.scale
    rect = layout.scrubber
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
    cards.window("###scrubber", rect)
    imgui.pop_style_var()
    row = imgui.get_frame_height()
    imgui.set_cursor_pos(imgui.ImVec2(10.0 * scale, 0.5 * (rect.height - row)))
    paused = loop.paused
    if icon_button(
        controls,
        "replay/back",
        back_icon,
        "One tick back (while paused)",
        shortcut("step_back"),
        enabled=paused,
    ):
        session.step_back()
    imgui.same_line()
    if paused:
        if icon_button(controls, "replay/play", play_icon, "Play", shortcut("toggle_pause")):
            session.toggle_pause()
    elif icon_button(controls, "replay/pause", pause_icon, "Pause", shortcut("toggle_pause")):
        session.toggle_pause()
    imgui.same_line()
    if icon_button(
        controls,
        "replay/step",
        step_icon,
        "One tick on (while paused)",
        shortcut("single_step"),
        enabled=paused,
    ):
        session.single_step()
    imgui.same_line(0.0, 14.0 * scale)
    replay = loop.replay
    last = replay.last
    imgui.align_text_to_frame_padding()
    imgui.text(f"{clock(loop.position * loop.tick_length)} / {clock(last * loop.tick_length)}")
    imgui.same_line(0.0, 16.0 * scale)

    # The timeline.
    start = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x - 10.0 * scale
    height = row
    track = Rect(start.x + 6.0 * scale, start.y, width - 12.0 * scale, height)
    imgui.invisible_button("timeline", imgui.ImVec2(width, height))
    controls.record("replay/timeline")
    draw_list = imgui.get_window_draw_list()
    middle = start.y + 0.5 * height
    line = 4.0 * scale
    draw_list.add_rect_filled(
        imgui.ImVec2(track.x, middle - 0.5 * line),
        imgui.ImVec2(track.right, middle + 0.5 * line),
        u32(theme.panel_line),
        line,
    )
    playhead = marker_x(track, loop.position, last)
    draw_list.add_rect_filled(
        imgui.ImVec2(track.x, middle - 0.5 * line),
        imgui.ImVec2(playhead, middle + 0.5 * line),
        u32(theme.text, 0.55),
        line,
    )
    ships = session.frame.ships
    markers = []
    for index, event in enumerate(loop.markers()):
        x = marker_x(track, event.tick, last)
        if event.kind == "kill":
            team = int(ships.team[event.other]) if event.other >= 0 else -1
            colour = theme.text if team < 0 else theme.team[team]
            draw_list.add_rect_filled(
                imgui.ImVec2(x - 1.2 * scale, middle - 0.42 * height),
                imgui.ImVec2(x + 1.2 * scale, middle + 0.42 * height),
                u32(colour),
                1.0,
            )
        elif event.kind == "capture":
            colour = theme.team[event.team]
            size = 5.0 * scale
            draw_list.add_quad_filled(
                imgui.ImVec2(x, middle - size),
                imgui.ImVec2(x + size, middle),
                imgui.ImVec2(x, middle + size),
                imgui.ImVec2(x - size, middle),
                u32(colour),
            )
        else:
            colour = theme.text if event.team < 0 else theme.team[event.team]
            draw_list.add_rect_filled(
                imgui.ImVec2(x - 1.0 * scale, middle - 0.45 * height),
                imgui.ImVec2(x + 1.0 * scale, middle + 0.45 * height),
                u32(colour),
            )
            draw_list.add_triangle_filled(
                imgui.ImVec2(x + 1.0 * scale, middle - 0.45 * height),
                imgui.ImVec2(x + 9.0 * scale, middle - 0.3 * height),
                imgui.ImVec2(x + 1.0 * scale, middle - 0.15 * height),
                u32(colour),
            )
        reach = MARKER_REACH * scale
        controls.current[f"replay/marker/{index}"] = Rect(x - reach, start.y, 2 * reach, height)
        markers.append((x, event))
    draw_list.add_circle_filled(imgui.ImVec2(playhead, middle), 6.0 * scale, u32(theme.text))
    draw_list.add_circle(imgui.ImVec2(playhead, middle), 6.0 * scale, u32(theme.sky), 0, 1.5)

    mouse = imgui.get_io().mouse_pos
    near = [
        (abs(x - mouse.x), event)
        for x, event in markers
        if abs(x - mouse.x) <= MARKER_REACH * scale
    ]
    hovered = imgui.is_item_hovered()
    if hovered and near and not imgui.is_item_active():
        event = min(near, key=lambda pair: pair[0])[1]
        tooltip(_describe(session, event) + " (click: watch it)")
    elif hovered:
        tooltip("Drag to seek")
    if imgui.is_item_clicked(0) and near:
        session.seek_marker(min(near, key=lambda pair: pair[0])[1])
        interface.scrubbing = False
    elif imgui.is_item_active() and (not near or interface.scrubbing):
        interface.scrubbing = True
        fraction = (mouse.x - track.x) / max(track.width, 1.0)
        tick = round(min(max(fraction, 0.0), 1.0) * last)
        if tick != loop.position:
            session.seek(tick)
    if not imgui.is_item_active():
        interface.scrubbing = False
    imgui.end()


def _describe(session, event) -> str:
    from .text import TEAM_NAMES, ship_name

    ships = session.frame.ships
    if event.kind == "kill":
        victim = ship_name(ships.team, ships.ship_type, event.ship)
        if event.other < 0:
            return f"the boundary killed {TEAM_NAMES[int(ships.team[event.ship])]} {victim}"
        killer = ship_name(ships.team, ships.ship_type, event.other)
        return f"{TEAM_NAMES[int(ships.team[event.other])]} {killer} killed {victim}"
    if event.kind == "capture":
        return f"{TEAM_NAMES[event.team]} captured a zone"
    return "the match ended: " + ("a draw" if event.team < 0 else f"{TEAM_NAMES[event.team]} won")
