"""The interface over the battlefield, in Dear ImGui (v2's play shell's look): the top bar, the
kill feed, the minimap card, the collision labels, the result card and, in the live window, the
step scrubber. The video draws it as it is; the live window also clicks it (`Request`).

Flat fills, thin outlines and small rounding, every colour from the theme. Fonts: Roboto
(regular and bold), which imgui-bundle ships. The interface scale multiplies the font and every
size. Reads snapshots (numpy) and never imports torch.
"""

from __future__ import annotations

import dataclasses
import pathlib

import imgui_bundle
import numpy as np
from imgui_bundle import imgui

from .marks import Mark, fade
from .play.camera import Camera
from .snapshot import SPECTATOR, Snapshot
from .play.themes import Colour, Theme

FONTS = pathlib.Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "Roboto"
FONT_SIZE = 15.0  # pixels at interface scale 1
TOP_BAR = 44.0  # pixels at interface scale 1
FEED_LINES = 5
FEED_SECONDS = 6.0
FEED_FADE_SECONDS = 1.0
MINIMAP_SIDE = 210.0
SCRUBBER = 52.0
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
VIEW_LABELS = {SPECTATOR: "Spectator", 0: "Team 0", 1: "Team 1"}
FLAGS = (
    imgui.WindowFlags_.no_decoration
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_saved_settings
    | imgui.WindowFlags_.no_bring_to_front_on_focus
    | imgui.WindowFlags_.no_focus_on_appearing
    | imgui.WindowFlags_.no_nav
    | imgui.WindowFlags_.no_scrollbar
)


@dataclasses.dataclass(frozen=True)
class Fonts:
    regular: imgui.ImFont
    bold: imgui.ImFont


def start(width: int, height: int) -> Fonts:
    """Make the ImGui context (no ini file) and load the fonts."""
    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")
    io.display_size = imgui.ImVec2(width, height)
    atlas = io.fonts
    regular = atlas.add_font_from_file_ttf(str(FONTS / "Roboto-Regular.ttf"), FONT_SIZE)
    bold = atlas.add_font_from_file_ttf(str(FONTS / "Roboto-Bold.ttf"), FONT_SIZE)
    return Fonts(regular, bold)


def u32(colour: Colour, alpha: float = 1.0) -> int:
    """A theme colour as ImGui's packed colour, its alpha times `alpha`."""
    red, green, blue, own = colour.floats()
    return imgui.get_color_u32(imgui.ImVec4(red, green, blue, own * alpha))


def mix(first: Colour, second: Colour, amount: float) -> Colour:
    """The colour `amount` of the way from first to second (alpha included)."""
    return Colour.from_floats(
        a + (b - a) * amount for a, b in zip(first.floats(), second.floats(), strict=True)
    )


def apply_style(theme: Theme, scale: float) -> None:
    """Set the current ImGui context's style from the theme, at this interface scale."""
    style = imgui.get_style()
    style.font_size_base = FONT_SIZE * scale
    for name, value in dict(
        window_padding=(10, 9),
        frame_padding=(8, 4),
        item_spacing=(7, 6),
        item_inner_spacing=(6, 4),
        window_rounding=6,
        frame_rounding=4,
        popup_rounding=5,
        grab_rounding=3,
        grab_min_size=10,
        window_min_size=(20, 20),
    ).items():
        if isinstance(value, tuple):
            setattr(style, name, imgui.ImVec2(value[0] * scale, value[1] * scale))
        else:
            setattr(style, name, value * scale)
    style.window_border_size = style.frame_border_size = style.popup_border_size = 1.0
    style.image_border_size = 0.0
    style.anti_aliased_lines = style.anti_aliased_fill = True
    hover = mix(theme.control, theme.control_active, 0.5)
    opaque_panel = dataclasses.replace(theme.panel, alpha=255)
    colours = {
        imgui.Col_.text: theme.text,
        imgui.Col_.text_disabled: mix(theme.text, theme.panel_line, 0.6),
        imgui.Col_.window_bg: theme.panel,
        imgui.Col_.popup_bg: opaque_panel,
        imgui.Col_.border: theme.panel_line,
        imgui.Col_.border_shadow: dataclasses.replace(theme.panel_line, alpha=0),
        imgui.Col_.frame_bg: theme.control,
        imgui.Col_.frame_bg_hovered: hover,
        imgui.Col_.frame_bg_active: theme.control_active,
        imgui.Col_.slider_grab: mix(theme.text, theme.panel_line, 0.3),
        imgui.Col_.slider_grab_active: theme.text,
        imgui.Col_.button: theme.control,
        imgui.Col_.button_hovered: hover,
        imgui.Col_.button_active: theme.control_active,
        imgui.Col_.header: theme.control_active,
        imgui.Col_.header_hovered: hover,
        imgui.Col_.header_active: theme.control_active,
        imgui.Col_.separator: theme.panel_line,
    }
    for key, colour in colours.items():
        style.set_color_(key, imgui.ImVec4(*colour.floats()))


def clock(seconds: float) -> str:
    """Minutes and seconds ("2:05")."""
    whole = max(int(seconds), 0)
    return f"{whole // 60}:{whole % 60:02d}"


def ship_name(frame: Snapshot, index: int) -> str:
    """A ship by its type and its slot in its team counted from 1 ("P-51D 2")."""
    ships = frame.ships
    slot = int((ships.team[:index] == ships.team[index]).sum())
    return f"{frame.type_names[int(ships.ship_type[index])]} {slot + 1}"


@dataclasses.dataclass(frozen=True)
class Kill:
    """A death at game time `time`: the dead ship and the ship that did it the most damage on
    that tick (-1: the boundary)."""

    time: float
    victim: int
    killer: int


def kills_at(frames: list[Snapshot], index: int) -> list[Kill]:
    """The kills of the last FEED_SECONDS before frames[index], oldest first."""
    now = frames[index].clock
    found = []
    for frame in frames[: index + 1][::-1]:
        if now - frame.clock > FEED_SECONDS:
            break
        events = frame.events
        found.extend(
            Kill(frame.clock, int(victim), int(events.killer[victim]))
            for victim in np.flatnonzero(events.died)
        )
    return found[::-1]


def result_text(frame: Snapshot) -> tuple[str, str]:
    """The result card's two lines (the last frame of a game)."""
    alive = frame.ships.alive
    left = [int((alive & (frame.ships.team == team)).sum()) for team in (0, 1)]
    when = clock(frame.clock)
    head = "Draw" if frame.result == 2 else f"Team {frame.result} wins"
    if min(left) == 0:
        how = f"wipe at {when}"
    elif left[0] != left[1]:
        how = f"clock: {left[0]} - {left[1]} ships alive"
    else:
        how = f"clock: {left[0]} - {left[1]}, decided on shield"
    return head, how


@dataclasses.dataclass
class Shown:
    """What the interface shows this frame. frames and index: the recorded ticks and the one
    on screen (the scrubber's range in the live window); live: the window (interactive, with
    the scrubber) or the video. title: a short description of the match (right of the bar)."""

    frames: list[Snapshot]
    index: int
    view: int
    theme: Theme
    camera: Camera
    minimap_camera: Camera
    minimap_texture: int
    marks: list[Mark]
    title: str
    draw_vision: bool = False
    playing: bool = True
    speed: float = 1.0
    live: bool = False
    show_result: bool = False


@dataclasses.dataclass
class Request:
    """What the user clicked this frame (the live window acts on it)."""

    view: int | None = None
    toggle_playing: bool = False
    step: int = 0
    speed: float | None = None
    seek: int | None = None
    toggle_vision: bool = False
    next_theme: bool = False


class Interface:
    """Draws the interface for a `Shown` in the current ImGui frame."""

    def __init__(self, fonts: Fonts, scale: float) -> None:
        self.fonts, self.scale = fonts, scale

    def draw(self, shown: Shown) -> Request:
        request = Request()
        apply_style(shown.theme, self.scale)
        width, height = imgui.get_io().display_size
        self._top_bar(shown, request, width)
        self._kill_feed(shown, width)
        self._collision_labels(shown)
        bottom = height - (SCRUBBER * self.scale if shown.live else 0.0)
        self._minimap(shown, width, bottom)
        if shown.live:
            self._scrubber(shown, request, width, height)
        if shown.show_result and shown.frames[shown.index].result >= 0:
            self._result(shown, width, height)
        return request

    # ------------------------------------------------------------------ pieces

    def _window(self, name: str, x: float, y: float, w: float, h: float, square: bool = False):
        if square:
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
            imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.set_next_window_pos(imgui.ImVec2(x, y), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.always)
        imgui.begin(name, None, FLAGS)
        if square:
            imgui.pop_style_var(2)

    def _button(self, label: str, active: bool = False, colour: Colour | None = None) -> bool:
        if active:
            imgui.push_style_color(
                imgui.Col_.button, imgui.get_style().color_(imgui.Col_.button_active)
            )
        if colour is not None:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour.floats()))
        pressed = imgui.button(label)
        if colour is not None:
            imgui.pop_style_color()
        if active:
            imgui.pop_style_color()
        return pressed

    def _top_bar(self, shown: Shown, request: Request, width: float) -> None:
        theme, scale = shown.theme, self.scale
        frame = shown.frames[shown.index]
        bar = TOP_BAR * scale
        self._window("###top bar", 0.0, 0.0, width, bar, square=True)
        draw_list = imgui.get_window_draw_list()
        draw_list.add_line(
            imgui.ImVec2(0.0, bar - 1.0), imgui.ImVec2(width, bar - 1.0), u32(theme.panel_line)
        )
        row = imgui.get_frame_height()
        top = 0.5 * (bar - row)
        # Left: the view selector (the video shows its one view).
        imgui.set_cursor_pos(imgui.ImVec2(10.0 * scale, top))
        views = (SPECTATOR, 0, 1) if shown.live else (shown.view,)
        for k, view in enumerate(views):
            if k:
                imgui.same_line(0.0, 2.0 * scale)
            colour = None if view == SPECTATOR else theme.team[view]
            if self._button(VIEW_LABELS[view], active=view == shown.view, colour=colour):
                request.view = view
        if shown.live or shown.draw_vision:
            imgui.same_line(0.0, 14.0 * scale)
            if self._button("Vision", active=shown.draw_vision):
                request.toggle_vision = True
        if shown.live:
            imgui.same_line(0.0, 2.0 * scale)
            if self._button(f"Theme: {theme.name}###theme"):
                request.next_theme = True
        # Middle: ships alive by team, then the clock.
        alive = frame.ships.alive
        counts = [int((alive & (frame.ships.team == team)).sum()) for team in (0, 1)]
        imgui.push_font(self.fonts.bold, 0.0)
        parts = [
            (f"{counts[0]}", theme.team[0], self.fonts.bold),
            ("  :  ", theme.text, self.fonts.bold),
            (f"{counts[1]}", theme.team[1], self.fonts.bold),
            ("    ships left", mix(theme.text, theme.panel_line, 0.5), self.fonts.regular),
            ("        " + clock(frame.clock), theme.text, self.fonts.bold),
            (f"  / {clock(frame.episode_length)}", mix(theme.text, theme.panel_line, 0.5),
             self.fonts.regular),
        ]
        total = 0.0
        for text, _, font in parts:
            imgui.push_font(font, 0.0)
            total += imgui.calc_text_size(text).x
            imgui.pop_font()
        x = 0.5 * (width - total)
        y = 0.5 * (bar - imgui.get_text_line_height())
        for text, colour, font in parts:
            imgui.push_font(font, 0.0)
            draw_list.add_text(imgui.ImVec2(x, y), u32(colour), text)
            x += imgui.calc_text_size(text).x
            imgui.pop_font()
        imgui.pop_font()
        # Right: the playback controls (live) or the match's title.
        if shown.live:
            labels = [("Pause" if shown.playing else "Play"), "Step"] + [
                f"x{speed:g}" for speed in SPEEDS
            ]
            spacing = 2.0 * scale
            widths = [imgui.calc_text_size(label).x + 2 * imgui.get_style().frame_padding.x
                      for label in labels]
            x = width - sum(widths) - spacing * (len(labels) - 1) - 10.0 * scale
            imgui.same_line(0.0, 0.0)
            imgui.set_cursor_pos(imgui.ImVec2(x, top))
            if self._button(labels[0]):
                request.toggle_playing = True
            imgui.same_line(0.0, spacing)
            if self._button(labels[1]):
                request.step = 1
            for speed in SPEEDS:
                imgui.same_line(0.0, spacing)
                if self._button(f"x{speed:g}", active=speed == shown.speed):
                    request.speed = speed
        else:
            text = shown.title
            size = imgui.calc_text_size(text).x
            draw_list.add_text(
                imgui.ImVec2(width - size - 14.0 * scale, y),
                u32(mix(theme.text, theme.panel_line, 0.4)),
                text,
            )
        imgui.end()

    def _kill_feed(self, shown: Shown, width: float) -> None:
        theme, scale = shown.theme, self.scale
        frame = shown.frames[shown.index]
        draw_list = imgui.get_background_draw_list()
        line_height = imgui.get_text_line_height() + 8.0 * scale
        y = TOP_BAR * scale + 10.0 * scale
        right = width - 10.0 * scale
        for kill in kills_at(shown.frames, shown.index)[::-1][:FEED_LINES]:
            age = frame.clock - kill.time
            opacity = min(max((FEED_SECONDS - age) / FEED_FADE_SECONDS, 0.0), 1.0)
            ships = frame.ships
            victim = (ship_name(frame, kill.victim), theme.team[int(ships.team[kill.victim])])
            if kill.killer >= 0:
                killer = (ship_name(frame, kill.killer), theme.team[int(ships.team[kill.killer])])
                parts = [killer, ("  ›  ", theme.text), victim]
            else:
                parts = [("the boundary  ›  ", theme.text), victim]
            text_width = sum(imgui.calc_text_size(text).x for text, _ in parts)
            padding = 7.0 * scale
            x = right - text_width - padding
            draw_list.add_rect_filled(
                imgui.ImVec2(x - padding, y),
                imgui.ImVec2(right, y + line_height - 2.0 * scale),
                u32(theme.panel, opacity),
                4.0 * scale,
            )
            draw_list.add_rect(
                imgui.ImVec2(x - padding, y),
                imgui.ImVec2(right, y + line_height - 2.0 * scale),
                u32(theme.panel_line, opacity),
                4.0 * scale,
            )
            for text, colour in parts:
                draw_list.add_text(imgui.ImVec2(x, y + 3.0 * scale), u32(colour, opacity), text)
                x += imgui.calc_text_size(text).x
            y += line_height

    def _collision_labels(self, shown: Shown) -> None:
        theme, scale = shown.theme, self.scale
        now = shown.frames[shown.index].clock
        draw_list = imgui.get_background_draw_list()
        for mark in shown.marks:
            if mark.kind != "collision":
                continue
            px, py = shown.camera.pixel(mark.x, mark.y)
            text = f"collision {mark.damage[0]:.0f} / {mark.damage[1]:.0f}"
            draw_list.add_text(
                imgui.ImVec2(px + 24.0 * scale, py - 22.0 * scale),
                u32(theme.outline, fade(mark, now)),
                text,
            )

    def _minimap(self, shown: Shown, width: float, bottom: float) -> None:
        theme, scale = shown.theme, self.scale
        side = MINIMAP_SIDE * scale
        padding = 4.0 * scale
        x, y = width - side - 2 * padding - 10.0 * scale, bottom - side - 2 * padding - 10.0 * scale
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(padding, padding))
        self._window("###minimap", x, y, side + 2 * padding, side + 2 * padding)
        imgui.pop_style_var()
        imgui.set_cursor_screen_pos(imgui.ImVec2(x + padding, y + padding))
        imgui.image(
            imgui.ImTextureRef(shown.minimap_texture),
            imgui.ImVec2(side, side),
            imgui.ImVec2(0.0, 1.0),
            imgui.ImVec2(1.0, 0.0),
        )
        # The battlefield camera's view as a rectangle on the minimap.
        minimap, camera = shown.minimap_camera, shown.camera
        corners = [camera.world(0.0, 0.0), camera.world(camera.width, camera.height)]
        (left, top), (right, low) = (minimap.pixel(*corner) for corner in corners)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(
            imgui.ImVec2(x + padding, y + padding), imgui.ImVec2(x + padding + side, y + padding + side)
        )
        draw_list.add_rect(
            imgui.ImVec2(x + padding + left, y + padding + top),
            imgui.ImVec2(x + padding + right, y + padding + low),
            u32(theme.text, 0.85),
            0.0,
            1.5 * scale,
        )
        draw_list.pop_clip_rect()
        imgui.end()

    def _scrubber(self, shown: Shown, request: Request, width: float, height: float) -> None:
        theme, scale = shown.theme, self.scale
        bar = SCRUBBER * scale
        self._window("###scrubber", 0.0, height - bar, width, bar, square=True)
        imgui.get_window_draw_list().add_line(
            imgui.ImVec2(0.0, height - bar), imgui.ImVec2(width, height - bar), u32(theme.panel_line)
        )
        frame = shown.frames[shown.index]
        imgui.set_cursor_pos(imgui.ImVec2(10.0 * scale, 0.5 * (bar - imgui.get_frame_height())))
        if self._button("|<"):
            request.seek = 0
        imgui.same_line(0.0, 2.0 * scale)
        if self._button("<"):
            request.step = -1
        imgui.same_line(0.0, 2.0 * scale)
        if self._button(">"):
            request.step = 1
        imgui.same_line(0.0, 10.0 * scale)
        last = shown.frames[-1]
        imgui.align_text_to_frame_padding()
        imgui.text(f"{clock(frame.clock)} / {clock(last.clock)}")
        imgui.same_line(0.0, 12.0 * scale)
        imgui.set_next_item_width(imgui.get_content_region_avail().x - 10.0 * scale)
        changed, value = imgui.slider_int("##ticks", shown.index, 0, len(shown.frames) - 1, "")
        if changed:
            request.seek = value
        # Kill ticks as marks on the track, in the dead ship's team colour.
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_list = imgui.get_window_draw_list()
        span = max(len(shown.frames) - 1, 1)
        for k, recorded in enumerate(shown.frames):
            for victim in np.flatnonzero(recorded.events.died):
                mx = low.x + (high.x - low.x) * k / span
                colour = theme.team[int(recorded.ships.team[victim])]
                draw_list.add_line(imgui.ImVec2(mx, low.y), imgui.ImVec2(mx, high.y), u32(colour),
                                   2.0 * scale)
        imgui.end()

    def _result(self, shown: Shown, width: float, height: float) -> None:
        theme, scale = shown.theme, self.scale
        frame = shown.frames[shown.index]
        head, how = result_text(frame)
        imgui.push_font(self.fonts.bold, FONT_SIZE * scale * 2.0)
        head_size = imgui.calc_text_size(head)
        imgui.pop_font()
        how_size = imgui.calc_text_size(how)
        w = max(head_size.x, how_size.x) + 48.0 * scale
        h = head_size.y + how_size.y + 40.0 * scale
        x, y = 0.5 * (width - w), (TOP_BAR + 16.0) * scale  # under the bar, off the action
        self._window("###result", x, y, w, h)
        draw_list = imgui.get_window_draw_list()
        colour = theme.text if frame.result == 2 else theme.team[frame.result]
        imgui.push_font(self.fonts.bold, FONT_SIZE * scale * 2.0)
        draw_list.add_text(
            imgui.ImVec2(x + 0.5 * (w - head_size.x), y + 14.0 * scale), u32(colour), head
        )
        imgui.pop_font()
        draw_list.add_text(
            imgui.ImVec2(x + 0.5 * (w - how_size.x), y + 20.0 * scale + head_size.y),
            u32(theme.text),
            how,
        )
        imgui.end()
