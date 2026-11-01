"""Pass 9: the player's HUD, through ImGui's foreground draw list: the clock, the scores
(team deathmatch) or the front (Frontline), the flown ship's respawn timer, and the view,
theme and time scale; the key reference on demand."""

import dataclasses
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..themes import Theme

if TYPE_CHECKING:
    from ..snapshot import Snapshot

TEXT_SIZE = 20.0
SMALL_SIZE = 16.0
MARGIN = 14.0
KEYS = (
    ("W / S", "boost / reverse (neither: coast)"),
    ("A / D", "turn left / right"),
    ("Shift / Ctrl", "full / lightest turn"),
    ("Space", "shoot"),
    ("Tab", "fly the next ship"),
    ("F", "camera follows the flown ship"),
    ("V", "view: spectator, team 0, team 1"),
    ("T", "next theme"),
    ("E", "theme editor"),
    ("P", "pause"),
    ("N", "one tick while paused"),
    ("+ / -", "faster / slower"),
    ("drag / wheel", "pan / zoom"),
    ("Home", "fit the arena"),
    ("F1", "this list"),
    ("Esc", "quit"),
)


@dataclasses.dataclass(frozen=True)
class HudState:
    """What the HUD shows besides the frame."""

    view: str
    theme: str
    time_scale: float
    paused: bool
    pilot: int | None
    follow: bool
    help: bool


def _colour(colour) -> int:
    return imgui.get_color_u32(colour.floats())


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def draw_hud(frame: Snapshot, state: HudState, theme: Theme, font) -> None:
    """Into the current ImGui frame's foreground draw list."""
    draw = imgui.get_foreground_draw_list()
    width, height = imgui.get_io().display_size
    text = _colour(theme.text)
    team = [_colour(c) for c in theme.team]

    def put(x: float, y: float, colour: int, line: str, size: float = TEXT_SIZE) -> float:
        draw.add_text(font, size, (x, y), colour, line)
        return x + imgui.calc_text_size(line).x * size / imgui.get_font_size()

    def measure(line: str, size: float = TEXT_SIZE) -> float:
        return imgui.calc_text_size(line).x * size / imgui.get_font_size()

    # Top middle: the score or the front, then the clock.
    if frame.mode == "frontline":
        parts = [(text, "Front "), (team[0] if frame.front > 0 else team[1] if frame.front < 0
                                    else text, f"{frame.front:+d}"),
                 (text, f"  (wins at {frame.front_threshold})")]
    elif frame.mode == "team_deathmatch":
        parts = [(team[0], f"{int(frame.score[0])}"), (text, "  :  "),
                 (team[1], f"{int(frame.score[1])}"), (text, f"   (first to {frame.score_target})")]
    else:
        parts = [(text, "Sandbox")]
    clock = f"{_clock(frame.clock)} / {_clock(frame.episode_length)}"
    line = "".join(p for _, p in parts) + "     " + clock
    x = 0.5 * (width - measure(line))
    for colour, part in parts:
        x = put(x, MARGIN, colour, part)
    put(x, MARGIN, text, "     " + clock)

    # Top left: view, theme, time scale.
    speed = "paused" if state.paused else f"x{state.time_scale:g}"
    put(MARGIN, MARGIN, text, f"{state.view}   {state.theme}   {speed}", SMALL_SIZE)

    # Bottom middle: the flown ship.
    if state.pilot is not None:
        ships = frame.ships
        pilot = state.pilot
        label = f"Flying team {int(ships.team[pilot])}, ship {pilot}"
        if state.follow:
            label += " (following)"
        if not ships.alive[pilot]:
            label += f"   respawn in {ships.respawn_wait[pilot]:.1f} s"
        put(0.5 * (width - measure(label)), height - MARGIN - TEXT_SIZE,
            team[int(ships.team[pilot])], label)

    if state.help:
        y = 2.0 * MARGIN + TEXT_SIZE
        draw.add_rect_filled((MARGIN - 6, y - 6),
                             (MARGIN + 420, y + len(KEYS) * (SMALL_SIZE + 4) + 6),
                             imgui.get_color_u32((1.0, 1.0, 1.0, 0.85)), 6.0)
        for key, meaning in KEYS:
            put(MARGIN, y, text, key, SMALL_SIZE)
            put(MARGIN + 130, y, text, meaning, SMALL_SIZE)
            y += SMALL_SIZE + 4
    else:
        put(MARGIN, height - MARGIN - SMALL_SIZE, text, "F1: keys", SMALL_SIZE)
