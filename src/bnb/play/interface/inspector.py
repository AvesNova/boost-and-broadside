"""The selection card (right): what the player selected, on the battlefield, the minimap
or a card.

- A ship: its team, type and controller, its live state (shield, energy, speed, angle of
  attack, power and stick, the last shot), kills and deaths; "Take control" (or
  "Release" on the flown ship) and "Follow". With the prediction overlays' readouts on
  (`bnb.play.overlays`), what the predicting side's agent thinks of it: an enemy's action
  distribution (power, stick, P(shoot)) beside the action it took; an own ship's value per
  reward level, the outcome head, and its next shot's P(enemy hit).
- A field: its refractive index, radius and core.
- A zone: its role and owner, its progress, and the ships contesting it.

Nothing selected: the card does not show.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from imgui_bundle import imgui

from . import cards
from .controls import bar, button, tooltip, u32
from .fleets import controller_label, controller_label_of_setup
from .shortcuts import shortcut
from .text import TEAM_NAMES, ship_name, type_name

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

POWER = ("coast", "boost", "reverse")
ROLE_NAMES = ("Team 0 spawn", "Team 0 defense", "Neutral", "Team 1 defense", "Team 1 spawn")
LABEL_COLUMN = 104.0  # pixels at scale 1
LEVEL_COLUMN = 150.0  # the critic's rows: a reward level's name is longer


def stick_text(stick: int) -> str:
    """A stick action as words: centred, or the level (1 lightest to 3 full) and side."""
    if stick == 0:
        return "centred"
    level = (stick + 1) // 2
    return f"{'right' if stick % 2 else 'left'} {level}"


def _row(label: str, value: str, scale: float, column: float = LABEL_COLUMN) -> None:
    imgui.text_disabled(label)
    imgui.same_line(column * scale)
    imgui.text(value)


def _meter(label: str, fraction: float, colour, theme, scale: float) -> None:
    imgui.text_disabled(label)
    imgui.same_line(LABEL_COLUMN * scale)
    corner = imgui.get_cursor_screen_pos()
    line = imgui.get_text_line_height()
    width = imgui.get_content_region_avail().x - 44.0 * scale
    height = 7.0 * scale
    bar(
        imgui.get_window_draw_list(),
        corner.x,
        corner.y + 0.5 * (line - height),
        width,
        height,
        fraction,
        u32(colour),
        u32(theme.panel_line),
    )
    imgui.dummy(imgui.ImVec2(width, line))
    imgui.same_line()
    imgui.text(f"{100.0 * fraction:.0f}%")


def draw(interface: Interface, layout: Layout) -> None:
    """The selection card, its tab, or nothing."""
    session = interface.session
    rect = layout.selection
    if rect is None or session.selection is None:
        return
    if "selection" not in interface.open:
        cards.tab(interface, "selection", rect)
        return
    cards.window("###selection", rect, fit_height=True)
    selection = session.selection
    titles = {"ship": "Ship", "field": "Field", "zone": "Zone"}
    cards.header(interface, "selection", titles[selection.kind])
    if selection.kind == "ship":
        _ship(interface, selection.index)
    elif selection.kind == "field":
        _field(interface, selection.index)
    else:
        _zone(interface, selection.index)
    imgui.spacing()
    if button(interface.controls, "selection/clear", "Clear selection", "Deselect"):
        session.select(None)
    imgui.end()


def _ship(interface: Interface, index: int) -> None:
    session, controls = interface.session, interface.controls
    loop, frame, theme, scale = session.loop, session.frame, session.theme, session.scale
    ships = frame.ships
    team = int(ships.team[index])
    imgui.push_font(session.fonts.bold, imgui.get_font_size() * 1.15)
    imgui.text_colored(
        imgui.ImVec4(*theme.team[team].floats()), ship_name(ships.team, ships.ship_type, index)
    )
    imgui.pop_font()
    if loop.pilot == index:
        controller = "you"
    elif loop.sides:
        controller = controller_label(loop.sides[team])
    else:
        controller = controller_label_of_setup(loop.setup.sides[team])
    _row("team", TEAM_NAMES[team], scale)
    _row("type", type_name(int(ships.ship_type[index])), scale)
    _row("controller", controller, scale)
    _row("kills / deaths", f"{loop.kills[index]} / {loop.deaths[index]}", scale)
    if not session.frame.fogged or session.view_team < 0 or ships.visible[session.view_team, index]:
        imgui.separator()
        if ships.alive[index]:
            _meter("shield", float(ships.shield[index]), theme.shield[team], theme, scale)
            _meter("energy", float(ships.energy[index]), theme.energy, theme, scale)
            current, previous = loop.current.ships, loop.previous.ships
            moved = math.hypot(
                current.x[index] - previous.x[index], current.y[index] - previous.y[index]
            )
            speed = 0.0 if current.spawned[index] else moved / loop.tick_length
            _row("speed", f"{speed:.0f} m/s", scale)
            _row(
                "angle of attack",
                f"{math.degrees(float(ships.angle_of_attack[index])):+.1f}°",
                scale,
            )
            power, stick, shoot = (int(value) for value in loop.last_actions[index])
            _row("power", POWER[power] if 0 <= power < len(POWER) else str(power), scale)
            _row("stick", stick_text(stick) + ("   firing" if shoot else ""), scale)
            now = loop.ticks * loop.tick_length
            shot = session.last_shot[index]
            _row(
                "last shot",
                "none yet" if not np.isfinite(shot) else f"{now - shot:.1f} s ago",
                scale,
            )
        else:
            _row("respawns in", f"{ships.respawn_wait[index]:.1f} s", scale)
    else:
        imgui.text_disabled("out of this view's sight")
    _predictions(interface, index)
    imgui.spacing()
    flying = loop.pilot == index
    if flying:
        if button(
            controls,
            "selection/release",
            "Release",
            "Hand the ship back to its side",
            shortcut("toggle_control"),
        ):
            session.release()
    elif button(
        controls,
        "selection/take control",
        "Take control",
        "Fly this ship (W S A D, Shift, Ctrl, Space)"
        + ("; forks a live match from this moment" if loop.replaying else ""),
        shortcut("toggle_control"),
    ):
        session.take_control(index)
    imgui.same_line()
    following = session.following == index
    if button(
        controls,
        "selection/follow",
        "Following" if following else "Follow",
        "Keep the camera on this ship",
        shortcut("toggle_follow"),
        active=following,
    ):
        session.follow(None if following else index)


STICK_ORDER = (6, 4, 2, 0, 1, 3, 5)  # the stick actions from full left to full right


def _bars(values, chosen: int, colour, theme, scale: float) -> None:
    """A row of small bars, one per value in [0, 1], the `chosen` one outlined (the
    action taken)."""
    draw = imgui.get_window_draw_list()
    corner = imgui.get_cursor_screen_pos()
    line = imgui.get_text_line_height()
    width, gap = 14.0 * scale, 3.0 * scale
    for slot, value in enumerate(values):
        left = corner.x + slot * (width + gap)
        top = corner.y + line * (1.0 - float(value))
        draw.add_rect_filled(
            imgui.ImVec2(left, corner.y),
            imgui.ImVec2(left + width, corner.y + line),
            u32(theme.panel_line),
        )
        draw.add_rect_filled(
            imgui.ImVec2(left, top), imgui.ImVec2(left + width, corner.y + line), u32(colour)
        )
        if slot == chosen:
            draw.add_rect(
                imgui.ImVec2(left - 1, corner.y - 1),
                imgui.ImVec2(left + width + 1, corner.y + line + 1),
                u32(theme.truth),
                0.0,
                1.5 * scale,
            )
    imgui.dummy(imgui.ImVec2(len(values) * (width + gap), line))


def _predictions(interface: Interface, index: int) -> None:
    """The readouts of the overlays on: the predicting side's view of this ship."""
    session = interface.session
    theme, scale = session.theme, session.scale
    drawn = session.overlay_arrays()
    team = int(session.frame.ships.team[index])
    colour = theme.prediction
    power, stick, shoot = (int(value) for value in session.loop.last_actions[index])
    if "enemy_action" in drawn and drawn["enemy_action"][0] != team:
        predictor, actions = drawn["enemy_action"]
        rows = np.flatnonzero(actions.ship == index)
        if len(rows):
            row = int(rows[0])
            imgui.separator()
            imgui.text_disabled(f"{TEAM_NAMES[predictor]} predicts its action")
            tooltip("The enemy-action head's marginals; the outline is the action it took")
            imgui.text_disabled("power")
            imgui.same_line(LABEL_COLUMN * scale)
            _bars(actions.power[row], power, colour, theme, scale)
            tooltip("coast, boost, reverse")
            imgui.text_disabled("stick")
            imgui.same_line(LABEL_COLUMN * scale)
            _bars(
                actions.stick[row][list(STICK_ORDER)],
                STICK_ORDER.index(stick),
                colour,
                theme,
                scale,
            )
            tooltip("full left to full right")
            _meter("P(shoot)", float(actions.shoot[row, 1]), colour, theme, scale)
    if "value" in drawn and drawn["value"][0] == team:
        critic = drawn["value"][1]
        rows = np.flatnonzero(critic.ship == index)
        if len(rows):
            row = int(rows[0])
            imgui.separator()
            imgui.text_disabled(f"{TEAM_NAMES[team]}'s critic")
            tooltip("The value per reward level (the critic's units) and the outcome head")
            for level, value in zip(critic.levels, critic.values[row], strict=True):
                _row(level.replace("_", " "), f"{float(value):+.3f}", scale, LEVEL_COLUMN)
            if critic.win is not None:  # a critic with an outcome head
                _meter("P(win)", critic.win, theme.team[team], theme, scale)
                _meter("P(tie)", critic.tie, theme.panel_line, theme, scale)
                _meter("P(loss)", critic.loss, theme.team[1 - team], theme, scale)
                _meter("P(unresolved)", critic.unresolved, theme.panel_line, theme, scale)
    if "shot_path" in drawn:
        shot = drawn["shot_path"][1]
        _meter("next shot hits", float(shot.enemy_hit[index]), colour, theme, scale)
        tooltip("The shot-outcome head's P(enemy hit) for the shot it would fire now")


def _field(interface: Interface, index: int) -> None:
    session = interface.session
    fields, scale = session.frame.fields, session.scale
    level = int(fields.level[index])
    index_value = math.sqrt(2.0) ** level
    air = "thick air" if level > 0 else "thin air" if level < 0 else "ambient"
    _row("refractive index", f"{index_value:.2f}  ({air})", scale)
    _row("speed in it", f"{100.0 / index_value:.0f}% of the open sky's", scale)
    _row("radius", f"{fields.radius[index]:.0f} m", scale)
    _row("core", f"{fields.core_radius[index]:.0f} m (blocks sight)", scale)
    _row("band", f"{fields.band_width[index]:.0f} m", scale)
    tooltip("The core blocks sight lines that cross it")


def _zone(interface: Interface, index: int) -> None:
    session = interface.session
    frame, theme, scale = session.frame, session.theme, session.scale
    zones, ships = frame.zones, frame.ships
    role = int(zones.role[index])
    owner, attacker = int(zones.owner[index]), int(zones.attacker[index])
    _row("role", ROLE_NAMES[role] if 0 <= role < len(ROLE_NAMES) else str(role), scale)
    _row("owner", TEAM_NAMES[owner] if owner >= 0 else "nobody", scale)
    if attacker >= 0:
        _meter("progress", float(zones.meter[index]), theme.team[attacker], theme, scale)
        _row("attacker", TEAM_NAMES[attacker], scale)
    inside = ships.alive & (
        np.hypot(ships.x - zones.x[index], ships.y - zones.y[index]) <= zones.radius[index]
    )
    imgui.separator()
    imgui.text_disabled("contesting")
    shown = False
    for team in range(2):
        members = np.flatnonzero(inside & (ships.team == team))
        if session.view_team >= 0 and frame.fogged:
            members = [ship for ship in members if ships.visible[session.view_team, ship]]
        if len(members):
            names = ", ".join(ship_name(ships.team, ships.ship_type, int(ship)) for ship in members)
            imgui.text_colored(imgui.ImVec4(*theme.team[team].floats()), names)
            shown = True
    if not shown:
        imgui.text("nobody")
