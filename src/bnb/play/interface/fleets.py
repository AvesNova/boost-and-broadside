"""The fleets card (left): one section per team, a row per ship with its type, shield and
energy bars, alive or respawning (with the time to its respawn, its team's respawn wave
included), and its controller (you, scripted, the scripted agent and its doctrine, or the
agent's name). In Frontline a team's section opens with its next respawn wave's countdown:
the player's own team's only while they fly a ship (the enemy's clock is hidden from them,
as from the agents), both teams' while they watch. Clicking a row selects its ship. A
team's controller is changed here: the scripted chaser, the scripted agent flying r9m2 (the
default) or r9 (`bnb.play.sides.DOCTRINES`), or one of the local runs' policies, and an
agent's sampling temperature. The card is wide enough for the longest
ship name (`width`), whose column both teams share, so their bars line up."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..loop import SideSetup
from ..selection import Selection
from ..sides import DOCTRINES, describe, label_of
from . import cards
from .controls import bar, right_edge, tooltip, u32
from .layout import FLEETS_WIDTH
from .text import TEAM_NAMES, ship_name

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

BAR_WIDTH = 46.0
BAR_HEIGHT = 6.0
LONGEST_WAIT = "99.9 s"  # the widest a row's respawn wait reads


SCRIPTED_LABELS = {"scripted": "Chaser", "scripted_agent": "Scripted agent"}
# The combo's scripted choices: the chaser, then the scripted agent with each doctrine.
SCRIPTED_CHOICES = ((SideSetup("scripted"), "Chaser"),) + tuple(
    (SideSetup("scripted_agent", doctrine=doctrine), f"Scripted agent {doctrine}")
    for doctrine in DOCTRINES
)


def scripted_label(side: SideSetup) -> str | None:
    """A scripted side's label in the combo (None for an agent)."""
    for choice, label in SCRIPTED_CHOICES:
        if side.controller == choice.controller and (
            side.controller != "scripted_agent" or side.doctrine == choice.doctrine
        ):
            return label
    return SCRIPTED_LABELS.get(side.controller)


def controller_label(controller) -> str:
    """A side's controller as the card names it."""
    return controller.label


def controller_combo(
    interface: Interface, team: int, name: str, current: SideSetup, width: float
) -> SideSetup | None:
    """A combo of the controllers a side can have (scripted, each local policy, or a
    policy directory typed in); returns the chosen one when it changes. Recorded as
    `name` and `name/<choice>`."""
    session, controls = interface.session, interface.controls
    preview = scripted_label(current) or label_of(current.policy)
    imgui.set_next_item_width(width)
    chosen = None
    opened = imgui.begin_combo(f"##{name}", preview)
    controls.record(name)
    tooltip(
        f"Who controls {TEAM_NAMES[team]}: the scripted chaser, the scripted agent (it "
        "remembers the enemies it saw; r9m2 is r9 with the maneuver doctrine elements, r9 "
        "the training's doctrine) or a trained agent"
    )
    if opened:
        for choice, label in SCRIPTED_CHOICES:
            if imgui.selectable(label, scripted_label(current) == label)[0]:
                chosen = choice
            controls.record(f"{name}/{label.lower()}")
        policies = interface.menu.policies()
        if not policies:
            imgui.text_disabled(f"no policies under {session.options.runs}")
        for path in policies:
            label = label_of(path)
            selected = current.controller == "agent" and current.policy == str(path)
            if imgui.selectable(label, selected)[0]:
                chosen = SideSetup("agent", str(path), current.temperature)
            controls.record(f"{name}/{label}")
            tooltip(str(path) + "  (" + describe(path) + ")")
        imgui.separator()
        imgui.set_next_item_width(-1.0)
        entered, typed = imgui.input_text_with_hint(
            f"##{name} path",
            "or a policy directory, then Enter",
            interface.typed_policy,
            imgui.InputTextFlags_.enter_returns_true,
        )
        controls.record(f"{name}/path")
        interface.typed_policy = typed
        if entered and typed.strip():
            chosen = SideSetup("agent", typed.strip(), current.temperature)
            imgui.close_current_popup()
        imgui.end_combo()
    return chosen


def draw(interface: Interface, layout: Layout) -> None:
    """The fleets card, or its tab."""
    session, controls = interface.session, interface.controls
    if "fleets" not in interface.open:
        cards.tab(interface, "fleets", layout.fleets)
        return
    loop, frame, theme, scale = session.loop, session.frame, session.theme, session.scale
    cards.window("###fleets", layout.fleets, fit_height=True)
    cards.header(interface, "fleets", "Fleets")
    ships = frame.ships
    track = u32(theme.panel_line)
    per_team = loop.rules.ships_per_team
    names = name_width(ships)
    for team in range(2):
        right = right_edge()
        draw_list = imgui.get_window_draw_list()
        position = imgui.get_cursor_screen_pos()
        row = imgui.get_frame_height()
        swatch = 0.55 * row
        draw_list.add_rect_filled(
            imgui.ImVec2(position.x, position.y + 0.5 * (row - swatch)),
            imgui.ImVec2(position.x + swatch, position.y + 0.5 * (row + swatch)),
            u32(theme.team[team]),
            2.0 * scale,
        )
        imgui.dummy(imgui.ImVec2(swatch, row))
        imgui.same_line()
        imgui.push_font(session.fonts.bold, 0.0)
        imgui.align_text_to_frame_padding()
        imgui.text(TEAM_NAMES[team])
        imgui.pop_font()
        combo_width = 150.0 * scale
        imgui.same_line(right - combo_width)
        if loop.replaying:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(controller_label_of_setup(loop.setup.sides[team]))
        else:
            side = loop.setup.sides[team]
            chosen = controller_combo(
                interface, team, f"fleet/team{team}/controller", side, combo_width
            )
            if chosen is not None:
                session.set_side(team, chosen)
            if side.controller == "agent":
                imgui.set_next_item_width(-1.0)
                changed, value = imgui.slider_float(
                    f"##temperature{team}", side.temperature, 0.0, 2.0, "temperature %.2f"
                )
                controls.record(f"fleet/team{team}/temperature")
                tooltip(
                    "Sampling temperature: 1 samples the policy as trained, 0 takes its "
                    "most likely action"
                )
                if changed:
                    session.set_temperature(team, value)
        if wave_shown(frame.mode, loop.pilot, per_team, team):
            imgui.text_disabled("next wave in")
            imgui.same_line()
            imgui.text(f"{float(frame.next_wave[team]):.1f} s")
            tooltip(
                "Dead ships respawn together at their team's respawn wave, once their "
                "minimum wait is over; each team's waves have their own clock"
            )
        flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_pad_outer_x
        if imgui.begin_table(f"team {team}", 5, flags):
            # As wide as the longest name of either team, so both teams' bars line up.
            imgui.table_setup_column("ship", imgui.TableColumnFlags_.width_fixed, names)
            for index in range(team * per_team, (team + 1) * per_team):
                imgui.table_next_row()
                imgui.table_next_column()
                selected = session.selection == Selection("ship", index)
                label = ship_name(ships.team, ships.ship_type, index)
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*theme.team[team].floats()))
                clicked = imgui.selectable(
                    f"{label}##ship{index}",
                    selected,
                    imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap,
                )[0]
                imgui.pop_style_color()
                controls.record(f"fleet/ship/{index}")
                if clicked:
                    session.select(Selection("ship", index))
                for value, colour in (
                    (ships.shield[index], theme.shield[team]),
                    (ships.energy[index], theme.energy),
                ):
                    imgui.table_next_column()
                    corner = imgui.get_cursor_screen_pos()
                    height = BAR_HEIGHT * scale
                    width = BAR_WIDTH * scale
                    line = imgui.get_text_line_height()
                    alive = bool(ships.alive[index])
                    bar(
                        imgui.get_window_draw_list(),
                        corner.x,
                        corner.y + 0.5 * (line - height),
                        width,
                        height,
                        float(value) if alive else 0.0,
                        u32(colour),
                        track,
                    )
                    imgui.dummy(imgui.ImVec2(width, line))
                imgui.table_next_column()
                if ships.alive[index]:
                    imgui.text_disabled("alive")
                else:
                    imgui.text(f"{ships.respawn_wait[index]:.1f} s")
                imgui.table_next_column()
                if loop.pilot == index:
                    imgui.text("you")
                else:
                    imgui.text_disabled(row_controller(loop, team))
            imgui.end_table()
        if team == 0:
            imgui.spacing()
    imgui.end()


def wave_shown(mode: str, pilot: int | None, ships_per_team: int, team: int) -> bool:
    """Whether the card shows this team's next respawn wave: in Frontline, the pilot's own
    team only (the enemy's clock is hidden information), or both teams with no pilot."""
    if mode != "frontline":
        return False
    return pilot is None or pilot // ships_per_team == team


def row_controller(loop, team: int) -> str:
    """A team's controller as its ship rows name it."""
    if loop.sides:
        return short(controller_label(loop.sides[team]))
    return short(controller_label_of_setup(loop.setup.sides[team]))


def name_width(ships) -> float:
    """The widest ship name's width, in pixels (the type's name and the slot)."""
    return max(
        imgui.calc_text_size(ship_name(ships.team, ships.ship_type, i)).x
        for i in range(len(ships.team))
    )


def width(interface: Interface) -> float:
    """The fleets card's width at interface scale 1: FLEETS_WIDTH, or wider so that a row
    holds its ship's whole name (the type's and the slot's) beside the bars, the wait and
    the controller."""
    session = interface.session
    loop, ships, scale = session.loop, session.frame.ships, session.scale
    style = imgui.get_style()

    def text(value: str) -> float:
        return imgui.calc_text_size(value).x

    name = name_width(ships)
    controller = max(text(row_controller(loop, team)) for team in range(2))
    row = (
        name
        + 2.0 * BAR_WIDTH * scale
        + text(LONGEST_WAIT)
        + max(controller, text("you"))
        + 4.0 * 2.0 * style.cell_padding.x  # between the five columns
        + 2.0 * style.window_padding.x
        + style.scrollbar_size
    )
    return max(FLEETS_WIDTH, row / scale)


def short(label: str) -> str:
    """A controller's label short enough for a row: the policy's directory."""
    return label.rsplit("/", 1)[-1] if "/" in label else label


def controller_label_of_setup(side: SideSetup) -> str:
    """A recorded side's controller label (replays run no controllers)."""
    if side.controller == "agent":
        return label_of(side.policy)
    return (scripted_label(side) or side.controller).lower()
