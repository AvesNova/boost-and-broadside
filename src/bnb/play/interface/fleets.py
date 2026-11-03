"""The fleets card (left): one section per team, a row per ship with its type, shield and
energy bars, alive or respawning (with the wait), and its controller (you, scripted, or
the agent's name). Clicking a row selects its ship. A team's controller is changed here:
scripted or one of the local runs' policies, and an agent's sampling temperature."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..loop import SideSetup
from ..selection import Selection
from ..sides import Scripted, describe, label_of
from . import cards
from .controls import bar, right_edge, tooltip, u32
from .text import TEAM_NAMES, ship_name

if TYPE_CHECKING:
    from .interface import Interface
    from .layout import Layout

BAR_WIDTH = 46.0
BAR_HEIGHT = 6.0


def controller_label(controller) -> str:
    """A side's controller as the card names it."""
    return "scripted" if isinstance(controller, Scripted) else controller.label


def controller_combo(
    interface: Interface, team: int, name: str, current: SideSetup, width: float
) -> SideSetup | None:
    """A combo of the controllers a side can have (scripted, each local policy, or a
    policy directory typed in); returns the chosen one when it changes. Recorded as
    `name` and `name/<choice>`."""
    session, controls = interface.session, interface.controls
    preview = "Scripted" if current.controller != "agent" else label_of(current.policy)
    imgui.set_next_item_width(width)
    chosen = None
    opened = imgui.begin_combo(f"##{name}", preview)
    controls.record(name)
    tooltip(f"Who controls {TEAM_NAMES[team]}: the scripted chaser or a trained agent")
    if opened:
        if imgui.selectable("Scripted", current.controller != "agent")[0]:
            chosen = SideSetup()
        controls.record(f"{name}/scripted")
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
        flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_pad_outer_x
        if imgui.begin_table(f"team {team}", 5, flags):
            imgui.table_setup_column("ship", imgui.TableColumnFlags_.width_stretch)
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
                    imgui.text_disabled(
                        short(
                            controller_label(loop.sides[team])
                            if loop.sides
                            else controller_label_of_setup(loop.setup.sides[team])
                        )
                    )
            imgui.end_table()
        if team == 0:
            imgui.spacing()
    imgui.end()


def short(label: str) -> str:
    """A controller's label short enough for a row: the policy's directory."""
    return label.rsplit("/", 1)[-1] if "/" in label else label


def controller_label_of_setup(side: SideSetup) -> str:
    """A recorded side's controller label (replays run no controllers)."""
    return "scripted" if side.controller != "agent" else label_of(side.policy)
