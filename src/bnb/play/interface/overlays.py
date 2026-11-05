"""The overlays submenu: the top bar's Overlays button opens it under the button (a click
elsewhere closes it). It lists the overlays the match offers (`bnb.play.overlays`), each
with a checkbox and, for a side's predictions, the team whose it draws; whose predictions
(the default, or a team); the truth beside them; and the density's side and map.

The readouts (the selected enemy's predicted action, the critic's values and the outcome)
are in the selection card (`inspector`)."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ..overlays import (
    DENSITY_SHOWN,
    DENSITY_SIDES,
    HELP,
    LABELS,
    PHYSICS,
    TRUTH,
    offered_by,
)
from .controls import button, tooltip
from .text import TEAM_NAMES

if TYPE_CHECKING:
    from .interface import Interface

POPUP = "overlays popup"


def draw_button(interface: Interface) -> float:
    """The top bar's Overlays button (shown chosen while an overlay is on) and its submenu;
    returns the button's right edge (screen x)."""
    session, controls = interface.session, interface.controls
    if button(
        controls,
        "overlays",
        "Overlays",
        "Draw what the agents predict, beside what is true",
        active=bool(session.active_overlays()),
    ):
        imgui.open_popup(POPUP)
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    imgui.set_next_window_pos(imgui.ImVec2(low.x, high.y + 4.0 * session.scale))
    if imgui.begin_popup(POPUP):
        _submenu(interface)
        imgui.end_popup()
    return high.x


def _team_text(interface: Interface, team: int) -> None:
    theme = interface.session.theme
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(*theme.team[team].floats()), f"({TEAM_NAMES[team]})")


def _choice(controls, name: str, label: str, chosen: bool, help: str) -> bool:
    imgui.same_line(0.0, 2.0)
    return button(controls, name, label, help, active=chosen)


def _submenu(interface: Interface) -> None:
    session, controls = interface.session, interface.controls
    settings, sides = session.overlays, session.loop.sides
    imgui.push_font(session.fonts.bold, 0.0)
    imgui.text("Overlays")
    imgui.pop_font()
    imgui.separator()
    offered = session.offered_overlays()
    for name in offered:
        changed, _ = imgui.checkbox(f"{LABELS[name]}##overlay {name}", name in settings.enabled)
        controls.record(f"overlays/{name}")
        tooltip(HELP[name])
        if changed:
            session.toggle_overlay(name)
        team = session.overlay_provider(name)
        if name not in PHYSICS and team is not None:
            _team_text(interface, team)
    if offered == PHYSICS:
        imgui.text_disabled("No side's controller makes predictions")
        tooltip("A trained Yemong agent predicts; the scripted agent dead reckons")
    providing = [team for team, side in enumerate(sides) if offered_by(side)]
    if providing:
        imgui.separator()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Whose")
        if _choice(
            controls,
            "overlays/side/default",
            "Default",
            settings.side is None,
            "The selected ship's side when it predicts this, else the other side",
        ):
            session.set_overlay_side(None)
        for team in providing:
            if _choice(
                controls,
                f"overlays/side/team{team}",
                TEAM_NAMES[team],
                settings.side == team,
                f"Draw {TEAM_NAMES[team]}'s predictions",
            ):
                session.set_overlay_side(team)
    if any(name in offered for name in TRUTH):
        changed, value = imgui.checkbox("Show the truth##overlay truth", settings.truth)
        controls.record("overlays/truth")
        tooltip("The true positions beside the ghosts, and the true density and its error")
        if changed:
            settings.truth = value
    if "density" in settings.enabled and "density" in offered:
        imgui.separator()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Density")
        for side in DENSITY_SIDES:
            if _choice(
                controls,
                f"overlays/density/{side}",
                side.capitalize(),
                settings.density_side == side,
                f"The predicting side's {side} cells",
            ):
                settings.density_side = side
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Map")
        for shown in DENSITY_SHOWN:
            imgui.begin_disabled(shown != "predicted" and not settings.truth)
            if _choice(
                controls,
                f"overlays/density/{shown}",
                shown.capitalize(),
                settings.density_shown == shown,
                {
                    "predicted": "The head's expected count per cell",
                    "true": "The true count per cell (needs the truth)",
                    "error": "Predicted minus true: over in the prediction colour, under in "
                    "the truth colour (needs the truth)",
                }[shown],
            ):
                settings.density_shown = shown
            imgui.end_disabled()
