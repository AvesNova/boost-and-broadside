"""Lineups as the menu's pages choose them: each slot's ship type, the same for both teams,
and the play shell's default."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from bnb.game import api

from .controls import button, tooltip

if TYPE_CHECKING:
    from .interface import Interface

LINEUP_PER_ROW = 5  # lineup dropdowns per row


# The play shell's default pair of types, alternating, in a set that has both.
DEFAULT_PAIR = ("P-51D", "Spitfire XIV")


def default_lineup(size: int, ship_set: str) -> list[int]:
    """The play shell's lineup: `DEFAULT_PAIR` alternating, or the set's first two types
    in a set without both."""
    names = api.type_names(ship_set)
    pair = [names.index(name) for name in DEFAULT_PAIR if name in names]
    if len(pair) < len(DEFAULT_PAIR):
        pair = list(range(min(2, len(names))))
    return [pair[i % len(pair)] for i in range(size)]


def resized_lineup(lineup: list[int], size: int, ship_set: str) -> list[int]:
    """A lineup cut or grown to `size` slots, a new slot taking the default lineup's type."""
    default = default_lineup(size, ship_set)
    return [lineup[i] if i < len(lineup) else default[i] for i in range(size)]


def lineup_row(interface: Interface, name: str, lineup: list[int], ship_set: str, size: int):
    """Each slot's ship type, a dropdown of the set's type names (LINEUP_PER_ROW to a row),
    and Default (the play shell's default lineup, `default_lineup`); `lineup` changes in
    place. Recorded as `<name>/<slot>`, `<name>/<slot>/<type name>` and `<name>/default`."""
    controls = interface.controls
    scale = interface.session.scale
    names = api.type_names(ship_set)
    spacing = 3.0 * scale
    imgui.begin_group()
    width = (imgui.get_content_region_avail().x - (LINEUP_PER_ROW - 1) * spacing) / (LINEUP_PER_ROW)
    for slot, ship_type in enumerate(lineup):
        if slot % LINEUP_PER_ROW:
            imgui.same_line(0.0, spacing)
        ship_type = min(ship_type, len(names) - 1)
        imgui.set_next_item_width(width)
        slot_name = f"{name}/{slot}"
        opened = imgui.begin_combo(
            f"##{name}{slot}",
            names[ship_type],
            imgui.ComboFlags_.no_arrow_button | imgui.ComboFlags_.height_largest,
        )
        controls.record(slot_name)
        tooltip(f"Slot {slot + 1}: {names[ship_type]}, on both teams")
        if opened:
            for kind, type_label in enumerate(names):
                if imgui.selectable(type_label, kind == ship_type)[0]:
                    lineup[slot] = kind
                controls.record(f"{slot_name}/{type_label}")
            imgui.end_combo()
    if button(
        controls,
        f"{name}/default",
        "Default",
        "The P-51D and the Spitfire XIV alternating (else the set's first two types)",
    ):
        lineup[:] = default_lineup(size, ship_set)
    imgui.end_group()
