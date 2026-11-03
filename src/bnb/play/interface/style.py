"""The interface's look: Dear ImGui themed to the brainstorm's flat, light style.

Flat fills, thin outlines and small rounding, no gradients or shadows; every colour comes
from the theme (`panel`, `panel_line`, `control`, `control_active`, `text`), so the theme
editor edits the cards too. The interface scale multiplies the font and every size.
Fonts: Roboto (regular and bold), which imgui-bundle ships, at any size (ImGui 1.92 bakes
glyphs as text needs them).
"""

import dataclasses
import pathlib

import imgui_bundle
from imgui_bundle import imgui

from ..themes import Colour, Theme

FONTS = pathlib.Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "Roboto"
FONT_SIZE = 15.0  # pixels at interface scale 1
SCALES = (0.75, 1.0, 1.25, 1.5, 2.0)


@dataclasses.dataclass(frozen=True)
class Fonts:
    """The interface's fonts: regular text and bold headings."""

    regular: imgui.ImFont
    bold: imgui.ImFont


def load_fonts() -> Fonts:
    """Add the fonts to the current ImGui context's atlas (the first added is the default)."""
    atlas = imgui.get_io().fonts
    regular = atlas.add_font_from_file_ttf(str(FONTS / "Roboto-Regular.ttf"), FONT_SIZE)
    bold = atlas.add_font_from_file_ttf(str(FONTS / "Roboto-Bold.ttf"), FONT_SIZE)
    return Fonts(regular, bold)


def mix(first: Colour, second: Colour, amount: float) -> Colour:
    """The colour `amount` of the way from first to second (alpha included)."""
    return Colour.from_floats(
        a + (b - a) * amount for a, b in zip(first.floats(), second.floats(), strict=True)
    )


def with_alpha(colour: Colour, alpha: float) -> Colour:
    """The colour at this opacity (0 to 1)."""
    return dataclasses.replace(colour, alpha=round(255 * alpha))


def vector(colour: Colour) -> imgui.ImVec4:
    """The colour as ImGui's float vector."""
    return imgui.ImVec4(*colour.floats())


def apply_style(theme: Theme, scale: float) -> None:
    """Set the current ImGui context's style from the theme, at this interface scale."""
    style = imgui.get_style()
    style.font_size_base = FONT_SIZE * scale
    sizes = dict(
        window_padding=(10, 9),
        frame_padding=(8, 4),
        item_spacing=(7, 6),
        item_inner_spacing=(6, 4),
        cell_padding=(5, 3),
        indent_spacing=14,
        scrollbar_size=10,
        grab_min_size=10,
        window_rounding=6,
        child_rounding=4,
        frame_rounding=4,
        popup_rounding=5,
        scrollbar_rounding=4,
        grab_rounding=3,
        tab_rounding=4,
        window_border_size=1,
        child_border_size=1,
        popup_border_size=1,
        frame_border_size=0,
        window_min_size=(20, 20),
    )
    for name, value in sizes.items():
        if isinstance(value, tuple):
            setattr(style, name, imgui.ImVec2(value[0] * scale, value[1] * scale))
        else:
            setattr(style, name, value * scale if "border" not in name else value)
    style.anti_aliased_lines = style.anti_aliased_fill = True
    style.image_border_size = 0.0
    hover = mix(theme.control, theme.control_active, 0.5)
    opaque_panel = dataclasses.replace(theme.panel, alpha=255)
    colours = {
        imgui.Col_.text: theme.text,
        imgui.Col_.text_disabled: mix(theme.text, theme.panel_line, 0.6),
        imgui.Col_.window_bg: theme.panel,
        imgui.Col_.child_bg: with_alpha(theme.control, 0.0),
        imgui.Col_.popup_bg: opaque_panel,
        imgui.Col_.border: theme.panel_line,
        imgui.Col_.border_shadow: with_alpha(theme.panel_line, 0.0),
        imgui.Col_.frame_bg: theme.control,
        imgui.Col_.frame_bg_hovered: hover,
        imgui.Col_.frame_bg_active: theme.control_active,
        imgui.Col_.title_bg: opaque_panel,
        imgui.Col_.title_bg_active: opaque_panel,
        imgui.Col_.title_bg_collapsed: opaque_panel,
        imgui.Col_.menu_bar_bg: opaque_panel,
        imgui.Col_.scrollbar_bg: with_alpha(theme.panel_line, 0.0),
        imgui.Col_.scrollbar_grab: theme.panel_line,
        imgui.Col_.scrollbar_grab_hovered: mix(theme.panel_line, theme.text, 0.3),
        imgui.Col_.scrollbar_grab_active: mix(theme.panel_line, theme.text, 0.5),
        imgui.Col_.check_mark: theme.text,
        imgui.Col_.slider_grab: mix(theme.text, theme.panel_line, 0.3),
        imgui.Col_.slider_grab_active: theme.text,
        imgui.Col_.button: theme.control,
        imgui.Col_.button_hovered: hover,
        imgui.Col_.button_active: theme.control_active,
        imgui.Col_.header: theme.control_active,
        imgui.Col_.header_hovered: hover,
        imgui.Col_.header_active: theme.control_active,
        imgui.Col_.separator: theme.panel_line,
        imgui.Col_.separator_hovered: theme.panel_line,
        imgui.Col_.separator_active: theme.panel_line,
        imgui.Col_.resize_grip: with_alpha(theme.panel_line, 0.0),
        imgui.Col_.resize_grip_hovered: with_alpha(theme.panel_line, 0.0),
        imgui.Col_.resize_grip_active: with_alpha(theme.panel_line, 0.0),
        imgui.Col_.tab: theme.control,
        imgui.Col_.tab_hovered: hover,
        imgui.Col_.tab_selected: theme.control_active,
        imgui.Col_.text_selected_bg: with_alpha(theme.control_active, 0.8),
        imgui.Col_.nav_cursor: theme.panel_line,
        imgui.Col_.modal_window_dim_bg: with_alpha(theme.text, 0.12),
        imgui.Col_.table_row_bg: with_alpha(theme.control, 0.0),
        imgui.Col_.table_row_bg_alt: with_alpha(theme.control, 0.5),
        imgui.Col_.table_border_light: theme.panel_line,
        imgui.Col_.table_border_strong: theme.panel_line,
        imgui.Col_.table_header_bg: theme.control,
        imgui.Col_.plot_histogram: theme.text,
    }
    for key, colour in colours.items():
        style.set_color_(key, vector(colour))
