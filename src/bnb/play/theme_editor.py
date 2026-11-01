"""The theme editor: an ImGui panel with a picker for every colour of the current theme.

Each colour shows its hex code, which can be typed (`#1F3A93`, or `#RRGGBBAA` where alpha
matters); the swatch opens ImGui's picker. An edit makes a new `Theme`, which the next
frame draws with. "Save as theme" writes it under a new name; "Revert" goes back to the
theme's file.
"""

import dataclasses
import pathlib

from imgui_bundle import imgui

from .themes import COLOUR_FIELDS, NUMBER_FIELDS, THEMES, Colour, Theme, save

HEX_FLAGS = imgui.ColorEditFlags_.display_hex | imgui.ColorEditFlags_.input_rgb
ALPHA_FLAGS = HEX_FLAGS | imgui.ColorEditFlags_.alpha_bar | imgui.ColorEditFlags_.alpha_preview_half
PANEL_WIDTH = 380.0


def with_colour(theme: Theme, name: str, colour: Colour, team: int | None = None) -> Theme:
    """The theme with one colour replaced (team: which of a pair)."""
    if team is None:
        return dataclasses.replace(theme, **{name: colour})
    pair = list(getattr(theme, name))
    pair[team] = colour
    return dataclasses.replace(theme, **{name: tuple(pair)})


class ThemeEditor:
    """open: whether the panel shows. name: the name field for "Save as theme"."""

    def __init__(self, directory: pathlib.Path = THEMES) -> None:
        self.directory = directory
        self.open = False
        self.name = ""
        self.message = ""

    def _colour(self, theme: Theme, field: dataclasses.Field, team: int | None) -> Theme:
        value = getattr(theme, field.name)
        colour = value if team is None else value[team]
        label = field.metadata["label"] + ("" if team is None else f" {team}")
        if field.metadata["alpha"]:
            changed, floats = imgui.color_edit4(label, list(colour.floats()), ALPHA_FLAGS)
        else:
            changed, floats = imgui.color_edit3(label, list(colour.floats()[:3]),
                                                HEX_FLAGS | imgui.ColorEditFlags_.no_alpha)
        if not changed:
            return theme
        return with_colour(theme, field.name, Colour.from_floats(floats), team)

    def draw(self, theme: Theme, original: Theme) -> tuple[Theme, Theme | None]:
        """The panel, in the current ImGui frame. Returns the (possibly edited) theme, and
        the theme just saved to a file, if any."""
        if not self.open:
            return theme, None
        saved = None
        width, _ = imgui.get_io().display_size
        imgui.set_next_window_pos((width - PANEL_WIDTH - 10.0, 10.0), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((PANEL_WIDTH, 0.0), imgui.Cond_.first_use_ever)
        expanded, self.open = imgui.begin(f"Theme: {theme.name}", self.open)
        if expanded:
            fields = {f.name: f for f in dataclasses.fields(Theme)}
            for name in COLOUR_FIELDS:
                field = fields[name]
                teams = (0, 1) if field.metadata.get("pair") else (None,)
                for team in teams:
                    theme = self._colour(theme, field, team)
            for name in NUMBER_FIELDS:
                changed, value = imgui.slider_float(fields[name].metadata["label"],
                                                    getattr(theme, name), 0.0, 1.0)
                if changed:
                    theme = dataclasses.replace(theme, **{name: value})
            imgui.separator()
            _, self.name = imgui.input_text("name", self.name)
            if imgui.button("Save as theme"):
                try:
                    saved = dataclasses.replace(theme, name=self.name.strip())
                    path = save(saved, self.directory)
                    theme, self.message = saved, f"saved {path.name}"
                except ValueError as error:
                    saved, self.message = None, str(error)
            imgui.same_line()
            if imgui.button("Revert"):
                theme, self.message = original, f"reverted to {original.name}"
            if self.message:
                imgui.text_wrapped(self.message)
        imgui.end()
        return theme, saved
