"""The theme editor: an ImGui panel with a picker for every colour of the current theme.

Each colour shows its hex code, which can be typed (`#1F3A93`, or `#RRGGBBAA` where alpha
matters); the swatch opens ImGui's picker. An edit makes a new `Theme`, which the next
frame draws with. "Save as theme" writes it to the user's themes under a new name, and
asks before it replaces a user theme of that name; "Revert" goes back to the theme's file.
"""

import dataclasses

from imgui_bundle import imgui

from .themes import COLOUR_FIELDS, NUMBER_FIELDS, Colour, Library, Theme

HEX_FLAGS = imgui.ColorEditFlags_.display_hex | imgui.ColorEditFlags_.input_rgb
ALPHA_FLAGS = HEX_FLAGS | imgui.ColorEditFlags_.alpha_bar | imgui.ColorEditFlags_.alpha_preview_half
PANEL_WIDTH = 440.0
LABEL_WIDTH = 170.0  # pixels kept for the labels right of the fields


def with_colour(theme: Theme, name: str, colour: Colour, team: int | None = None) -> Theme:
    """The theme with one colour replaced (team: which of a pair)."""
    if team is None:
        return dataclasses.replace(theme, **{name: colour})
    pair = list(getattr(theme, name))
    pair[team] = colour
    return dataclasses.replace(theme, **{name: tuple(pair)})


class ThemeEditor:
    """open: whether the panel shows. name: the name field for "Save as theme". pending:
    a theme waiting for the user to confirm it replaces the user theme of its name."""

    def __init__(self, library: Library) -> None:
        self.library = library
        self.open = False
        self.name = ""
        self.pending: Theme | None = None
        shadowed = library.shadowed()
        self.message = (
            f"ignored user themes with shipped names: {', '.join(shadowed)}" if shadowed else ""
        )

    def save_as(self, theme: Theme, name: str) -> Theme | None:
        """Save the theme under `name`; returns the saved theme, or None when nothing was
        written: the name is invalid (`message` says why), or a user theme has it, which
        waits in `pending` for `confirm`."""
        named = dataclasses.replace(theme, name=name.strip())
        try:
            path = self.library.user_path(named.name)
        except ValueError as error:
            self.message = str(error)
            return None
        if path.exists():
            self.pending, self.message = named, f"replace your theme {named.name}?"
            return None
        return self._write(named)

    def confirm(self) -> Theme | None:
        """Replace the user theme with the pending one; returns it."""
        named, self.pending = self.pending, None
        return None if named is None else self._write(named)

    def cancel(self) -> None:
        """Drop the pending theme without saving it."""
        self.pending, self.message = None, "not saved"

    def _write(self, theme: Theme) -> Theme:
        path = self.library.save(theme)
        self.message = f"saved {path}"
        return theme

    def _colour(self, theme: Theme, field: dataclasses.Field, team: int | None) -> Theme:
        value = getattr(theme, field.name)
        colour = value if team is None else value[team]
        label = field.metadata["label"] + ("" if team is None else f" {team}")
        if field.metadata["alpha"]:
            changed, floats = imgui.color_edit4(label, list(colour.floats()), ALPHA_FLAGS)
        else:
            changed, floats = imgui.color_edit3(
                label, list(colour.floats()[:3]), HEX_FLAGS | imgui.ColorEditFlags_.no_alpha
            )
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
        expanded, self.open = imgui.begin(f"Theme: {theme.name}###theme editor", self.open)
        if expanded:
            imgui.push_item_width(-LABEL_WIDTH)
            fields = {field.name: field for field in dataclasses.fields(Theme)}
            for name in COLOUR_FIELDS:
                field = fields[name]
                teams = (0, 1) if field.metadata.get("pair") else (None,)
                for team in teams:
                    theme = self._colour(theme, field, team)
            for name in NUMBER_FIELDS:
                changed, value = imgui.slider_float(
                    fields[name].metadata["label"], getattr(theme, name), 0.0, 1.0
                )
                if changed:
                    theme = dataclasses.replace(theme, **{name: value})
            imgui.separator()
            _, self.name = imgui.input_text("name", self.name)
            if self.pending is None:
                if imgui.button("Save as theme"):
                    saved = self.save_as(theme, self.name)
                imgui.same_line()
                if imgui.button("Revert"):
                    theme, self.message = original, f"reverted to {original.name}"
            else:
                if imgui.button("Replace"):
                    saved = self.confirm()
                imgui.same_line()
                if imgui.button("Cancel"):
                    self.cancel()
            if saved is not None:
                theme = saved
            if self.message:
                imgui.text_wrapped(self.message)
            imgui.pop_item_width()
        imgui.end()
        return theme, saved
