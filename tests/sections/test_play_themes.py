"""Themes: the shipped files load and validate, hex codes round-trip, an edited theme
saves to the user's directory and reloads equal, and the shipped themes stay read-only."""

import dataclasses
import tomllib

import pytest

from bnb.play import themes
from bnb.play.theme_editor import ThemeEditor, with_colour
from bnb.play.themes import Colour


def test_shipped_themes_load_and_validate():
    names = themes.shipped_names()
    assert names[0] == themes.DEFAULT and len(names) >= 3
    for name in names:
        theme = themes.load(name)
        assert theme.name == name
        assert len(theme.team) == len(theme.shield) == len(theme.trail) == 2


@pytest.mark.parametrize("code", ["#1F3A93", "#E8A200", "#000000", "#FFFFFF", "#7A889959"])
def test_hex_round_trips_through_the_colour_type(code):
    colour = Colour.from_hex(code)
    assert colour.hex() == code
    assert Colour.from_floats(colour.floats()) == colour
    assert Colour.from_hex(code.lower()) == colour


@pytest.mark.parametrize("code", ["1F3A93", "#1F3A9", "#GG0000", "#1F3A93F"])
def test_malformed_hex_fails(code):
    with pytest.raises(ValueError):
        Colour.from_hex(code)


def test_an_edited_theme_saves_and_reloads_equal(tmp_path):
    theme = themes.load(themes.DEFAULT)
    edited = with_colour(theme, "sky", Colour.from_hex("#F0F4FA"))
    edited = with_colour(edited, "team", Colour.from_hex("#123456"), team=1)
    edited = with_colour(edited, "vortex", Colour.from_hex("#11223380"))
    edited = dataclasses.replace(edited, name="my-sky")
    library = themes.Library(tmp_path / "user")
    path = library.save(edited)
    assert path == tmp_path / "user" / "my-sky.toml"
    assert library.load("my-sky") == edited
    assert library.names() == themes.shipped_names() + ["my-sky"]


def test_the_shipped_themes_are_read_only(tmp_path):
    shipped = {path: path.read_bytes() for path in themes.SHIPPED.glob("*.toml")}
    library = themes.Library(tmp_path)
    theme = with_colour(themes.load(themes.DEFAULT), "sky", Colour.from_hex("#000000"))
    with pytest.raises(ValueError, match="shipped"):
        library.save(theme)
    library.save(dataclasses.replace(theme, name="night"))
    assert {path: path.read_bytes() for path in themes.SHIPPED.glob("*.toml")} == shipped
    assert themes.USER != themes.SHIPPED and themes.SHIPPED not in themes.USER.parents


def test_a_user_theme_with_a_shipped_name_is_reported_not_loaded(tmp_path):
    dark = with_colour(themes.load(themes.DEFAULT), "sky", Colour.from_hex("#000000"))
    (tmp_path / f"{themes.DEFAULT}.toml").write_text(themes.to_toml(dark))
    library = themes.Library(tmp_path)
    assert library.load(themes.DEFAULT) == themes.read(themes.SHIPPED / f"{themes.DEFAULT}.toml")
    assert library.shadowed() == [themes.DEFAULT]
    assert library.names() == themes.shipped_names()
    assert themes.DEFAULT in ThemeEditor(library).message


def test_replacing_a_user_theme_asks_first(tmp_path):
    library = themes.Library(tmp_path)
    editor = ThemeEditor(library)
    first = dataclasses.replace(themes.load(themes.DEFAULT), name="mine")
    assert editor.save_as(first, "mine") == first
    second = with_colour(first, "sky", Colour.from_hex("#102030"))
    assert editor.save_as(second, "mine") is None  # waits for the answer
    assert editor.pending == second and library.load("mine") == first
    editor.cancel()
    assert editor.pending is None and library.load("mine") == first
    editor.save_as(second, "mine")
    assert editor.confirm() == second and library.load("mine") == second


def test_a_theme_file_must_list_every_value(tmp_path):
    values = tomllib.loads(themes.to_toml(themes.load(themes.DEFAULT)))
    del values["sky"]
    with pytest.raises(ValueError, match="missing"):
        themes.parse("broken", values)
    values = tomllib.loads(themes.to_toml(themes.load(themes.DEFAULT))) | {"glow": "#FFFFFF"}
    with pytest.raises(ValueError, match="unknown"):
        themes.parse("broken", values)
    with pytest.raises(ValueError):
        themes.Library(tmp_path).save(dataclasses.replace(themes.load(themes.DEFAULT),
                                                          name="Bad Name"))
