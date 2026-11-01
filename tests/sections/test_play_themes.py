"""Themes: the shipped files load and validate, hex codes round-trip, and an edited theme
saves and reloads equal."""

import dataclasses
import tomllib

import pytest

from bnb.play import themes
from bnb.play.theme_editor import with_colour
from bnb.play.themes import Colour


def test_shipped_themes_load_and_validate():
    names = themes.available()
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
    path = themes.save(edited, tmp_path)
    assert path.name == "my-sky.toml"
    assert themes.load("my-sky", tmp_path) == edited
    assert themes.available(tmp_path) == ["my-sky"]


def test_a_theme_file_must_list_every_value(tmp_path):
    values = tomllib.loads(themes.to_toml(themes.load(themes.DEFAULT)))
    del values["sky"]
    with pytest.raises(ValueError, match="missing"):
        themes.parse("broken", values)
    values = tomllib.loads(themes.to_toml(themes.load(themes.DEFAULT))) | {"glow": "#FFFFFF"}
    with pytest.raises(ValueError, match="unknown"):
        themes.parse("broken", values)
    with pytest.raises(ValueError):
        themes.save(dataclasses.replace(themes.load(themes.DEFAULT), name="Bad Name"), tmp_path)
