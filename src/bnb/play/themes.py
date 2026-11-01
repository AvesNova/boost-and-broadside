"""Themes: the colours the renderer draws with, as data files in `themes/` (TOML).

Every colour is a hex code, "#RRGGBB" or "#RRGGBBAA" (alpha where it matters). A theme
file lists every field of `Theme`, no more and no fewer; `load` checks that. The colours
reach the shaders as uniforms each frame, so an edited theme draws on the next frame.
"""

import dataclasses
import pathlib
import re
import tomllib

THEMES = pathlib.Path(__file__).parent / "themes"
DEFAULT = "mint-pink"
HEX = re.compile(r"#([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?")
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


@dataclasses.dataclass(frozen=True)
class Colour:
    """An sRGB colour with alpha, 8 bits per channel (what a hex code holds)."""

    red: int
    green: int
    blue: int
    alpha: int = 255

    @classmethod
    def from_hex(cls, code: str) -> Colour:
        match = HEX.fullmatch(code)
        if match is None:
            raise ValueError(f"{code!r} is not a colour (#RRGGBB or #RRGGBBAA)")
        digits = match.group(1) + (match.group(2) or "FF")
        return cls(*(int(digits[i:i + 2], 16) for i in range(0, 8, 2)))

    @classmethod
    def from_floats(cls, values) -> Colour:
        """From 3 or 4 floats in [0, 1] (alpha 1 when there are 3), rounded to 8 bits."""
        channels = [min(max(round(float(v) * 255.0), 0), 255) for v in values]
        return cls(*channels)

    def hex(self) -> str:
        code = f"#{self.red:02X}{self.green:02X}{self.blue:02X}"
        return code if self.alpha == 255 else code + f"{self.alpha:02X}"

    def floats(self) -> tuple[float, float, float, float]:
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0, self.alpha / 255.0)


def _colour(label: str, alpha: bool = False):
    return dataclasses.field(metadata={"label": label, "alpha": alpha})


def _pair(label: str, alpha: bool = False):
    return dataclasses.field(metadata={"label": label, "alpha": alpha, "pair": True})


@dataclasses.dataclass(frozen=True)
class Theme:
    """A whole look. Colours are `Colour`s; a pair holds team 0's then team 1's. fog
    multiplies what a team cannot see (#999999 darkens to 60%, saturation kept).
    fill_strength: how far a field's deepest fill goes from the sky toward its air colour.
    vortex_threshold: the angle of attack, as a fraction of the stall angle, beyond which
    a wingtip vortex trails."""

    name: str
    sky: Colour = _colour("sky")
    outside: Colour = _colour("outside the arena")
    boundary: Colour = _colour("boundary hatching")
    thin_air: Colour = _colour("thin air")
    thick_air: Colour = _colour("thick air")
    thin_contour: Colour = _colour("thin-air contours", alpha=True)
    thick_contour: Colour = _colour("thick-air contours", alpha=True)
    thin_core: Colour = _colour("thin-air core line")
    thick_core: Colour = _colour("thick-air core line")
    fog: Colour = _colour("fog darkening")
    neutral: Colour = _colour("neutral zone")
    team: tuple[Colour, Colour] = _pair("team")
    outline: Colour = _colour("outline")
    shield: tuple[Colour, Colour] = _pair("shield")
    energy: Colour = _colour("energy")
    trail: tuple[Colour, Colour] = _pair("trail", alpha=True)
    vortex: Colour = _colour("wingtip vortex", alpha=True)
    text: Colour = _colour("HUD text")
    fill_strength: float = dataclasses.field(metadata={"label": "fill strength"})
    vortex_threshold: float = dataclasses.field(metadata={"label": "vortex threshold"})


NUMBER_FIELDS = ("fill_strength", "vortex_threshold")
COLOUR_FIELDS = tuple(f.name for f in dataclasses.fields(Theme)
                      if f.name != "name" and f.name not in NUMBER_FIELDS)


def parse(name: str, values: dict) -> Theme:
    """A theme from a file's table; fails on a missing, unknown or malformed value."""
    expected = {f.name for f in dataclasses.fields(Theme)} - {"name"}
    missing, unknown = expected - values.keys(), values.keys() - expected
    if missing or unknown:
        raise ValueError(f"theme {name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    out = {}
    for field in dataclasses.fields(Theme):
        if field.name == "name":
            continue
        value = values[field.name]
        if field.name in NUMBER_FIELDS:
            if not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
                raise ValueError(f"theme {name}: {field.name} must be a number in [0, 1]")
            out[field.name] = float(value)
        elif field.metadata.get("pair"):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"theme {name}: {field.name} needs two colours, one per team")
            out[field.name] = tuple(Colour.from_hex(v) for v in value)
        else:
            out[field.name] = Colour.from_hex(value)
    return Theme(name=name, **out)


def load(name: str, directory: pathlib.Path = THEMES) -> Theme:
    """The theme in `directory/name.toml`."""
    with open(directory / f"{name}.toml", "rb") as file:
        return parse(name, tomllib.load(file))


def available(directory: pathlib.Path = THEMES) -> list[str]:
    """The theme names in a directory, the default first, then alphabetical."""
    names = sorted(path.stem for path in directory.glob("*.toml"))
    return sorted(names, key=lambda n: n != DEFAULT)


def to_toml(theme: Theme) -> str:
    lines = [f"# Theme {theme.name}, saved by the play shell's theme editor."]
    for field in dataclasses.fields(Theme):
        if field.name == "name":
            continue
        value = getattr(theme, field.name)
        if field.name in NUMBER_FIELDS:
            text = repr(value)
        elif field.metadata.get("pair"):
            text = "[" + ", ".join(f'"{colour.hex()}"' for colour in value) + "]"
        else:
            text = f'"{value.hex()}"'
        lines.append(f"{field.name} = {text}")
    return "\n".join(lines) + "\n"


def save(theme: Theme, directory: pathlib.Path = THEMES) -> pathlib.Path:
    """Write the theme to `directory/<its name>.toml`; returns the path. Names are
    lowercase words joined by hyphens."""
    if NAME.fullmatch(theme.name) is None:
        raise ValueError(f"theme name {theme.name!r}: use lowercase words joined by hyphens")
    path = directory / f"{theme.name}.toml"
    path.write_text(to_toml(theme))
    return path
