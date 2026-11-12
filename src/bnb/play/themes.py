"""Themes: the colours the renderer draws with, and the rest of the play shell's look,
as data files (TOML).

Every colour is a hex code, "#RRGGBB" or "#RRGGBBAA" (alpha where it matters). The look
settings (`LOOK_FIELDS`: true size, the dot grid's pattern, the contours' shapes, the
look-ahead, the bullets' look) are plain values. A theme file lists every colour and
number of `Theme` and no unknown value; `load` checks that. A look setting a file
omits takes its default (`LOOK_DEFAULTS`), so a theme written before a setting existed
still loads and looks as it did. The values reach the shaders as uniforms each frame,
so an edited theme draws on the next frame.

Themes live in two places (`Library`): the shipped ones in the package's `themes/`, which
are read-only, and the user's in a per-user configuration directory, where "Save as"
writes. A shipped name always means the shipped theme: a user file with that name is
listed by `Library.shadowed` and not loaded.
"""

import dataclasses
import pathlib
import re
import tomllib

import platformdirs

SHIPPED = pathlib.Path(__file__).parent / "themes"
USER = pathlib.Path(platformdirs.user_config_dir("boost-and-broadside")) / "themes"
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
        """The colour of a hex code, "#RRGGBB" or "#RRGGBBAA"."""
        match = HEX.fullmatch(code)
        if match is None:
            raise ValueError(f"{code!r} is not a colour (#RRGGBB or #RRGGBBAA)")
        digits = match.group(1) + (match.group(2) or "FF")
        return cls(*(int(digits[i : i + 2], 16) for i in range(0, 8, 2)))

    @classmethod
    def from_floats(cls, values) -> Colour:
        """From 3 or 4 floats in [0, 1] (alpha 1 when there are 3), rounded to 8 bits."""
        channels = [min(max(round(float(value) * 255.0), 0), 255) for value in values]
        return cls(*channels)

    def hex(self) -> str:
        """The colour's hex code ("#RRGGBB", or "#RRGGBBAA" when it is not opaque)."""
        code = f"#{self.red:02X}{self.green:02X}{self.blue:02X}"
        return code if self.alpha == 255 else code + f"{self.alpha:02X}"

    def floats(self) -> tuple[float, float, float, float]:
        """The colour as four floats in [0, 1], alpha last."""
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0, self.alpha / 255.0)


def _colour(label: str, alpha: bool = False):
    return dataclasses.field(metadata={"label": label, "alpha": alpha})


def _pair(label: str, alpha: bool = False):
    return dataclasses.field(metadata={"label": label, "alpha": alpha, "pair": True})


GRID_PATTERNS = ("off", "square", "triangle", "hex")  # `render.grid.PATTERNS` and off
CONTOUR_SETS = (  # `render.contours.SHAPE_SETS` and off
    "off",
    "fields",
    "edge",
    "fields+edge",
    "fields+edge+zones",
    "zones",
)
LOOK_AHEADS = ("off", "flying", "always")  # when the camera leads a followed ship
BULLET_STYLES = ("tracer", "trail")  # `render.bullets.STYLES`


def _look(label: str, default):
    return dataclasses.field(default=default, metadata={"label": label, "look": True})


@dataclasses.dataclass(frozen=True)
class Theme:
    """A whole look. Colours are `Colour`s; a pair holds team 0's then team 1's. fog
    multiplies what a team cannot see (#999999 darkens to 60%, saturation kept). grid
    colours the dot grid over the sky and contours the distance contours, each alpha how
    faint (`render.grid`, `render.contours`). The
    interface's cards draw with panel (their fill, semi-opaque), panel_line (outlines and
    separators), control (buttons and fields), control_active (a pressed or chosen control)
    and the HUD text; selection outlines the selected ship, field or zone. The prediction
    overlays draw an agent's predictions in prediction and the truth beside them in truth
    (`bnb.play.overlays`).
    fill_strength: how far a field's deepest fill goes from the sky toward its air colour.
    vortex_threshold: the angle of attack, as a fraction of the stall angle, beyond which
    a wingtip vortex trails.
    The look settings, after the colours: true_size (ships at their size in metres, no
    minimum on screen), grid_pattern (the dot grid's pattern, or "off"; grid is its
    colour), contour_shapes (the distance contours' shape set, or "off"), look_ahead
    (when the camera leads a followed ship: "off", "flying", "always") and the
    bullets' look (`render.bullets.BulletLook`): bullet_style ("tracer" or "trail"),
    bullet_head_size (the head's radius), the trail's width at the head and at the
    tail (screen pixels) and bullet_trail_seconds."""

    name: str
    sky: Colour = _colour("sky")
    outside: Colour = _colour("outside the arena")
    boundary: Colour = _colour("boundary hatching")
    grid: Colour = _colour("grid dots", alpha=True)
    contours: Colour = _colour("distance contours", alpha=True)
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
    panel: Colour = _colour("panel", alpha=True)
    panel_line: Colour = _colour("panel lines")
    control: Colour = _colour("controls")
    control_active: Colour = _colour("active control")
    selection: Colour = _colour("selection")
    prediction: Colour = _colour("predictions")
    truth: Colour = _colour("truth")
    fill_strength: float = dataclasses.field(metadata={"label": "fill strength"})
    vortex_threshold: float = dataclasses.field(metadata={"label": "vortex threshold"})
    true_size: bool = _look("ships at true size", False)
    grid_pattern: str = _look("dot grid pattern", "triangle")
    contour_shapes: str = _look("distance contour shapes", "off")
    look_ahead: str = _look("camera look-ahead", "always")
    bullet_style: str = _look("bullet style", "tracer")
    bullet_head_size: float = _look("bullet head radius (px)", 2.5)
    bullet_trail_head_width: float = _look("bullet trail width at the head (px)", 5.0)
    bullet_trail_tail_width: float = _look("bullet trail width at the tail (px)", 0.5)
    bullet_trail_seconds: float = _look("bullet trail length (s)", 0.04)


NUMBER_FIELDS = ("fill_strength", "vortex_threshold")
LOOK_FIELDS = tuple(f.name for f in dataclasses.fields(Theme) if f.metadata.get("look"))
LOOK_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Theme) if f.name in LOOK_FIELDS}
LOOK_CHOICES = {
    "grid_pattern": GRID_PATTERNS,
    "contour_shapes": CONTOUR_SETS,
    "look_ahead": LOOK_AHEADS,
    "bullet_style": BULLET_STYLES,
}
COLOUR_FIELDS = tuple(
    field.name
    for field in dataclasses.fields(Theme)
    if field.name != "name" and field.name not in NUMBER_FIELDS and field.name not in LOOK_FIELDS
)


def _look_value(name: str, field: str, value):
    """A look setting from a file's value; fails when it is not of the setting's kind."""
    default = LOOK_DEFAULTS[field]
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"theme {name}: {field} must be true or false")
        return value
    if field in LOOK_CHOICES:
        if value not in LOOK_CHOICES[field]:
            raise ValueError(f"theme {name}: {field} must be one of {list(LOOK_CHOICES[field])}")
        return value
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0.0:
        raise ValueError(f"theme {name}: {field} must be a number, zero or more")
    return float(value)


def parse(name: str, values: dict) -> Theme:
    """A theme from a file's table; fails on a missing colour or number, an unknown value
    or a malformed one. A missing look setting takes its default."""
    expected = {field.name for field in dataclasses.fields(Theme)} - {"name"}
    missing = expected - values.keys() - set(LOOK_FIELDS)
    unknown = values.keys() - expected
    if missing or unknown:
        raise ValueError(f"theme {name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    out = {}
    for field in dataclasses.fields(Theme):
        if field.name == "name":
            continue
        if field.name in LOOK_FIELDS:
            if field.name in values:
                out[field.name] = _look_value(name, field.name, values[field.name])
            continue
        value = values[field.name]
        if field.name in NUMBER_FIELDS:
            if not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
                raise ValueError(f"theme {name}: {field.name} must be a number in [0, 1]")
            out[field.name] = float(value)
        elif field.metadata.get("pair"):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"theme {name}: {field.name} needs two colours, one per team")
            out[field.name] = tuple(Colour.from_hex(code) for code in value)
        else:
            out[field.name] = Colour.from_hex(value)
    return Theme(name=name, **out)


def read(path: pathlib.Path) -> Theme:
    """The theme in a file, named for its stem."""
    with open(path, "rb") as file:
        return parse(path.stem, tomllib.load(file))


def _names(directory: pathlib.Path) -> set[str]:
    return {path.stem for path in directory.glob("*.toml")} if directory.is_dir() else set()


def shipped_names() -> list[str]:
    """The shipped themes, the default first, then alphabetical."""
    return sorted(_names(SHIPPED), key=lambda name: (name != DEFAULT, name))


@dataclasses.dataclass(frozen=True)
class Library:
    """The theme files: the shipped ones (read-only) and the user's (where `save`
    writes). A test passes its own user directory."""

    user_directory: pathlib.Path = USER

    def names(self) -> list[str]:
        """Every theme: the shipped ones (`shipped_names`), then the user's,
        alphabetical."""
        return shipped_names() + sorted(_names(self.user_directory) - _names(SHIPPED))

    def shadowed(self) -> list[str]:
        """User themes not loaded because a shipped theme has their name."""
        return sorted(_names(self.user_directory) & _names(SHIPPED))

    def path(self, name: str) -> pathlib.Path:
        """The file a theme name loads from: the shipped one, else the user's."""
        shipped = SHIPPED / f"{name}.toml"
        return shipped if shipped.is_file() else self.user_directory / f"{name}.toml"

    def load(self, name: str) -> Theme:
        """The theme of this name (`path` says which file)."""
        return read(self.path(name))

    def user_path(self, name: str) -> pathlib.Path:
        """Where `save` writes a theme of this name. Fails for a name that is not
        lowercase words joined by hyphens, or that a shipped theme has."""
        if NAME.fullmatch(name) is None:
            raise ValueError(f"theme name {name!r}: use lowercase words joined by hyphens")
        if (SHIPPED / f"{name}.toml").is_file():
            raise ValueError(f"theme name {name!r} is a shipped theme's; choose another")
        return self.user_directory / f"{name}.toml"

    def save(self, theme: Theme) -> pathlib.Path:
        """Write the theme to the user's directory (created if missing), replacing any
        file of its name; returns the path."""
        path = self.user_path(theme.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_toml(theme))
        return path


def load(name: str) -> Theme:
    """A theme by name, from the shipped themes or the user's (`Library()`)."""
    return Library().load(name)


def to_toml(theme: Theme) -> str:
    """The theme as a theme file's text, every value listed."""
    lines = [f"# Theme {theme.name}, saved by the play shell's theme editor."]
    for field in dataclasses.fields(Theme):
        if field.name == "name":
            continue
        value = getattr(theme, field.name)
        if field.name in LOOK_FIELDS:
            text = (
                repr(value).lower()
                if isinstance(value, bool)
                else (repr(value) if isinstance(value, float) else f'"{value}"')
            )
        elif field.name in NUMBER_FIELDS:
            text = repr(value)
        elif field.metadata.get("pair"):
            text = "[" + ", ".join(f'"{colour.hex()}"' for colour in value) + "]"
        else:
            text = f'"{value.hex()}"'
        lines.append(f"{field.name} = {text}")
    return "\n".join(lines) + "\n"
