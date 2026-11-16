"""The figures' one theme: every chart and still draws its colours, line weights, fonts,
grid and sizes from here, in a light and a dark variant.

The palette is the reference data-visualisation palette (categorical slots in a fixed
order, validated for colour-vision deficiency against both surfaces), stepped separately
for the light and the dark surface: the dark variant is its own steps from the same ramps,
not an inversion.

The active variant is a module state, set with `use`:

    with theme.use("dark"):
        figure, axes = output.new_figure()      # dark surface, dark ink
        axis.plot(xs, ys, color=theme.mode_colour("frontline"))

`theme.INK`, `theme.PREDICTED` and the other role names read the active variant at the
moment they are used (`output.INK` too), so a module that reads them inside its drawing
code follows the variant. A file drawn in the dark variant carries `suffix()` ("_dark")
after its name; the light one has none.

Sizes (`SIZES`): "web" (the docs and the README), "column" (a paper column, 3.4 in) and
"page" (a paper page's text width, 7 in). A chart drawn for the web is fitted to a paper
size by `fit` (fonts and lines keep their point sizes, so they read at print size).
"""

import contextlib
import dataclasses

# Categorical slots, in their fixed order: blue, orange, aqua, yellow, magenta, green,
# violet, red. Light steps for the light surface, dark steps for the dark one.
CATEGORICAL = {
    "light": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#0ca30c",
        "#9085e9",
        "#e66767",
    ),
}

# Team ramps (a still's ships, light to dark along their team's hue): blue is the
# reference sequential ramp, orange the same lightness steps at slot 2's hue. The dark
# variant takes the lighter steps so every ship clears the dark surface.
TEAM_RAMPS = {
    "light": (
        ("#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
        ("#fe743e", "#e15102", "#b43f03", "#8a2e00", "#611e01"),
    ),
    "dark": (
        ("#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"),
        ("#ffbfa9", "#fe9068", "#f55d16", "#cb4801", "#9e3601"),
    ),
}

# Chrome and ink by role.
CHROME = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        secondary="#52514e",
        muted="#52514e",
        grid="#e4e3df",
        baseline="#c3c2b7",
        neutral="#7d7c78",
        field="#8a8984",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        secondary="#c3c2b7",
        muted="#a3a29e",
        grid="#383835",
        baseline="#383835",
        neutral="#8f8e89",
        field="#8f8e89",
    ),
}
VARIANTS = ("light", "dark")

# The modes in their one order (`bnb.figures.ladder.MODES`), each with its categorical slot.
MODE_ORDER = ("frontline", "deathmatch", "elimination")
MODE_SLOTS = {"frontline": 0, "deathmatch": 1, "elimination": 2}
TEAM_NAMES = ("blue team", "orange team")  # team 0's and team 1's, by their colour

# Diverging maps (polarity around a neutral middle): a field's ln n, a signed error.
DIVERGING = "RdBu_r"
# An ordered quantity (a ratio's ramp): perceptual and sequential, sampled short of its pale
# end in the light variant and of its dark end in the dark one, so every line clears the
# surface.
SEQUENTIAL = "viridis"
SEQUENTIAL_RANGE = {"light": (0.0, 0.82), "dark": (0.25, 1.0)}

FONT = ("DejaVu Sans", "sans-serif")  # the system sans; no display or serif face


@dataclasses.dataclass(frozen=True)
class Size:
    """A figure size: its width (in), the raster dpi, the line width and the base font
    size (pt)."""

    width: float
    dpi: int
    line: float
    font: float


SIZES = {
    "web": Size(6.4, 160, 1.0, 8.0),
    "column": Size(3.4, 300, 0.6, 5.5),
    "page": Size(7.0, 300, 0.9, 7.5),
}
WEB_DPI = 150  # a results chart's PNG
GRID_WIDTH = 0.6  # pt, the recessive grid

_ACTIVE = ["light"]


def active() -> str:
    """The active variant's name: "light" or "dark"."""
    return _ACTIVE[-1]


@contextlib.contextmanager
def use(variant: str):
    """Draw in `variant` ("light" or "dark") inside the block: the role names and the
    matplotlib defaults (text, legends, fonts: `rc`) follow it."""
    import matplotlib

    if variant not in VARIANTS:
        raise ValueError(f"theme {variant!r}: one of {VARIANTS}")
    _ACTIVE.append(variant)
    try:
        with matplotlib.rc_context(rc()):
            yield variant
    finally:
        _ACTIVE.pop()


def suffix(variant: str | None = None) -> str:
    """A file name's suffix for the variant: "" for light, "_dark" for dark."""
    return "" if (variant or active()) == "light" else f"_{variant or active()}"


def chrome(role: str, variant: str | None = None) -> str:
    """A chrome colour by role (surface, ink, secondary, muted, grid, baseline, neutral,
    field) in the variant (the active one by default)."""
    return CHROME[variant or active()][role]


def slot(index: int, variant: str | None = None) -> str:
    """The categorical slot `index` (0 first) in the variant."""
    return CATEGORICAL[variant or active()][index]


def mode_colour(mode: str, variant: str | None = None) -> str:
    """A mode's colour (the first mode named in `mode`, so a preset works too); the
    neutral grey for anything else."""
    found = next((name for name in MODE_ORDER if name in mode), None)
    return slot(MODE_SLOTS[found], variant) if found else chrome("neutral", variant)


def team_ramp(team: int, variant: str | None = None) -> tuple[str, ...]:
    """A team's ramp, light to dark, in the variant."""
    return TEAM_RAMPS[variant or active()][team]


def team_colour(team: int, variant: str | None = None) -> str:
    """A team's middle step."""
    ramp = team_ramp(team, variant)
    return ramp[len(ramp) // 2]


def sequential(count: int, variant: str | None = None) -> list:
    """`count` colours along the sequential ramp, lowest first."""
    import matplotlib

    ramp = matplotlib.colormaps[SEQUENTIAL]
    low, high = SEQUENTIAL_RANGE[variant or active()]
    if count == 1:
        return [ramp(low)]
    return [ramp(low + (high - low) * index / (count - 1)) for index in range(count)]


# The role names the figures read (`theme.INK`, `output.INK`): the active variant's.
_ROLES = {
    "SURFACE": lambda: chrome("surface"),
    "INK": lambda: chrome("ink"),
    "MUTED": lambda: chrome("muted"),
    "GRID": lambda: chrome("grid"),
    "BASELINE": lambda: chrome("neutral"),  # a cheap baseline's series
    "FIELD": lambda: chrome("field"),
    "PREDICTED": lambda: slot(0),  # the model
    "TRUE": lambda: slot(1),  # the truth
    "SECOND": lambda: slot(2),  # a third series
}


def __getattr__(name: str):
    if name in _ROLES:
        return _ROLES[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def style(axis) -> None:
    """Recessive axes in the active variant: no top or right spine, muted ticks and
    labels, a light grid under the marks."""
    axis.set_facecolor(chrome("surface"))
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(chrome("muted"))
    axis.tick_params(colors=chrome("muted"), labelsize=8)
    axis.xaxis.label.set_color(chrome("ink"))
    axis.yaxis.label.set_color(chrome("ink"))
    axis.title.set_color(chrome("ink"))
    axis.grid(True, color=chrome("grid"), linewidth=GRID_WIDTH)
    axis.set_axisbelow(True)


def rc() -> dict:
    """matplotlib rc settings for the active variant (text, legends, fonts); `use`
    applies them."""
    return {
        "font.family": FONT,
        "text.color": chrome("ink"),
        "axes.labelcolor": chrome("ink"),
        "axes.edgecolor": chrome("muted"),
        "axes.facecolor": chrome("surface"),
        "figure.facecolor": chrome("surface"),
        "savefig.facecolor": chrome("surface"),
        "xtick.color": chrome("muted"),
        "ytick.color": chrome("muted"),
        "legend.labelcolor": chrome("ink"),
        "legend.facecolor": chrome("surface"),
        "legend.edgecolor": chrome("grid"),
        "grid.color": chrome("grid"),
    }


def fit(figure, size: str) -> None:
    """Resize a drawn figure to a paper size's width (its aspect kept), fonts and lines
    at their point sizes, and lay it out again."""
    width, height = figure.get_size_inches()
    target = SIZES[size].width
    figure.set_size_inches(target, height * target / width)
    try:
        figure.tight_layout()
    except ValueError, RuntimeError:  # a layout that cannot fit: keep the drawn one
        pass
