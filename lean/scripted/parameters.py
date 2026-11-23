"""The scripted agent's parameters: one table of named rows, the yardstick and its personas.

Each row is a dimensionless number: a multiple of one named reference of the ship's own
performance (a game constant by its name in `state.constants`), or an angle in radians, a
store fraction, a probability, a ratio (core-decisions.md "Scripted BFM"). Each row has its
own box for the persona population: at spread s it is drawn uniformly around its default,
`width` times s either side (in natural-log units for the "log" law), independently of every
other row. The yardstick (spread 0) reads the defaults as Python numbers; a persona reads (E,
N) tensors under the same names.

The logistic gates are named too (`GATES`): each a quantity's natural log against a centre
row, with a fixed scale (log units) and a direction; the persona's temperature divides the
sharpness of every gate it applies to.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from lean.contract import random_keywords

LOG, LINEAR, UPWARD, FIXED = "log", "linear", "upward", "fixed"


@dataclass(frozen=True)
class Parameter:
    """One row: its default, the reference it is a multiple of, and its box. Laws at spread s,
    u uniform on [-1, 1] (u01 on [0, 1]): "log" default x e^(s width u); "linear" default +
    s width u; "upward" default + s width u01; "fixed" the default for every persona."""

    default: float
    reference: str
    law: str = FIXED
    width: float = 0.0
    meaning: str = ""


@dataclass(frozen=True)
class Gate:
    """A logistic gate: P = floor + (1 - 2 floor) sigmoid(sharpness ln(quantity / centre)),
    sharpness +-1 / (scale x temperature), positive when the gate rises with its quantity.
    `centre` names the parameter row; `tempered`: the persona's temperature divides it."""

    centre: str
    scale: float
    rising: bool
    floor: float = 0.0
    tempered: bool = True


# fmt: off
PARAMETERS: dict[str, Parameter] = {
    # Gate centres (the gates below).
    "turn": Parameter(0.002868, "radians", LOG, 0.4, "pointing error: turn at all"),
    "level_two": Parameter(0.02097, "radians", LOG, 0.4, "pointing error: sustained stick"),
    "level_three": Parameter(0.1044, "radians", LOG, 0.4, "pointing error: the full stick"),
    "dive_slow": Parameter(1.8, "stall_speed", LOG, 0.4, "speed: dive because slow"),
    "dive_in_turn": Parameter(0.95, "corner_speed", LOG, 0.4, "speed: dive below it, turning hard"),
    "climb_fast": Parameter(1.2, "corner_speed", LOG, 0.4, "speed: climb because fast"),
    "climb_close": Parameter(0.21, "gun_range", LOG, 0.4, "distance: climb because very close"),
    "regain_far": Parameter(3.0, "gun_range", LOG, 0.4, "distance: far from every known enemy"),
    "regain_low": Parameter(0.3, "store fraction", LOG, 0.4, "store: low"),
    "aligned": Parameter(1.01, "target angular size", LOG, 0.4, "gun aim error: the gun bears"),
    "in_range": Parameter(1.04, "gun_range", LOG, 0.4, "distance: the target is within reach"),
    # Decisiveness: divides every tempered gate's sharpness.
    "temperature": Parameter(1.0, "ratio", LOG, 0.4, "softens every tempered gate"),
    # Probability floors.
    "side_floor": Parameter(0.02, "probability", meaning="turning the long way round"),
    "power_floor": Parameter(0.02, "probability", meaning="each of coast, dive and climb"),
    "regain_share": Parameter(0.5, "probability", LINEAR, 0.25,
                              "far and low on store: climb at most this share of the ticks"),
    "head_on_band": Parameter(0.1, "radians", meaning="want this near behind: kept side"),
    # Assessment (lean.scripted.situation).
    "turning_floor": Parameter(0.25, "sustained_rate", LOG, 0.4,
                               "a nose rate this small counts as straight"),
    # Target choice.
    "target_keep_factor": Parameter(0.8, "ratio", meaning="switch only to one this much closer"),
    "weakest_weight": Parameter(0.0, "ratio", UPWARD, 1.0,
                                "0 the nearest enemy; larger prefers enemies with less store"),
    # Lead: first-order intercept with the drag law's time of flight in the medium.
    "lead_cap": Parameter(4.78, "gun_range", meaning="the lead's distance is capped here"),
    "lead_gain": Parameter(1.0, "ratio", LINEAR, 0.2, "share of the computed lead applied"),
    # Boundary steer: inside the margin of the edge and flying outward, a steer to the centre
    # blended in, its weight ramping from 0 to 1 between two depths (0 at the margin's inner
    # edge, 1 at the arena edge). The margin is in turn radii at the ship's speed: the
    # sustained radius up to corner speed, growing as speed squared above it.
    "boundary_margin": Parameter(2.2, "turn radius", LOG, 0.3,
                                 "depth of the edge band"),
    "boundary_ramp_low": Parameter(0.0554, "margin fraction", meaning="the steer starts"),
    "boundary_ramp_high": Parameter(0.52, "margin fraction", meaning="the steer is whole"),
    # The gun.
    "target_size_minimum": Parameter(math.radians(1.0), "radians", meaning="target size clamp"),
    "target_size_maximum": Parameter(math.radians(45.0), "radians", meaning="target size clamp"),
    "ally_clearance": Parameter(2.0, "hit_radius", meaning="a team-mate this near the line blocks"),
}

GATES: dict[str, Gate] = {
    "turn": Gate("turn", 0.172, True, 0.02),
    "level_two": Gate("level_two", 0.328, True, 0.02),
    "level_three": Gate("level_three", 0.267, True, 0.02),
    "dive_slow": Gate("dive_slow", 0.05, False),
    "dive_in_turn": Gate("dive_in_turn", 0.05, False),
    "climb_fast": Gate("climb_fast", 0.05, True),
    "climb_close": Gate("climb_close", 0.16, False),
    "regain_far": Gate("regain_far", 0.2, True),
    "regain_low": Gate("regain_low", 0.2, False),
    "aligned": Gate("aligned", 0.243, False),
    "in_range": Gate("in_range", 0.127, False),
}
# fmt: on
SAMPLED = tuple(name for name, row in PARAMETERS.items() if row.law != FIXED)


def _defaults() -> dict[str, float]:
    return {name: row.default for name, row in PARAMETERS.items()}


@dataclass
class ScriptedConfig:
    """The yardstick: a value for every row of `PARAMETERS` (the defaults unless replaced,
    `ScriptedConfig().replace(name=value)`) and the persona population's spread law. Holds no
    ship characteristic: those are the game's constants."""

    values: dict[str, float] = field(default_factory=_defaults)
    # Stick layout: True when even stick indices turn left (counter-clockwise).
    left_is_even: bool = True

    def __post_init__(self):
        unknown = set(self.values) ^ set(PARAMETERS)
        if unknown:
            raise ValueError(f"scripted parameters not in the table: {sorted(unknown)}")

    def replace(self, **values: float) -> ScriptedConfig:
        """A copy with some rows changed (by name)."""
        return ScriptedConfig({**self.values, **values}, self.left_is_even)


def personas(
    config: ScriptedConfig,
    envs: int,
    slots: int,
    generator: torch.Generator | None,
    device: torch.device | str,
    spread: float,
) -> dict[str, Tensor] | None:
    """Per-ship parameters, drawn once per match.

    spread = 0 is the yardstick: returns None and uses no draw (the opponent reads the
    config's numbers, no per-ship tensors). 0 < spread <= 1 draws every sampled row (`SAMPLED`)
    of every ship independently on its box (`Parameter`), around the config's value: a dense
    population whose every member stays competent at spread 1.

    Returns {name: (E, N) float32 on `device`} for every sampled row.
    """
    if not 0.0 <= spread <= 1.0:
        raise ValueError(f"persona spread {spread} outside [0, 1]")
    if spread == 0.0:
        return None
    unit = torch.rand((envs, slots, len(SAMPLED)), **random_keywords(generator, device))
    drawn = {}
    for column, name in enumerate(SAMPLED):
        row, value, u = PARAMETERS[name], config.values[name], unit[..., column]
        reach = spread * row.width
        if row.law == LOG:
            drawn[name] = value * torch.exp(reach * (2.0 * u - 1.0))
        elif row.law == LINEAR:
            drawn[name] = value + reach * (2.0 * u - 1.0)
        else:  # UPWARD
            drawn[name] = value + reach * u
    return drawn


def value(config: ScriptedConfig, persona: dict[str, Tensor] | None, name: str) -> Tensor | float:
    """A row's value: the persona's (E, N) tensor when it samples the row, else the config's
    number."""
    if persona is not None and name in persona:
        return persona[name]
    return config.values[name]
