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
# Defaults: BFM-CMA2, the provisional yardstick (author 2026-11-24; values from
# docs/rewrite/sections/scripted-screening/cma_candidate_round2.json, a CMA search around the
# earlier defaults). Provisional: they change when a better search lands.
PARAMETERS: dict[str, Parameter] = {
    # Target choice.
    "target_keep_factor": Parameter(0.8, "ratio", meaning="switch only to one this much closer"),
    "weakest_weight": Parameter(0.0, "ratio", UPWARD, 1.0,
                                "0 the nearest enemy; larger prefers enemies with less store"),
    # Assessment (lean.scripted.situation).
    "turning_floor": Parameter(0.25, "sustained_rate", LOG, 0.4,
                               "a nose rate this small counts as straight"),
    # Tactics: engaged or not, and the quadrant by the 3/9 line (merge, offensive, turning,
    # defensive).
    "engage_range": Parameter(3.0, "gun_range", LOG, 0.4, "beyond it from the target: disengaged"),
    "three_nine_line": Parameter(math.pi / 2, "radians", LINEAR, 0.2,
                                 "ahead of me, or behind him, inside this angle"),
    "defend_range": Parameter(3.0, "gun_range", LOG, 0.3, "defensive: break inside it"),
    # Angles off (alpha + |lambda|, pi minus Shaw's angular advantage; his advantage is
    # angles off - pi): the two-circle energy plan cashes its store when his advantage
    # reaches 90 deg (Shaw p. 110); a break relaxes to a hard turn below the relax angle
    # (Shaw p. 134: about 90 deg of his advantage, 1.5 pi; the yardstick breaks whenever
    # defensive within three gun ranges: graded defence lost to the old agent, measured).
    "cash_angle": Parameter(1.5 * math.pi, "radians", LINEAR, 0.3, "two-circle: cash beyond it"),
    "relax_angle": Parameter(math.pi, "radians", LINEAR, 0.3, "break beyond it"),
    "scissors_range": Parameter(1.0, "sustained_radius", LOG, 0.3,
                                "one-circle inside it: slow to the speed floor"),
    # The committed turn sense and the plan (one- or two-circle).
    "tie_band": Parameter(0.1, "radians", meaning="him this near dead astern: the drawn side"),
    "release_band": Parameter(0.2, "radians", meaning="release this far forward of the 3/9 line"),
    "plan_threshold": Parameter(-0.33279597121084487, "store fraction", LINEAR, 0.25,
                                "energy advantage above it: two-circle"),
    "lead_turn_scale": Parameter(0.042853310306349236, "ratio", UPWARD, 1.0,
                                 "the lead turn's time, of its geometric value"),
    "switch_gain": Parameter(0.17, "radians", LOG, 0.4,
                             "a pass gaining less angles off than this switches the plan"),
    "overshoot_range": Parameter(0.67, "sustained_radius", LOG, 0.3, "he overshoots inside it"),
    "overshoot_rate": Parameter(1.0, "sustained_rate", LOG, 0.3,
                                "crossing my six faster than this"),
    "overshoot_crossing": Parameter(1.0, "radians", LOG, 0.3,
                                    "track-crossing angle below it"),
    # Yo-yos: y = tanh((closure - wanted closure) / scale), the wanted closure reaching the
    # nose-tail distance in the closing time; y > 0 climb (and lag by the yo-yo time), y < 0
    # dive (and lead). The yardstick flies the gun's solution (yo-yo time and lead turn 0:
    # either lost about 0.2 of the score against the old agent, measured); personas sample
    # them upward.
    "nose_tail": Parameter(0.46550273408948645, "sustained_radius", LOG, 0.4,
                           "where the chase wants to sit"),
    "closing_time": Parameter(0.9124776658863046, "quarter_turn_time", LOG, 0.4,
                              "time to close to it"),
    "yo_yo_scale": Parameter(0.1, "corner_speed", LOG, 0.4, "closure error for a full yo-yo"),
    "yo_yo_time": Parameter(0.19070978159535185, "quarter_turn_time", UPWARD, 0.5,
                            "lag or lead of a full yo-yo"),
    # Speeds: the power tracks a speed target v*, mixed in log space over the tactics.
    "merge_speed": Parameter(1.0, "corner_speed", LOG, 0.1, "v* disengaged and at the merge"),
    "fight_speed": Parameter(1.2943094805067583, "sustained_speed", LOG, 0.2,
                             "v* in the rate fight"),
    "speed_floor": Parameter(1.3, "stall_speed", LOG, 0.1, "v* never below it"),
    "spend_share": Parameter(1.0, "store x store_turn", LOG, 0.4,
                             "a shot is near when the turn it needs is within this"),
    "spend_need": Parameter(0.1, "radians", LOG, 0.4, "a turn this large is worth the store"),
    "spend_range": Parameter(1.0, "gun_range", LOG, 0.3, "a shot is near within this range"),
    # The power law: x = ln(v / v*); climb above the band, dive below it.
    "speed_band": Parameter(0.05, "log units", LINEAR, 0.03, "half-width of the coast band"),
    "power_scale": Parameter(0.03, "log units", LOG, 0.4, "the power's sharpness"),
    "power_floor": Parameter(0.02, "probability", meaning="each of coast, dive and climb"),
    # The stick cap folds the full level into the sustained one (no turn lost above the
    # middle stick speed; banking speed in a high yo-yo above the fight speed).
    "level_cap_speed": Parameter(1.0, "middle_stick_speed", meaning="cap above it"),
    "yo_yo_cap_speed": Parameter(1.0, "fight speed", meaning="high yo-yo: cap above it"),
    # Stick: three nested gates on the pointing error, then the side.
    "turn": Parameter(0.002868, "radians", LOG, 0.4, "pointing error: turn at all"),
    "level_two": Parameter(0.02097, "radians", LOG, 0.4, "pointing error: middle stick"),
    "level_three": Parameter(0.1044, "radians", LOG, 0.4, "pointing error: the full stick"),
    "side_floor": Parameter(0.02, "probability", meaning="turning the long way round"),
    # The gun. Lead: first-order intercept with the drag law's time of flight in the medium.
    "aligned": Parameter(0.9769193740967238, "target angular size", LOG, 0.4,
                         "gun aim error: the gun bears"),
    "in_range": Parameter(1.1319620031840072, "gun_range", LOG, 0.4,
                          "distance: the target is within reach"),
    "lead_cap": Parameter(4.78, "gun_range", meaning="the lead's distance is capped here"),
    "lead_gain": Parameter(1.2209131619484321, "ratio", LINEAR, 0.2,
                           "share of the computed lead applied"),
    "target_size_minimum": Parameter(math.radians(1.0), "radians", meaning="target size clamp"),
    "target_size_maximum": Parameter(math.radians(45.0), "radians", meaning="target size clamp"),
    "ally_clearance": Parameter(2.0, "hit_radius", meaning="a team-mate this near the line blocks"),
    # Boundary steer: inside the margin of the edge and flying outward, a steer to the centre
    # blended in, its weight ramping from 0 to 1 between two depths (0 at the margin's inner
    # edge, 1 at the arena edge). The margin is in turn radii at the ship's speed: the
    # sustained radius up to corner speed, growing as speed squared above it.
    "boundary_margin": Parameter(2.2, "turn radius", LOG, 0.3, "depth of the edge band"),
    "boundary_ramp_low": Parameter(0.0554, "margin fraction", meaning="the steer starts"),
    "boundary_ramp_high": Parameter(0.52, "margin fraction", meaning="the steer is whole"),
    # Decisiveness: divides the sharpness of every tempered gate (all but the gun's: aim is
    # its own skill, author 2026-11-23) and of the power law.
    "temperature": Parameter(0.974789678735119, "ratio", LOG, 0.4, "softens every tempered gate"),
}

# The fight's gates, in one logistic before the stick (assessment, tactics, power, gun).
GATES: dict[str, Gate] = {
    "engaged": Gate("engage_range", 0.2, False),  # range in gun ranges
    "front": Gate("three_nine_line", 0.1, False),  # |bearing|, radians: he is ahead
    "rear": Gate("three_nine_line", 0.1, False),  # aspect, radians: I am behind his 3/9 line
    "defend": Gate("defend_range", 0.127, False),  # range in gun ranges
    "cash": Gate("cash_angle", 0.05, True),  # angles off, radians
    "relax": Gate("relax_angle", 0.05, True),  # angles off, radians
    "scissors": Gate("scissors_range", 0.2, False),  # range in sustained radii
    "shot_near": Gate("spend_share", 0.2, False),  # turn needed / (store x store_turn)
    "spend_need": Gate("spend_need", 0.2, True),  # turn needed, radians
    "spend_range": Gate("spend_range", 0.127, False),  # range in gun ranges
    "level_cap": Gate("level_cap_speed", 0.05, True),  # speed / middle stick speed
    "yo_yo_cap": Gate("yo_yo_cap_speed", 0.05, True),  # speed / fight speed
    "aligned": Gate("aligned", 0.243, False, tempered=False),  # aim error in target sizes
    "in_range": Gate("in_range", 0.127, False, tempered=False),  # range in gun ranges
}
# The stick's gates, in a second logistic on the error to the wanted direction (radians).
STICK_GATES: dict[str, Gate] = {
    "turn": Gate("turn", 0.172, True, 0.02),
    "level_two": Gate("level_two", 0.328, True, 0.02),
    "level_three": Gate("level_three", 0.267, True, 0.02),
}
# fmt: on
# The previous defaults (before BFM-CMA2), as the rows that differ: for comparison,
# `ScriptedConfig().replace(**BFM_DEFAULT_2026_11_24)`.
BFM_DEFAULT_2026_11_24: dict[str, float] = {
    "plan_threshold": 0.0,
    "lead_turn_scale": 0.0,
    "nose_tail": 0.67,
    "closing_time": 1.0,
    "yo_yo_time": 0.0,
    "fight_speed": 1.0,
    "aligned": 1.01,
    "in_range": 1.04,
    "lead_gain": 1.0,
    "temperature": 1.0,
}
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
    return box_point(config, unit, spread)


def box_point(config: ScriptedConfig, unit: Tensor, spread: float = 1.0) -> dict[str, Tensor]:
    """The sampled rows (`SAMPLED`) at a point `unit` (..., len(SAMPLED)) of the unit cube,
    each on its box at `spread` around the config's value (`Parameter`'s laws): a persona's
    draw, or a Sobol point of the box (lean.eval)."""
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


def vector(config: ScriptedConfig) -> tuple[float, ...]:
    """Every row's value in the table's order (`PARAMETERS`): a scripted agent's identity."""
    return tuple(float(config.values[name]) for name in PARAMETERS)


def from_vector(values) -> ScriptedConfig:
    """The ScriptedConfig of a full vector in the table's order (`vector`)."""
    values = [float(item) for item in values]
    if len(values) != len(PARAMETERS):
        raise ValueError(f"a scripted vector has {len(PARAMETERS)} rows, not {len(values)}")
    return ScriptedConfig(dict(zip(PARAMETERS, values, strict=True)))


def rows_of(table: Tensor, agent: Tensor) -> dict[str, Tensor]:
    """Per-ship parameters for every row of the table: `table` (A, len(PARAMETERS)) holds A
    agents' vectors, `agent` (E, N) long each ship's agent. Every row is a tensor, so a new
    vector (a candidate, a new yardstick) is new data for a compiled graph, never a new
    graph; the config passed beside it is then not read."""
    rows = table.index_select(0, agent.flatten()).view(*agent.shape, table.shape[1])
    return dict(zip(PARAMETERS, rows.unbind(-1), strict=True))


def value(config: ScriptedConfig, persona: dict[str, Tensor] | None, name: str) -> Tensor | float:
    """A row's value: the persona's (E, N) tensor when it samples the row, else the config's
    number."""
    if persona is not None and name in persona:
        return persona[name]
    return config.values[name]
