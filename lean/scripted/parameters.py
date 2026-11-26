"""The scripted agent's parameters: one table of named rows, the yardstick and its personas.

The look-ahead agent (lean.scripted.opponent) scores each of its 21 flight candidates in one
unit, the **score second**: one second of a full gun solution (my nose on his lead point, in
range, the hit at full worth). Every term is a rate in score seconds per second of the
look-ahead, summed over the look-ahead points with the time weights; a weight row says what one
unit of its term is worth against one second of my own gun solution (the gun's weight is 1 by
definition). The other rows are physical (a multiple of a named game constant, in
`state.constants`) or dimensionless with a stated meaning. Each sampled row has its own box for
the persona population: at spread s it is drawn uniformly around its default, `width` times s
either side (in natural-log units for the "log" law), independently of every other row. The
yardstick (spread 0) reads the defaults as Python numbers; a persona reads (E, N) tensors under
the same names.
"""

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


# fmt: off
# Defaults chosen by reasoning (2026-11-26, docs/rewrite/sections/scripted-lookahead/
# progress.md), not tuned.
PARAMETERS: dict[str, Parameter] = {
    # Term weights, against my gun's 1.
    "threat_weight": Parameter(1.0, "ratio", LOG, 0.3,
                               "his gun solution on me, per second, against mine"),
    "angles_weight": Parameter(0.3, "ratio", LOG, 0.4,
                               "angular advantage (+1 on his tail nose on, -1 he on mine) per "
                               "second, at closeness 1"),
    # A full store buys `store_turn` (about 2.1 rad on the P-51D) of turn beyond the sustained
    # rate, about 0.75 of the angles term's unit: so a store is worth about 0.75 x the angles
    # weight.
    "energy_weight": Parameter(0.25, "ratio", LOG, 0.4,
                               "energy advantage (speed^2 / 2 + store) in full stores per "
                               "second, at closeness 1"),
    "speed_weight": Parameter(0.3, "ratio", LOG, 0.4,
                              "|ln(speed / preferred speed)| per second"),
    "collision_weight": Parameter(3.0, "ratio", LOG, 0.3,
                                  "a predicted contact (in score seconds, not per second)"),
    "boundary_weight": Parameter(2.0, "ratio", LOG, 0.3,
                                 "a second at the arena edge flying outward"),
    "field_weight": Parameter(0.3, "ratio", LOG, 0.4,
                              "a second in a slow medium per half-octave; a second in cover "
                              "from an enemy with the angles on me"),
    # The look-ahead's time weights: each point stands for the time since the one before,
    # discounted by e^(-t / horizon).
    "horizon": Parameter(1.5, "seconds", LOG, 0.3, "the look-ahead's discount time"),
    "preferred_speed": Parameter(0.9, "corner_speed", LOG, 0.15, "the speed term's centre"),
    # His reply and my choice (opponent.collapse): w(a, b) = softmax_b S_him[a, b] / tau_him;
    # score(a) = log(sum_b w e^(beta S_me[a, b])) / beta.
    "reply_temperature": Parameter(0.2, "score seconds", LOG, 0.5,
                                   "tau_him: how sensible I expect his reply (small: his best)"),
    "caution": Parameter(-2.0, "per score second", LINEAR, 2.0,
                         "beta: < 0 weighs his replies that hurt me most, > 0 the ones that "
                         "suit me, 0 the plain mean"),
    "temperature": Parameter(0.05, "score seconds", LOG, 0.4,
                             "tau_me: the flight softmax's temperature"),
    "stickiness": Parameter(0.05, "score seconds", LOG, 0.5,
                            "bonus for the flight action of the last tick"),
    # No enemy known (author 2026-11-26: no goal point, only the own-ship terms and the seek).
    "disengaged_energy_weight": Parameter(0.5, "ratio", LOG, 0.5,
                                          "no enemy known: energy (speed^2 / 2 + store) in "
                                          "full stores, per second"),
    "seek_weight": Parameter(0.1, "ratio", LOG, 0.7,
                             "no enemy known: heading toward the freshest dropped enemy's "
                             "last-seen position, per second (eager to patient)"),
    # Information.
    "roam_cutoff": Parameter(1.0, "arena_radius",
                             meaning="a hidden enemy's roam (seconds since seen x sustained "
                                     "speed) beyond this: dropped; its weight falls linearly "
                                     "to 0 there"),
    # The gun (fire now).
    "fire_tolerance": Parameter(1.0, "target angular size", LOG, 0.3,
                                "aim error at which the trigger is pulled half the time"),
    "fire_worth": Parameter(1.0, "ticks of shield recharge",
                            meaning="a hit worth less than this is not fired (or the target's "
                                    "shield)"),
    # Fixed shapes of the terms (not personality).
    "closeness_range": Parameter(3.0, "gun_range",
                                 meaning="the angles and energy terms halve at this distance"),
    "aim_blur": Parameter(0.1, "radians per second",
                          meaning="the look-ahead gun's tolerance grows by this with time"),
    "lead_cap": Parameter(2.0, "gun_range", meaning="the lead's distance is capped here"),
    "fire_range": Parameter(1.1, "gun_range", meaning="the trigger's range at half"),
    "fire_aim_scale": Parameter(0.243, "log units", meaning="the trigger's aim sharpness"),
    "fire_range_scale": Parameter(0.127, "log units", meaning="the trigger's range sharpness"),
    "collision_margin": Parameter(2.0, "summed hit radii",
                                  meaning="a predicted pass this near counts as a contact"),
    "boundary_margin": Parameter(2.2, "turn radius",
                                 meaning="depth of the edge band (turn radius at the speed)"),
    "target_size_minimum": Parameter(0.017453292519943295, "radians",
                                     meaning="the target's angular size, clamped below (1 deg)"),
    "ally_clearance": Parameter(2.0, "hit_radius",
                                meaning="a team-mate this near the line of fire blocks"),
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

    def __post_init__(self):
        unknown = set(self.values) ^ set(PARAMETERS)
        if unknown:
            raise ValueError(f"scripted parameters not in the table: {sorted(unknown)}")

    def replace(self, **values: float) -> ScriptedConfig:
        """A copy with some rows changed (by name)."""
        return ScriptedConfig({**self.values, **values})


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
