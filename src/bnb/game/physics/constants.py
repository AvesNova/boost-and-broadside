"""The constants of SPEC.md 1, 1.1 and 3.2, read from spec/presets.json.

Everything here is a plain Python number. The compiled step specializes on them, so a
constant costs nothing at run time and a different preset is a different compiled graph.

Ship types (SPEC 1.1) are data: presets.json "types" lists each type's overrides. A
constant some type overrides is per-type: it lives in `Preset.types` (and `Preset.flights`
for flight constants), not as a shared `Preset` attribute, and code reads it per ship with
`sim.ship_types.type_value` / `flight_of`. Making another constant per-type is a change to
presets.json (and dropping its shared field below), not to the game code.
"""

import dataclasses
import json
import math
import os

_SPEC = os.path.join(os.path.dirname(__file__), "presets.json")
with open(_SPEC) as _fh:
    _DATA = json.load(_fh)

STICK_LEVELS: tuple = tuple(_DATA["stick_levels"])  # sigma by stick index, positive turns left
POWER_MODES: tuple = tuple(_DATA["power_modes"])  # pi by power index: coast, boost (-1), reverse (+1)
FIELD_LEVELS = (-2, -1, 1, 2)
LN_SQRT2 = 0.5 * math.log(2.0)


@dataclasses.dataclass(frozen=True)
class Flight:
    """SPEC 3.2, keys as in presets.json "flight"."""

    mass: float
    g: float
    cl_max: float
    alpha_s: float
    a_max: float
    cd0: float
    k_ind: float
    power: float
    p_min: float
    e_knee: float
    f_static: float
    e_max: float
    band: float
    a0: float
    a_climb: float
    a_dive: float
    soft_p: float
    omega_c: float
    zeta: float
    omega_floor: float
    eps: float

    @property
    def corner_speed(self):
        return (self.a_max / self.cl_max) ** 0.5  # a float, or per ship (see flight_of)


FLIGHT = Flight(**_DATA["flight"])

TYPE_NAMES: tuple = tuple(_DATA["types"])  # by ship_type id: 0 fighter, 1 heavy
_TYPE_SPECS: tuple = tuple(_DATA["types"].values())
# Game constants (outside "flight") that some type overrides: the constants marked † in SPEC 1.
PER_TYPE_KEYS: tuple = tuple(sorted({k for spec in _TYPE_SPECS for k in spec if k != "flight"}))


def type_flights(overrides: dict | None = None) -> tuple:
    """Each type's flight constants, by ship_type id; `overrides` apply to every type."""
    return tuple(dataclasses.replace(FLIGHT, **{**spec.get("flight", {}), **(overrides or {})})
                 for spec in _TYPE_SPECS)


@dataclasses.dataclass(frozen=True)
class Preset:
    """SPEC 1: sizes plus the per-game constants, with the derived values precomputed."""

    name: str
    num_ships: int
    num_fields: int
    arena_radius: float
    h: float
    episode_ticks: int
    slots: int
    bullet_moves: int
    spread: float
    head_on_fraction: float
    shield_delay: float
    shield_regen: float
    boundary_base: float
    boundary_per_m: float
    respawn_ticks: int
    spawn_speed: float
    spawn_offset: float
    spawn_radius: float
    field_width: float
    vision: float
    ship_types: bool  # the slot rule of SPEC 1.1 applies (otherwise every ship is a fighter)
    types: tuple  # by ship_type id: {key: value} for every key in PER_TYPE_KEYS
    flights: tuple  # by ship_type id: Flight

    @property
    def flight(self) -> Flight:
        """The flight constants every type shares (read only those from it)."""
        return self.flights[0]

    def per_type(self, key: str) -> tuple:
        """A per-type constant's value for each ship type, by ship_type id."""
        return tuple(t[key] for t in self.types)

    @property
    def slot_types(self) -> tuple:
        """ship_type of each slot: within a team, slots alternate starting with a fighter."""
        return tuple((i % self.half) % 2 if self.ship_types else 0 for i in range(self.num_ships))

    @property
    def half(self) -> int:
        """Ships per team."""
        return self.num_ships // 2

    @property
    def map_scale(self) -> float:
        return self.arena_radius / 2600.0

    @property
    def num_uniforms(self) -> int:
        return 4 * self.num_fields + 1


def _preset(name: str, values: dict) -> Preset:
    shared = {k: v for k, v in values.items() if k not in PER_TYPE_KEYS}
    types = tuple({k: spec.get(k, values[k]) for k in PER_TYPE_KEYS} for spec in _TYPE_SPECS)
    return Preset(name=name, types=types, flights=type_flights(), **shared)


PRESETS = {name: _preset(name, values) for name, values in _DATA["presets"].items()}
SIZE_KEYS = ("num_ships", "num_fields", "arena_radius")


def preset_of(spec) -> Preset:
    """A preset by name, or a custom size {num_ships, num_fields, arena_radius} (SPEC 1.2)
    with every other constant shared."""
    if isinstance(spec, str):
        return PRESETS[spec]
    n, m, r = (spec[k] for k in SIZE_KEYS)
    assert n >= 2 and n % 2 == 0 and 1 <= m <= 64 and r > 0, spec
    return _preset(f"N{n}M{m}R{r:g}", {**_DATA["presets"]["squad"], **{k: spec[k] for k in SIZE_KEYS}})
