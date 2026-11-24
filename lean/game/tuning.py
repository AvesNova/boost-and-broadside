"""Ship design: a real aircraft's data and a design's inputs into a lean card row
(lean.game.card) at the game's time scale, under the game's flight model
(docs/rewrite/ship-design-proposal.md, option C; core-decisions.md "Ship design").

The game plays at TIME_FACTOR (2) times real time (v2's real_scale.EQUIVALENT_FACTOR): lengths
are real metres, game time is real time over 2, so speeds are x2, accelerations x4 (gravity
39.24 = 4 x 9.81), powers per kg x8 and turn rates x2; turn radii, and the lift and drag
coefficients per metre, are unchanged. `to_game` and `to_real` convert.

Two layers (docs/rewrite/p51d-model.md, "Design inputs"):

- **The reference** (`reference`): an aircraft's documents and best guesses. Its physical data
  (`Airframe`) go through `fit_point_mass` (C_D0 and the propeller efficiency set so the point
  mass flies the flight test's top speed and climb) into its point mass (`airframe_row`, the
  game terms off). The reference's design values are the data's (stall speed, g limit, static
  thrust, stall angle), the flight test's (top speed, best climb and its speed), the point
  mass's (the best sustained turn rate) and the aircraft's choices (the gap, the game terms, the
  guns...).
- **The design** (`DESIGN_INPUTS`, one ordered table: name, label, unit, the card field it sets,
  how): the sliders, which start at the reference's values. `design_row(reference, design)`
  turns them into the card row: closed forms first (the stall speed sets the lift, the g limit
  the maximum acceleration, every other input its own field), then the five flown targets
  (top speed, best climb, best-climb speed, best sustained rate, the gap) solved for the
  zero-lift drag, thrust power, thrust softness, induced drag and separation drag: a
  point-mass first guess, then Newton in logs (Broyden in the web tool's fast path).
  `design_of(row)` is its inverse: the design a row flies.

`implied(row, airframe)` reads a row back as the airframe's physical quantities (mass, wing
area, aspect ratio, shaft power and propeller diameter held): C_Lmax, C_D0, Oswald e, propeller
efficiency, static figure of merit; `warnings` checks them against `PLAUSIBILITY` (warnings,
never refusals; a solve that does not converge is the one refusal).

The store size, the spawn store and the shot's cost are the game's own (two loops at corner
speed, half of it, free), and the stick levels and the bullet's damage reference speed are
derived by the game (core._derived_characteristics), so no row carries them.
`performance(..., plain=True)` turns the game terms off for any row (POINT_MASS_TERMS).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import math
import os
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch

from . import card, core

TIME_FACTOR = 2.0  # the game's time against real time (v2 real_scale.EQUIVALENT_FACTOR)
REAL_GRAVITY = 9.81
AIR_DENSITY = 1.225  # kg/m^3, sea level
HORSEPOWER = 745.7  # W
FIGURE_OF_MERIT = 0.75  # a propeller's static figure of merit (planes.py)
SPEED_OF_SOUND = 340.3  # m/s, ISA sea level
LIFE_TOLERANCE = 1e-4  # v2 weapons.LIFE_TOLERANCE
MPH, FEET_PER_MINUTE, POUND = 0.44704, 0.00508, 0.45359237  # m/s, m/s, kg
assert abs(card.GRAVITY - REAL_GRAVITY * TIME_FACTOR**2) < 1e-9


def to_game(value: float, time_power: int) -> float:
    """A real quantity in game units, given the power of time in its units (speed -1,
    acceleration -2, power per kg -3, rate -1; lengths and per-metre coefficients 0)."""
    return value * TIME_FACTOR ** (-time_power)


def to_real(value: float, time_power: int) -> float:
    return value * TIME_FACTOR**time_power


# --------------------------------------------------------------------------- the airframe


@dataclasses.dataclass(frozen=True)
class Airframe:
    """An aircraft's physical data, SI, real time. power is the shaft power (W) at the rating
    the drag data were fitted at; propeller_efficiency turns it into thrust power; g_limit is
    rated at g_limit_mass (the structure's limit at another mass scales as 1 / mass)."""

    mass: float  # kg
    wing_area: float  # m^2
    max_lift_coefficient: float  # clean C_Lmax
    zero_lift_drag_coefficient: float  # C_D0
    aspect_ratio: float
    oswald: float  # span efficiency e
    power: float  # W, shaft
    propeller_efficiency: float
    propeller_diameter: float  # m
    g_limit: float
    g_limit_mass: float  # kg
    stall_angle: float  # rad, angle of attack at C_Lmax

    @property
    def per_metre(self) -> float:
        """rho S / (2 m): a coefficient's acceleration per v^2, 1/m."""
        return AIR_DENSITY * self.wing_area / (2.0 * self.mass)

    @property
    def propeller_disc(self) -> float:
        return math.pi * self.propeller_diameter**2 / 4.0

    def static_thrust(self) -> float:
        """Static thrust (N) by momentum theory: FM P = T^1.5 / sqrt(2 rho A)."""
        momentum = FIGURE_OF_MERIT**2 * self.power**2 * 2.0 * AIR_DENSITY * self.propeller_disc
        return momentum ** (1.0 / 3.0)


# The P-51D's data (docs/rewrite/aircraft-references.md, p51d-inputs-sources.md): 4,427 kg and
# 1,630 hp at 67 in Hg (war emergency rating) as tested (AAF TSCEP5E-1908); wing area 21.67 m^2
# (NACA ACR 4K02); AR 5.83; C_Lmax 1.6 (the AAF's power-on value for its turn calculation;
# stall-derived values span 1.50-1.70); Oswald e 0.8 (assumed); Hamilton Standard propeller
# 3.40 m (11 ft 2 in); the g limit 8 at the 8,000 lb design weight (forum, F; 6.6 g at the
# tested 9,760 lb) and the stall angle 15.5 deg, from v2's research (bnb.reference.real_planes).
# C_D0 and the propeller efficiency are not data: fit_point_mass sets them so the point mass
# flies the tested 375 mph and 3,600 ft/min at sea level (the October procedure,
# physics-checks/real_fit.py), every time the card is derived; the values here are its start.
P51D = Airframe(
    mass=4427.0,
    wing_area=21.67,
    max_lift_coefficient=1.6,
    zero_lift_drag_coefficient=0.015036438034450334,
    aspect_ratio=5.83,
    oswald=0.8,
    power=1630.0 * HORSEPOWER,
    propeller_efficiency=0.8211724151486224,
    propeller_diameter=3.40,
    g_limit=8.0,
    g_limit_mass=8000.0 * POUND,
    stall_angle=math.radians(15.5),
)

# The game terms switched off: no drag rise, no separation drag, soft minima near hard (exponent
# 400), no flaps. A real aircraft's data fly as its point mass with these.
POINT_MASS_TERMS = dict(
    critical_speed=math.inf,
    separation_drag=0.0,
    lift_softmin_exponent=400.0,
    thrust_softmin_exponent=400.0,
    flap_lift=0.0,
    flap_drag_scale=0.0,
)


def point_mass(row: dict[str, float]) -> dict[str, float]:
    """A row with the game terms off (POINT_MASS_TERMS); the other flight terms kept."""
    return {**row, **POINT_MASS_TERMS}


def airframe_row(airframe: Airframe, base: dict[str, float]) -> dict[str, float]:
    """The aircraft's point mass as a card row: the flight fields from the airframe, the game
    terms off (POINT_MASS_TERMS), everything else from `base`. The point-mass conversion of the
    October fit (docs/rewrite/physics-checks/planes.py `from_real`): lift and zero-lift drag per
    metre rho S C / (2 m), induced drag factor k / (rho S / 2 m) with k = 1 / (pi e AR), thrust
    power P eta / m, static thrust by momentum theory (Airframe.static_thrust); dive and climb
    trade at 1 g."""
    a = airframe
    induced = 1.0 / (math.pi * a.oswald * a.aspect_ratio) / a.per_metre  # m
    flight = dict(
        max_lift_coefficient=a.per_metre * a.max_lift_coefficient,
        max_acceleration=a.g_limit * card.GRAVITY,
        zero_lift_drag=a.per_metre * a.zero_lift_drag_coefficient,
        induced_drag_scale=induced / card.INDUCED_DRAG_FACTOR,
        engine_power=to_game(a.power * a.propeller_efficiency / a.mass, -3),
        static_thrust=to_game(a.static_thrust() / a.mass, -2),
        stall_angle_scale=a.stall_angle / card.STALL_ANGLE,
        exchange_acceleration=card.GRAVITY,  # 1 g: a vertical dive or zoom
    )
    return {**base, **flight, **POINT_MASS_TERMS}


def fit_point_mass(
    airframe: Airframe, top_speed: float, best_climb: float, base: dict[str, float]
) -> Airframe:
    """The airframe with C_D0 and the propeller efficiency set so its point mass flies
    `top_speed` and `best_climb` (real m/s, sea level); the other data kept (the October
    procedure: docs/rewrite/physics-checks/real_fit.py). `base`: the row's other fields."""
    targets = (to_game(top_speed, -1), to_game(best_climb, -1))

    def fitted(logs):
        return dataclasses.replace(
            airframe,
            zero_lift_drag_coefficient=math.exp(logs[0]),
            propeller_efficiency=math.exp(logs[1]),
        )

    def residual(logs):
        p = performance(airframe_row(fitted(logs), base))
        return [math.log(p["top_speed"] / targets[0]), math.log(p["best_climb"] / targets[1])]

    start = [math.log(airframe.zero_lift_drag_coefficient), math.log(airframe.propeller_efficiency)]
    return fitted(_newton(residual, start, 1e-7, name="fit_point_mass").logs)


# --------------------------------------------------------------------------- the design inputs


@dataclasses.dataclass(frozen=True)
class DesignInput:
    """One design input, the whole of what the solve, the reference and the web tool know of it.

    - `sets`: the card field it mainly decides.
    - `measure(row, flown)`: the value a row has or flies (`flown`: performance(row)).
    - `apply(value, design)`: for a closed input, the field's value in closed form (it may read
      the other design values); None for a solved input, which design_row meets by Newton on
      its `sets` (in logs, from `first_guess` when there is no start row and the point mass
      has no usable value).
    - `source`: the reference's provenance (the P-51D's).
    - `kind`: the quantity's kind in the web tool's units (lean.tool.units.KINDS; game units
      here), None for a number shown as it is with `unit`.
    - the slider: `low` to `high`, multiples of the reference's value if `relative`."""

    name: str
    label: str
    group: str
    sets: str
    measure: Callable[[dict, dict], float]
    apply: Callable[[float, dict], float] | None = None
    first_guess: float | None = None
    source: str = "game choice"
    kind: str | None = None
    unit: str = ""
    low: float = 0.5
    high: float = 2.0
    relative: bool = True

    @property
    def method(self) -> str:
        return "solved" if self.apply is None else "closed"


def flown(name, label, group, sets, key=None, **rest) -> DesignInput:
    """A solved input: what performance() reports under `key` (default its name)."""
    key = key or name
    return DesignInput(name, label, group, sets, lambda row, numbers: numbers[key], **rest)


def scaled(name, label, group, sets=None, scale=lambda: 1.0, **rest) -> DesignInput:
    """A closed input: its field is the value times `scale()` (a callable, so it reads the card's
    constants when used); `sets` defaults to the name (a short circuit)."""
    sets = sets or name
    return DesignInput(name, label, group, sets, lambda row, numbers: row[sets] / scale(),
                       lambda value, design: value * scale(), **rest)


def _bullet_life(gun_range: float, design: dict, tick_length: float = 1.0 / 30.0) -> float:
    """The bullet's life in whole ticks: long enough to fly the gun range when fired at the top
    speed against the bullet drag (v2's weapons.combat_card)."""
    drag = design["bullet_drag"]
    flight = math.expm1(drag * gun_range) / (drag * (design["top_speed"] + design["muzzle_speed"]))
    return math.ceil(flight / tick_length - LIFE_TOLERANCE) * tick_length


def _gun_range(row: dict, numbers: dict) -> float:
    """The range of a bullet's whole life, fired at the top speed (at least the designed range)."""
    drag = row["bullet_drag_scale"] * card.BULLET_DRAG
    launch = numbers["top_speed"] + row["muzzle_speed"]
    return math.log1p(drag * launch * row["bullet_life"]) / drag


GROUPS = ("Performance", "Airframe", "Flight terms", "Flaps", "Stick and nose", "Guns",
          "Shield, head-on and vision")
_GRAVITY = lambda: card.GRAVITY  # noqa: E731
P, A, T, F, S, G, H = GROUPS
# The design inputs, in the tool's order. To add, remove or swap one, edit this table only (see
# docs/rewrite/p51d-model.md, "Design inputs"); every card field must be set exactly once.
DESIGN_INPUTS = (
    DesignInput("stall_speed", "stall speed (1 g)", P, "max_lift_coefficient",
                lambda row, numbers: numbers["stall_speed"],
                lambda value, design: card.GRAVITY / value**2,
                source="B: C_Lmax 1.6, AAF turn calculation", kind="speed", low=0.6, high=1.5),
    scaled("g_limit", "g limit", P, "max_acceleration", _GRAVITY, unit="g",
           source="F: forum 8 g (author's choice)", low=0.4, high=1.6),
    flown("top_speed", "top level speed", P, "zero_lift_drag",
          source="A: flight test, 375 mph", kind="speed", low=0.6, high=1.4),
    flown("best_climb", "best climb rate", P, "engine_power",
          source="A: flight test, 3,600 ft/min", kind="climb", low=0.4, high=2.0),
    flown("best_climb_speed", "best-climb speed", P, "thrust_softmin_exponent", first_guess=4.0,
          source="A: flight test, 165 mph", kind="speed", low=0.85, high=1.3),
    flown("sustained_rate", "best sustained turn rate", P, "induced_drag_scale",
          source="model: the point mass (e 0.8 assumed)", kind="angular_rate", low=0.6, high=1.5),
    flown("gap", "the gap: tightest over best-rate sustained speed", P, "separation_drag",
          first_guess=0.4, source="design (author 2026-11-24)", low=0.7, high=0.99,
          relative=False),
    scaled("static_thrust", "static thrust / weight", A, scale=_GRAVITY,
           source="momentum theory, FM 0.75", low=0.3, high=2.0),
    scaled("stall_angle", "stall angle", A, "stall_angle_scale", lambda: 1.0 / card.STALL_ANGLE,
           unit="rad", source="guess: v2's 15.5 deg", low=0.5, high=1.6),
    scaled("exchange", "dive and zoom trade", A, "exchange_acceleration", _GRAVITY, unit="g",
           source="design: 1 g", low=0.3, high=1.6),
    scaled("critical_mach", "drag rise: critical Mach", T, "critical_speed",
           lambda: to_game(SPEED_OF_SOUND, -1), unit="Mach", source="A: NACA ACR 4K02 Fig. 12",
           low=0.5, high=1.5, relative=False),
    scaled("drag_rise_exponent", "drag rise: exponent", T, source="A: NACA ACR 4K02 Fig. 12",
           low=2.0, high=40.0, relative=False),
    scaled("separation_exponent", "separation drag: exponent", T,
           source="design (author 2026-11-24)", low=4.0, high=32.0, relative=False),
    scaled("lift_softmin_exponent", "lift softness: exponent (low = soft)", T,
           source="design: v2's 16", low=2.0, high=64.0, relative=False),
    scaled("flap_lift", "extra lift", F, unit="of clean", source="design: off", low=0.0,
           high=0.5, relative=False),
    scaled("flap_blowback_ratio", "blown back at", F, unit="x clean corner speed", low=0.3,
           high=2.0, relative=False),
    scaled("flap_drag_scale", "drag of the extra lift", F, low=0.0, high=3.0),
    scaled("nose_damping", "nose damping", S, low=0.2, high=1.5, relative=False),
    scaled("corner_nose_frequency", "nose frequency at corner speed", S, kind="angular_rate",
           low=0.3, high=2.0),
    scaled("fine_stick_fraction", "fine stick", S, unit="of sustained",
           source="design: half (author 2026-11-24)", low=0.05, high=1.0, relative=False),
    scaled("hit_damage", "hit damage", G, low=0.3, high=3.0),
    scaled("explosive_share", "explosive share", G, low=0.0, high=1.0, relative=False),
    scaled("fire_rate", "fire rate", G, kind="frequency", low=0.3),
    scaled("muzzle_speed", "muzzle speed", G, kind="speed"),
    scaled("bullet_spread", "bullet spread", G, "bullet_spread_scale",
           lambda: 1.0 / card.BULLET_SPREAD, kind="speed", low=0.0, high=3.0),
    scaled("bullet_drag", "bullet drag", G, "bullet_drag_scale", lambda: 1.0 / card.BULLET_DRAG,
           kind="inverse_length", low=0.0, high=4.0),
    DesignInput("gun_range", "gun range (from top speed)", G, "bullet_life", _gun_range,
                _bullet_life, kind="length", low=0.3, high=2.5),
    scaled("hit_radius", "hit radius", G, kind="length", low=0.3, high=3.0),
    scaled("core_radius", "core radius", G, kind="length", low=0.3, high=3.0),
    scaled("max_shield", "shield", H, low=0.3, high=3.0),
    scaled("shield_recharge_rate", "shield recharge rate", H, "shield_recharge_rate_scale",
           lambda: 1.0 / card.SHIELD_RECHARGE_RATE, kind="frequency", low=0.2, high=4.0),
    scaled("shield_recharge_delay", "shield recharge wait", H, "shield_recharge_delay_scale",
           lambda: 1.0 / card.SHIELD_RECHARGE_DELAY, kind="time", low=0.2, high=4.0),
    scaled("head_on_fraction", "head-on damage share", H, "head_on_fraction_scale",
           lambda: 1.0 / card.HEAD_ON_FRACTION, low=0.0, high=1.0, relative=False),
    scaled("head_on_angle", "head-on angle", H, "head_on_angle_scale",
           lambda: 1.0 / card.HEAD_ON_ANGLE, unit="rad", low=0.1, high=1.57, relative=False),
    scaled("head_on_sharpness", "head-on sharpness", H, low=0.3, high=3.0),
    scaled("vision_range", "vision range", H, "vision_range_scale",
           lambda: 1.0 / card.VISION_RANGE, kind="length", low=0.3, high=3.0),
)
del P, A, T, F, S, G, H


def check_inputs(inputs: tuple[DesignInput, ...]) -> None:
    """A design-input table must set every card field exactly once, with unique names."""
    fields = sorted(spec.sets for spec in inputs)
    if fields != sorted(card.CARD_FIELDS):
        off = sorted(set(fields) ^ set(card.CARD_FIELDS))
        raise ValueError(f"the inputs do not set each card field once: {off}")
    if len({spec.name for spec in inputs}) != len(inputs):
        raise ValueError("design input names repeat")


check_inputs(DESIGN_INPUTS)


def inputs_version(inputs: tuple[DesignInput, ...] = DESIGN_INPUTS) -> str:
    """12 hex digits naming a table's inputs (names, what each sets, how): a saved design's
    values mean the same only under the same version."""
    text = json.dumps([(spec.name, spec.sets, spec.method) for spec in inputs])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _closed_row(inputs, design: dict[str, float]) -> dict[str, float]:
    """Every card field a closed input sets."""
    return {spec.sets: spec.apply(design[spec.name], design) for spec in inputs if spec.apply}


# --------------------------------------------------------------------------- the reference


@dataclasses.dataclass(frozen=True)
class RealAircraft:
    """A real aircraft: its data (C_D0 and the propeller efficiency only a start), its flight
    test (real m/s, sea level: top speed and best climb, which fit the point mass) and its
    documented design values (game units, design-input names: the flight test's, the gap, the
    flight terms, guns, shield...). An input with no documented value takes its point mass's."""

    airframe: Airframe
    top_speed: float
    best_climb: float
    values: dict


# The P-51D's values (design units, game time). The flight test: 375 mph, 3,600 ft/min at
# 165 mph (AAF TSCEP5E-1908). The gap (core-decisions.md 2026-11-24): the tightest sustained turn
# 15% slower than the best sustained rate, with the separation exponent 16 (where its
# coefficient still moves the gap, docs/rewrite/ship-design-proposal.md 5). The drag rise doubles
# the zero-lift drag at Mach 0.77, exponent 20 (NACA ACR 4K02 Fig. 12, the P-51B propeller off:
# flat to about M 0.68, then +40% at M 0.735 and +67% at M 0.755). The lift softness 16 (v2's).
# The flaps off (author 2026-11-24; switchable per type; on, the extra lift costs Delta C_D =
# 1.5 Delta C_L^2 with the flaps fully used). Guns, damage, shield, the nose and vision are v2's
# Vizia's; the fine stick is half the sustained level (author 2026-11-24).
P51D_TOP_SPEED, P51D_BEST_CLIMB = 375.0 * MPH, 3600.0 * FEET_PER_MINUTE  # real m/s
P51D_FLAP_DRAG = 1.5  # Delta C_D per (Delta C_L)^2 beyond the clean maximum, flaps fully used
P51D_VALUES = dict(
    top_speed=to_game(P51D_TOP_SPEED, -1),
    best_climb=to_game(P51D_BEST_CLIMB, -1),
    best_climb_speed=to_game(165.0 * MPH, -1),
    gap=0.85,
    critical_mach=0.77,
    drag_rise_exponent=20.0,
    separation_exponent=16.0,
    lift_softmin_exponent=16.0,
    flap_lift=0.0,
    flap_blowback_ratio=1.0,
    flap_drag_scale=P51D_FLAP_DRAG / (P51D.per_metre * card.INDUCED_DRAG_FACTOR),
    nose_damping=0.7,
    corner_nose_frequency=11.23144817352295,
    fine_stick_fraction=0.5,
    hit_damage=21.0,
    explosive_share=0.0,
    fire_rate=30.0,
    muzzle_speed=1712.0,
    bullet_spread=3.084482431411743,
    bullet_drag=0.0005,
    gun_range=600.0,
    hit_radius=12.0,
    core_radius=5.0,
    max_shield=100.0,
    shield_recharge_rate=24.0,
    shield_recharge_delay=3.0,
    head_on_fraction=0.0,
    head_on_angle=math.pi / 4.0,
    head_on_sharpness=3.0,
    vision_range=1000.0,
)
REAL_AIRCRAFT = {"p51d": RealAircraft(P51D, P51D_TOP_SPEED, P51D_BEST_CLIMB, P51D_VALUES)}


@dataclasses.dataclass(frozen=True)
class Reference:
    """A reference ship: its design-input table, the fitted airframe, its point mass (a card
    row, the game terms off) and that row's performance, and every input's reference value
    (game units)."""

    inputs: tuple
    airframe: Airframe
    point_mass_row: dict
    point_mass: dict
    design: dict


def reference_of(aircraft: RealAircraft, inputs: tuple = DESIGN_INPUTS) -> Reference:
    """The reference from the aircraft's documents: the point mass fitted to the flight test
    (fit_point_mass); each input's reference value its documented one, else what the point
    mass has or flies (its measure: e.g. the stall speed, g limit, static thrust and the best
    sustained rate)."""
    check_inputs(inputs)
    names = {spec.name for spec in inputs}
    values = {name: value for name, value in aircraft.values.items() if name in names}
    base = {spec.sets: spec.apply(values[spec.name], values)
            for spec in inputs if spec.apply and spec.name in values}
    base = {name: base.get(name, 1.0) for name in card.CARD_FIELDS}  # what the airframe sets
    fitted = fit_point_mass(aircraft.airframe, aircraft.top_speed, aircraft.best_climb, base)
    row = airframe_row(fitted, base)
    numbers = performance(row)
    design = {spec.name: values.get(spec.name, spec.measure(row, numbers)) for spec in inputs}
    return Reference(inputs, fitted, row, numbers, design)


@functools.cache
def reference(name: str) -> Reference:
    """REAL_AIRCRAFT[name]'s reference (reference_of), once per process."""
    return reference_of(REAL_AIRCRAFT[name])


# --------------------------------------------------------------------------- the solve


@dataclasses.dataclass(frozen=True)
class Solved:
    """A design's row and its solve: the largest |log(flown / target)|, the Newton iterations
    and the condition number of the last log-log Jacobian (None if none was needed)."""

    row: dict
    residual: float
    iterations: int
    condition: float | None


def design_row(
    reference: Reference,
    design: dict[str, float] | None = None,
    start: dict[str, float] | None = None,
    fast: bool = False,
) -> Solved:
    """The card row of the reference with the design inputs `design` changed (its table's
    names, game units). The closed inputs first; then every solved input met by Newton in the
    logs of the fields they set, from `start` (a row, e.g. the reference's own), else from the
    point mass's value (an input's first_guess where the point mass has none to take a log of),
    to 1e-7 in log (`fast`: Broyden updates after the first Jacobian, to 1e-5: the web tool's
    path). A design with flaps is solved clean first, then with them from there: the flaps'
    low-speed part of the sustained turn is lift-limited, so no drag coefficient moves it, and a
    start whose clean best turn lies below it sees no way to the target. Raises ValueError for
    an unknown input, RuntimeError when the solve does not converge."""
    inputs = reference.inputs
    values = {**reference.design, **(design or {})}
    unknown = set(values) - {spec.name for spec in inputs}
    if unknown:
        raise ValueError(f"not design inputs: {sorted(unknown)}")
    row = _closed_row(inputs, values)
    solved = [spec for spec in inputs if spec.apply is None]
    point = reference.point_mass_row

    def first(spec):
        if start is not None:
            return start[spec.sets]
        usable = 0.0 < point[spec.sets] < math.inf and spec.sets not in POINT_MASS_TERMS
        return point[spec.sets] if usable or spec.first_guess is None else spec.first_guess

    logs = [math.log(first(spec)) for spec in solved]
    targets = [values[spec.name] for spec in solved]
    stages = [{**row, "flap_lift": 0.0}, row] if row["flap_lift"] > 0.0 else [row]
    solve = SimpleNamespace(logs=logs, residual=0.0, iterations=0, condition=None)
    for stage in stages:

        def residual(logs, stage=stage):
            trial = {**stage, **{spec.sets: math.exp(x) for spec, x in zip(solved, logs)}}
            numbers = performance(trial)
            return [math.log(spec.measure(trial, numbers) / t) for spec, t in zip(solved, targets)]

        if solved:
            tolerance = 1e-5 if fast else 1e-7
            solve = _newton(residual, solve.logs, tolerance, broyden=fast, name="design_row")
    row.update({spec.sets: math.exp(x) for spec, x in zip(solved, solve.logs)})
    return Solved(row, solve.residual, solve.iterations, solve.condition)


def design_of(row: dict[str, float], inputs: tuple = DESIGN_INPUTS) -> dict[str, float]:
    """The design a row has and flies (game units): every input's measure.
    design_row(reference, design_of(row)) is the row again, within the solve's tolerance."""
    numbers = performance(row)
    return {spec.name: spec.measure(row, numbers) for spec in inputs}


DESIGN = {spec.name: spec for spec in DESIGN_INPUTS}  # the game's table by name
SOLVED = tuple(spec for spec in DESIGN_INPUTS if spec.apply is None)


# --------------------------------------------------------------------------- plausibility


def implied(row: dict[str, float], airframe: Airframe) -> dict[str, float]:
    """The row read back as the airframe's physical quantities (real units), holding its mass,
    wing area, aspect ratio, shaft power and propeller diameter: C_Lmax, C_D0, Oswald e,
    propeller efficiency (thrust power over shaft power), static figure of merit (momentum
    theory), static thrust over weight, the g limit over the structure's at this mass, and the
    held wing loading (kg/m^2) and shaft power loading (W/kg)."""
    a = airframe
    induced = card.INDUCED_DRAG_FACTOR * row["induced_drag_scale"] * a.per_metre
    thrust = to_real(row["static_thrust"], -2) * a.mass  # N
    structure = a.g_limit * a.g_limit_mass / a.mass  # the structure's g limit at this mass
    return dict(
        max_lift_coefficient=row["max_lift_coefficient"] / a.per_metre,
        zero_lift_drag_coefficient=row["zero_lift_drag"] / a.per_metre,
        oswald=1.0 / (math.pi * a.aspect_ratio * induced),
        propeller_efficiency=to_real(row["engine_power"], -3) * a.mass / a.power,
        figure_of_merit=math.sqrt(thrust**3 / (2.0 * AIR_DENSITY * a.propeller_disc)) / a.power,
        static_thrust_over_weight=row["static_thrust"] / card.GRAVITY,
        structural_load=row["max_acceleration"] / card.GRAVITY / structure,
        wing_loading=a.mass / a.wing_area,
        power_loading=a.power / a.mass,
    )


@dataclasses.dataclass(frozen=True)
class Check:
    """A plausibility check: `quantity` (implied's, or a flown one) outside `low` to `high`
    (None: unbounded) warns, with `reason`. `severity` "limit" for a physical limit, "range"
    for outside the documented aircraft; `input` the design input it points at."""

    quantity: str
    low: float | None
    high: float | None
    severity: str
    reason: str
    input: str


# docs/rewrite/ship-design-proposal.md section 7, as data.
PLAUSIBILITY = (
    Check("oswald", None, 1.0, "limit", "above the elliptic wing's 1.0", "sustained_rate"),
    Check("oswald", 0.78, 0.93, "range",
          "outside 0.78-0.93 (Fw 190 0.78-0.84, Bf 109 0.93; NACA P-51B 0.88-0.91)",
          "sustained_rate"),
    Check("propeller_efficiency", None, 1.0, "limit", "more thrust power than shaft power",
          "best_climb"),
    Check("propeller_efficiency", 0.7, 0.9, "range", "outside 0.7-0.9 (P-47D measured 0.78-0.80)",
          "best_climb"),
    Check("zero_lift_drag_coefficient", 0.0, None, "limit", "no drag", "top_speed"),
    Check("zero_lift_drag_coefficient", 0.0148, 0.034, "range",
          "outside 0.0148-0.034 (P-51D powered fit 0.0148-0.0163; Fw 190 A-8 0.0265-0.034)",
          "top_speed"),
    Check("max_lift_coefficient", None, 1.7, "range",
          "above the P-51D's stall-derived 1.50-1.70", "stall_speed"),
    Check("figure_of_merit", None, 1.0, "limit", "beyond momentum theory", "static_thrust"),
    Check("structural_load", None, 1.0, "range",
          "above the structure's limit at this mass (P-51D: 8 g at 8,000 lb, 6.6 g at 9,760 lb)",
          "g_limit"),
    Check("gap", None, 0.9, "range", "less than the 10% gap the flight test asks", "gap"),
    Check("sustained_speed_change", -0.12, 0.12, "range",
          "best sustained speed over 12% from the point mass's", "sustained_rate"),
    Check("full_stick_fall", None, 1e-6, "range",
          "the full-stick rate falls with speed below corner", "flap_lift"),
    Check("thrust_loss", None, 0.1, "range",
          "the soft thrust limit takes over a tenth of the thrust at the best-climb speed"
          " (v2's softness 2 did)", "best_climb_speed"),
)


def _thrust_loss(row: dict[str, float], speed: float) -> float:
    """The share of the thrust power's thrust P / v the soft static-thrust limit takes at
    `speed` (core.softmin, as the flight tick)."""
    free = row["engine_power"] / speed
    thrust = core.softmin(*(torch.tensor(x, dtype=torch.float64) for x in (
        free, row["static_thrust"], row["thrust_softmin_exponent"])))
    return 1.0 - float(thrust) / free


def warnings(row: dict[str, float], reference: Reference, flown: dict | None = None) -> list[dict]:
    """The checks of PLAUSIBILITY the row fails, each as {quantity, value, severity, input,
    message}. `flown`: the row's performance, if already computed."""
    flown = flown or performance(row)
    values = dict(
        implied(row, reference.airframe),
        gap=flown["gap"],
        sustained_speed_change=(
            flown["sustained_speed"] / reference.point_mass["sustained_speed"] - 1.0
        ),
        full_stick_fall=flown["full_stick_fall"],
        thrust_loss=_thrust_loss(row, flown["best_climb_speed"]),
    )
    out = []
    for check in PLAUSIBILITY:
        value = values[check.quantity]
        below = check.low is not None and value < check.low
        if below or (check.high is not None and value > check.high):
            message = f"{check.quantity} {value:.4g}: {check.reason}"
            out.append(dict(quantity=check.quantity, value=value, severity=check.severity,
                            input=check.input, message=message))
    return out


# --------------------------------------------------------------------------- the derived card

# The source the derived rows depend on: the flight model, the card's constants, this route.
FLIGHT_SOURCES = tuple(
    Path(__file__).parent / name for name in ("core.py", "card.py", "tuning.py")
) + (Path(__file__).parent.parent / "contract.py",)


def real_row(aircraft: RealAircraft) -> dict[str, float]:
    """A real aircraft's card row: its reference's design, unchanged (design_row)."""
    return design_row(reference_of(aircraft)).row


def row_key(aircraft: RealAircraft) -> str:
    """16 hex digits of the SHA-256 of the aircraft (data, flight test, values) and
    FLIGHT_SOURCES: a derived row's cache key (any change to the flight model's source
    re-solves)."""
    digest = hashlib.sha256()
    targets = [dataclasses.asdict(aircraft.airframe), aircraft.top_speed, aircraft.best_climb,
               aircraft.values]
    digest.update(json.dumps(targets, sort_keys=True).encode())
    for path in FLIGHT_SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def cache_directory() -> Path:
    """Where derived rows are kept: $BNB_CARD_CACHE, default ~/.cache/bnb-card."""
    return Path(os.environ.get("BNB_CARD_CACHE", "~/.cache/bnb-card")).expanduser()


def cached_real_row(name: str) -> dict[str, float]:
    """REAL_AIRCRAFT[name]'s row: from the disk cache when its key is there, else solved
    (real_row, a few seconds) and written (atomically; an unwritable cache is skipped)."""
    aircraft = REAL_AIRCRAFT[name]
    path = cache_directory() / f"{name}-{row_key(aircraft)}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    row = real_row(aircraft)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_text(json.dumps(row, sort_keys=True))
        os.replace(partial, path)
    except OSError:
        pass
    return row


_CARD_LOCK = threading.RLock()
_DERIVING = threading.local()


def load_card() -> dict[str, tuple[float, ...]]:
    """card.CARD, derived once per process (card.__getattr__ calls this on first use): every
    type's row (cached_real_row) as {field: one value per type in TYPE_NAMES' order}."""
    with _CARD_LOCK:
        if "CARD" in vars(card):
            return card.CARD
        if getattr(_DERIVING, "active", False):
            raise RuntimeError("card.CARD read while it is being derived")
        _DERIVING.active = True
        try:
            rows = [cached_real_row(name) for name in card.TYPE_NAMES]
        finally:
            _DERIVING.active = False
        for name, row in zip(card.TYPE_NAMES, rows):
            if set(row) != set(card.CARD_FIELDS):
                off = sorted(set(row) ^ set(card.CARD_FIELDS))
                raise ValueError(f"{name}'s derived row differs from the card's fields: {off}")
        card.CARD = {field: tuple(float(row[field]) for row in rows) for field in card.CARD_FIELDS}
        return card.CARD




# --------------------------------------------------------------------------- Newton


def _newton(residual, logs, tolerance, iterations=40, broyden=False, name="solve"):
    """Newton in log parameters with a finite-difference Jacobian (v2's solver): `residual(logs)`
    a list of log ratios flown / target, driven below `tolerance`. `broyden`: after the first
    Jacobian, rank-one updates with no new evaluations (the web tool's fast path). The step is a
    pseudo-inverse's (a parameter with no sensitivity stays where it is), clamped to 0.5 in each
    log. Returns logs, residual (the largest |r|), iterations and condition (of the last
    Jacobian, None if none was needed)."""
    step = 1e-5
    jacobian, last = None, None
    for iteration in range(iterations):
        r = residual(logs)
        if max(abs(x) for x in r) < tolerance:
            condition = None if jacobian is None else float(torch.linalg.cond(jacobian))
            return SimpleNamespace(logs=logs, residual=max(abs(x) for x in r),
                                   iterations=iteration, condition=condition)
        r_tensor = torch.tensor(r, dtype=torch.float64)
        if broyden and jacobian is not None:
            moved_by, change = last[0], r_tensor - last[1]
            jacobian = jacobian + torch.outer(change - jacobian @ moved_by, moved_by) / (moved_by @ moved_by)
        else:
            jacobian = torch.zeros(len(r), len(logs), dtype=torch.float64)
            for j in range(len(logs)):
                moved = list(logs)
                moved[j] += step
                jacobian[:, j] = (torch.tensor(residual(moved), dtype=torch.float64) - r_tensor) / step
        delta = (torch.linalg.pinv(jacobian, rtol=1e-9) @ -r_tensor).clamp(-0.5, 0.5)
        last = (delta, r_tensor)
        logs = [x + float(d) for x, d in zip(logs, delta)]
    raise RuntimeError(f"{name} did not converge: residual {residual(logs)}")


# --------------------------------------------------------------------------- quasi-steady checks


@contextlib.contextmanager
def _with_row(row: dict[str, float]):
    """The card with `row` appended as a last type, restored afterwards; the core's built
    ship cards are dropped on entry and exit (as in time_scale). While the card is itself being
    derived (load_card) the row is its only type."""
    saved = None if getattr(_DERIVING, "active", False) else card.CARD
    base = saved if saved is not None else {name: () for name in card.CARD_FIELDS}
    card.CARD = {name: (*values, float(row[name])) for name, values in base.items()}
    core._SHIP_CARDS.clear()
    try:
        yield len(next(iter(base.values())))
    finally:
        if saved is None:
            del card.CARD
        else:
            card.CARD = saved
        core._SHIP_CARDS.clear()


def _bisect(function, low, high, steps=40):
    """Per element, the root of a function positive at low and negative at high."""
    for _ in range(steps):
        middle = 0.5 * (low + high)
        positive = function(middle) > 0.0
        low, high = torch.where(positive, middle, low), torch.where(positive, high, middle)
    return 0.5 * (low + high)



REFINE_POINTS = 33  # a local grid across the two grid steps around an extremum


def _extremum(function, speeds: torch.Tensor, values: torch.Tensor):
    """The speed and value of the maximum of `function` (speeds -> values, -inf where it does
    not hold), from its values on the grid `speeds`: a local grid of REFINE_POINTS between the
    best grid point's neighbours, then a parabola through the best three, the value flown at its
    vertex. Smooth in the row's coefficients to far below the grid (the solve's finite
    differences need it; the grid's argmax alone moves in steps)."""
    k = int(torch.argmax(values))
    low, high = speeds[max(k - 1, 0)], speeds[min(k + 1, len(speeds) - 1)]
    local = torch.linspace(float(low), float(high), REFINE_POINTS, dtype=torch.float64)
    y = function(local)
    j = int(torch.argmax(y))
    if 0 < j < REFINE_POINTS - 1 and bool(torch.isfinite(y[j - 1 : j + 2]).all()):
        y0, y1, y2 = (float(x) for x in y[j - 1 : j + 2])
        curvature = y0 - 2.0 * y1 + y2
        if curvature < 0.0:
            vertex = float(local[j]) + 0.5 * (y0 - y2) / curvature * float(local[1] - local[0])
            speed = torch.tensor([vertex], dtype=torch.float64)
            return vertex, float(function(speed)[0])
    return float(local[j]), float(y[j])


def performance(
    row: dict[str, float],
    plain: bool = False,
    config: core.GameConfig | None = None,
    speed_step: float = 0.5,
) -> dict[str, float]:
    """The row's quasi-steady numbers in the game's flight model (game units, coasting,
    mid-store): stall (1 g) speed, corner speed and rate, top speed, best climb (Ps, m/s) and
    its speed, best sustained turn (rate, speed, radius), the tightest sustained turn (min_radius
    and its speed), the gap (that speed over the best sustained rate's), the settled full-stick
    turn (rate, speed), the low-speed acceleration (straight at the stall speed, empty store),
    and full_stick_fall (the most the full-stick rate falls with rising speed below corner,
    rad/s; 0 when it never does). Rates in rad/s. Each extremum is refined below the
    `speed_step` grid (`_extremum`), so the numbers are smooth in the row's coefficients."""
    config = config or core.GameConfig()
    if plain:
        row = point_mass(row)
    with torch.no_grad(), _with_row(row) as index:
        constants = core.per_type_constants(config, torch.tensor([index]))
        c = SimpleNamespace(
            **{
                k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in vars(constants).items()
            }
        )
        energy = 0.5 * c.max_energy
        stall = math.sqrt(card.GRAVITY / row["max_lift_coefficient"])

        def flown(speeds, stick, store=energy):
            return core.quasi_steady(c, speeds, torch.full_like(speeds, stick), 0, store)

        def full_rate(speeds):
            return flown(speeds, 1.0)[0]

        def climb(speeds):
            return torch.where(speeds > stall, flown(speeds, 0.0)[2] / card.GRAVITY, -math.inf)

        def sustained(speeds):
            """Per speed the stick where the energy rate is zero (full stick if it gains), and
            whether the turn holds (above stall, energy held, straight flight holds energy)."""
            zero, one = torch.zeros_like(speeds), torch.ones_like(speeds)
            stick = _bisect(lambda s: core.quasi_steady(c, speeds, s, 0, energy)[2], zero, one)
            stick = torch.where(flown(speeds, 1.0)[2] >= 0.0, one, stick)
            rate, _, energy_rate = core.quasi_steady(c, speeds, stick, 0, energy)
            straight = flown(speeds, 0.0)[2]
            held = (speeds > stall) & (energy_rate > -1e-6) & (straight >= 0.0) & (rate > 1e-6)
            return rate, held

        def sustained_rate(speeds):
            rate, held = sustained(speeds)
            return torch.where(held, rate, -math.inf)

        def tightness(speeds):  # minus the sustained radius
            rate, held = sustained(speeds)
            return torch.where(held, -speeds / rate.clamp_min(1e-9), -math.inf)

        speeds = torch.arange(30.0, 700.0, speed_step, dtype=torch.float64)
        out = {"stall_speed": stall}
        out["corner_speed"] = math.sqrt(row["max_acceleration"] / row["max_lift_coefficient"])
        full = full_rate(speeds)
        out["corner_speed_flown"], out["corner_rate"] = _extremum(full_rate, speeds, full)
        below = full[speeds < out["corner_speed_flown"]]
        fall = torch.cummax(below, 0).values - below
        out["full_stick_fall"] = float(fall.max()) if len(below) else 0.0
        straight = flown(speeds, 0.0)[1]
        top = int(torch.nonzero((speeds > stall) & (straight > 0.0))[-1])
        out["top_speed"] = float(
            _bisect(lambda v: flown(v, 0.0)[1], speeds[top : top + 1], speeds[top + 1 : top + 2])
        )
        out["best_climb_speed"], out["best_climb"] = _extremum(climb, speeds, climb(speeds))
        turning = speeds[(speeds > stall - speed_step) & (speeds < out["top_speed"] + speed_step)]
        rate, held = sustained(turning)  # held only between stall and top speed
        out["sustained_speed"], out["sustained_rate"] = _extremum(
            sustained_rate, turning, torch.where(held, rate, -math.inf)
        )
        out["sustained_radius"] = out["sustained_speed"] / out["sustained_rate"]
        speed, tight = _extremum(
            tightness, turning, torch.where(held, -turning / rate.clamp_min(1e-9), -math.inf)
        )
        # The radius has a kink where the sustained turn leaves full stick (above it full stick
        # loses energy): the tightest turn may sit on it, which no parabola finds.
        gains = torch.nonzero(flown(turning, 1.0)[2] >= 0.0)
        if len(gains) and int(gains[-1]) + 1 < len(turning):
            k = int(gains[-1])
            departure = _bisect(
                lambda v: flown(v, 1.0)[2], turning[k : k + 1], turning[k + 1 : k + 2]
            )
            if float(tightness(departure)[0]) > tight:
                speed, tight = float(departure[0]), float(tightness(departure)[0])
        out["min_radius_speed"], out["min_radius"] = speed, -tight
        out["gap"] = out["min_radius_speed"] / out["sustained_speed"]
        # the settled full-stick turn: the speed where full stick holds speed
        k = int(torch.nonzero(flown(speeds, 1.0)[1] > 0.0)[-1])
        settled = _bisect(lambda v: flown(v, 1.0)[1], speeds[k : k + 1], speeds[k + 1 : k + 2])
        out["settled_speed"] = float(settled)
        out["settled_turn_rate"] = float(full_rate(settled)[0])
        stall_speed = torch.tensor([stall], dtype=torch.float64)
        empty = torch.zeros_like(energy)
        out["low_speed_acceleration"] = float(flown(stall_speed, 0.0, empty)[1][0])
    return out


def envelope(
    row: dict[str, float],
    speeds,
    sticks,
    power: int = 0,
    store_fraction: float | None = None,
    plain: bool = False,
    config: core.GameConfig | None = None,
    minimum_speed: float = 100.0,
) -> dict:
    """The row's quasi-steady turn envelope (game units, the nose settled, no fields, no guns)
    on a (stick, speed) grid (core.steady_rates): `speeds` (m/s) and `sticks` (0 to 1) as 1-D sequences, `power`
    0 coast, 1 dive, 2 climb, at `store_fraction` of the store (default the spawn store).
    Returns numpy arrays shaped (len(sticks), len(speeds)): rate (rad/s), radius (m), ps
    (m/s altitude-equivalent total energy rate), speed_rate, store_rate; and per speed the
    sustained turn (the largest rate with Ps >= 0, nan if none): sustained_rate,
    sustained_radius; and the key numbers from the full-stick line (corner_*) and, at `minimum_speed`
    and above (below it the turn cannot be held in flight), the sustained line
    (best_sustained_*, tightest_sustained_*; `minimum_speed`, game m/s), or None when there are none. The charts
    (docs/rewrite/sections/turns/doghouse.py) and the web tool draw from this."""
    import numpy as np

    config = config or core.GameConfig()
    if plain:
        row = point_mass(row)
    speeds = torch.as_tensor(speeds, dtype=torch.float32)
    sticks = torch.as_tensor(sticks, dtype=torch.float32)
    v = speeds.repeat(len(sticks))
    s = sticks.repeat_interleave(len(speeds))
    b = v.shape[0]
    with torch.no_grad(), _with_row(row) as index:
        p = core.per_type_constants(config, torch.full((b,), index))
        energy = (
            p.spawn_energy.clone()
            if store_fraction is None
            else store_fraction * p.max_energy
        )
        rate, speed_rate, store_rate = core.steady_rates(p, v, s, power, energy)
    shape = (len(sticks), len(speeds))
    rate = rate.reshape(shape).numpy()
    speed_rate = speed_rate.reshape(shape).numpy()
    store_rate = store_rate.reshape(shape).numpy()
    speed_axis = speeds.numpy()
    radius = speed_axis[None, :] / np.maximum(rate, 1e-9)
    ps = (speed_axis[None, :] * speed_rate + store_rate) / card.GRAVITY
    held_rate = np.full(len(speeds), np.nan)
    held_radius = np.full(len(speeds), np.nan)
    for j in range(len(speeds)):
        ok = np.where(ps[:, j] >= 0.0)[0]
        if len(ok):
            i = ok[np.argmax(rate[ok, j])]
            held_rate[j], held_radius[j] = rate[i, j], radius[i, j]
    out = dict(
        speeds=speed_axis, sticks=sticks.numpy(), rate=rate, radius=radius, ps=ps,
        speed_rate=speed_rate, store_rate=store_rate,
        sustained_rate=held_rate, sustained_radius=held_radius,
    )
    full = rate[-1]
    k = int(np.nanargmax(full))
    out["corner_speed"], out["corner_rate"] = float(speed_axis[k]), float(full[k])
    usable = np.where(speed_axis >= minimum_speed, held_rate, np.nan)
    if np.any(np.isfinite(usable)):
        k = int(np.nanargmax(usable))
        out["best_sustained_speed"], out["best_sustained_rate"] = float(speed_axis[k]), float(usable[k])
        tight = np.where(np.isfinite(usable), held_radius, np.nan)
        k = int(np.nanargmin(tight))
        out["tightest_sustained_speed"], out["tightest_sustained_radius"] = float(speed_axis[k]), float(tight[k])
    else:
        for name in ("best_sustained_speed", "best_sustained_rate", "tightest_sustained_speed", "tightest_sustained_radius"):
            out[name] = None
    return out


def stick_levels(row: dict[str, float], config: core.GameConfig | None = None) -> dict[str, float]:
    """The row's stick levels as fractions of the most lateral acceleration at each speed
    (core.stick_table, derived from its flight: the sustained level holds energy at the best
    sustained turn, the fine one is fine_stick_fraction of it): {"fine", "sustained", "full"}."""
    with torch.no_grad(), _with_row(row) as index:
        p = core.per_type_constants(config or core.GameConfig(), torch.tensor([index]))
    return dict(fine=float(p.fine_stick[0]), sustained=float(p.sustained_stick[0]), full=1.0)


def ps_over_rate(grid: dict, rate_axis, rate=None, ps=None):
    """Ps on a regular (rate, speed) grid: per speed, interpolated over the stick sweep of
    `grid` (envelope's output; `rate` and `ps` override its arrays, e.g. in other units), nan
    above the full-stick rate. The doghouse's shaded field."""
    import numpy as np

    rate = grid["rate"] if rate is None else rate
    ps = grid["ps"] if ps is None else ps
    field = np.full((len(rate_axis), rate.shape[1]), np.nan)
    for j in range(rate.shape[1]):
        order = np.argsort(rate[:, j])
        inside = rate_axis <= rate[order, j][-1]
        field[inside, j] = np.interp(rate_axis[inside], rate[order, j], ps[order, j])
    return field


@contextlib.contextmanager
def flying(row: dict[str, float], factor: float = TIME_FACTOR):
    """Inside, the card holds `row` (game units at TIME_FACTOR) as its only type, index 0, at
    `factor` times real time (time_scale): a tool's way of flying or measuring a tuned design.
    Process-global like time_scale; restored afterwards."""
    saved = card.CARD
    card.CARD = {name: (float(row[name]),) for name in saved}
    try:
        with time_scale(factor):
            yield
    finally:
        card.CARD = saved
        core._SHIP_CARDS.clear()


# Each performance number's power of time (for to_real); radii, coefficients 0.
TIME_POWERS = dict(
    stall_speed=-1,
    corner_speed=-1,
    corner_speed_flown=-1,
    corner_rate=-1,
    top_speed=-1,
    best_climb=-1,
    best_climb_speed=-1,
    sustained_rate=-1,
    sustained_speed=-1,
    sustained_radius=0,
    min_radius=0,
    min_radius_speed=-1,
    settled_speed=-1,
    settled_turn_rate=-1,
    low_speed_acceleration=-2,
    gap=0,
    full_stick_fall=-1,
)


def in_real_units(numbers: dict[str, float]) -> dict[str, float]:
    """performance() in real units (speeds and rates halved, accelerations quartered, radii
    unchanged; Ps is a speed: halved)."""
    return {name: to_real(value, TIME_POWERS[name]) for name, value in numbers.items()}


def row_of(name: str) -> dict[str, float]:
    """A card type's row by name."""
    index = card.TYPE_NAMES.index(name)
    return {field: values[index] for field, values in card.CARD.items()}


# --------------------------------------------------------------------------- other time scales

# Each card column's power of time in its units (to_game): speeds -1, accelerations and J/kg -2,
# powers per kg -3, rates -1, durations +1; lengths, per-metre coefficients, fractions, angles,
# damage and the scales of rule constants 0 (the rule constants scale instead: RULE_TIME_POWERS).
CARD_TIME_POWERS = dict(
    max_lift_coefficient=0, max_acceleration=-2, zero_lift_drag=0, engine_power=-3,
    static_thrust=-2, exchange_acceleration=-2, corner_nose_frequency=-1, max_shield=0,
    hit_radius=0, core_radius=0, hit_damage=0, muzzle_speed=-1, fire_rate=-1, bullet_life=1,
    explosive_share=0, induced_drag_scale=0, stall_angle_scale=0,
    bullet_spread_scale=0, bullet_drag_scale=0, shield_recharge_rate_scale=0,
    shield_recharge_delay_scale=0, head_on_fraction_scale=0, head_on_angle_scale=0,
    head_on_sharpness=0, fine_stick_fraction=0, vision_range_scale=0,
    critical_speed=-1, drag_rise_exponent=0, separation_drag=0, separation_exponent=0,
    lift_softmin_exponent=0, thrust_softmin_exponent=0, nose_damping=0, flap_lift=0,
    flap_blowback_ratio=0, flap_drag_scale=0,
)
# The rule constants (lean.game.card) with time in their units.
RULE_TIME_POWERS = dict(
    GRAVITY=-2, NOSE_FREQUENCY_FLOOR=-1, SPEED_FLOOR=-1,
    BULLET_SPREAD=-1, SHIELD_RECHARGE_RATE=-1, SHIELD_RECHARGE_DELAY=1, BOUNDARY_MAX_DAMAGE=-1,
    SPAWN_SPEED=-1,
)
assert set(CARD_TIME_POWERS) == set(card.CARD_FIELDS)


def time_scaled(factor: float, tick_length: float = 1.0 / 30.0):
    """The card and the rule constants at `factor` times real time (the game's are TIME_FACTOR):
    each value converted to real time and back at the new factor, so gravity is factor^2 x 9.81.
    The tick stays tick_length of game time. Guns too; a bullet's life is capped at what
    contract.BULLET_SLOTS holds at the fire interval (whole ticks): at 1.5x it is 11 ticks, not
    15. Returns (card dict, {rule constant: value})."""
    from lean.contract import BULLET_SLOTS

    ratio = TIME_FACTOR / factor

    def scaled(value, time_power):
        return value * ratio**time_power

    table = {name: tuple(scaled(v, CARD_TIME_POWERS[name]) for v in values) for name, values in card.CARD.items()}
    lives = []
    for rate, life in zip(table["fire_rate"], table["bullet_life"]):
        interval = max(1, round(1.0 / (tick_length * rate)))
        lives.append(min(life, BULLET_SLOTS * interval * tick_length))
    table["bullet_life"] = tuple(lives)
    rules = {name: scaled(getattr(card, name), power) for name, power in RULE_TIME_POWERS.items()}
    return table, rules


@contextlib.contextmanager
def time_scale(factor: float):
    """Inside, lean.game.card holds the card and rules at `factor` times real time (time_scaled),
    restored afterwards; the core's built ship cards are dropped on entry and exit. For flight
    testing tools (the sandbox's --time-scale); training never enters it. At TIME_FACTOR the
    card is the game's."""
    saved = card.CARD, {name: getattr(card, name) for name in RULE_TIME_POWERS}
    if factor != TIME_FACTOR:
        table, rules = time_scaled(factor)
        card.CARD = table
        for name, value in rules.items():
            setattr(card, name, value)
    core._SHIP_CARDS.clear()
    try:
        yield
    finally:
        card.CARD = saved[0]
        for name, value in saved[1].items():
            setattr(card, name, value)
        core._SHIP_CARDS.clear()
