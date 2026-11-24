"""Ship tuning: a real aircraft, or a design's handles, into a lean card row (lean.game.card) at
the game's time scale, under the game's flight model.

The game plays at TIME_FACTOR (2) times real time (v2's real_scale.EQUIVALENT_FACTOR): lengths
are real metres, game time is real time over 2, so speeds are x2, accelerations x4 (gravity
39.24 = 4 x 9.81), powers per kg x8 and turn rates x2; turn radii, and the lift and drag
coefficients per metre, are unchanged. `to_game` and `to_real` convert.

Routes (docs/rewrite/p51d-model.md):

- `airframe_row`: an aircraft's physical data (mass, wing area, C_Lmax, C_D0, aspect ratio and
  Oswald e, engine power and propeller) converted directly: no fitting, no gameplay tuning.
  The point-mass conversion of the October fit (docs/rewrite/physics-checks/planes.py
  `from_real`): lift and zero-lift drag per metre rho S C / (2 m), induced drag factor
  k / (rho S / 2 m) with k = 1 / (pi e AR), thrust power P eta / m, static thrust by momentum
  theory (FM^2 P^2 2 rho A)^(1/3) / m; dive and climb trade at 1 g. With the game terms off
  (POINT_MASS_TERMS) the row is the aircraft's point mass. `fit_point_mass` sets C_D0 and the
  propeller efficiency so the point mass flies a tested top speed and climb.
- `game_row`: a real aircraft with the game terms on (a type's drag rise, separation drag,
  soft minima and automatic flaps): the airframe converted, then its zero-lift drag, thrust
  power and induced drag solved so it still flies its point mass's top speed, best climb and
  best sustained turn rate (core-decisions.md 2026-11-24). The P-51D's row.
- `handles_row`: a design's handles (v2's bnb.game.handling.Handles names, in game units): the
  direct ones in v2's closed forms (bnb.game.handling.solver._flight_card, weapons.combat_card),
  and the three flown handles that mean something under the game's flight model solved
  quasi-steadily (cruise speed -> engine power, low-speed acceleration -> static thrust, the
  settled full-stick turn -> induced drag; Newton in log parameters with finite differences, as
  v2's solver). The dive and zoom accelerations at cruise set the trade (their mean).

The store size, the spawn store and the shot's cost are the game's own (two loops at corner
speed, half of it, free), so no row carries them. The nose frequency and the fine stick's
fraction are control feel, not physics: they come from the base row (GAME_FIELDS for a new
design); the stick levels themselves are derived from the flight (core.stick_table,
`stick_levels`). The
flight terms (card.FLIGHT_TERMS) are per type; `performance(..., plain=True)` turns the game
terms off for any row (POINT_MASS_TERMS).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

from . import card, core

TIME_FACTOR = 2.0  # the game's time against real time (v2 real_scale.EQUIVALENT_FACTOR)
REAL_GRAVITY = 9.81
AIR_DENSITY = 1.225  # kg/m^3, sea level
HORSEPOWER = 745.7  # W
FIGURE_OF_MERIT = 0.75  # a propeller's static figure of merit (planes.py)
DAMAGE_REFERENCE_DISTANCE = 300.0  # m, the set's rule (v2 ships/real-scale/rules.toml)
LIFE_TOLERANCE = 1e-4  # v2 weapons.LIFE_TOLERANCE
assert abs(card.GRAVITY - REAL_GRAVITY * TIME_FACTOR**2) < 1e-9


def to_game(value: float, time_power: int) -> float:
    """A real quantity in game units, given the power of time in its units (speed -1,
    acceleration -2, power per kg -3, rate -1; lengths and per-metre coefficients 0)."""
    return value * TIME_FACTOR ** (-time_power)


def to_real(value: float, time_power: int) -> float:
    return value * TIME_FACTOR**time_power


# --------------------------------------------------------------------------- airframe route


@dataclasses.dataclass(frozen=True)
class Airframe:
    """An aircraft's physical data, SI, real time. power is the shaft power (W) at the rating
    the drag data were fitted at; propeller_efficiency turns it into thrust power."""

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
    stall_angle: float  # rad, angle of attack at C_Lmax

    @property
    def per_metre(self) -> float:
        """rho S / (2 m): a coefficient's acceleration per v^2, 1/m."""
        return AIR_DENSITY * self.wing_area / (2.0 * self.mass)

    def static_thrust(self) -> float:
        """Static thrust (N) by momentum theory: FM P = T^1.5 / sqrt(2 rho A)."""
        disc = math.pi * self.propeller_diameter**2 / 4.0
        return (FIGURE_OF_MERIT**2 * self.power**2 * 2.0 * AIR_DENSITY * disc) ** (1.0 / 3.0)


# The P-51D's data (docs/rewrite/aircraft-references.md, p51d-model.md): 4,427 kg and 1,630 hp
# at 67 in Hg (war emergency rating) as tested (AAF TSCEP5E-1908); wing area 21.67 m^2 (NACA ACR
# 4K02); AR 5.83; C_Lmax 1.6 (the AAF's power-on value for its turn calculation; stall-derived
# values span 1.50-1.70); Oswald e 0.8 (assumed); Hamilton Standard propeller 3.40 m (11 ft 2 in);
# from v2's research (bnb.reference.real_planes) the g limit 8 and the stall angle 15.5 deg. C_D0
# and the propeller efficiency are not data: fit_point_mass sets them so the point mass flies
# the tested 375 mph and 3,600 ft/min at sea level (the October procedure, physics-checks/
# real_fit.py), every time the card is derived (real_row); the values here are its start.
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
    stall_angle=math.radians(15.5),
)
MPH, FEET_PER_MINUTE = 0.44704, 0.00508  # m/s
P51D_TOP_SPEED = 375.0 * MPH  # sea level, 67 in Hg, 4,427 kg (flight test), real m/s
P51D_BEST_CLIMB = 3600.0 * FEET_PER_MINUTE  # the same test, real m/s
SPEED_OF_SOUND = 340.3  # m/s, ISA sea level

# A design's fields that are not its airframe: guns, damage, shield, the nose's reversal and
# vision (v2's Vizia's), the fine stick's fraction of the sustained level (author 2026-11-24:
# half; the sustained level is derived, core.stick_table), and the flight terms the point mass
# keeps (v2's).
GAME_FIELDS = dict(
    corner_nose_frequency=11.23144817352295,
    max_shield=100.0,
    hit_radius=12.0,
    core_radius=5.0,
    hit_damage=21.0,
    muzzle_speed=1712.0,
    fire_rate=30.0,
    bullet_life=0.36666667461395264,
    explosive_share=0.0,
    reference_bullet_speed=1753.606201171875,
    bullet_spread_scale=0.25704020261764526,
    bullet_drag_scale=1.0,
    shield_recharge_rate_scale=2.0,
    shield_recharge_delay_scale=0.5,
    head_on_fraction_scale=0.0,
    head_on_angle_scale=1.0644670724868774,
    head_on_sharpness=3.0,
    fine_stick_fraction=0.5,  # the fine level over the sustained one (the levels: core.stick_table)
    vision_range_scale=0.3255208432674408,
    drag_rise_exponent=12.0,
    separation_exponent=8.0,
    nose_damping=0.7,
    flap_lift=0.0,
    flap_blowback_ratio=1.0,
    flap_drag_scale=0.0,
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

# The P-51D's game terms (core-decisions.md 2026-11-24: the terms on, the solver compensating).
# - Drag rise: the zero-lift drag doubles at Mach 0.77, exponent 20: NACA ACR 4K02 Fig. 12 (the
#   P-51B, propeller off) is flat to about M 0.68, then +40% at M 0.735 and +67% at M 0.755.
# - Separation drag 0.5, exponent 24 (the induced drag x 1.5 at the wing's maximum lift, +4.0%
#   at 0.9 of it): it bends the sustained turn's line away from the lift limit, so the tightest
#   sustained turn sits clearly slower than the best sustained rate (speed ratio 0.848). The
#   2026-11-24 sweep (docs/rewrite/p51d-model.md): at exponent 8 no coefficient gives both a
#   10% gap and Oswald e <= 1 (0.35: 0.897 with e 1.07); at 24, 0.5 gives e 0.952.
# - Soft minima: lift against the g limit 16 (v2's), thrust against static thrust 8.
# - Automatic flaps (core._wing): off (flap_lift 0, author 2026-11-24); the machinery stays,
#   switchable per type. A game choice when on (the P-51D had no combat flap setting): flap_lift
#   x the maximum lift when the stick asks for more than the clean wing gives, blown back by the
#   clean corner speed (over its last 40%, card.FLAP_BLOWBACK_BAND); the extra lift costs
#   Delta C_D = 1.5 Delta C_L^2 with the flaps fully used (P51D_FLAP_DRAG; cubic below that).
P51D_FLAP_DRAG = 1.5  # Delta C_D per (Delta C_L)^2 of lift beyond the clean maximum, flaps fully used
P51D_TERMS = dict(
    critical_speed=to_game(0.77 * SPEED_OF_SOUND, -1),
    drag_rise_exponent=20.0,
    separation_drag=0.5,
    separation_exponent=24.0,
    lift_softmin_exponent=16.0,
    thrust_softmin_exponent=8.0,
    flap_lift=0.0,
    flap_blowback_ratio=1.0,
    flap_drag_scale=P51D_FLAP_DRAG / (P51D.per_metre * card.INDUCED_DRAG_FACTOR),
)


def point_mass(row: dict[str, float]) -> dict[str, float]:
    """A row with the game terms off (POINT_MASS_TERMS); the other flight terms kept."""
    return {**row, **POINT_MASS_TERMS}


def airframe_row(
    airframe: Airframe, base: dict[str, float], terms: dict[str, float] | None = None
) -> dict[str, float]:
    """A card row: the flight fields from the airframe, the flight terms `terms` over the base's
    (default POINT_MASS_TERMS: the real aircraft's point mass), everything else from `base`."""
    a = airframe
    induced = 1.0 / (math.pi * a.oswald * a.aspect_ratio) / a.per_metre  # m
    power = to_game(a.power * a.propeller_efficiency / a.mass, -3)
    flight = dict(
        max_lift_coefficient=a.per_metre * a.max_lift_coefficient,
        max_acceleration=a.g_limit * card.GRAVITY,
        zero_lift_drag=a.per_metre * a.zero_lift_drag_coefficient,
        induced_drag_scale=induced / card.INDUCED_DRAG_FACTOR,
        engine_power=power,
        static_thrust=to_game(a.static_thrust() / a.mass, -2),
        stall_angle_scale=a.stall_angle / card.STALL_ANGLE,
        exchange_acceleration=card.GRAVITY,  # 1 g: a vertical dive or zoom
    )
    return {**base, **flight, **(POINT_MASS_TERMS if terms is None else terms)}


def fit_point_mass(airframe: Airframe, top_speed: float, best_climb: float) -> Airframe:
    """The airframe with C_D0 and the propeller efficiency set so its point mass (the game terms
    off) flies `top_speed` and `best_climb` (real m/s, sea level); the other data kept (the
    October procedure: docs/rewrite/physics-checks/real_fit.py)."""
    targets = (to_game(top_speed, -1), to_game(best_climb, -1))

    def fitted(logs):
        return dataclasses.replace(
            airframe,
            zero_lift_drag_coefficient=math.exp(logs[0]),
            propeller_efficiency=math.exp(logs[1]),
        )

    def residual(logs):
        p = performance(airframe_row(fitted(logs), GAME_FIELDS))
        return [math.log(p["top_speed"] / targets[0]), math.log(p["best_climb"] / targets[1])]

    start = [math.log(airframe.zero_lift_drag_coefficient), math.log(airframe.propeller_efficiency)]
    return fitted(_newton(residual, start, 1e-7, name="fit_point_mass"))


GAME_TARGETS = ("top_speed", "best_climb", "sustained_rate")
GAME_SOLVED = ("zero_lift_drag", "engine_power", "induced_drag_scale")


def game_row(airframe: Airframe, terms: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    """A real aircraft's card row with the game terms on (core-decisions.md 2026-11-24): the
    airframe converted (airframe_row) with `terms`, then the low-level coefficients GAME_SOLVED
    (zero-lift drag, thrust power, induced drag) solved so the row flies its point mass's top
    speed, best climb and best sustained turn rate (GAME_TARGETS; Newton, as handles_row). The
    rest is the airframe's: the lift and the stall speed, the g limit and so the corner speed,
    the stall angle, the dive and climb trade, and the static thrust (momentum theory; no target
    sees it). The best sustained turn's speed is not solved: the terms are chosen to keep it
    near the point mass's (test_p51d_game_row_flies_its_point_mass).

    Solved first without the flaps, then with them from there: the flaps' low-speed part of
    the sustained turn is lift-limited, so no drag coefficient moves it, and a start whose
    clean best turn lies below it would see no way to the target."""
    row = airframe_row(airframe, base, terms)
    reference = performance(airframe_row(airframe, base))
    targets = [reference[name] for name in GAME_TARGETS]
    logs = [math.log(row[name]) for name in GAME_SOLVED]
    for stage in ({**row, "flap_lift": 0.0}, row):

        def residual(logs, stage=stage):
            trial = {**stage, **{name: math.exp(x) for name, x in zip(GAME_SOLVED, logs)}}
            flown = performance(trial)
            return [math.log(flown[name] / target) for name, target in zip(GAME_TARGETS, targets)]

        logs = _newton(residual, logs, 1e-7, name="game_row")
    row.update({name: math.exp(x) for name, x in zip(GAME_SOLVED, logs)})
    return row


# --------------------------------------------------------------------------- the derived card


@dataclasses.dataclass(frozen=True)
class RealAircraft:
    """A real aircraft's targets: its data (C_D0 and the propeller efficiency only a start),
    its flight test (real m/s, sea level) and its game terms."""

    airframe: Airframe
    top_speed: float
    best_climb: float
    terms: dict


# The card's real aircraft by type name (card.TYPE_NAMES).
REAL_AIRCRAFT = {"p51d": RealAircraft(P51D, P51D_TOP_SPEED, P51D_BEST_CLIMB, P51D_TERMS)}
# The source the derived rows depend on: the flight model, the card's constants, this route.
FLIGHT_SOURCES = tuple(
    Path(__file__).parent / name for name in ("core.py", "card.py", "tuning.py")
) + (Path(__file__).parent.parent / "contract.py",)


def real_row(aircraft: RealAircraft) -> dict[str, float]:
    """A real aircraft's card row from its targets: the point mass fitted to the flight test
    (fit_point_mass), then the game terms on with the low-level coefficients solved (game_row)."""
    fitted = fit_point_mass(aircraft.airframe, aircraft.top_speed, aircraft.best_climb)
    return game_row(fitted, aircraft.terms, GAME_FIELDS)


def row_key(aircraft: RealAircraft) -> str:
    """16 hex digits of the SHA-256 of the aircraft's targets, the base fields and FLIGHT_SOURCES:
    a derived row's cache key (any change to the flight model's source re-solves)."""
    digest = hashlib.sha256()
    targets = [dataclasses.asdict(aircraft.airframe), aircraft.top_speed, aircraft.best_climb,
               aircraft.terms, GAME_FIELDS]
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


# --------------------------------------------------------------------------- handles route


def direct_fields(
    handles: dict[str, float], cruise_speed: float, tick_length: float = 1.0 / 30.0
) -> dict[str, float]:
    """The card fields v2 sets in closed form from the handles (solver._flight_card's direct
    ones and weapons.combat_card); cruise_speed is the flown cruise (the bullet's launch)."""
    launch = cruise_speed + handles["muzzle_speed"]
    drag = handles["bullet_drag"]
    flight = math.expm1(drag * handles["gun_range"]) / (drag * launch)
    life = math.ceil(flight / tick_length - LIFE_TOLERANCE) * tick_length
    return dict(
        max_lift_coefficient=card.GRAVITY / handles["stall_speed"] ** 2,
        max_acceleration=handles["g_limit"] * card.GRAVITY,
        zero_lift_drag=handles["zero_lift_drag"],
        stall_angle_scale=handles["stall_angle"] / card.STALL_ANGLE,
        vision_range_scale=handles["vision_range"] / card.VISION_RANGE,
        hit_damage=handles["hit_damage"],
        explosive_share=handles["explosive_share"],
        reference_bullet_speed=launch * math.exp(-drag * DAMAGE_REFERENCE_DISTANCE),
        fire_rate=handles["fire_rate"],
        bullet_life=life,
        muzzle_speed=handles["muzzle_speed"],
        bullet_spread_scale=handles["bullet_spread"] / card.BULLET_SPREAD,
        bullet_drag_scale=drag / card.BULLET_DRAG,
        max_shield=handles["max_shield"],
        shield_recharge_rate_scale=handles["shield_recharge_rate"] / card.SHIELD_RECHARGE_RATE,
        shield_recharge_delay_scale=handles["shield_recharge_delay"] / card.SHIELD_RECHARGE_DELAY,
        head_on_fraction_scale=handles["head_on_fraction"] / card.HEAD_ON_FRACTION,
        head_on_angle_scale=handles["head_on_angle"] / card.HEAD_ON_ANGLE,
        head_on_sharpness=handles["head_on_sharpness"],
        hit_radius=handles["hit_radius"],
        core_radius=handles["core_radius"],
    )


SOLVED = ("engine_power", "static_thrust", "induced_drag_scale")


def _flown(row: dict[str, float], speed_step: float = 0.05) -> list[float]:
    """The three flown handles the flight model keeps: cruise, low-speed acceleration, settled
    full-stick turn rate (game units)."""
    p = performance(row, speed_step=speed_step, flown_only=True)
    return [p["top_speed"], p["low_speed_acceleration"], p["settled_turn_rate"]]


def handles_row(
    handles: dict[str, float],
    base: dict[str, float],
    iterations: int = 40,
    speed_step: float = 0.05,
    fast: bool = False,
) -> dict[str, float]:
    """A card row from a design's handles (game units); fields the handles do not set come
    from `base` (nose frequency, the fine stick's fraction, the flight terms). `speed_step`:
    the speed grid of the flown numbers the solve matches (a tool may take a coarser one). `fast` (for an
    interactive tool; the default solve is unchanged): start from the base's engine power and induced drag, Broyden-update
    the Jacobian, stop at a residual of 1e-5 (the flown handles to 0.001%).
    `handles["low_speed_acceleration"]` may be None (or absent): the static thrust is then the
    base's and the low-speed acceleration is not matched (a power-limited ship such as the P-51D
    flies the same whatever its static thrust).
    See the module docstring."""
    if abs(handles["half_lift_drag"] - 0.25) > 1e-9:
        raise ValueError(
            "lean's card has the square polar only (half-lift drag 0.25, bleed exponent 2)"
        )
    row = {**base, **direct_fields(handles, handles["cruise_speed"])}
    # Dive and zoom at cruise: the commanded trade is the speed's rate change.
    trade = 0.5 * (handles["dive_acceleration"] - handles["zoom_acceleration"])
    row["exchange_acceleration"] = trade
    target = [
        handles["cruise_speed"],
        handles.get("low_speed_acceleration"),
        handles["sustained_turn_rate"],
    ]
    free_static = target[1] is None  # the static thrust kept as the base's: a tool's handles

    def residual(logs):
        trial = {**row, **{name: math.exp(x) for name, x in zip(SOLVED, logs)}}
        return [
            0.0 if t is None else math.log(f / t)
            for f, t in zip(_flown(trial, speed_step), target)
        ]

    # First guess: thrust power to hold cruise against the zero-lift and 1 g induced drag; the
    # static thrust from the low-speed acceleration; the base's induced drag.
    k = card.INDUCED_DRAG_FACTOR * base["induced_drag_scale"]
    v = handles["cruise_speed"]
    power = (row["zero_lift_drag"] * v**2 + k * card.GRAVITY**2 / v**2) * v
    stall = handles["stall_speed"]
    static = (
        (handles["low_speed_acceleration"] if not free_static else 0.0)
        + row["zero_lift_drag"] * stall**2
        + k * card.GRAVITY**2 / stall**2
    )
    logs = [math.log(power), math.log(static), math.log(base["induced_drag_scale"])]
    if free_static:
        logs[1] = math.log(base["static_thrust"])
    if fast:
        logs[0], logs[2] = math.log(base["engine_power"]), math.log(base["induced_drag_scale"])
    logs = _newton(residual, logs, 1e-5 if fast else 1e-7, iterations, fast, "handles_row")
    row.update({name: math.exp(x) for name, x in zip(SOLVED, logs)})
    return row


def _newton(residual, logs, tolerance, iterations=40, broyden=False, name="solve"):
    """Newton in log parameters with a finite-difference Jacobian (v2's solver): `residual(logs)`
    a list of log ratios flown / target, driven below `tolerance`. `broyden`: after the first
    Jacobian, rank-one updates with no new evaluations (the interactive tool's fast path). The
    step is a pseudo-inverse's (a parameter with no sensitivity, such as the static thrust of a
    power-limited design, stays where it is), clamped to 0.5 in each log."""
    step = 1e-5
    jacobian, last = None, None
    for _ in range(iterations):
        r = residual(logs)
        if max(abs(x) for x in r) < tolerance:
            return logs
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


def performance(
    row: dict[str, float],
    plain: bool = False,
    config: core.GameConfig | None = None,
    speed_step: float = 0.05,
    flown_only: bool = False,
) -> dict[str, float]:
    """The row's quasi-steady numbers in the game's flight model (game units, coasting,
    mid-store): stall (1 g) speed, corner speed and rate, top speed, best climb (Ps, m/s) and
    its speed, best sustained turn (rate, speed, radius), minimum sustained radius and its
    speed, the settled full-stick turn (rate, speed), the low-speed acceleration (straight at
    the stall speed, empty store). Rates in rad/s. `flown_only`: just the three flown handles'
    numbers (top speed, settled turn rate, low-speed acceleration, with the speeds around them),
    which handles_row's solve needs, a third of the work."""
    config = config or core.GameConfig()
    if plain:
        row = point_mass(row)
    with torch.no_grad(), _with_row(row) as index:
        speeds = torch.arange(30.0, 700.0, speed_step, dtype=torch.float64)
        n = speeds.shape[0]
        constants = core.per_type_constants(config, torch.full((n,), index))
        c = SimpleNamespace(
            **{
                k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in vars(constants).items()
            }
        )
        energy = 0.5 * c.max_energy
        zero, one = torch.zeros_like(speeds), torch.ones_like(speeds)
        out = {}
        out["stall_speed"] = math.sqrt(card.GRAVITY / row["max_lift_coefficient"])
        out["corner_speed"] = math.sqrt(row["max_acceleration"] / row["max_lift_coefficient"])
        full_rate, full_speed_rate, _ = core.quasi_steady(c, speeds, one, 0, energy)
        k = int(torch.argmax(full_rate))
        out["corner_rate"], out["corner_speed_flown"] = float(full_rate[k]), float(speeds[k])
        _, straight, straight_energy = core.quasi_steady(c, speeds, zero, 0, energy)
        above = speeds > out["stall_speed"]
        top = int(torch.nonzero(above & (straight > 0.0))[-1])
        out["top_speed"] = float(
            _bisect(
                lambda v: core.quasi_steady(_take(c, top), v, zero[:1], 0, energy[:1])[1],
                speeds[top : top + 1],
                speeds[top + 1 : top + 2],
            )
        )
        if not flown_only:
            climb = torch.where(above, straight_energy / card.GRAVITY, -math.inf)
            k = int(torch.argmax(climb))
            out["best_climb"], out["best_climb_speed"] = float(climb[k]), float(speeds[k])
            # sustained: per speed the stick where the energy rate is zero (full stick if positive)
            stick = _bisect(lambda s: core.quasi_steady(c, speeds, s, 0, energy)[2], zero, one)
            stick = torch.where(core.quasi_steady(c, speeds, one, 0, energy)[2] >= 0.0, one, stick)
            rate, _, energy_rate = core.quasi_steady(c, speeds, stick, 0, energy)
            held = above & (energy_rate > -1e-6) & (straight_energy >= 0.0)
            sustained = torch.where(held, rate, 0.0)
            k = int(torch.argmax(sustained))
            out["sustained_rate"], out["sustained_speed"] = float(sustained[k]), float(speeds[k])
            out["sustained_radius"] = out["sustained_speed"] / out["sustained_rate"]
            radius = torch.where(
                held & (sustained > 1e-6), speeds / sustained.clamp_min(1e-9), math.inf
            )
            k = int(torch.argmin(radius))
            out["min_radius"], out["min_radius_speed"] = float(radius[k]), float(speeds[k])
        # the settled full-stick turn: the speed where full stick holds speed
        k = int(torch.nonzero(full_speed_rate > 0.0)[-1])
        settled = float(
            _bisect(
                lambda v: core.quasi_steady(_take(c, k), v, one[:1], 0, energy[:1])[1],
                speeds[k : k + 1],
                speeds[k + 1 : k + 2],
            )
        )
        out["settled_speed"] = settled
        out["settled_turn_rate"] = float(
            core.quasi_steady(
                _take(c, k), torch.tensor([settled], dtype=torch.float64), one[:1], 0, energy[:1]
            )[0]
        )
        stall = torch.tensor([out["stall_speed"]], dtype=torch.float64)
        out["low_speed_acceleration"] = float(
            core.quasi_steady(_take(c, 0), stall, zero[:1], 0, zero[:1])[1]
        )
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


def handles_of(row: dict[str, float], gun_range: float = 600.0) -> dict[str, float]:
    """A row's handles (game units): the inverse of handles_row's direct fields, with the three
    flown handles (cruise speed, low-speed acceleration, settled full-stick turn rate) as the
    row flies them, so handles_row(handles_of(row), row) returns the row's flight."""
    flown = performance(row)
    return dict(
        stall_speed=math.sqrt(card.GRAVITY / row["max_lift_coefficient"]),
        g_limit=row["max_acceleration"] / card.GRAVITY,
        zero_lift_drag=row["zero_lift_drag"],
        stall_angle=row["stall_angle_scale"] * card.STALL_ANGLE,
        vision_range=row["vision_range_scale"] * card.VISION_RANGE,
        hit_damage=row["hit_damage"],
        explosive_share=row["explosive_share"],
        fire_rate=row["fire_rate"],
        muzzle_speed=row["muzzle_speed"],
        bullet_spread=row["bullet_spread_scale"] * card.BULLET_SPREAD,
        bullet_drag=row["bullet_drag_scale"] * card.BULLET_DRAG,
        gun_range=gun_range,
        max_shield=row["max_shield"],
        shield_recharge_rate=row["shield_recharge_rate_scale"] * card.SHIELD_RECHARGE_RATE,
        shield_recharge_delay=row["shield_recharge_delay_scale"] * card.SHIELD_RECHARGE_DELAY,
        head_on_fraction=row["head_on_fraction_scale"] * card.HEAD_ON_FRACTION,
        head_on_angle=row["head_on_angle_scale"] * card.HEAD_ON_ANGLE,
        head_on_sharpness=row["head_on_sharpness"],
        hit_radius=row["hit_radius"],
        core_radius=row["core_radius"],
        half_lift_drag=0.25,
        dive_acceleration=row["exchange_acceleration"],
        zoom_acceleration=-row["exchange_acceleration"],
        cruise_speed=flown["top_speed"],
        low_speed_acceleration=flown["low_speed_acceleration"],
        sustained_turn_rate=flown["settled_turn_rate"],
    )


def _take(c, index):
    """The constants of one row (index) as a namespace of (1,) tensors."""
    return SimpleNamespace(
        **{
            k: (
                v[index : index + 1]
                if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] > index
                else v
            )
            for k, v in vars(c).items()
        }
    )


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
    explosive_share=0, reference_bullet_speed=-1, induced_drag_scale=0, stall_angle_scale=0,
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
