"""Ship tuning: a real aircraft, or a design's handles, into a lean card row (lean.game.card) at
the game's time scale, under the game's flight model.

The game plays at TIME_FACTOR (2) times real time (v2's real_scale.EQUIVALENT_FACTOR): lengths
are real metres, game time is real time over 2, so speeds are x2, accelerations x4 (gravity
39.24 = 4 x 9.81), powers per kg x8 and turn rates x2; turn radii, and the lift and drag
coefficients per metre, are unchanged. `to_game` and `to_real` convert.

Two routes (docs/rewrite/p51d-model.md):

- `airframe_row`: an aircraft's physical data (mass, wing area, C_Lmax, C_D0, aspect ratio and
  Oswald e, engine power and propeller) converted directly: no fitting, no gameplay tuning.
  The point-mass conversion of the October fit (docs/rewrite/physics-checks/planes.py
  `from_real`): lift and zero-lift drag per metre rho S C / (2 m), induced drag factor
  k / (rho S / 2 m) with k = 1 / (pi e AR), thrust power P eta / m, static thrust by momentum
  theory (FM^2 P^2 2 rho A)^(1/3) / m; dive and climb trade at 1 g.
- `handles_row`: a design's handles (v2's bnb.game.handling.Handles names, in game units): the
  direct ones in v2's closed forms (bnb.game.handling.solver._flight_card, weapons.combat_card),
  and the three flown handles that mean something under the game's flight model solved
  quasi-steadily (cruise speed -> engine power, low-speed acceleration -> static thrust, the
  settled full-stick turn -> induced drag; Newton in log parameters with finite differences, as
  v2's solver). The dive and zoom accelerations at cruise set the trade (their mean).

The store size, the spawn store and the shot's cost are the game's own (two loops at corner
speed, half of it, free), so no row carries them. The nose frequency and the stick levels are
control feel, not physics: they come from the base row (GAME_FIELDS for a new design). The
flight terms (card.FLIGHT_TERMS) are per type; a real aircraft converted directly flies with
the game terms off (POINT_MASS_TERMS: the pure point mass of the October fit), and
`performance(..., plain=True)` turns them off for any row.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
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


# The P-51D: the October fit to the author's flight-test data (docs/rewrite/physics-checks/
# real_fit.py, decisions.md 2026-10-23): 4,427 kg and 1,630 hp (67 in Hg) as tested, 21.83 m^2,
# AR 5.83, C_Lmax 1.50 from the clean 101-102 mph stall, Oswald e 0.8 (assumed), C_D0 0.0149
# and propeller efficiency 0.82 fitted to the tested 375 mph and 3,600 ft/min at sea level,
# Hamilton Standard propeller 3.40 m (11 ft 2 in). From v2's research
# (bnb.reference.real_planes): the g limit 8, the stall angle 15.5 deg.
P51D = Airframe(
    mass=4427.0,
    wing_area=21.83,
    max_lift_coefficient=1.50,
    zero_lift_drag_coefficient=0.0149,
    aspect_ratio=5.83,
    oswald=0.8,
    power=1630.0 * HORSEPOWER,
    propeller_efficiency=0.82,
    propeller_diameter=3.40,
    g_limit=8.0,
    stall_angle=math.radians(15.5),
)

# A design's fields that are not its airframe: guns, damage, shield, the nose's reversal, the
# stick levels and vision (v2's Vizia's), and the flight terms the point mass keeps (v2's).
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
    fine_stick=0.21010109782218933,
    sustained_stick=0.7003369927406311,
    vision_range_scale=0.3255208432674408,
    drag_rise_exponent=12.0,
    separation_exponent=8.0,
    nose_damping=0.7,
)


# The game terms switched off: no drag rise, no separation drag, soft minima near hard (exponent
# 400). A real aircraft flies as its point mass with these (test_p51d_point_mass_is_the_october_fit).
POINT_MASS_TERMS = dict(
    critical_speed=math.inf,
    separation_drag=0.0,
    lift_softmin_exponent=400.0,
    thrust_softmin_exponent=400.0,
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


def _flown(row: dict[str, float]) -> list[float]:
    """The three flown handles the flight model keeps: cruise, low-speed acceleration, settled
    full-stick turn rate (game units)."""
    p = performance(row)
    return [p["top_speed"], p["low_speed_acceleration"], p["settled_turn_rate"]]


def handles_row(
    handles: dict[str, float], base: dict[str, float], iterations: int = 40
) -> dict[str, float]:
    """A card row from a design's handles (game units); fields the handles do not set come
    from `base` (nose frequency, stick levels, the flight terms). See the module docstring."""
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
        handles["low_speed_acceleration"],
        handles["sustained_turn_rate"],
    ]

    def residual(logs):
        trial = {**row, **{name: math.exp(x) for name, x in zip(SOLVED, logs)}}
        return [math.log(f / t) for f, t in zip(_flown(trial), target)]

    # First guess: thrust power to hold cruise against the zero-lift and 1 g induced drag; the
    # static thrust from the low-speed acceleration; the base's induced drag.
    k = card.INDUCED_DRAG_FACTOR * base["induced_drag_scale"]
    v = handles["cruise_speed"]
    power = (row["zero_lift_drag"] * v**2 + k * card.GRAVITY**2 / v**2) * v
    stall = handles["stall_speed"]
    static = (
        handles["low_speed_acceleration"]
        + row["zero_lift_drag"] * stall**2
        + k * card.GRAVITY**2 / stall**2
    )
    logs = [math.log(power), math.log(static), math.log(base["induced_drag_scale"])]
    step = 1e-5
    for _ in range(iterations):
        r = residual(logs)
        if max(abs(x) for x in r) < 1e-7:
            break
        jacobian = torch.zeros(3, 3, dtype=torch.float64)
        for j in range(3):
            moved = list(logs)
            moved[j] += step
            jacobian[:, j] = (
                torch.tensor(residual(moved), dtype=torch.float64)
                - torch.tensor(r, dtype=torch.float64)
            ) / step
        delta = torch.linalg.solve(jacobian, -torch.tensor(r, dtype=torch.float64)).clamp(-0.5, 0.5)
        logs = [x + float(d) for x, d in zip(logs, delta)]
    else:
        raise RuntimeError(f"handles_row did not converge: residual {residual(logs)}")
    row.update({name: math.exp(x) for name, x in zip(SOLVED, logs)})
    return row


# --------------------------------------------------------------------------- quasi-steady checks


@contextlib.contextmanager
def _with_row(row: dict[str, float]):
    """The card with `row` appended as a last type, restored afterwards."""
    saved = card.CARD
    card.CARD = {name: (*values, float(row[name])) for name, values in saved.items()}
    try:
        yield len(next(iter(saved.values())))
    finally:
        card.CARD = saved


def _evaluate(c, speed, stick, power, energy):
    """Quasi-steady (doghouse.py's evaluate): turn rate (rad/s), speed rate, total energy rate."""
    control = SimpleNamespace(
        thrust_scale=1.0,
        stick_size=stick,
        stick_sign=torch.ones_like(speed),
        commanded=c.commanded_table[..., power],
        demand=c.demand_table[..., power],
    )
    setting = core._setting(c, control, speed, energy)
    lateral, _, _, thrust = setting
    reach = (lateral.abs() / (speed * speed) * c.inverse_lift).clamp(0.0, 1.0)
    angle = c.stall * (1.0 - torch.sqrt(1.0 - reach))
    lift = core._lift_coefficient(c, angle)
    rate = (lift * speed * speed + thrust * torch.sin(angle)) / speed
    speed_rate, store_rate = core._rates(
        c, setting, torch.ones_like(speed), speed, lift, torch.cos(angle), 1.0 / (speed * speed)
    )
    return rate, speed_rate, speed * speed_rate + store_rate


def _bisect(function, low, high, steps=60):
    """Per element, the root of a function positive at low and negative at high."""
    for _ in range(steps):
        middle = 0.5 * (low + high)
        positive = function(middle) > 0.0
        low, high = torch.where(positive, middle, low), torch.where(positive, high, middle)
    return 0.5 * (low + high)


def performance(
    row: dict[str, float], plain: bool = False, config: core.GameConfig | None = None
) -> dict[str, float]:
    """The row's quasi-steady numbers in the game's flight model (game units, coasting,
    mid-store): stall (1 g) speed, corner speed and rate, top speed, best climb (Ps, m/s) and
    its speed, best sustained turn (rate, speed, radius), minimum sustained radius and its
    speed, the settled full-stick turn (rate, speed), the low-speed acceleration (straight at
    the stall speed, empty store). Rates in rad/s."""
    config = config or core.GameConfig()
    if plain:
        row = point_mass(row)
    with torch.no_grad(), _with_row(row) as index:
        speeds = torch.arange(30.0, 700.0, 0.05, dtype=torch.float64)
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
        full_rate, full_speed_rate, _ = _evaluate(c, speeds, one, 0, energy)
        k = int(torch.argmax(full_rate))
        out["corner_rate"], out["corner_speed_flown"] = float(full_rate[k]), float(speeds[k])
        _, straight, straight_energy = _evaluate(c, speeds, zero, 0, energy)
        above = speeds > out["stall_speed"]
        top = int(torch.nonzero(above & (straight > 0.0))[-1])
        out["top_speed"] = float(
            _bisect(
                lambda v: _evaluate(_take(c, top), v, zero[:1], 0, energy[:1])[1],
                speeds[top : top + 1],
                speeds[top + 1 : top + 2],
            )
        )
        climb = torch.where(above, straight_energy / card.GRAVITY, -math.inf)
        k = int(torch.argmax(climb))
        out["best_climb"], out["best_climb_speed"] = float(climb[k]), float(speeds[k])
        # sustained: per speed the stick where the energy rate is zero (full stick if positive)
        stick = _bisect(lambda s: _evaluate(c, speeds, s, 0, energy)[2], zero, one)
        stick = torch.where(_evaluate(c, speeds, one, 0, energy)[2] >= 0.0, one, stick)
        rate, _, energy_rate = _evaluate(c, speeds, stick, 0, energy)
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
                lambda v: _evaluate(_take(c, k), v, one[:1], 0, energy[:1])[1],
                speeds[k : k + 1],
                speeds[k + 1 : k + 2],
            )
        )
        out["settled_speed"] = settled
        out["settled_turn_rate"] = float(
            _evaluate(
                _take(c, k), torch.tensor([settled], dtype=torch.float64), one[:1], 0, energy[:1]
            )[0]
        )
        stall = torch.tensor([out["stall_speed"]], dtype=torch.float64)
        out["low_speed_acceleration"] = float(
            _evaluate(_take(c, 0), stall, zero[:1], 0, zero[:1])[1]
        )
    return out


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
    head_on_sharpness=0, fine_stick=0, sustained_stick=0, vision_range_scale=0,
    critical_speed=-1, drag_rise_exponent=0, separation_drag=0, separation_exponent=0,
    lift_softmin_exponent=0, thrust_softmin_exponent=0, nose_damping=0,
)
# The rule constants (lean.game.card) with time in their units.
RULE_TIME_POWERS = dict(
    GRAVITY=-2, NOSE_FREQUENCY_FLOOR=-1, SPEED_FLOOR=-1,
    BULLET_SPREAD=-1, SHIELD_RECHARGE_RATE=-1, SHIELD_RECHARGE_DELAY=1, BOUNDARY_MAX_DAMAGE=-1,
    SPAWN_SPEED=-1,
)
assert set(CARD_TIME_POWERS) == set(card.CARD)


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
