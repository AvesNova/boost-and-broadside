"""Flight constants from handling targets.

Designers state handling -- turn radius, sustained and peak turn rate, corner
speed, top speed, nose onset -- and the flight constants of ``ShipConfig`` are
solved from it. Most closed-form handling quantities are products of powers of
the constants, ``h = scale * prod(theta_j ** A_ij)``, so in logs the map is
linear, ``log h = A log theta + log scale``, and pinning targets is a linear
system in ``log theta``:

* square and full rank: a unique, exact solution;
* fewer targets than constants: the smallest change from a base configuration
  (minimum norm in log space), with the directions left free reported;
* redundant targets: the dependencies among them reported, and if the pinned
  values disagree, the least-squares compromise and its residuals.

Speeds are proper px/s at n = 1, angles degrees, rates degrees per second.
``max_power`` is held fixed: power units are a gauge freedom, since K only
appears as K * max_power, K * P and K * bullet cost.

The closed forms ignore the slipped thrust's share of turning, which makes the
sharp turn about 5-7% tighter than stated, and the transient shape of the nose
response. :func:`correct` closes such gaps against a simulated forward model;
``benchmarks/flight_spec.py`` drives it with the flight-envelope harness.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace

import numpy as np

from boost_and_broadside.config.core import ShipConfig

# Free constants of the flight model, by ShipConfig field name. reverse_thrust
# is negative; it is solved by magnitude and the sign restored.
NEGATIVE = frozenset({"reverse_thrust"})
PARAMETERS = (
    "base_thrust",
    "boost_thrust",
    "reverse_thrust",
    "power_speed_constant",
    "passive_power_gain",
    "zero_slip_drag_coeff",
    "normal_slip_drag_coeff",
    "stall_drag_coeff",
    "max_lift_coeff",
    "normal_lift_coeff",
    "max_lateral_accel",
    "stall_angle",
    "nose_frequency",
    "max_slip_rate",
    "slip_fade_speed",
    "min_speed",
)

#: ``omega * t`` at which a critically damped step reaches 90% of its command:
#: the root of ``1 - (1 + s) exp(-s) = 0.9``.
ONSET_90_PERCENT = 3.889720169867429

_DEGREES = math.degrees(1.0)


@dataclass(frozen=True)
class Quantity:
    """A handling quantity ``scale * prod(theta ** exponent)``."""

    name: str
    exponents: dict[str, float]
    unit: str
    description: str
    scale: float = 1.0


def quantities(max_power: float) -> tuple[Quantity, ...]:
    """The closed-form handling quantities of the flight model."""
    third = 1.0 / 3.0
    regen = {"power_speed_constant": third, "passive_power_gain": third}

    def sustained(drag: str, lift: str, level: str) -> tuple[Quantity, Quantity]:
        speed = {**regen, drag: -third}
        return (
            Quantity(
                f"{level}_sustained_speed",
                speed,
                "px/s",
                f"energy-limited boost terminal in a {level} turn (valid below corner)",
            ),
            Quantity(
                f"{level}_sustained_rate",
                {**speed, lift: 1.0},
                "°/s",
                "turn rate at that speed (valid below corner)",
                scale=_DEGREES,
            ),
        )

    return (
        Quantity(
            "cruise_speed",
            {"base_thrust": 0.5, "zero_slip_drag_coeff": -0.5},
            "px/s",
            "coast terminal, straight",
        ),
        Quantity(
            "top_speed",
            {**regen, "zero_slip_drag_coeff": -third},
            "px/s",
            "energy-limited boost terminal, straight",
        ),
        Quantity(
            "unlimited_top_speed",
            {"boost_thrust": 0.5, "zero_slip_drag_coeff": -0.5},
            "px/s",
            "boost terminal with power held full",
        ),
        Quantity("boost_accel", {"boost_thrust": 1.0}, "px/s²", "boost acceleration from rest"),
        Quantity(
            "reverse_decel", {"reverse_thrust": 1.0}, "px/s²", "reverse deceleration, no drag"
        ),
        Quantity(
            "recharge_time",
            {"passive_power_gain": -1.0},
            "s",
            "empty to full power by regeneration",
            scale=max_power,
        ),
        Quantity(
            "tank_speed",
            {"power_speed_constant": 0.5},
            "px/s",
            "speed a full tank buys from rest, no drag",
            scale=math.sqrt(2.0 * max_power),
        ),
        Quantity("sharp_radius", {"max_lift_coeff": -1.0}, "px", "sharp turn radius below corner"),
        Quantity(
            "normal_radius", {"normal_lift_coeff": -1.0}, "px", "normal turn radius below corner"
        ),
        *sustained("stall_drag_coeff", "max_lift_coeff", "sharp"),
        *sustained("normal_slip_drag_coeff", "normal_lift_coeff", "normal"),
        Quantity(
            "coast_sharp_speed",
            {"base_thrust": 0.5, "stall_drag_coeff": -0.5},
            "px/s",
            "coast terminal, sharp turn",
        ),
        Quantity(
            "coast_normal_speed",
            {"base_thrust": 0.5, "normal_slip_drag_coeff": -0.5},
            "px/s",
            "coast terminal, normal turn",
        ),
        Quantity(
            "corner_speed",
            {"max_lateral_accel": 0.5, "max_lift_coeff": -0.5},
            "px/s",
            "slowest speed at which the g-limit caps the sharp turn",
        ),
        Quantity(
            "max_turn_rate",
            {"max_lateral_accel": 0.5, "max_lift_coeff": 0.5},
            "°/s",
            "sharp turn rate at corner speed",
            scale=_DEGREES,
        ),
        Quantity(
            "turn_rate_at_top",
            {
                "max_lateral_accel": 1.0,
                "power_speed_constant": -third,
                "passive_power_gain": -third,
                "zero_slip_drag_coeff": third,
            },
            "°/s",
            "g-limited sharp turn rate at top speed",
            scale=_DEGREES,
        ),
        Quantity(
            "onset_time",
            {"nose_frequency": -1.0},
            "s",
            "straight to 90% of a commanded slip",
            scale=ONSET_90_PERCENT,
        ),
        Quantity(
            "nose_slew_rate",
            {"max_slip_rate": 1.0},
            "°/s",
            "fastest the nose moves against the flight path",
            scale=_DEGREES,
        ),
        Quantity(
            "sharp_gun_offset",
            {"stall_angle": 1.0},
            "°",
            "nose angle to the path in a sharp turn below corner",
            scale=_DEGREES,
        ),
        Quantity("fade_speed", {"slip_fade_speed": 1.0}, "px/s", "below this, slip fades"),
        Quantity("stall_speed", {"min_speed": 1.0}, "px/s", "below this, no lift"),
    )


@dataclass(frozen=True)
class HandlingSpec:
    """Handling targets; ``None`` leaves a quantity unpinned.

    Field names match :func:`quantities`. Units are those listed there.
    """

    cruise_speed: float | None = None
    top_speed: float | None = None
    unlimited_top_speed: float | None = None
    boost_accel: float | None = None
    reverse_decel: float | None = None
    recharge_time: float | None = None
    tank_speed: float | None = None
    sharp_radius: float | None = None
    normal_radius: float | None = None
    sharp_sustained_speed: float | None = None
    sharp_sustained_rate: float | None = None
    normal_sustained_speed: float | None = None
    normal_sustained_rate: float | None = None
    coast_sharp_speed: float | None = None
    coast_normal_speed: float | None = None
    corner_speed: float | None = None
    max_turn_rate: float | None = None
    turn_rate_at_top: float | None = None
    onset_time: float | None = None
    nose_slew_rate: float | None = None
    sharp_gun_offset: float | None = None
    fade_speed: float | None = None
    stall_speed: float | None = None

    def targets(self) -> dict[str, float]:
        """The pinned quantities."""
        return {
            f.name: float(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


def exponent_matrix(chosen: tuple[Quantity, ...]) -> np.ndarray:
    """``A`` with one row per quantity and one column per parameter."""
    matrix = np.zeros((len(chosen), len(PARAMETERS)))
    for row, quantity in enumerate(chosen):
        for name, exponent in quantity.exponents.items():
            matrix[row, PARAMETERS.index(name)] = exponent
    return matrix


def log_parameters(config: ShipConfig) -> np.ndarray:
    return np.log([abs(getattr(config, name)) for name in PARAMETERS])


def handling(config: ShipConfig) -> dict[str, float]:
    """Closed-form value of every handling quantity for ``config``."""
    catalogue = quantities(config.max_power)
    log_h = exponent_matrix(catalogue) @ log_parameters(config)
    return {q.name: q.scale * math.exp(v) for q, v in zip(catalogue, log_h)}


@dataclass
class Solution:
    config: ShipConfig
    rank: int
    pinned: tuple[str, ...]
    #: Relations the pinned quantities satisfy among themselves, as
    #: ``{quantity: exponent}`` with ``prod(h ** exponent)`` fixed by the model.
    dependencies: list[dict[str, float]] = field(default_factory=list)
    #: Parameter directions no pinned quantity constrains.
    free_directions: list[dict[str, float]] = field(default_factory=list)
    #: Pinned value / achieved value - 1, nonzero only for inconsistent pins.
    residuals: dict[str, float] = field(default_factory=dict)


def _readable(vector: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    """Normalise a null-space vector to max |component| 1 and drop small terms."""
    vector = vector / np.abs(vector).max()
    if vector[np.abs(vector).argmax()] < 0:
        vector = -vector
    return {n: round(float(v), 3) for n, v in zip(names, vector) if abs(v) > 1e-6}


def _with_parameters(base: ShipConfig, values: Mapping[str, float]) -> ShipConfig:
    try:
        return replace(base, **values)
    except ValueError as error:
        raise ValueError(f"the solved flight constants are not a valid model: {error}") from error


def solve(targets: Mapping[str, float], base: ShipConfig, tol: float = 1e-9) -> Solution:
    """Constants that hit ``targets`` with the smallest log change from ``base``."""
    by_name = {q.name: q for q in quantities(base.max_power)}
    unknown = sorted(set(targets) - set(by_name))
    if unknown:
        raise KeyError(f"unknown handling quantities: {unknown}")
    names = tuple(targets)
    if not names:
        return Solution(base, 0, names, free_directions=[{p: 1.0} for p in PARAMETERS])
    chosen = tuple(by_name[name] for name in names)
    matrix = exponent_matrix(chosen)  # (S, P)
    target = np.log([targets[q.name] / q.scale for q in chosen])  # (S,)
    x0 = log_parameters(base)  # (P,)

    # Minimum-norm step: x = x0 + A⁺ (y − A x0). Exact when consistent, least
    # squares otherwise; parameters outside A's row space keep their base.
    x = x0 + np.linalg.pinv(matrix, rcond=tol) @ (target - matrix @ x0)

    u, singular, vt = np.linalg.svd(matrix)
    rank = int((singular > tol * max(singular.max(), 1.0)).sum())
    dependencies = [_readable(u[:, i], names) for i in range(rank, matrix.shape[0])]
    free = [_readable(vt[i], PARAMETERS) for i in range(rank, matrix.shape[1])]

    achieved = matrix @ x
    residuals = {
        q.name: float(math.exp(t - a) - 1.0)
        for q, t, a in zip(chosen, target, achieved)
        if abs(t - a) > 1e-9
    }
    values = {}
    for name, log_value, log_base in zip(PARAMETERS, x, x0):
        # Constants the solve does not move keep their exact base value.
        value = (
            abs(getattr(base, name)) if abs(log_value - log_base) < 1e-12 else math.exp(log_value)
        )
        values[name] = -value if name in NEGATIVE else value
    config = _with_parameters(base, values)
    return Solution(config, rank, names, dependencies, free, residuals)


def infeasibilities(config: ShipConfig) -> list[str]:
    """Handling orderings the flight model needs, each one that fails.

    Drag rising with slip and the lift ordering are enforced by ``ShipConfig``
    itself; these are the orderings between handling quantities.
    """
    h = handling(config)
    problems = []
    if not h["sharp_sustained_speed"] < h["corner_speed"] < h["top_speed"]:
        problems.append(
            "need sharp sustained speed < corner speed < top speed, got "
            f"{h['sharp_sustained_speed']:.4g} / {h['corner_speed']:.4g} / {h['top_speed']:.4g}"
        )
    if not h["normal_radius"] > h["sharp_radius"]:
        problems.append("need normal radius > sharp radius")
    if not h["fade_speed"] < h["coast_sharp_speed"]:
        problems.append(
            f"need fade speed < coasting sharp-turn speed, got {h['fade_speed']:.4g} "
            f"/ {h['coast_sharp_speed']:.4g}"
        )
    if not h["stall_speed"] < h["fade_speed"]:
        problems.append("need stall speed < fade speed")
    return problems


def ship_config_from_handling(
    spec: HandlingSpec,
    base: ShipConfig,
    max_residual: float = 0.01,
) -> ShipConfig:
    """Derive ``base``'s flight constants from ``spec``.

    Unpinned directions keep ``base``'s values. Redundant targets that disagree
    by more than ``max_residual`` are an error, as is any infeasible ordering.

    Raises:
        ValueError: The spec is inconsistent or infeasible.
    """
    solution = solve(spec.targets(), base)
    off = {name: r for name, r in solution.residuals.items() if abs(r) > max_residual}
    if off:
        raise ValueError(f"handling targets are inconsistent; residuals {off}")
    problems = infeasibilities(solution.config)
    if problems:
        raise ValueError("infeasible handling spec: " + "; ".join(problems))
    return solution.config


@dataclass
class Correction:
    """Outcome of :func:`correct`."""

    solution: Solution
    #: Pinned targets after correction: what the closed forms were asked for
    #: so that the measured envelope lands on the original targets.
    corrected_targets: dict[str, float]
    #: Measured / target - 1 for each measured pinned quantity, last iteration.
    errors: dict[str, float]
    iterations: int


def correct(
    targets: Mapping[str, float],
    base: ShipConfig,
    measure: Callable[[ShipConfig], Mapping[str, float]],
    *,
    iterations: int = 6,
    tolerance: float = 0.005,
) -> Correction:
    """Adjust closed-form targets until a forward model meets ``targets``.

    Each pass solves the linear system, measures the result, and scales every
    measured pinned target by ``target / measured``. Quantities ``measure``
    does not report are left as stated.
    """
    asked = dict(targets)
    errors: dict[str, float] = {}
    solution = solve(asked, base)
    for iteration in range(1, iterations + 1):
        measured = measure(solution.config)
        errors = {n: measured[n] / targets[n] - 1.0 for n in targets if n in measured}
        if not errors or max(abs(e) for e in errors.values()) < tolerance:
            return Correction(solution, asked, errors, iteration)
        for name, error in errors.items():
            asked[name] /= 1.0 + error
        solution = solve(asked, base)
    return Correction(solution, asked, errors, iterations)
