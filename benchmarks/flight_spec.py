"""Prototype: solve between flight constants and handling quantities.

Every closed-form handling quantity of the current flight model is a product
of powers of the physics constants, ``h = scale · Π θ_j^{A_ij}``, so in logs
the map is linear: ``log h = A · log θ + log scale``. Pinning a set of
handling targets is then a linear system in ``log θ``:

* square and full rank: a unique, exact solution;
* fewer targets than constants: the smallest change from a base config
  (minimum norm in log space), and the directions left free are reported;
* redundant targets: the dependencies between them are reported, and if the
  pinned values disagree, the least-squares compromise and its residuals.

Speeds are proper px/s at n = 1. ``max_power`` is held fixed: power units are
a gauge freedom, since K only appears as K·max_power, K·P and K·bullet cost.

The closed forms ignore the slipped thrust's share of turning (radii read
about 5% wide) and time-step effects; ``--verify`` measures the solved config
with the flight-envelope harness.

Example:
    uv run --no-sync python -m benchmarks.flight_spec --verify
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.frontline import frontline_ship_config

# Free constants θ of the current flight model, by ShipConfig field name.
# reverse_thrust is negative; it is solved by magnitude and the sign restored.
NEGATIVE = frozenset({"reverse_thrust"})
PARAMETERS = (
    "base_thrust",
    "boost_thrust",
    "reverse_thrust",
    "power_speed_constant",
    "passive_power_gain",
    "no_turn_drag_coeff",
    "normal_turn_drag_coeff",
    "sharp_turn_drag_coeff",
    "normal_turn_lift_coeff",
    "sharp_turn_lift_coeff",
    "normal_turn_angle",
    "sharp_turn_angle",
    "min_speed",
)


@dataclass(frozen=True)
class Quantity:
    """A handling quantity ``scale · Π θ^exponent``."""

    name: str
    exponents: dict[str, float]
    unit: str
    description: str
    scale: float = 1.0


def quantities(max_power: float) -> tuple[Quantity, ...]:
    """The closed-form handling quantities of the current model."""
    third = 1.0 / 3.0
    regen = {"power_speed_constant": third, "passive_power_gain": third}
    return (
        Quantity(
            "cruise_speed",
            {"base_thrust": 0.5, "no_turn_drag_coeff": -0.5},
            "px/s",
            "coast terminal, straight",
        ),
        Quantity(
            "top_speed",
            {**regen, "no_turn_drag_coeff": -third},
            "px/s",
            "energy-limited boost terminal, straight",
        ),
        Quantity(
            "unlimited_top_speed",
            {"boost_thrust": 0.5, "no_turn_drag_coeff": -0.5},
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
        Quantity("sharp_radius", {"sharp_turn_lift_coeff": -1.0}, "px", "sharp turn radius"),
        Quantity("normal_radius", {"normal_turn_lift_coeff": -1.0}, "px", "normal turn radius"),
        Quantity(
            "sharp_sustained_speed",
            {**regen, "sharp_turn_drag_coeff": -third},
            "px/s",
            "energy-limited boost terminal, sharp turn",
        ),
        Quantity(
            "sharp_sustained_rate",
            {**regen, "sharp_turn_drag_coeff": -third, "sharp_turn_lift_coeff": 1.0},
            "°/s",
            "turn rate at that speed",
            scale=math.degrees(1.0),
        ),
        Quantity(
            "normal_sustained_speed",
            {**regen, "normal_turn_drag_coeff": -third},
            "px/s",
            "energy-limited boost terminal, normal turn",
        ),
        Quantity(
            "normal_sustained_rate",
            {**regen, "normal_turn_drag_coeff": -third, "normal_turn_lift_coeff": 1.0},
            "°/s",
            "turn rate at that speed",
            scale=math.degrees(1.0),
        ),
        Quantity(
            "coast_sharp_speed",
            {"base_thrust": 0.5, "sharp_turn_drag_coeff": -0.5},
            "px/s",
            "coast terminal, sharp turn",
        ),
        Quantity(
            "coast_normal_speed",
            {"base_thrust": 0.5, "normal_turn_drag_coeff": -0.5},
            "px/s",
            "coast terminal, normal turn",
        ),
        Quantity(
            "normal_gun_offset",
            {"normal_turn_angle": 1.0},
            "°",
            "gun offset in a normal turn",
            scale=math.degrees(1.0),
        ),
        Quantity(
            "sharp_gun_offset",
            {"sharp_turn_angle": 1.0},
            "°",
            "gun offset in a sharp turn",
            scale=math.degrees(1.0),
        ),
        Quantity("stall_speed", {"min_speed": 1.0}, "px/s", "below this, no lift"),
    )


def exponent_matrix(chosen: tuple[Quantity, ...]) -> np.ndarray:
    """``A`` with one row per quantity and one column per parameter."""
    matrix = np.zeros((len(chosen), len(PARAMETERS)))
    for row, quantity in enumerate(chosen):
        for name, exponent in quantity.exponents.items():
            matrix[row, PARAMETERS.index(name)] = exponent
    return matrix


def log_parameters(config: ShipConfig) -> np.ndarray:
    return np.log([abs(getattr(config, name)) for name in PARAMETERS])


def evaluate(config: ShipConfig) -> dict[str, float]:
    """Closed-form value of every handling quantity for ``config``."""
    catalogue = quantities(config.max_power)
    log_h = exponent_matrix(catalogue) @ log_parameters(config)
    return {q.name: q.scale * math.exp(v) for q, v in zip(catalogue, log_h)}


@dataclass
class Solution:
    config: ShipConfig
    rank: int
    pinned: tuple[str, ...]
    # Relations the pinned quantities must satisfy among themselves, as
    # {quantity: exponent} with Π h^exponent fixed by the model alone.
    dependencies: list[dict[str, float]] = field(default_factory=list)
    # Parameter directions no pinned quantity constrains.
    free_directions: list[dict[str, float]] = field(default_factory=list)
    # Pinned value / achieved value − 1, nonzero only for inconsistent pins.
    residuals: dict[str, float] = field(default_factory=dict)


def _readable(vector: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    """Normalize a null-space vector to max |component| 1 and drop small terms."""
    vector = vector / np.abs(vector).max()
    if vector[np.abs(vector).argmax()] < 0:
        vector = -vector
    return {n: round(float(v), 3) for n, v in zip(names, vector) if abs(v) > 1e-6}


def solve(targets: dict[str, float], base: ShipConfig, tol: float = 1e-9) -> Solution:
    """Constants that hit ``targets`` with the smallest log change from ``base``."""
    by_name = {q.name: q for q in quantities(base.max_power)}
    unknown = sorted(set(targets) - set(by_name))
    if unknown:
        raise KeyError(f"unknown quantities: {unknown}")
    chosen = tuple(by_name[name] for name in targets)
    matrix = exponent_matrix(chosen)  # (S, P)
    target = np.log([targets[q.name] / q.scale for q in chosen])  # (S,)
    x0 = log_parameters(base)  # (P,)

    # Minimum-norm step: x = x0 + A⁺ (y − A x0). Exact when consistent,
    # least squares otherwise; parameters outside A's row space keep base.
    x = x0 + np.linalg.pinv(matrix, rcond=tol) @ (target - matrix @ x0)

    u, singular, vt = np.linalg.svd(matrix)
    rank = int((singular > tol * max(singular.max(), 1.0)).sum())
    dependencies = [_readable(u[:, i], tuple(targets)) for i in range(rank, matrix.shape[0])]
    free = [_readable(vt[i], PARAMETERS) for i in range(rank, matrix.shape[1])]

    achieved = matrix @ x
    residuals = {
        q.name: round(float(math.exp(t - a) - 1.0), 6)
        for q, t, a in zip(chosen, target, achieved)
        if abs(t - a) > 1e-9
    }
    values = {}
    for name, log_value, log_base in zip(PARAMETERS, x, x0):
        # Constants the solve does not move keep their exact base value.
        if abs(log_value - log_base) < 1e-12:
            value = abs(getattr(base, name))
        else:
            value = math.exp(log_value)
        values[name] = -value if name in NEGATIVE else value
    return Solution(replace(base, **values), rank, tuple(targets), dependencies, free, residuals)


# The handling of today's game, pinned as one full-rank set of 13.
SPEC_QUANTITIES = (
    "cruise_speed",
    "top_speed",
    "boost_accel",
    "reverse_decel",
    "recharge_time",
    "tank_speed",
    "sharp_radius",
    "normal_radius",
    "sharp_sustained_rate",
    "normal_sustained_rate",
    "normal_gun_offset",
    "sharp_gun_offset",
    "stall_speed",
)


def verify(config: ShipConfig, device: torch.device) -> dict[str, tuple[float, float]]:
    """Closed form against the flight-envelope harness at n = 1: {name: (closed, measured)}."""
    from benchmarks.flight_envelope import measure_terminal

    rows = measure_terminal(config, device, indices=(1.0,), seconds=90.0)
    measured = {(r["power_mode"], r["turn"]): r for r in rows}
    closed = evaluate(config)
    pairs = {
        "cruise_speed": ("coast", "GO_STRAIGHT", "proper_speed"),
        "top_speed": ("boost", "GO_STRAIGHT", "proper_speed"),
        "unlimited_top_speed": ("boost_unlimited", "GO_STRAIGHT", "proper_speed"),
        "sharp_sustained_speed": ("boost", "SHARP_LEFT", "proper_speed"),
        "normal_sustained_speed": ("boost", "TURN_LEFT", "proper_speed"),
        "coast_sharp_speed": ("coast", "SHARP_LEFT", "proper_speed"),
        "coast_normal_speed": ("coast", "TURN_LEFT", "proper_speed"),
        "sharp_radius": ("coast", "SHARP_LEFT", "radius_world_px"),
        "normal_radius": ("coast", "TURN_LEFT", "radius_world_px"),
        "sharp_sustained_rate": ("boost", "SHARP_LEFT", "turn_rate_deg_s"),
        "normal_sustained_rate": ("boost", "TURN_LEFT", "turn_rate_deg_s"),
    }
    return {
        name: (closed[name], abs(measured[(mode, turn)][key]))
        for name, (mode, turn, key) in pairs.items()
    }


def _print_solution(title: str, solution: Solution, base: ShipConfig) -> None:
    print(f"\n== {title}")
    print(f"pinned {len(solution.pinned)}, rank {solution.rank}, parameters {len(PARAMETERS)}")
    for dependency in solution.dependencies:
        print(f"  dependency (Π h^e fixed): {dependency}")
    for direction in solution.free_directions:
        print(f"  free direction: {direction}")
    for name, residual in solution.residuals.items():
        print(f"  residual {name}: {residual:+.2%}")
    print(f"  {'parameter':<24}{'base':>12}{'solved':>12}")
    for name in PARAMETERS:
        before, after = getattr(base, name), getattr(solution.config, name)
        mark = "" if math.isclose(before, after, rel_tol=1e-6) else "  *"
        print(f"  {name:<24}{before:>12.5g}{after:>12.5g}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verify", action="store_true", help="measure solved configs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    base = frontline_ship_config(SHIP_CONFIG)
    today = evaluate(base)
    catalogue = {q.name: q for q in quantities(base.max_power)}

    print("== Today's handling (closed form, n = 1)")
    for name, value in today.items():
        q = catalogue[name]
        print(f"  {name:<24}{value:>10.4g} {q.unit:<6} {q.description}")

    spec = {name: today[name] for name in SPEC_QUANTITIES}
    _print_solution(
        "Round trip: today's 13 quantities from a perturbed base",
        solve(spec, replace(base, **{p: abs(getattr(base, p)) * 1.7 for p in PARAMETERS[:3]})),
        base,
    )

    redundant = {**spec, "unlimited_top_speed": today["unlimited_top_speed"]}
    _print_solution("Redundant: add unlimited top speed", solve(redundant, base), base)

    faster = {name: today[name] for name in SPEC_QUANTITIES if name != "recharge_time"}
    faster["top_speed"] = 155.0
    _print_solution("Top speed 155, turning pinned, recharge time free", solve(faster, base), base)
    faster["unlimited_top_speed"] = today["unlimited_top_speed"]
    faster_solution = solve(faster, base)
    _print_solution(
        "Top speed 155, turning pinned, unlimited top speed pinned instead",
        faster_solution,
        base,
    )

    if args.verify:
        device = torch.device(args.device)
        for title, config in (("today", base), ("top speed 155", faster_solution.config)):
            print(f"\n== Harness check: {title}")
            print(f"  {'quantity':<24}{'closed':>10}{'measured':>10}{'error':>9}")
            for name, (closed, measured) in verify(config, device).items():
                print(f"  {name:<24}{closed:>10.4g}{measured:>10.4g}{measured / closed - 1:>9.1%}")


if __name__ == "__main__":
    main()
