"""measure (card -> handles) and solve (handles -> card), ship-card spec 4.

The direct handles set their parameters in closed form, the half-lift drag the bleed
exponent among them (e = log2(1 / half_lift_drag)). The seven flown handles set seven
parameters (FLOWN_PARAMETERS) by Levenberg-Marquardt on flights of the game's own code: a
closed-form first guess, then per iteration one batch that flies the base card and its seven
perturbations (log space, finite differences), one environment each. The cruise fix
derives the engine power: P = P_min + m a_0 u_cruise, with P_min (the null power) solved
so that the flown cruise meets its handle.
"""

import functools
import math
import warnings
from collections.abc import Callable
from typing import NamedTuple

import torch

from ..physics.flight import Airframe, airframe_of, lift_coefficient, rates, setting
from ..rules import Rules
from ..ship_types import DEFAULTS, ShipTypes
from .errors import Infeasible, RedundantHandles
from .handles import FLOWN, NAMES, Handles
from .manoeuvres import (
    CLIMB_LIMIT,
    REFERENCE,
    REFERENCE_STALL_SPEED,
    REVERSAL_LIMIT,
    TURNING_STORE,
    batched,
    fly,
    steady_turn,
    stretch,
)
from .weapons import combat_card, combat_handles

# The parameter each flown handle sets, in FLOWN's order (physical values, not scales). The
# induced drag is solved as its factor at full lift, k C_max^(e - 2): the induced drag
# k S^(e/2) is then that times C_max^2 (S / C_max^2)^(e/2), so the bleed exponent (a direct
# handle, the half-lift drag) reshapes the polar below full lift and the sustained turn
# sets its size at full lift.
FLOWN_PARAMETERS = (
    "engine_power_full_store",
    "coast_exchange",
    "static_thrust",
    "full_lift_induced_drag",
    "corner_nose_frequency",
    "dive_exchange",
    "climb_exchange",
)
P = len(FLOWN_PARAMETERS)
# A handle's error is relative to max(|target|, floor), so handles near zero have a floor:
# these at the standard card's scales (manoeuvres.REFERENCE), each a multiple of its unit in
# the type's own scales (ERROR_UNITS: speed V_s, time V_s / g, acceleration g, rate g / V_s).
ERROR_FLOOR = dict(
    cruise_speed=1.0,
    climb_time=0.1,
    low_speed_acceleration=0.1,
    sustained_turn_rate=0.01,
    nose_reversal_time=1e-3,
    dive_acceleration=0.1,
    zoom_acceleration=0.1,
)
# Each flown handle's unit as powers of the type's (time, speed) scales; g = speed / time.
ERROR_UNITS = dict(
    cruise_speed=(0, 1),
    climb_time=(1, 0),
    low_speed_acceleration=(-1, 1),
    sustained_turn_rate=(-1, 0),
    nose_reversal_time=(1, 0),
    dive_acceleration=(-1, 1),
    zoom_acceleration=(-1, 1),
)
TOLERANCE = 1e-4  # stop when every flown handle is this close (the spec's bar is 5e-3)
ACCEPTANCE = 5e-3  # the spec's 0.5%: beyond it after the budget, the solve fails
ITERATIONS = 16
STEP = 1e-5  # the finite-difference step in log parameters
STEP_LIMIT = 0.5  # the largest Newton step in any log parameter
DAMPING = (1e-6, 1e6)  # Levenberg-Marquardt's: the first, and the bounds
EXPONENT_BOUNDS = (0.5, 5.0)  # the bleed exponents a card may have
CONDITION_WARNING = 1e4  # a Jacobian condition number above this warns
# The flown handles measured over a bounded flight: (its window's constant, its name, the
# stretch it flies by: `manoeuvres.Scales`' field).
WINDOWS = dict(
    nose_reversal_time=(REVERSAL_LIMIT, "REVERSAL_LIMIT", "time"),
    climb_time=(CLIMB_LIMIT, "CLIMB_LIMIT", "climb"),
)
# Why a flown handle is not measured (NaN), where it can be; {window} is its window in s.
UNMEASURED = dict(
    cruise_speed="no straight equilibrium: the drag outruns the engine at every speed",
    climb_time="the store does not fill within {window:g} s, or stops rising",
    nose_reversal_time="the nose does not reverse within {window:g} s",
)


def windows(rules: Rules, handles: Handles) -> dict[str, float]:
    """The window (s) each WINDOWS handle is flown over for a batch of these handles: its
    constant stretched by the batch's largest scales (`manoeuvres.stretch`)."""
    scale = stretch(rules.flight, handles.stall_speed, handles.ceiling)
    return {name: limit * getattr(scale, kind) for name, (limit, _, kind) in WINDOWS.items()}


def error_floor(rules: Rules, handles: Handles) -> torch.Tensor:
    """(..., num_types, P) each flown handle's error floor in its type's own scales."""
    time_scale = handles.stall_speed / rules.flight.gravity
    floors = []
    for name in FLOWN:
        time_power, speed_power = ERROR_UNITS[name]
        reference = REFERENCE.time**time_power * REFERENCE_STALL_SPEED**speed_power
        own = time_scale**time_power * handles.stall_speed**speed_power
        floors.append(ERROR_FLOOR[name] * own / reference)
    return torch.stack(floors, -1)


def exponent_of(half_lift_drag: torch.Tensor) -> torch.Tensor:
    """The bleed exponent e of a half-lift drag 0.5^e."""
    return torch.log(half_lift_drag) / math.log(0.5)


def _flight_card(rules: Rules, handles: Handles, parameters: torch.Tensor) -> dict:
    """Card fields from the direct flight handles and the flown parameters (..., P)."""
    constants = rules.flight
    value = dict(zip(FLOWN_PARAMETERS, parameters.unbind(-1)))
    exponent = exponent_of(handles.half_lift_drag)
    lift = constants.gravity / handles.stall_speed**2  # C_max
    induced = value["full_lift_induced_drag"] * lift ** (2.0 - exponent)  # k
    return dict(
        max_lift_coefficient=lift,
        max_acceleration=handles.g_limit * constants.gravity,
        zero_lift_drag=handles.zero_lift_drag,
        engine_power=(
            value["engine_power_full_store"]
            + constants.mass * value["coast_exchange"] * handles.cruise_speed
        ),
        engine_power_full_store=value["engine_power_full_store"],
        static_thrust=value["static_thrust"],
        max_energy=handles.ceiling,
        corner_nose_frequency=value["corner_nose_frequency"],
        bleed_exponent=exponent,
        induced_drag_scale=induced / constants.induced_drag_factor,
        stall_angle_scale=handles.stall_angle / constants.stall_angle,
        coast_exchange_scale=value["coast_exchange"] / constants.coast_exchange,
        dive_exchange_scale=value["dive_exchange"] / constants.dive_exchange,
        climb_exchange_scale=value["climb_exchange"] / constants.climb_exchange,
        vision_range_scale=handles.vision_range / rules.vision_range,
    )


def _card(rules: Rules, handles: Handles, parameters: torch.Tensor, combat: dict) -> ShipTypes:
    fields = _flight_card(rules, handles, parameters) | combat
    assert set(fields) == set(DEFAULTS)
    return ShipTypes(**fields)


def _batched_handles(handles: Handles) -> Handles:
    if handles.ceiling.dim() == 2:
        return handles
    return Handles(**{name: getattr(handles, name)[None] for name in NAMES})


def measure(types: ShipTypes, rules: Rules, tick_function=None) -> Handles:
    """Every handle of a card ((num_types,) or (B, num_types) fields), flown in its dtype."""
    one_card = types.max_energy.dim() == 1
    types = batched(types)
    constants = rules.flight
    flown = fly(rules, types, tick_function)
    handles = Handles(
        stall_speed=torch.sqrt(constants.gravity / types.max_lift_coefficient),
        g_limit=types.max_acceleration / constants.gravity,
        ceiling=types.max_energy,
        stall_angle=constants.stall_angle * types.stall_angle_scale,
        vision_range=rules.vision_range * types.vision_range_scale,
        zero_lift_drag=types.zero_lift_drag,
        half_lift_drag=0.5**types.bleed_exponent,
        **{name: getattr(flown, name) for name in FLOWN},
        **combat_handles(rules, types, flown.cruise_speed),
    )
    return Handles(**{n: getattr(handles, n)[0] for n in NAMES}) if one_card else handles


# ------------------------------------------------------------------ feasibility


def check(rules: Rules, handles: Handles) -> None:
    """Infeasible with the first reason found, for sets no card can meet (spec 4)."""
    constants = rules.flight
    positive = (
        "stall_speed",
        "cruise_speed",
        "climb_time",
        "ceiling",
        "sustained_turn_rate",
        "nose_reversal_time",
        "stall_angle",
        "vision_range",
        "zero_lift_drag",
        "fire_rate",
        "muzzle_speed",
        "gun_range",
        "shots_per_store",
        "max_shield",
        "hit_radius",
    )
    for name in positive:
        if not (getattr(handles, name) > 0.0).all():
            raise Infeasible(f"{name} must be positive")
    for name in (
        "bullet_spread",
        "hit_damage",
        "shield_recharge_rate",
        "shield_recharge_delay",
        "spawn_energy",
    ):
        if (getattr(handles, name) < 0.0).any():
            raise Infeasible(f"{name} must not be negative")
    if (handles.g_limit <= 1.0).any():
        raise Infeasible(
            "a g limit of at most 1 cannot hold level flight: the lift budget must exceed gravity"
        )
    if (handles.cruise_speed <= handles.stall_speed).any():
        raise Infeasible("the cruise speed must be above the stall speed")
    if (handles.stall_angle > math.pi / 4).any():
        raise Infeasible("an angle-of-attack limit above 45 degrees is outside the lift model")
    corner = handles.stall_speed * torch.sqrt(handles.g_limit)
    acceleration = handles.g_limit * constants.gravity
    peak = torch.sqrt(acceleration**2 - constants.gravity**2) / corner
    if (handles.sustained_turn_rate >= peak).any():
        raise Infeasible(
            "the sustained turn rate must be below the instantaneous peak turn "
            "sqrt(a_max^2 - g^2) / corner speed"
        )
    if (handles.spawn_energy > handles.ceiling).any():
        raise Infeasible("the spawn energy must fit in the store")
    low, high = (0.5**bound for bound in reversed(EXPONENT_BOUNDS))
    if ((handles.half_lift_drag < low) | (handles.half_lift_drag > high)).any():
        raise Infeasible(
            f"the half-lift drag must lie in [{low:.3g}, {high:.3g}] (a bleed exponent in "
            f"{list(EXPONENT_BOUNDS)})"
        )
    window_of = windows(rules, handles)
    for name, (_, constant, _) in WINDOWS.items():
        window = window_of[name]
        if (getattr(handles, name) >= window).any():
            raise Infeasible(
                f"a {name} beyond its measurement window: the manoeuvre is flown "
                f"for at most {window:g} s (manoeuvres.{constant})"
            )


def _unmeasured(residual: torch.Tensor, window_of: dict[str, float]) -> str:
    """Why the first guess flies no measurement: for each card and type that misses a flown
    handle (residual (B, types, P), NaN where not measured), the handles and their reasons
    (with the windows the handles were flown over, `windows`)."""
    missing = residual.isnan()  # (B, types, P)
    reasons = []
    for card, kind in missing.any(-1).nonzero().tolist():
        names = [FLOWN[i] for i in missing[card, kind].nonzero()[:, 0].tolist()]
        reasons.append(
            f"card {card} type {kind}: "
            + ", ".join(
                f"{name} "
                f"({UNMEASURED.get(name, 'not measured').format(window=window_of.get(name))})"
                for name in names
            )
        )
    return "; ".join(reasons)


# ------------------------------------------------------------------ first guess


@functools.cache
def _climb_integral(knee: float, band: float) -> float:
    """The climb time in units of E_max / (m u a_0): the store fraction's integral of
    1 / (1 - phi) from empty to 1 - band (midpoint rule)."""
    steps = 200_000
    end = 1.0 - band
    total = 0.0
    for i in range(steps):
        fraction = (i + 0.5) * end / steps
        t = min(max((fraction - knee) / (1.0 - knee), 0.0), 1.0)
        total += 1.0 / (1.0 - t * t * (3.0 - 2.0 * t))
    return total * end / steps


def _step_response(damping: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """The fraction of a unit step a second-order system starting at rest has covered at
    phase = omega t, for any damping ratio (under-, critically and over-damped)."""
    d2 = 1.0 - damping * damping
    d = d2.abs().sqrt().clamp_min(1e-6)
    x = d * phase
    under = torch.cos(x) + damping * torch.sin(x) / d
    over = torch.cosh(x) + damping * torch.sinh(x) / d
    return 1.0 - torch.exp(-damping * phase) * torch.where(d2 > 0.0, under, over)


def _reversal_frequency(rules: Rules, airframe: Airframe, handles: Handles, turning):
    """The corner nose frequency omega_c that reverses the nose in the reversal handle's time.

    The reversal (`manoeuvres.fly`) flies at corner speed, where the nose frequency is
    omega_c, from the steady full-stick turn's angle of attack to its mirror, to 95% of the
    step. The nose oscillator (game spec 3.5) is a damped oscillator in the angle error, but
    lift turning the flight path adds damping: e'' + (kappa + 2 zeta Omega) e' + Omega^2 e = 0,
    so its damping ratio is zeta + kappa / (2 omega_c), with kappa the path slope. kappa is
    taken at its mean over the swing: the lift's mean slope between the two angles, and the
    thrust's mean cos. Leaving that damping out (zeta alone) guessed omega_c about a third
    low; with it, the guess is within a few percent of the card that flies the time."""
    constants = rules.flight
    corner = airframe.corner_speed
    ones = torch.ones_like(corner)
    steady = steady_turn(rules, airframe, corner, turning, ones)
    angle = steady.angle.abs().clamp_min(1e-6)
    mean_slope = (
        lift_coefficient(constants, airframe, angle) - lift_coefficient(constants, airframe, -angle)
    ) / (2.0 * angle)
    current = setting(constants, airframe, corner, turning, ones, torch.zeros_like(corner))
    kappa = mean_slope * corner + current.thrust * (torch.sin(angle) / angle) / corner
    time = handles.nose_reversal_time
    # 95% of the step by `time`: monotone in omega (a faster and less damped nose), so bisect
    # it in log omega.
    low, high = torch.full_like(time, math.log(1e-2)), torch.full_like(time, math.log(1e4))
    for _ in range(80):
        middle = 0.5 * (low + high)
        omega = middle.exp()
        damping = constants.nose_damping + kappa / (2.0 * omega)
        slow = _step_response(damping, omega * time) < 0.95
        low, high = torch.where(slow, middle, low), torch.where(slow, high, middle)
    return (0.5 * (low + high)).exp()


def _drag_and_exchange(rules: Rules, airframe, speed, energy, stick, angle):
    """Drag and exchange (m/s^2) at a state, from the game's setting and rates (n = 1)."""
    current = setting(rules.flight, airframe, speed, energy, stick, torch.zeros_like(speed))
    lift = lift_coefficient(rules.flight, airframe, angle)
    speed_rate, _ = rates(
        rules.flight, airframe, current, 1.0, speed, lift, torch.cos(angle), 1.0 / (speed * speed)
    )
    drag = current.thrust * torch.cos(angle) - current.exchange - speed_rate
    return drag, current


def first_guess(rules: Rules, handles: Handles, combat: dict) -> torch.Tensor:
    """(B, num_types, P) flown parameters from closed forms in the handles, each inverting
    the game's own instantaneous equations at its manoeuvre's state (no flight), at the
    bleed exponent the half-lift drag sets."""
    constants = rules.flight
    mass = constants.mass
    stall, cruise, store = handles.stall_speed, handles.cruise_speed, handles.ceiling
    B, type_count = stall.shape
    columns = torch.arange(type_count, dtype=torch.int32, device=stall.device).expand(B, type_count)
    coast = (
        store
        * _climb_integral(constants.engine_knee, constants.store_fade_band)
        / (mass * cruise * handles.climb_time)
    )
    dive = handles.dive_acceleration - coast
    climb = coast - handles.zoom_acceleration
    if (dive <= 0.0).any() or (climb <= 0.0).any():
        raise Infeasible(
            "the dive acceleration must exceed, and the zoom acceleration fall "
            "short of, the coast exchange the climb implies"
        )
    lift = constants.gravity / stall**2  # C_max
    exponent = exponent_of(handles.half_lift_drag)
    full_lift = torch.full_like(stall, constants.induced_drag_factor)  # k C_max^(e - 2)
    thrust_cap = torch.full_like(stall, 1e6)
    zeros, ones = torch.zeros_like(stall), torch.ones_like(stall)
    turning = TURNING_STORE * store
    null_power = ones

    def airframe_at() -> Airframe:
        parameters = torch.stack([null_power, coast, thrust_cap, full_lift, ones, dive, climb], -1)
        return airframe_of(_card(rules, handles, parameters, combat), columns)

    for _ in range(3):
        airframe = airframe_at()
        # The null power at cruise, full store, drag only (the cap is the flight's to meet).
        drag, _ = _drag_and_exchange(rules, airframe, cruise, store, zeros, zeros)
        null_power = drag * cruise
        power = null_power + mass * coast * cruise
        # The static-thrust cap from the low-speed acceleration (store empty).
        airframe = airframe_at()
        drag, current = _drag_and_exchange(rules, airframe, stall, zeros, zeros, zeros)
        needed = handles.low_speed_acceleration + drag + current.exchange
        free = power / stall
        capped = (needed**-4 - free**-4).clamp_min(1e-30) ** -0.25  # softmin_4 inverted
        thrust_cap = torch.where(needed < free, capped, 10.0 * needed)
        airframe = airframe_at()
        # The induced drag k S^(e/2) where the sustained turn settles: thrust meets drag and
        # exchange.
        low, high = stall * 1.0001, stall * torch.sqrt(handles.g_limit)
        for _ in range(60):
            middle = 0.5 * (low + high)
            rate = steady_turn(rules, airframe, middle, turning, ones).path_rate
            slow = rate < handles.sustained_turn_rate
            low, high = torch.where(slow, middle, low), torch.where(slow, high, middle)
        speed = 0.5 * (low + high)
        turn = steady_turn(rules, airframe, speed, turning, ones)
        _, current = _drag_and_exchange(rules, airframe, speed, turning, ones, turn.angle)
        coefficient = lift_coefficient(constants, airframe, turn.angle)
        lifting = coefficient**2 + (current.vertical_lift / speed**2) ** 2  # S
        total = current.thrust * torch.cos(turn.angle) - current.exchange
        induced_drag = (total / speed**2 - handles.zero_lift_drag).clamp_min(1e-12)
        full_lift = induced_drag / lifting ** (0.5 * exponent) * lift ** (exponent - 2.0)
    frequency = _reversal_frequency(rules, airframe_at(), handles, turning)
    return torch.stack([null_power, coast, thrust_cap, full_lift, frequency, dive, climb], -1)


# ------------------------------------------------------------------ Newton


class Solution(NamedTuple):
    """What `solve` returns: the card, how close it came, and how well determined it is."""

    types: ShipTypes  # float32, shaped like the handles
    error: torch.Tensor  # (..., num_types, P) achieved relative error of each flown handle
    condition: torch.Tensor  # (..., num_types) the flown Jacobian's condition number
    redundant: list  # (card, type, handle, handle) for every warned pair
    iterations: int
    converged: torch.Tensor  # (..., num_types) bool: every flown handle within 0.5%
    failures: list  # the reason for each card and type that did not converge


def _flown_values(flown) -> torch.Tensor:
    return torch.stack([getattr(flown, name) for name in FLOWN], -1)


def flown_parameters(rules: Rules, types: ShipTypes) -> torch.Tensor:
    """(..., num_types, P) a card's flown parameters (FLOWN_PARAMETERS, physical values): what
    `solve` iterates on, so a solve can start from a card (`start`)."""
    constants = rules.flight
    induced = constants.induced_drag_factor * types.induced_drag_scale
    return torch.stack(
        [
            types.engine_power_full_store,
            types.coast_exchange_scale * constants.coast_exchange,
            types.static_thrust,
            induced * types.max_lift_coefficient ** (types.bleed_exponent - 2.0),
            types.corner_nose_frequency,
            types.dive_exchange_scale * constants.dive_exchange,
            types.climb_exchange_scale * constants.climb_exchange,
        ],
        -1,
    )


class Progress(NamedTuple):
    """What `solve` reports after each iteration (its `progress` callback): the best card so
    far (float32, shaped like the handles), the flown handles that card flies (FLOWN order,
    (..., num_types, P)), its worst relative error, and the iteration."""

    types: ShipTypes
    flown: torch.Tensor
    worst: torch.Tensor
    iteration: int


def marquardt_step(
    jacobian: torch.Tensor, residual: torch.Tensor, damping: torch.Tensor
) -> torch.Tensor:
    """The Levenberg-Marquardt step (..., P) of residuals (..., h) with this Jacobian
    (..., h, P) and damping (...). A parameter no flown handle feels (a zero Jacobian
    column) makes the system singular: such a card takes the least-squares step, which
    leaves that parameter alone."""
    normal = jacobian.transpose(-1, -2) @ jacobian
    normal = normal + damping[..., None, None] * torch.diag_embed(
        normal.diagonal(dim1=-2, dim2=-1)
    )
    gradient = jacobian.transpose(-1, -2) @ residual[..., None]
    solved, info = torch.linalg.solve_ex(normal, gradient)
    singular = info != 0
    if singular.any():
        least = torch.linalg.pinv(normal) @ gradient
        solved = torch.where(singular[..., None, None], least, solved)
    return -solved[..., 0]


def solve(
    handles: Handles,
    rules: Rules,
    iterations: int = ITERATIONS,
    strict: bool = True,
    start: ShipTypes | None = None,
    progress: Callable[[Progress], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    tick_function=None,
) -> Solution:
    """The card meeting these handles ((num_types,) or (B, num_types) fields), solved in
    float64 on the handles' device and stored in float32. Infeasible with a reason when no
    card can meet them; a RedundantHandles warning when two are nearly redundant.

    A batch converges or fails card by card. strict (the default) raises Infeasible with
    every failure's reason; otherwise the Solution lists them (`converged`, `failures`), and
    a failed card's types are its best attempt.

    Live tuning: `start`, a card shaped like the handles to start from instead of the
    closed-form first guess (near the answer, it converges in an iteration or two);
    `progress`, called after every iteration with the best card so far (`Progress`);
    `cancel`, asked before every iteration: True stops the solve, which then returns its
    best card so far as if its iterations had run out; `tick_function`, the flight tick the
    flights use (`manoeuvres.fly`)."""
    one_card = handles.ceiling.dim() == 1
    handles = _batched_handles(Handles(**{n: getattr(handles, n).double() for n in NAMES}))
    check(rules, handles)
    combat = combat_card(rules, handles, handles.cruise_speed)
    B, type_count = handles.ceiling.shape
    target = torch.stack([getattr(handles, name) for name in FLOWN], -1)  # (B, types, P)
    scale = torch.maximum(target.abs(), error_floor(rules, handles))

    def flown(log_parameters: torch.Tensor, copies: int) -> torch.Tensor:
        """Relative residuals (B, copies, types, P) of log parameters (B * copies, types, P)."""
        repeated = Handles(**{n: getattr(handles, n).repeat_interleave(copies, 0) for n in NAMES})
        types = _card(
            rules,
            repeated,
            log_parameters.exp(),
            {k: v.repeat_interleave(copies, 0) for k, v in combat.items()},
        )
        measured = _flown_values(fly(rules, types, tick_function)).reshape(B, copies, type_count, P)
        return (measured - target[:, None]) / scale[:, None]

    if start is not None:
        start = batched(ShipTypes(**{n: v.double() for n, v in vars(start).items()}))
        x = flown_parameters(rules, start).log().expand(B, type_count, P)
        if not torch.isfinite(x).all():
            raise Infeasible("the starting card has parameters a solve cannot start from")
    else:
        x = first_guess(rules, handles, combat).log()  # the closed forms
        if not torch.isfinite(x).all():
            raise Infeasible("the closed-form first guess has no finite parameters")
    eye = torch.eye(P, dtype=x.dtype, device=x.device)
    candidates = torch.cat([torch.zeros_like(eye[:1]), STEP * eye])  # (P + 1, P)
    best_x, best_cost = x, torch.full_like(x[..., 0], math.inf)
    best_base, best_jacobian = torch.full_like(x, math.inf), eye.expand(B, type_count, P, P)
    damping = torch.full_like(best_cost, DAMPING[0])
    iteration = 0
    for iteration in range(1, iterations + 1):
        if cancel is not None and cancel():
            iteration -= 1
            break
        residual = flown(
            (x[:, None] + candidates[None, :, None, :]).reshape(B * (P + 1), type_count, P), P + 1
        )
        base = residual[:, 0]  # (B, types, P)
        if iteration == 1 and start is None and not torch.isfinite(base).all():
            raise Infeasible(
                "the first guess does not fly every manoeuvre: "
                + _unmeasured(base, windows(rules, handles))
            )
        jacobian = ((residual[:, 1:] - base[:, None]) / STEP).permute(
            0, 2, 3, 1
        )  # (B, types, h, p)
        cost = (base * base).sum(-1)
        improved = cost < best_cost  # NaN never improves
        best_x = torch.where(improved[..., None], x, best_x)
        best_cost = torch.where(improved, cost, best_cost)
        best_base = torch.where(improved[..., None], base, best_base)
        best_jacobian = torch.where(improved[..., None, None], jacobian, best_jacobian)
        damping = torch.where(improved, damping / 3.0, damping * 4.0).clamp(*DAMPING)
        if progress is not None:
            best = _card(rules, handles, best_x.exp(), combat)
            best = ShipTypes(**{name: value.float() for name, value in vars(best).items()})
            measured = target + best_base * scale
            worst = best_base.abs().amax(-1)
            if one_card:
                best = ShipTypes(**{name: value[0] for name, value in vars(best).items()})
                measured, worst = measured[0], worst[0]
            progress(Progress(best, measured, worst, iteration))
        if (best_base.abs().amax(-1) < TOLERANCE).all():
            break
        # Levenberg-Marquardt from each card's best point: Gauss-Newton while it improves,
        # toward scaled gradient descent while it does not.
        step = marquardt_step(best_jacobian, best_base, damping)
        step = step * (STEP_LIMIT / step.abs().amax(-1, keepdim=True).clamp_min(STEP_LIMIT))
        x = best_x + step
    jacobian = best_jacobian
    error = best_base.abs()
    worst = error.amax(-1)
    condition, redundant = _conditioning(jacobian)
    converged = worst < ACCEPTANCE
    failures = []
    for card, kind in (~converged).nonzero().tolist():
        handle = FLOWN[int(error[card, kind].nan_to_num(math.inf).argmax())]
        weakest = torch.linalg.svd(jacobian[card, kind])[0][:, -1].abs().topk(2).indices
        failures.append(
            f"card {card} type {kind}: no convergence within {iterations} "
            f"iterations: misses {handle} by {float(worst[card, kind]):.2%} "
            f"(condition number {float(condition[card, kind]):.3g}, weakest "
            f"handles {FLOWN[int(weakest[0])]} and {FLOWN[int(weakest[1])]})"
        )
    if failures and strict:
        raise Infeasible("; ".join(failures))
    final = _card(rules, handles, best_x.exp(), combat)
    final = ShipTypes(**{name: value.float() for name, value in vars(final).items()})
    if one_card:
        final = ShipTypes(**{name: value[0] for name, value in vars(final).items()})
        error, condition, converged = error[0], condition[0], converged[0]
    return Solution(final, error, condition, redundant, iteration, converged, failures)


def _conditioning(jacobian: torch.Tensor):
    """Condition numbers (B, types) and, where they exceed CONDITION_WARNING, the two handles
    that dominate the least-determined combination (with a warning)."""
    left, values, _ = torch.linalg.svd(jacobian)
    condition = values[..., 0] / values[..., -1]
    redundant = []
    for card, kind in (condition > CONDITION_WARNING).nonzero().tolist():
        weakest = left[card, kind, :, -1].abs()
        first, second = weakest.topk(2).indices.tolist()
        redundant.append((card, kind, FLOWN[first], FLOWN[second]))
        warnings.warn(
            f"card {card} type {kind}: {FLOWN[first]} and {FLOWN[second]} are nearly "
            f"redundant (condition number {float(condition[card, kind]):.3g})",
            RedundantHandles,
            stacklevel=3,
        )
    return condition, redundant
