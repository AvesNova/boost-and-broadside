"""measure (card -> handles) and solve (handles -> card), ship-card spec 4.

The direct handles set their parameters in closed form. The eight flown handles set eight
parameters (FLOWN_PARAMETERS) by Levenberg-Marquardt on flights of the game's own code: a
closed-form first guess at each of several bleed exponents (flown, the best kept), then per
iteration one batch that flies the base card and its eight perturbations (log space, finite
differences), one environment each. The cruise fix
derives the engine power: P = P_min + m a_0 u_cruise, with P_min (the null power) solved
so that the flown cruise meets its handle.
"""

import functools
import math
import warnings
from typing import NamedTuple

import torch

from ..physics.flight import Airframe, airframe_of, lift_coefficient, rates, setting
from ..rules import Rules
from ..ship_types import DEFAULTS, ShipTypes
from .errors import Infeasible, RedundantHandles
from .handles import FLOWN, NAMES, Handles
from .manoeuvres import CLIMB_LIMIT, REVERSAL_LIMIT, TURNING_STORE, batched, fly, steady_turn
from .weapons import combat_card, combat_handles

# The parameter each flown handle sets, in FLOWN's order (physical values, not scales). The
# induced drag is solved as its factor at full lift, k C_max^(e - 2): the induced drag
# k S^(e/2) is then that times C_max^2 (S / C_max^2)^(e/2), so the bleed exponent reshapes
# the polar below full lift instead of rescaling all of it (k alone has units m^(e - 1),
# and its log moves with e: Newton in (log k, e) is badly conditioned).
FLOWN_PARAMETERS = ("engine_power_full_store", "coast_exchange", "static_thrust",
                    "full_lift_induced_drag", "bleed_exponent", "corner_nose_frequency",
                    "dive_exchange", "climb_exchange")
# A handle's error is relative to max(|target|, floor), so handles near zero have a floor.
ERROR_FLOOR = dict(cruise_speed=1.0, climb_time=0.1, low_speed_acceleration=0.1,
                   sustained_turn_rate=0.01, bleed_rate=10.0, nose_reversal_time=1e-3,
                   dive_acceleration=0.1, zoom_acceleration=0.1)
TOLERANCE = 1e-4  # stop when every flown handle is this close (the spec's bar is 5e-3)
ACCEPTANCE = 5e-3  # the spec's 0.5%: beyond it after the budget, the solve fails
ITERATIONS = 16
STEP = 1e-5  # the finite-difference step in log parameters
STEP_LIMIT = 0.5  # the largest Newton step in any log parameter
DAMPING = (1e-6, 1e6)  # Levenberg-Marquardt's: the first, and the bounds
EXPONENT_BOUNDS = (0.5, 5.0)  # the bleed exponents the solver considers
EXPONENT_SCAN = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # the first guess's
CONDITION_WARNING = 1e4  # a Jacobian condition number above this warns
# The flown handles measured over a bounded flight: (window in s, its constant's name).
WINDOWS = dict(nose_reversal_time=(REVERSAL_LIMIT, "REVERSAL_LIMIT"),
               climb_time=(CLIMB_LIMIT, "CLIMB_LIMIT"))
# Why a flown handle is not measured (NaN), where it can be.
UNMEASURED = dict(
    cruise_speed="no straight equilibrium: the drag outruns the engine at every speed",
    climb_time=f"the store does not fill within {CLIMB_LIMIT:g} s, or stops rising",
    nose_reversal_time=f"the nose does not reverse within {REVERSAL_LIMIT:g} s",
)


def _flight_card(rules: Rules, handles: Handles, parameters: torch.Tensor) -> dict:
    """Card fields from the direct flight handles and the flown parameters (..., 8)."""
    constants = rules.flight
    value = dict(zip(FLOWN_PARAMETERS, parameters.unbind(-1)))
    lift = constants.gravity / handles.stall_speed ** 2  # C_max
    induced = value["full_lift_induced_drag"] * lift ** (2.0 - value["bleed_exponent"])  # k
    return dict(
        max_lift_coefficient=lift,
        max_acceleration=handles.g_limit * constants.gravity,
        zero_lift_drag=handles.zero_lift_drag,
        engine_power=(value["engine_power_full_store"]
                      + constants.mass * value["coast_exchange"] * handles.cruise_speed),
        engine_power_full_store=value["engine_power_full_store"],
        static_thrust=value["static_thrust"],
        max_energy=handles.ceiling,
        corner_nose_frequency=value["corner_nose_frequency"],
        bleed_exponent=value["bleed_exponent"],
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


def measure(types: ShipTypes, rules: Rules) -> Handles:
    """Every handle of a card ((num_types,) or (B, num_types) fields), flown in its dtype."""
    one_card = types.max_energy.dim() == 1
    types = batched(types)
    constants = rules.flight
    flown = fly(rules, types)
    handles = Handles(
        stall_speed=torch.sqrt(constants.gravity / types.max_lift_coefficient),
        g_limit=types.max_acceleration / constants.gravity,
        ceiling=types.max_energy,
        stall_angle=constants.stall_angle * types.stall_angle_scale,
        vision_range=rules.vision_range * types.vision_range_scale,
        zero_lift_drag=types.zero_lift_drag,
        **flown._asdict(),
        **combat_handles(rules, types, flown.cruise_speed))
    return Handles(**{n: getattr(handles, n)[0] for n in NAMES}) if one_card else handles


# ------------------------------------------------------------------ feasibility

def check(rules: Rules, handles: Handles) -> None:
    """Infeasible with the first reason found, for sets no card can meet (spec 4)."""
    constants = rules.flight
    positive = ("stall_speed", "cruise_speed", "climb_time", "ceiling", "sustained_turn_rate",
                "nose_reversal_time", "stall_angle", "vision_range", "zero_lift_drag",
                "fire_rate", "muzzle_speed", "gun_range", "shots_per_store", "max_shield",
                "hit_radius")
    for name in positive:
        if not (getattr(handles, name) > 0.0).all():
            raise Infeasible(f"{name} must be positive")
    for name in ("bullet_spread", "hit_damage", "shield_recharge_rate", "shield_recharge_delay",
                 "spawn_energy"):
        if (getattr(handles, name) < 0.0).any():
            raise Infeasible(f"{name} must not be negative")
    if (handles.g_limit <= 1.0).any():
        raise Infeasible("a g limit of at most 1 cannot hold level flight: the lift budget "
                         "must exceed gravity")
    if (handles.cruise_speed <= handles.stall_speed).any():
        raise Infeasible("the cruise speed must be above the stall speed")
    if (handles.stall_angle > math.pi / 4).any():
        raise Infeasible("an angle-of-attack limit above 45 degrees is outside the lift model")
    corner = handles.stall_speed * torch.sqrt(handles.g_limit)
    acceleration = handles.g_limit * constants.gravity
    peak = torch.sqrt(acceleration ** 2 - constants.gravity ** 2) / corner
    if (handles.sustained_turn_rate >= peak).any():
        raise Infeasible("the sustained turn rate must be below the instantaneous peak turn "
                         "sqrt(a_max^2 - g^2) / corner speed")
    if (handles.spawn_energy > handles.ceiling).any():
        raise Infeasible("the spawn energy must fit in the store")
    for name, (window, constant) in WINDOWS.items():
        if (getattr(handles, name) >= window).any():
            raise Infeasible(f"a {name} beyond its measurement window: the manoeuvre is flown "
                             f"for at most {window:g} s (manoeuvres.{constant})")


def _unmeasured(residual: torch.Tensor) -> str:
    """Why the first guesses fly no measurement: for each card and type whose every candidate
    misses a flown handle (residual (B, copies, T, 8), NaN where not measured), the handles
    and their reasons."""
    missing = residual.isnan().all(1)  # (B, T, 8)
    reasons = []
    for card, kind in missing.any(-1).nonzero().tolist():
        names = [FLOWN[i] for i in missing[card, kind].nonzero()[:, 0].tolist()]
        reasons.append(f"card {card} type {kind}: "
                       + ", ".join(f"{name} ({UNMEASURED.get(name, 'not measured')})"
                                   for name in names))
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


@functools.cache
def _reversal_constant(damping: float) -> float:
    """omega t at which a damped oscillator stepping from -1 to +1 first reaches +0.9 (95% of
    the step): the reversal time is about this over the corner nose frequency."""
    damped = math.sqrt(1.0 - damping * damping)

    def reached(t):
        return 1.0 - math.exp(-damping * t) * (math.cos(damped * t)
                                               + damping / damped * math.sin(damped * t))

    low, high = 0.0, math.pi / damped  # rises monotonically to its first peak
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if reached(middle) < 0.95 else (low, middle)
    return 0.5 * (low + high)


def _drag_and_exchange(rules: Rules, airframe, speed, energy, stick, angle):
    """Drag and exchange (m/s^2) at a state, from the game's setting and rates (n = 1)."""
    current = setting(rules.flight, airframe, speed, energy, stick, torch.zeros_like(speed))
    lift = lift_coefficient(rules.flight, airframe, angle)
    speed_rate, _ = rates(rules.flight, airframe, current, 1.0, speed, lift, torch.cos(angle),
                          1.0 / (speed * speed))
    drag = current.thrust * torch.cos(angle) - current.exchange - speed_rate
    return drag, current


def first_guess(rules: Rules, handles: Handles, combat: dict, exponent: float) -> torch.Tensor:
    """(B, num_types, 8) flown parameters at this bleed exponent, from closed forms in the
    handles, each inverting the game's own instantaneous equations at its manoeuvre's state
    (no flight). The bleed rate is left to the exponent scan (`solve`)."""
    constants = rules.flight
    mass = constants.mass
    stall, cruise, store = handles.stall_speed, handles.cruise_speed, handles.ceiling
    B, T = stall.shape
    columns = torch.arange(T, dtype=torch.int32, device=stall.device).expand(B, T)
    coast = store * _climb_integral(constants.engine_knee, constants.store_fade_band) / (
        mass * cruise * handles.climb_time)
    dive = handles.dive_acceleration - coast
    climb = coast - handles.zoom_acceleration
    if (dive <= 0.0).any() or (climb <= 0.0).any():
        raise Infeasible("the dive acceleration must exceed, and the zoom acceleration fall "
                         "short of, the coast exchange the climb implies")
    lift = constants.gravity / stall ** 2  # C_max
    exponent = torch.full_like(stall, exponent)
    full_lift = torch.full_like(stall, constants.induced_drag_factor)  # k C_max^(e - 2)
    thrust_cap = torch.full_like(stall, 1e6)
    zeros, ones = torch.zeros_like(stall), torch.ones_like(stall)
    turning = TURNING_STORE * store
    null_power = ones

    def airframe_at() -> Airframe:
        parameters = torch.stack([null_power, coast, thrust_cap, full_lift, exponent, ones,
                                  dive, climb], -1)
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
        capped = (needed ** -4 - free ** -4).clamp_min(1e-30) ** -0.25  # softmin_4 inverted
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
        lifting = coefficient ** 2 + (current.vertical_lift / speed ** 2) ** 2  # S
        total = current.thrust * torch.cos(turn.angle) - current.exchange
        induced_drag = (total / speed ** 2 - handles.zero_lift_drag).clamp_min(1e-12)
        full_lift = induced_drag / lifting ** (0.5 * exponent) * lift ** (exponent - 2.0)
    frequency = _reversal_constant(constants.nose_damping) / handles.nose_reversal_time
    return torch.stack([null_power, coast, thrust_cap, full_lift, exponent, frequency, dive,
                        climb], -1)


# ------------------------------------------------------------------ Newton

class Solution(NamedTuple):
    """What `solve` returns: the card, how close it came, and how well determined it is."""

    types: ShipTypes  # float32, shaped like the handles
    error: torch.Tensor  # (..., num_types, 8) achieved relative error of each flown handle
    condition: torch.Tensor  # (..., num_types) the flown Jacobian's condition number
    redundant: list  # (card, type, handle, handle) for every warned pair
    iterations: int
    converged: torch.Tensor  # (..., num_types) bool: every flown handle within 0.5%
    failures: list  # the reason for each card and type that did not converge


def _flown_values(flown) -> torch.Tensor:
    return torch.stack([getattr(flown, name) for name in FLOWN], -1)


def solve(handles: Handles, rules: Rules, iterations: int = ITERATIONS,
          strict: bool = True) -> Solution:
    """The card meeting these handles ((num_types,) or (B, num_types) fields), solved in
    float64 on the handles' device and stored in float32. Infeasible with a reason when no
    card can meet them; a RedundantHandles warning when two are nearly redundant.

    A batch converges or fails card by card. strict (the default) raises Infeasible with
    every failure's reason; otherwise the Solution lists them (`converged`, `failures`), and
    a failed card's types are its best attempt."""
    one_card = handles.ceiling.dim() == 1
    handles = _batched_handles(Handles(**{n: getattr(handles, n).double() for n in NAMES}))
    check(rules, handles)
    combat = combat_card(rules, handles, handles.cruise_speed)
    B, T = handles.ceiling.shape
    target = torch.stack([getattr(handles, name) for name in FLOWN], -1)  # (B, T, 8)
    scale = torch.maximum(target.abs(), target.new_tensor([ERROR_FLOOR[n] for n in FLOWN]))
    def flown(log_parameters: torch.Tensor, copies: int) -> torch.Tensor:
        """Relative residuals (B, copies, T, 8) of log parameters (B * copies, T, 8)."""
        repeated = Handles(**{n: getattr(handles, n).repeat_interleave(copies, 0) for n in NAMES})
        types = _card(rules, repeated, log_parameters.exp(),
                      {k: v.repeat_interleave(copies, 0) for k, v in combat.items()})
        measured = _flown_values(fly(rules, types)).reshape(B, copies, T, 8)
        return (measured - target[:, None]) / scale[:, None]

    # The first guess: closed forms at each scanned bleed exponent, flown in one batch; each
    # card and type starts from its best.
    scan = torch.stack([first_guess(rules, handles, combat, e) for e in EXPONENT_SCAN], 1).log()
    if not torch.isfinite(scan).all():
        raise Infeasible("the closed-form first guess has no finite parameters")
    residual = flown(scan.reshape(B * len(EXPONENT_SCAN), T, 8), len(EXPONENT_SCAN))
    cost = residual.pow(2).sum(-1)
    if not torch.isfinite(cost).any(1).all():
        raise Infeasible(f"no first guess flies every manoeuvre: {_unmeasured(residual)}")
    choice = cost.nan_to_num(math.inf).argmin(1)  # (B, T)
    x = scan.gather(1, choice[:, None, :, None].expand(B, 1, T, 8))[:, 0]
    eye = torch.eye(8, dtype=x.dtype, device=x.device)
    candidates = torch.cat([torch.zeros_like(eye[:1]), STEP * eye])  # (9, 8)
    bounds = math.log(EXPONENT_BOUNDS[0]), math.log(EXPONENT_BOUNDS[1])
    best_x, best_cost = x, torch.full_like(x[..., 0], math.inf)
    best_base, best_jacobian = torch.full_like(x, math.inf), eye.expand(B, T, 8, 8)
    damping = torch.full_like(best_cost, DAMPING[0])
    for iteration in range(1, iterations + 1):
        residual = flown((x[:, None] + candidates[None, :, None, :]).reshape(B * 9, T, 8), 9)
        base = residual[:, 0]  # (B, T, 8)
        jacobian = ((residual[:, 1:] - base[:, None]) / STEP).permute(0, 2, 3, 1)  # (B, T, h, p)
        cost = (base * base).sum(-1)
        improved = cost < best_cost  # NaN never improves
        best_x = torch.where(improved[..., None], x, best_x)
        best_cost = torch.where(improved, cost, best_cost)
        best_base = torch.where(improved[..., None], base, best_base)
        best_jacobian = torch.where(improved[..., None, None], jacobian, best_jacobian)
        damping = torch.where(improved, damping / 3.0, damping * 4.0).clamp(*DAMPING)
        if (best_base.abs().amax(-1) < TOLERANCE).all():
            break
        # Levenberg-Marquardt from each card's best point: Gauss-Newton while it improves,
        # toward scaled gradient descent while it does not.
        normal = best_jacobian.transpose(-1, -2) @ best_jacobian
        normal = normal + damping[..., None, None] * torch.diag_embed(
            normal.diagonal(dim1=-2, dim2=-1))
        gradient = (best_jacobian.transpose(-1, -2) @ best_base[..., None])[..., 0]
        step = -torch.linalg.solve(normal, gradient[..., None])[..., 0]
        step = step * (STEP_LIMIT / step.abs().amax(-1, keepdim=True).clamp_min(STEP_LIMIT))
        x = best_x + step
        x = torch.cat([x[..., :4], x[..., 4:5].clamp(*bounds), x[..., 5:]], -1)  # the exponent
    jacobian = best_jacobian
    error = best_base.abs()
    worst = error.amax(-1)
    condition, redundant = _conditioning(jacobian)
    converged = worst < ACCEPTANCE
    failures = []
    for card, kind in (~converged).nonzero().tolist():
        handle = FLOWN[int(error[card, kind].nan_to_num(math.inf).argmax())]
        weakest = torch.linalg.svd(jacobian[card, kind])[0][:, -1].abs().topk(2).indices
        exponent = math.exp(float(best_x[card, kind, 4]))
        bound = [b for b in EXPONENT_BOUNDS if math.isclose(exponent, b, rel_tol=1e-6)]
        failures.append(f"card {card} type {kind}: no convergence within {iterations} "
                        f"iterations: misses {handle} by {float(worst[card, kind]):.2%} "
                        f"(condition number {float(condition[card, kind]):.3g}, weakest "
                        f"handles {FLOWN[int(weakest[0])]} and {FLOWN[int(weakest[1])]}"
                        + (f"; the bleed exponent is at its bound {bound[0]:g}" if bound else "")
                        + ")")
    if failures and strict:
        raise Infeasible("; ".join(failures))
    final = _card(rules, handles, best_x.exp(), combat)
    final = ShipTypes(**{name: value.float() for name, value in vars(final).items()})
    if one_card:
        final = ShipTypes(**{name: value[0] for name, value in vars(final).items()})
        error, condition, converged = error[0], condition[0], converged[0]
    return Solution(final, error, condition, redundant, iteration, converged, failures)


def _conditioning(jacobian: torch.Tensor):
    """Condition numbers (B, T) and, where they exceed CONDITION_WARNING, the two handles
    that dominate the least-determined combination (with a warning)."""
    left, values, _ = torch.linalg.svd(jacobian)
    condition = values[..., 0] / values[..., -1]
    redundant = []
    for card, kind in (condition > CONDITION_WARNING).nonzero().tolist():
        weakest = left[card, kind, :, -1].abs()
        first, second = weakest.topk(2).indices.tolist()
        redundant.append((card, kind, FLOWN[first], FLOWN[second]))
        warnings.warn(f"card {card} type {kind}: {FLOWN[first]} and {FLOWN[second]} are nearly "
                      f"redundant (condition number {float(condition[card, kind]):.3g})",
                      RedundantHandles, stacklevel=3)
    return condition, redundant

