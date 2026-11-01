"""measure (card -> handles) and solve (handles -> card), ship-card spec 4.

The direct handles set their parameters in closed form. The eight flown handles set eight
parameters (FLOWN_PARAMETERS) by Newton's method on flights of the game's own code: a
closed-form first guess, then per iteration one batch that flies the base card and its
eight perturbations (log space, finite differences), one environment each. The cruise fix
derives the engine power: P = P_min + m a_0 u_cruise, with P_min (the null power) solved
so that the flown cruise meets its handle.
"""

import functools
import math
import warnings
from typing import NamedTuple

import torch

from ..physics.flight import airframe_of, lift_coefficient, rates, setting
from ..rules import Rules
from ..ship_types import DEFAULTS, ShipTypes
from .errors import Infeasible, RedundantHandles
from .handles import FLOWN, NAMES, Handles
from .manoeuvres import TURNING_STORE, batched, fly, steady_turn
from .weapons import combat_card, combat_handles

# The parameter each flown handle sets, in FLOWN's order (physical values, not scales).
FLOWN_PARAMETERS = ("engine_power_full_store", "coast_exchange", "static_thrust",
                    "induced_drag_factor", "bleed_exponent", "corner_nose_frequency",
                    "dive_exchange", "climb_exchange")
# A handle's error is relative to max(|target|, floor), so handles near zero have a floor.
ERROR_FLOOR = dict(cruise_speed=1.0, climb_time=0.1, low_speed_acceleration=0.1,
                   sustained_turn_rate=0.01, bleed_rate=10.0, nose_reversal_time=1e-3,
                   dive_acceleration=0.1, zoom_acceleration=0.1)
TOLERANCE = 1e-4  # stop when every flown handle is this close (the spec's bar is 5e-3)
ACCEPTANCE = 5e-3  # the spec's 0.5%: beyond it after the budget, the solve fails
ITERATIONS = 12
STEP = 1e-5  # the finite-difference step in log parameters
STEP_LIMIT = 0.5  # the largest Newton step in any log parameter
CONDITION_WARNING = 1e4  # a Jacobian condition number above this warns
EXPONENT_BOUNDS = (0.25, 8.0)  # the first guess's bleed exponent stays in these


def _flight_card(rules: Rules, handles: Handles, parameters: torch.Tensor) -> dict:
    """Card fields from the direct flight handles and the flown parameters (..., 8)."""
    constants = rules.flight
    value = dict(zip(FLOWN_PARAMETERS, parameters.unbind(-1)))
    return dict(
        max_lift_coefficient=constants.gravity / handles.stall_speed ** 2,
        max_acceleration=handles.g_limit * constants.gravity,
        zero_lift_drag=handles.zero_lift_drag,
        engine_power=(value["engine_power_full_store"]
                      + constants.mass * value["coast_exchange"] * handles.cruise_speed),
        engine_power_full_store=value["engine_power_full_store"],
        static_thrust=value["static_thrust"],
        max_energy=handles.ceiling,
        corner_nose_frequency=value["corner_nose_frequency"],
        bleed_exponent=value["bleed_exponent"],
        induced_drag_scale=value["induced_drag_factor"] / constants.induced_drag_factor,
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
                "fire_interval", "muzzle_speed", "gun_range", "shots_per_store", "max_shield",
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


def first_guess(rules: Rules, handles: Handles, combat: dict) -> torch.Tensor:
    """(B, num_types, 8) flown parameters from closed forms in the handles, each inverting
    the game's own instantaneous equations at the manoeuvre's state (no flight)."""
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
    induced, exponent = (torch.full_like(stall, constants.induced_drag_factor),
                         torch.full_like(stall, 2.0))
    thrust_cap = torch.full_like(stall, 1e6)
    for _ in range(3):
        parameters = torch.stack([torch.ones_like(stall), coast, thrust_cap, induced, exponent,
                                  torch.ones_like(stall), dive, climb], -1)
        airframe = airframe_of(_card(rules, handles, parameters, combat), columns)
        zeros, ones = torch.zeros_like(stall), torch.ones_like(stall)
        # The null power at cruise, full store, drag only (the cap is the flight's to meet).
        drag, _ = _drag_and_exchange(rules, airframe, cruise, store, zeros, zeros)
        null_power = drag * cruise
        power = null_power + mass * coast * cruise
        # The static-thrust cap from the low-speed acceleration (store empty).
        parameters[..., 0] = null_power
        airframe = airframe_of(_card(rules, handles, parameters, combat), columns)
        drag, current = _drag_and_exchange(rules, airframe, stall, zeros, zeros, zeros)
        needed = handles.low_speed_acceleration + drag + current.exchange
        free = power / stall
        capped = (needed ** -4 - free ** -4).clamp_min(1e-30) ** -0.25  # softmin_4 inverted
        thrust_cap = torch.where(needed < free, capped, 10.0 * needed)
        parameters[..., 2] = thrust_cap
        airframe = airframe_of(_card(rules, handles, parameters, combat), columns)
        # Induced drag factor and bleed exponent from the sustained turn and the bleed rate:
        # each pins k S^(e/2) at its state, and the two states' S differ.
        turning = TURNING_STORE * store
        low, high = stall * 1.0001, stall * torch.sqrt(handles.g_limit)
        for _ in range(60):
            middle = 0.5 * (low + high)
            rate = steady_turn(rules, airframe, middle, turning, ones).path_rate
            slow = rate < handles.sustained_turn_rate
            low, high = torch.where(slow, middle, low), torch.where(slow, high, middle)
        sustained_speed = 0.5 * (low + high)
        corner = stall * torch.sqrt(handles.g_limit)
        factors = []
        for speed, kind in ((sustained_speed, "sustained"), (corner, "bleed")):
            turn = steady_turn(rules, airframe, speed, turning, ones)
            drag, current = _drag_and_exchange(rules, airframe, speed, turning, ones, turn.angle)
            lift = lift_coefficient(constants, airframe, turn.angle)
            lifting = lift * lift + (current.vertical_lift / speed ** 2) ** 2  # S
            if kind == "sustained":  # settled: thrust meets drag and exchange
                total = current.thrust * torch.cos(turn.angle) - current.exchange
            else:  # Ps = u (T cos(alpha) - D)
                total = current.thrust * torch.cos(turn.angle) - handles.bleed_rate / speed
            induced_drag = (total / speed ** 2 - handles.zero_lift_drag).clamp_min(1e-12)
            factors.append((induced_drag, lifting))  # k S^(e/2), S
        (sustained_factor, sustained_lift), (bleed_factor, bleed_lift) = factors
        exponent = (2.0 * torch.log(bleed_factor / sustained_factor)
                    / torch.log(bleed_lift / sustained_lift)).clamp(*EXPONENT_BOUNDS)
        induced = sustained_factor / sustained_lift ** (0.5 * exponent)
    frequency = _reversal_constant(constants.nose_damping) / handles.nose_reversal_time
    return torch.stack([null_power, coast, thrust_cap, induced, exponent, frequency, dive, climb],
                       -1)


# ------------------------------------------------------------------ Newton

class Solution(NamedTuple):
    types: ShipTypes  # float32, shaped like the handles
    error: torch.Tensor  # (..., num_types, 8) achieved relative error of each flown handle
    condition: torch.Tensor  # (..., num_types) the flown Jacobian's condition number
    redundant: list  # (card, type, handle, handle) for every warned pair
    iterations: int


def _flown_values(flown) -> torch.Tensor:
    return torch.stack([getattr(flown, name) for name in FLOWN], -1)


def solve(handles: Handles, rules: Rules, iterations: int = ITERATIONS) -> Solution:
    """The card meeting these handles ((num_types,) or (B, num_types) fields), solved in
    float64 on the handles' device and stored in float32. Infeasible with a reason when no
    card can meet them; a RedundantHandles warning when two are nearly redundant."""
    one_card = handles.ceiling.dim() == 1
    handles = _batched_handles(Handles(**{n: getattr(handles, n).double() for n in NAMES}))
    check(rules, handles)
    combat = combat_card(rules, handles, handles.cruise_speed)
    B, T = handles.ceiling.shape
    target = torch.stack([getattr(handles, name) for name in FLOWN], -1)  # (B, T, 8)
    scale = torch.maximum(target.abs(), target.new_tensor([ERROR_FLOOR[n] for n in FLOWN]))
    x = first_guess(rules, handles, combat).log()
    if not torch.isfinite(x).all():
        raise Infeasible("the closed-form first guess has no finite parameters")
    eye = torch.eye(8, dtype=x.dtype, device=x.device)
    candidates = torch.cat([torch.zeros_like(eye[:1]), STEP * eye])  # (9, 8)
    repeated = Handles(**{n: getattr(handles, n).repeat_interleave(9, 0) for n in NAMES})
    repeated_combat = {k: v.repeat_interleave(9, 0) for k, v in combat.items()}
    best_x, best_worst = x, torch.full_like(x[..., 0], math.inf)
    best_error = torch.full_like(x, math.inf)
    for iteration in range(1, iterations + 1):
        trial = (x[:, None] + candidates[None, :, None, :]).reshape(B * 9, T, 8)
        types = _card(rules, repeated, trial.exp(), repeated_combat)
        measured = _flown_values(fly(rules, types)).reshape(B, 9, T, 8)
        residual = (measured - target[:, None]) / scale[:, None]
        base = residual[:, 0]  # (B, T, 8)
        jacobian = ((residual[:, 1:] - base[:, None]) / STEP).permute(0, 2, 3, 1)  # (B, T, h, p)
        worst = base.abs().amax(-1)
        if iteration == 1 and not torch.isfinite(worst).all():
            raise Infeasible("the first guess cannot fly every manoeuvre (no straight "
                             "equilibrium, or a climb that never fills the store)")
        improved = worst < best_worst  # NaN never improves
        best_x = torch.where(improved[..., None], x, best_x)
        best_worst = torch.where(improved, worst, best_worst)
        best_error = torch.where(improved[..., None], base.abs(), best_error)
        if (best_worst < TOLERANCE).all():
            break
        # Newton from an element that improved; half way back to its best from one that did not.
        step = -torch.linalg.solve(torch.where(improved[..., None, None], jacobian, eye),
                                   torch.where(improved[..., None], base, 0.0)[..., None])[..., 0]
        step = step * (STEP_LIMIT / step.abs().amax(-1, keepdim=True).clamp_min(STEP_LIMIT))
        x = torch.where(improved[..., None], x + step, 0.5 * (x + best_x))
    error, worst = best_error, best_worst
    if not (worst < ACCEPTANCE).all():
        card, kind = divmod(int(worst.flatten().nan_to_num(math.inf).argmax()), T)
        handle = FLOWN[int(error[card, kind].nan_to_num(math.inf).argmax())]
        raise Infeasible(f"no convergence within {iterations} iterations: card {card} type "
                         f"{kind} misses {handle} by {float(worst[card, kind]):.2%}")
    condition, redundant = _conditioning(jacobian)
    final = _card(rules, handles, best_x.exp(), combat)
    final = ShipTypes(**{name: value if name == "ticks_between_shots" else value.float()
                         for name, value in vars(final).items()})
    if one_card:
        final = ShipTypes(**{name: value[0] for name, value in vars(final).items()})
        error, condition = error[0], condition[0]
    return Solution(final, error, condition, redundant, iteration)


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

