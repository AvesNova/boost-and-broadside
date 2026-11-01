"""The standard manoeuvres (ship-card spec 3), flown in the game's own flight tick.

Every manoeuvre of every type of every card flies at once: one environment per card and one
ship per (manoeuvre, type), at the standard field index (n = 1, no fields) and in the card's
dtype (float64 for the solver). Rates at the start of a manoeuvre are the slope at t = 0 of
a cubic fitted over its first FIT_WINDOW. The climb's time interpolates linearly between
ticks; the nose reversal's, which is only a few ticks long, through a cubic on the four
ticks around the crossing (linear interpolation alone leaves up to 0.5% between 25 and 60
Hz). Both are continuous in the parameters and comparable across tick rates.
"""

import dataclasses
import math
from typing import NamedTuple

import torch

from ..physics.flight import (
    Airframe,
    airframe_of,
    flight_tick,
    lift_coefficient,
    setting,
    stall_angle,
)
from ..rules import Rules
from ..ship_types import ShipTypes
from ..state import Fields, Ships

FIT_WINDOW = 0.2  # s
FIT_DEGREE = 3
SETTLE_TIME = 20.0  # s, a sustained turn's settling time
REVERSAL_LIMIT = 2.0  # s, the longest nose reversal measured
CLIMB_LIMIT = 900.0  # s, the longest climb measured
CRUISE_BISECTIONS = 60
TURNING_STORE = 0.25  # store fraction for the turns: below the knee, clear of the fade bands
DIVE_STORE = 0.5  # store fraction for the dive and the zoom (spec: half full)
FAR = 1e9  # m, where the one field of a flight-only map sits


def batched(types: ShipTypes) -> ShipTypes:
    """The card with a batch axis: (num_types,) fields become (1, num_types)."""
    if types.max_energy.dim() == 2:
        return types
    return ShipTypes(**{f.name: getattr(types, f.name)[None] for f in dataclasses.fields(types)})


def type_axis(types: ShipTypes) -> torch.Tensor:
    """(B, num_types) int32: each column's ship type."""
    B, T = types.max_energy.shape
    return torch.arange(T, dtype=torch.int32, device=types.max_energy.device).expand(B, T)


def ships(speed, energy, angle, nose_rate, ship_type) -> Ships:
    """Ships at the origin flying along +x (the nose at `angle` above the path), alive."""
    zeros = torch.zeros_like(speed)
    integers = torch.zeros_like(ship_type)
    flags = torch.zeros_like(speed, dtype=torch.bool)
    return Ships(x=zeros, y=zeros, heading_cos=torch.cos(angle), heading_sin=torch.sin(angle),
                 proper_speed=speed, angle_of_attack=angle, angle_of_attack_cos=torch.cos(angle),
                 angle_of_attack_sin=torch.sin(angle), nose_rate=nose_rate, energy=energy,
                 log_refractive_index=zeros, log_index_gradient_x=zeros,
                 log_index_gradient_y=zeros, shield=zeros, shield_recharge_wait=zeros,
                 alive=~flags, fired=flags, spawned=flags, respawn_wait=integers,
                 gun_cooldown=integers, next_slot=integers, ship_type=ship_type)


def empty_map(like: torch.Tensor) -> Fields:
    """One field too far away to touch anything: ln n = 0 everywhere a ship flies."""
    B = like.shape[0]
    ones = torch.ones((B, 1), dtype=like.dtype, device=like.device)
    return Fields(x=ones * FAR, y=ones * 0.0, radius=ones, band_width=ones,
                  level=torch.ones((B, 1), dtype=torch.int32, device=like.device))


class SteadyTurn(NamedTuple):
    angle: torch.Tensor  # the commanded (steady) angle of attack
    path_rate: torch.Tensor  # the flight path's turn rate there, also the steady nose rate


def steady_turn(rules: Rules, airframe: Airframe, speed, energy, stick) -> SteadyTurn:
    """The nose oscillator's rest point at a speed, store and stick, coasting (game spec 3.5
    steps 4 and 7 with the angle at its command)."""
    constants = rules.flight
    current = setting(constants, airframe, speed, energy, stick, torch.zeros_like(speed))
    lateral = current.lateral_lift
    reach = (lateral.abs() / (airframe.max_lift_coefficient * speed * speed)).clamp(0.0, 1.0)
    angle = torch.sign(lateral) * stall_angle(constants, airframe) * (1.0 - torch.sqrt(1.0 - reach))
    lift = lift_coefficient(constants, airframe, angle)
    return SteadyTurn(angle, (lift * speed * speed + current.thrust * torch.sin(angle)) / speed)


def derivative_weights(tick_length: float, like: torch.Tensor) -> torch.Tensor:
    """(K + 1,) weights whose dot product with samples at t = 0, h, ..., K h is the slope at
    t = 0 of their least-squares cubic over FIT_WINDOW."""
    count = max(round(FIT_WINDOW / tick_length), FIT_DEGREE + 1) + 1
    time = torch.arange(count, dtype=like.dtype, device=like.device) * tick_length
    vandermonde = torch.stack([time ** p for p in range(FIT_DEGREE + 1)], -1)
    return torch.linalg.pinv(vandermonde)[1]


CROSSING_BISECTIONS = 50


def crossing_time(series: torch.Tensor, threshold: torch.Tensor, tick_length: float):
    """The first time series (ticks + 1, ...) rises through threshold (...), on the cubic
    through the four ticks around the crossing (two before, two after where the series
    allows); NaN if it never rises through it after the start."""
    ticks = series.shape[0]
    above = series >= threshold
    crossed = above[1:].any(0) & ~above[0]
    first = torch.where(crossed, above.int().argmax(0), 1)  # the first tick at or above
    start = (first - 2).clamp(0, ticks - 4)
    stencil = start[None] + torch.arange(4, device=series.device).reshape((4,) + (1,) * first.dim())
    values = series.gather(0, stencil)
    times = stencil.to(series.dtype)  # in ticks

    def cubic(t):
        total = torch.zeros_like(t)
        for i in range(4):
            term = values[i]
            for j in range(4):
                if j != i:
                    term = term * (t - times[j]) / (times[i] - times[j])
            total = total + term
        return total

    low, high = (first - 1).to(series.dtype), first.to(series.dtype)
    for _ in range(CROSSING_BISECTIONS):
        middle = 0.5 * (low + high)
        rising = cubic(middle) < threshold
        low, high = torch.where(rising, middle, low), torch.where(rising, high, middle)
    return torch.where(crossed, tick_length * 0.5 * (low + high), torch.nan)


def path_turned(before: Ships, after: Ships) -> torch.Tensor:
    """The flight path's turn over a tick: the nose's turn less the angle of attack's."""
    cos, sin = before.heading_cos, before.heading_sin
    nose = torch.atan2(cos * after.heading_sin - sin * after.heading_cos,
                       cos * after.heading_cos + sin * after.heading_sin)
    return nose - (after.angle_of_attack - before.angle_of_attack)


def aitken(first, second, third):
    """The limit of three equally spaced samples of a geometric approach (the last sample
    where they are not one: converged, oscillating or noisy)."""
    step1, step2 = second - first, third - second
    ratio = step2 / torch.where(step1 == 0.0, 1.0, step1)
    geometric = (step1 != 0.0) & (ratio > 0.0) & (ratio < 0.9)
    return torch.where(geometric, third + step2 * ratio / (1.0 - ratio), third)


def cruise_speed(rules: Rules, types: ShipTypes, energy: torch.Tensor) -> torch.Tensor:
    """(B, T, G) the straight coasting equilibrium at stores energy (B, T, G): the speed the
    flight tick leaves unchanged, between the stall speed and 30 times it (bisection on
    one-tick flights). NaN where the tick slows the ship at the stall speed or still speeds
    it up at the top of the bracket."""
    types = batched(types)
    B, T, G = energy.shape
    ship_type = type_axis(types)[..., None].expand(B, T, G).reshape(B, T * G)
    airframe = airframe_of(types, ship_type)
    stall = torch.sqrt(rules.flight.gravity / airframe.max_lift_coefficient)
    low, high = stall, 30.0 * stall
    flat = energy.reshape(B, T * G)
    zeros = torch.zeros_like(flat)
    fields = empty_map(flat)

    def gain(speed):
        start = ships(speed, flat, zeros, zeros, ship_type)
        end = flight_tick(rules.flight, airframe, rules.tick_length, start, zeros, zeros, fields)
        return end.proper_speed - speed

    bracketed = (gain(low) > 0.0) & (gain(high) < 0.0)
    for _ in range(CRUISE_BISECTIONS):
        middle = 0.5 * (low + high)
        rising = gain(middle) > 0.0
        low, high = torch.where(rising, middle, low), torch.where(rising, high, middle)
    return torch.where(bracketed, 0.5 * (low + high), torch.nan).reshape(B, T, G)


class Flown(NamedTuple):
    """The flown handles, (B, num_types) each, in the units of `Handles`."""

    cruise_speed: torch.Tensor
    climb_time: torch.Tensor
    low_speed_acceleration: torch.Tensor
    sustained_turn_rate: torch.Tensor
    bleed_rate: torch.Tensor
    nose_reversal_time: torch.Tensor
    dive_acceleration: torch.Tensor
    zoom_acceleration: torch.Tensor


# The manoeuvres, in the order of the ship axis (each a block of num_types ships).
CLIMB, DIVE, ZOOM, LOW_SPEED, SUSTAINED, REVERSAL = range(6)


def fly(rules: Rules, types: ShipTypes) -> Flown:
    """Fly every manoeuvre of every card (B, num_types) in one batch and read the flown
    handles off it.

    The cruise is found first (cruise_speed at a full store), since the climb, the dive and
    the zoom start from it. The sustained turn starts at corner speed with the nose at its
    steady angle of attack, so its first FIT_WINDOW is also the bleed manoeuvre.
    """
    types = batched(types)
    constants, h = rules.flight, rules.tick_length
    columns = type_axis(types)
    airframe = airframe_of(types, columns)
    B, T = columns.shape
    full = airframe.max_energy
    cruise = cruise_speed(rules, types, full[..., None])[..., 0]
    stall = torch.sqrt(constants.gravity / airframe.max_lift_coefficient)
    corner = airframe.corner_speed
    turning = TURNING_STORE * full
    hard_left = torch.ones_like(full)
    steady = steady_turn(rules, airframe, corner, turning, hard_left)
    zeros, ones = torch.zeros_like(full), torch.ones_like(full)

    def blocks(*values):
        return torch.cat(values, -1)  # (B, T) per manoeuvre -> (B, 6 T)

    speed = blocks(cruise, cruise, cruise, stall, corner, corner)
    energy = blocks(zeros, DIVE_STORE * full, DIVE_STORE * full, zeros, turning, turning)
    angle = blocks(zeros, zeros, zeros, zeros, steady.angle, -steady.angle)
    nose_rate = blocks(zeros, zeros, zeros, zeros, steady.path_rate, -steady.path_rate)
    stick = blocks(zeros, zeros, zeros, zeros, ones, ones)
    power = blocks(zeros, -ones, ones, zeros, zeros, zeros)
    ship_type = columns.repeat(1, 6)
    state = ships(speed, energy, angle, nose_rate, ship_type)
    flight = airframe_of(types, ship_type)
    fields = empty_map(speed)

    weights = derivative_weights(h, speed)
    short = max(len(weights) - 1, math.ceil(REVERSAL_LIMIT / h))
    settle = round(SETTLE_TIME / h)
    samples = [settle - round(k / h) for k in (3.0, 2.0, 1.0, 0.0)]  # path angle at these ticks
    climb_target = (1.0 - constants.store_fade_band) * full
    climb_time = torch.full_like(full, torch.nan)
    history = {"speed": [speed], "energy": [energy], "angle": [angle]}
    path, path_samples = torch.zeros_like(speed), {}
    tick, limit = 0, round(CLIMB_LIMIT / h)
    while tick < limit and (tick < max(short, settle) or climb_time.isnan().any()):
        new = flight_tick(constants, flight, h, state, stick, power, fields)
        path = path + path_turned(state, new)
        tick += 1
        before, after = state.energy[:, :T], new.energy[:, :T]
        crossed = climb_time.isnan() & (after >= climb_target)
        fraction = (climb_target - before) / (after - before)
        climb_time = torch.where(crossed, h * (tick - 1 + fraction), climb_time)
        state = new
        if tick <= short:
            for key, value in (("speed", new.proper_speed), ("energy", new.energy),
                               ("angle", new.angle_of_attack)):
                history[key].append(value)
        if tick in samples:
            path_samples[tick] = path
    series = {key: torch.stack(values) for key, values in history.items()}

    def block(values, manoeuvre):
        return values[..., manoeuvre * T:(manoeuvre + 1) * T]

    def start_rate(values):
        return torch.einsum("k,k...->...", weights, values[:len(weights)])

    specific = 0.5 * series["speed"] ** 2 + series["energy"] / constants.mass
    rates = [(path_samples[later] - path_samples[earlier]) / ((later - earlier) * h)
             for earlier, later in zip(samples[:-1], samples[1:])]
    reversal = crossing_time(block(series["angle"], REVERSAL), 0.9 * steady.angle, h)
    return Flown(cruise_speed=cruise, climb_time=climb_time,
                 low_speed_acceleration=start_rate(block(series["speed"], LOW_SPEED)),
                 sustained_turn_rate=block(aitken(*rates), SUSTAINED),
                 bleed_rate=start_rate(block(specific, SUSTAINED)),
                 nose_reversal_time=reversal,
                 dive_acceleration=start_rate(block(series["speed"], DIVE)),
                 zoom_acceleration=start_rate(block(series["speed"], ZOOM)))
