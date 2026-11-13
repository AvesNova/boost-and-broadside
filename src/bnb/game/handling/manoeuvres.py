"""The standard manoeuvres (ship-card spec 3), flown in the game's own flight tick.

Every manoeuvre of every type of every card flies at once: one environment per card and one
ship per (manoeuvre, type), at the standard field index (n = 1, no fields) and in the card's
dtype (float64 for the solver). Rates at the start of a manoeuvre are the slope at t = 0 of
a quartic fitted over its first FIT_WINDOW. The climb's time interpolates linearly between
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
from ..rules import STICK_LEVELS, FlightConstants, Rules
from ..ship_types import ShipTypes
from ..state import Fields, Ships, replace

# The flights' windows are set in seconds at the standard card's scales (REFERENCE) and
# flown stretched by each card's own (`Stretch`): a card twice as slow flies them twice as
# long, and the standard card flies exactly these.
FIT_WINDOW = 0.2  # s
FIT_DEGREE = 4  # a cubic's slope is 0.9% off the bleed rate's (its curvature)
SETTLE_TIME = 20.0  # s, a sustained turn's settling time
SAMPLE_SPACING = 1.0  # s between the settled turn's path samples (the last three spans)
REVERSAL_LIMIT = 2.0  # s, the longest nose reversal measured
CLIMB_LIMIT = 900.0  # s, the longest climb measured (stretched by the climb's scale)
CLIMB_PROGRESS = 10.0  # s: a climb whose store has not risen over this long is given up
CRUISE_BISECTIONS = 50
CROSSING_BISECTIONS = 50  # on the cubic through the ticks around a crossing
CRUISE_GRID = 48  # speeds, log-spaced from the stall speed to 30 times it
CRUISE_DRAG_RISE_TOP = 1.6  # critical speeds: the cruise grid's top under a drag rise
TURNING_STORE = 0.25  # store fraction for the turns: below the knee, clear of the fade bands
DIVE_STORE = 0.5  # store fraction for the dive and the zoom (spec: half full)
# The bleed's stick: the middle of the game's three levels, so that the bleed and the
# full-stick sustained turn are turns at different lifts and pin the induced drag's factor
# and exponent apart.
BLEED_STICK = sorted(level for level in STICK_LEVELS if level > 0.0)[1]
FAR = 1e9  # m, where the one field of a flight-only map sits


class Scales(NamedTuple):
    """A ship type's own scales, with V_s its stall speed sqrt(g / C_max) and g gravity: its
    time V_s / g, and its climb's time E_max / (m g V_s), the store over the specific power
    scale g V_s (both in s)."""

    time: float
    climb: float


# The standard card's scales: its heavy type's, the larger of its two types' in both (C_max
# 0.022 /m and a 14 kJ store at g = 27 m/s^2 and m = 1 kg).
REFERENCE_GRAVITY = 27.0  # m/s^2
REFERENCE_STALL_SPEED = math.sqrt(REFERENCE_GRAVITY / 0.022)  # m/s, 35.03
REFERENCE = Scales(
    time=REFERENCE_STALL_SPEED / REFERENCE_GRAVITY,
    climb=14000.0 / (REFERENCE_GRAVITY * REFERENCE_STALL_SPEED),
)


def stretch(constants: FlightConstants, stall_speed, store) -> Scales:
    """How many times longer than the standard card's a batch of types (stall speeds and
    stores, any shape) flies its windows: its largest scales over REFERENCE's."""
    time = float((stall_speed / constants.gravity).max())
    climb = float((store / (constants.mass * constants.gravity * stall_speed)).max())
    return Scales(time / REFERENCE.time, climb / REFERENCE.climb)


def card_stretch(constants: FlightConstants, types: ShipTypes) -> Scales:
    """`stretch` of every type of a card (or a batch of cards)."""
    stall = torch.sqrt(constants.gravity / types.max_lift_coefficient)
    return stretch(constants, stall, types.max_energy)


def batched(types: ShipTypes) -> ShipTypes:
    """The card with a batch axis: (num_types,) fields become (1, num_types)."""
    if types.max_energy.dim() == 2:
        return types
    return ShipTypes(**{f.name: getattr(types, f.name)[None] for f in dataclasses.fields(types)})


def type_axis(types: ShipTypes) -> torch.Tensor:
    """(B, num_types) int32: each column's ship type."""
    B, type_count = types.max_energy.shape
    return torch.arange(type_count, dtype=torch.int32, device=types.max_energy.device).expand(
        B, type_count
    )


def ships(speed, energy, angle, nose_rate, ship_type) -> Ships:
    """Ships at the origin flying along +x (the nose at `angle` above the path), alive."""
    zeros = torch.zeros_like(speed)
    integers = torch.zeros_like(ship_type)
    flags = torch.zeros_like(speed, dtype=torch.bool)
    return Ships(
        x=zeros,
        y=zeros,
        heading_cos=torch.cos(angle),
        heading_sin=torch.sin(angle),
        proper_speed=speed,
        angle_of_attack=angle,
        angle_of_attack_cos=torch.cos(angle),
        angle_of_attack_sin=torch.sin(angle),
        nose_rate=nose_rate,
        energy=energy,
        log_refractive_index=zeros,
        log_index_gradient_x=zeros,
        log_index_gradient_y=zeros,
        shield=zeros,
        shield_recharge_wait=zeros,
        alive=~flags,
        fired=flags,
        spawned=flags,
        respawn_wait=integers,
        gun_cooldown=integers,
        next_slot=integers,
        ship_type=ship_type,
    )


def empty_map(like: torch.Tensor) -> Fields:
    """One field too far away to touch anything: ln n = 0 everywhere a ship flies."""
    B = like.shape[0]
    ones = torch.ones((B, 1), dtype=like.dtype, device=like.device)
    return Fields(
        x=ones * FAR,
        y=ones * 0.0,
        radius=ones,
        band_width=ones,
        level=torch.ones((B, 1), dtype=torch.int32, device=like.device),
    )


class SteadyTurn(NamedTuple):
    """A coasting turn's rest point: the nose at its commanded angle, turning with the
    path. Shaped like the speeds given."""

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


def derivative_weights(tick_length: float, like: torch.Tensor, time_stretch: float) -> torch.Tensor:
    """(K + 1,) weights whose dot product with samples at t = 0, h, ..., K h is the slope at
    t = 0 of their least-squares polynomial of degree FIT_DEGREE over FIT_WINDOW (times the
    card's time stretch)."""
    count = max(round(FIT_WINDOW * time_stretch / tick_length), FIT_DEGREE + 1) + 1
    time = torch.arange(count, dtype=like.dtype, device=like.device) * tick_length
    vandermonde = torch.stack([time**p for p in range(FIT_DEGREE + 1)], -1)
    return torch.linalg.pinv(vandermonde)[1]


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
    nose = torch.atan2(
        cos * after.heading_sin - sin * after.heading_cos,
        cos * after.heading_cos + sin * after.heading_sin,
    )
    return nose - (after.angle_of_attack - before.angle_of_attack)


def aitken(first, second, third):
    """The limit of three equally spaced samples of a geometric approach (the last sample
    where they are not one: converged, oscillating or noisy)."""
    step1, step2 = second - first, third - second
    ratio = step2 / torch.where(step1 == 0.0, 1.0, step1)
    geometric = (step1 != 0.0) & (ratio > 0.0) & (ratio < 0.9)
    return torch.where(geometric, third + step2 * ratio / (1.0 - ratio), third)


def cruise_speed(rules: Rules, types: ShipTypes, energy: torch.Tensor) -> torch.Tensor:
    """(B, types, points) the straight coasting equilibrium at stores energy (B, types,
    points): the fastest speed the flight tick leaves unchanged, the terminal speed a ship coasting
    straight reaches from above. Bracketed on a grid of one-tick flights from the stall speed to 30
    times it (the last sign change of the speed's gain from + to -), then bisected. NaN where there
    is none."""
    types = batched(types)
    B, type_count, points = energy.shape
    ship_type = (
        type_axis(types)[..., None].expand(B, type_count, points).reshape(B, type_count * points)
    )
    airframe = airframe_of(types, ship_type)
    stall = torch.sqrt(rules.flight.gravity / airframe.max_lift_coefficient)
    flat = energy.reshape(B, type_count * points)

    def gain(speed):  # (B, P, ...) speeds
        shape = speed.shape
        speed = speed.reshape(B, -1)
        stores = flat[..., None].expand(shape).reshape(B, -1)
        kinds = ship_type[..., None].expand(shape).reshape(B, -1)
        zeros = torch.zeros_like(speed)
        start = ships(speed, stores, zeros, zeros, kinds)
        flown = airframe_of(types, kinds)
        end = flight_tick(
            rules.flight, flown, rules.tick_length, start, zeros, zeros, empty_map(speed)
        )
        return (end.proper_speed - speed).reshape(shape)

    # The grid's top: 30 stall speeds, or below the drag rise's stiff range (where one
    # tick's drag outruns the speed and the explicit kick's gain changes sign spuriously).
    grid = stall[..., None] * torch.logspace(
        0.0, math.log10(30.0), CRUISE_GRID, dtype=stall.dtype, device=stall.device
    )
    if rules.flight.critical_speed > 0.0:
        top = (CRUISE_DRAG_RISE_TOP * rules.flight.critical_speed / stall).clamp_max(30.0)
        steps = torch.linspace(0.0, 1.0, CRUISE_GRID, dtype=stall.dtype, device=stall.device)
        grid = stall[..., None] * top[..., None] ** steps
    rising = gain(grid) > 0.0
    falls = rising[..., :-1] & ~rising[..., 1:]  # a stable root in (grid[i], grid[i + 1])
    found = falls.any(-1)
    last = (CRUISE_GRID - 2) - falls.flip(-1).int().argmax(-1)
    low = grid.gather(-1, last[..., None])[..., 0]
    high = grid.gather(-1, (last + 1)[..., None])[..., 0]
    for _ in range(CRUISE_BISECTIONS):
        middle = 0.5 * (low + high)
        up = gain(middle[..., None])[..., 0] > 0.0
        low, high = torch.where(up, middle, low), torch.where(up, high, middle)
    return torch.where(found, 0.5 * (low + high), torch.nan).reshape(B, type_count, points)


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
CLIMB, DIVE, ZOOM, LOW_SPEED, SUSTAINED, BLEED, REVERSAL = range(7)


def fly(rules: Rules, types: ShipTypes, tick_function=None) -> Flown:
    """Fly every manoeuvre of every card (B, num_types) in one batch and read the flown
    handles off it.

    The cruise is found first (cruise_speed at a full store), since the climb, the dive and
    the zoom start from it. The sustained turn (full stick) and the bleed (BLEED_STICK)
    start at corner speed with the nose at its steady angle of attack; the sustained turn's
    store is held below the knee. Every window is flown stretched by the batch's
    `card_stretch` (the slowest card's): the climb's by its climb scale, the rest by its time
    scale.

    tick_function: the flight tick to fly with (the game's `flight_tick` by default; the
    play shell's tuning passes a compiled one, which the runtime builds)."""
    step = flight_tick if tick_function is None else tick_function
    types = batched(types)
    constants, h = rules.flight, rules.tick_length
    columns = type_axis(types)
    airframe = airframe_of(types, columns)
    B, type_count = columns.shape
    full = airframe.max_energy
    cruise = cruise_speed(rules, types, full[..., None])[..., 0]
    stall = torch.sqrt(constants.gravity / airframe.max_lift_coefficient)
    corner = airframe.corner_speed
    turning = TURNING_STORE * full
    hard_left = torch.ones_like(full)
    steady = steady_turn(rules, airframe, corner, turning, hard_left)
    bleeding = steady_turn(rules, airframe, corner, turning, BLEED_STICK * hard_left)
    zeros, ones = torch.zeros_like(full), torch.ones_like(full)

    def blocks(*values):
        return torch.cat(values, -1)  # (B, types) per manoeuvre -> (B, 7 types)

    speed = blocks(cruise, cruise, cruise, stall, corner, corner, corner)
    energy = blocks(zeros, DIVE_STORE * full, DIVE_STORE * full, zeros, turning, turning, turning)
    angle = blocks(zeros, zeros, zeros, zeros, steady.angle, bleeding.angle, -steady.angle)
    nose_rate = blocks(
        zeros, zeros, zeros, zeros, steady.path_rate, bleeding.path_rate, -steady.path_rate
    )
    stick = blocks(zeros, zeros, zeros, zeros, ones, BLEED_STICK * ones, ones)
    power = blocks(zeros, -ones, ones, zeros, zeros, zeros, zeros)
    ship_type = columns.repeat(1, 7)
    # The sustained turn's store is held, so it stays below the knee however long it flies.
    held = blocks(
        *(torch.full_like(full, block == SUSTAINED, dtype=torch.bool) for block in range(7))
    )
    state = ships(speed, energy, angle, nose_rate, ship_type)
    flight = airframe_of(types, ship_type)
    fields = empty_map(speed)

    scale = card_stretch(constants, types)
    weights = derivative_weights(h, speed, scale.time)
    short = max(len(weights) - 1, round(REVERSAL_LIMIT * scale.time / h))
    settle = round(SETTLE_TIME * scale.time / h)
    samples = [  # the path angle at these ticks
        settle - round(k * SAMPLE_SPACING * scale.time / h) for k in (3.0, 2.0, 1.0, 0.0)
    ]
    climb_target = (1.0 - constants.store_fade_band) * full
    climb_time = torch.full_like(full, torch.nan)
    history = {"speed": [speed], "energy": [energy], "angle": [angle]}
    path, path_samples = torch.zeros_like(speed), {}
    limit = round(CLIMB_LIMIT * scale.climb / h)
    tick, check = 0, max(round(CLIMB_PROGRESS * scale.time / h), 1)
    pending, checkpoint = climb_time.isnan(), energy[:, :type_count]
    while tick < limit and (tick < max(short, settle) or pending.any()):
        new = step(constants, flight, h, state, stick, power, fields)
        path = path + path_turned(state, new)
        tick += 1
        before, after = state.energy[:, :type_count], new.energy[:, :type_count]
        crossed = climb_time.isnan() & (after >= climb_target)
        fraction = (climb_target - before) / (after - before)
        climb_time = torch.where(crossed, h * (tick - 1 + fraction), climb_time)
        pending = pending & ~crossed
        if tick % check == 0:  # a climb whose store stopped rising never finishes
            pending = pending & (after > checkpoint + 1e-6 * full)
            checkpoint = after
        state = replace(new, energy=torch.where(held, energy, new.energy))
        if tick <= short:
            for key, value in (
                ("speed", new.proper_speed),
                ("energy", new.energy),
                ("angle", new.angle_of_attack),
            ):
                history[key].append(value)
        if tick in samples:
            path_samples[tick] = path
    series = {key: torch.stack(values) for key, values in history.items()}

    def block(values, manoeuvre):
        return values[..., manoeuvre * type_count : (manoeuvre + 1) * type_count]

    def start_rate(values):
        return torch.einsum("k,k...->...", weights, values[: len(weights)])

    specific = 0.5 * series["speed"] ** 2 + series["energy"] / constants.mass
    rates = [
        (path_samples[later] - path_samples[earlier]) / ((later - earlier) * h)
        for earlier, later in zip(samples[:-1], samples[1:])
    ]
    reversal = crossing_time(block(series["angle"], REVERSAL), 0.9 * steady.angle, h)
    return Flown(
        cruise_speed=cruise,
        climb_time=climb_time,
        low_speed_acceleration=start_rate(block(series["speed"], LOW_SPEED)),
        sustained_turn_rate=block(aitken(*rates), SUSTAINED),
        bleed_rate=start_rate(block(specific, BLEED)),
        nose_reversal_time=reversal,
        dive_acceleration=start_rate(block(series["speed"], DIVE)),
        zoom_acceleration=start_rate(block(series["speed"], ZOOM)),
    )
