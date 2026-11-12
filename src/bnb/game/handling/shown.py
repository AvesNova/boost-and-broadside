"""The quantities the ship card shows but does not set (ship-card spec 3), and the data of
the reference charts (spec 6), all flown in the game's own code.

Every function takes a batched card (B, num_types) and returns (B, num_types, ...) tensors.
Their windows are in seconds at the standard card's scales and flown stretched by the
card's own (`manoeuvres.card_stretch`): the settling turns by its time, V_s / g; the boost
and the firing, which last as long as the store, by its climb's, E_max / (m g V_s).
"""

import math
from typing import NamedTuple

import torch

from ..physics.bullets import fire
from ..physics.flight import airframe_of, flight_tick
from ..rules import STICK_LEVELS, Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, replace
from .manoeuvres import (
    TURNING_STORE,
    aitken,
    batched,
    card_stretch,
    cruise_speed,
    derivative_weights,
    empty_map,
    path_turned,
    ships,
    steady_turn,
    type_axis,
)
from .weapons import whole_tick_life

STICKS = tuple(level for level in STICK_LEVELS if level > 0.0)  # 0.4, 0.75, 1
SUSTAINED_CURVE = tuple(i / 12 for i in range(1, 13))  # stick levels drawn on the doghouse
SETTLE_TIME = 40.0  # s
AVERAGING_WINDOW = 1.0  # s, each of the last three settled windows
ENDURANCE_LIMIT = 120.0  # s: firing longer than this counts as unlimited
TOP_SPEED_LIMIT = 60.0  # s
SPEED_GRID = 160  # speeds on the charts' curves


def _grid(types: ShipTypes, *values: torch.Tensor):
    """(B, types, grid) values laid out as ships (B, types grid) with their types and airframes."""
    B, type_count, grid = values[0].shape
    ship_type = (
        type_axis(types)[..., None].expand(B, type_count, grid).reshape(B, type_count * grid)
    )
    return (
        ship_type,
        airframe_of(types, ship_type),
        [v.reshape(B, type_count * grid) for v in values],
    )


def trajectory(
    rules: Rules,
    types: ShipTypes,
    speed,
    energy,
    angle,
    nose_rate,
    stick,
    power,
    ticks: int,
    hold_store: bool = False,
    stop=None,
) -> dict[str, torch.Tensor]:
    """Fly (B, types, grid) starts for `ticks` ticks: (ticks + 1, B, types, grid) speed,
    energy, angle of attack and the flight path's angle (from 0). hold_store: the store is
    reset to its start after every tick. stop: called with the ships before and after each
    tick; True ends the flight there (fewer ticks are returned)."""
    B, type_count, grid = speed.shape
    ship_type, airframe, (speed, energy, angle, nose_rate, stick, power) = _grid(
        types, speed, energy, angle, nose_rate, stick, power
    )
    state = ships(speed, energy, angle, nose_rate, ship_type)
    fields = empty_map(speed)
    path = torch.zeros_like(speed)
    history = {"speed": [speed], "energy": [energy], "angle": [angle], "path": [path]}
    for _ in range(ticks):
        new = flight_tick(rules.flight, airframe, rules.tick_length, state, stick, power, fields)
        path = path + path_turned(state, new)
        before, state = state, replace(new, energy=energy) if hold_store else new
        for key, value in (
            ("speed", new.proper_speed),
            ("energy", new.energy),
            ("angle", new.angle_of_attack),
            ("path", path),
        ):
            history[key].append(value)
        if stop is not None and stop(before, new):
            break
    return {k: torch.stack(v).reshape(-1, B, type_count, grid) for k, v in history.items()}


class Instant(NamedTuple):
    """Rates at the start of a steady turn, (B, types, grid) over a speed grid."""

    speed: torch.Tensor
    turn_rate: torch.Tensor  # rad/s, the flight path's
    specific_power: torch.Tensor  # W/kg, Ps = u du/dt + (dE/dt)/m
    angle: torch.Tensor  # the steady angle of attack (the gun's offset from the path)


def instant(rules: Rules, types: ShipTypes, speed: torch.Tensor, stick: float) -> Instant:
    """At each speed (B, types, grid): the nose at its steady angle of attack for the stick,
    coasting, below the engine knee; the start rates fitted over the first ticks."""
    B, type_count, grid = speed.shape
    full = types.max_energy[..., None].expand(B, type_count, grid)
    energy = TURNING_STORE * full
    sticks = torch.full_like(speed, stick)
    _, airframe, (flat_speed, flat_energy, flat_stick) = _grid(types, speed, energy, sticks)
    steady = steady_turn(rules, airframe, flat_speed, flat_energy, flat_stick)
    weights = derivative_weights(rules.tick_length, speed, card_stretch(rules.flight, types).time)
    flown = trajectory(
        rules,
        types,
        speed,
        energy,
        steady.angle.reshape(B, type_count, grid),
        steady.path_rate.reshape(B, type_count, grid),
        sticks,
        torch.zeros_like(speed),
        len(weights) - 1,
    )

    def start_rate(values):
        return torch.einsum("k,k...->...", weights, values)

    specific = 0.5 * flown["speed"] ** 2 + flown["energy"] / rules.flight.mass
    return Instant(
        speed,
        start_rate(flown["path"]),
        start_rate(specific),
        steady.angle.reshape(B, type_count, grid),
    )


class Settled(NamedTuple):
    """Settled coasting turns, (B, types, sticks) over stick levels."""

    stick: torch.Tensor
    speed: torch.Tensor
    turn_rate: torch.Tensor
    angle: torch.Tensor  # the angle of attack: the gun's offset from the flight path


def settled(rules: Rules, types: ShipTypes, sticks: tuple[float, ...]) -> Settled:
    """The coast datum at each stick level: coasting from cruise with the store held below
    the engine knee until speed, angle of attack and turn rate settle (the Aitken limit of
    the means over the last three AVERAGING_WINDOWs), for SETTLE_TIME (both stretched by
    the card's time scale)."""
    B, type_count = types.max_energy.shape
    stick_count = len(sticks)
    stick = torch.tensor(sticks, dtype=types.max_energy.dtype).expand(B, type_count, stick_count)
    cruise = cruise_speed(rules, types, types.max_energy[..., None])
    zeros = torch.zeros_like(stick)
    time = card_stretch(rules.flight, types).time
    ticks = round(SETTLE_TIME * time / rules.tick_length)
    last = round(AVERAGING_WINDOW * time / rules.tick_length)
    energy = TURNING_STORE * types.max_energy[..., None].expand(B, type_count, stick_count)
    flown = trajectory(
        rules,
        types,
        cruise.expand(B, type_count, stick_count),
        energy,
        zeros,
        zeros,
        stick,
        zeros,
        ticks,
        hold_store=True,
    )
    ends = [ticks - k * last for k in (2, 1, 0)]  # three one-second windows

    def settled_value(values):
        return aitken(*(values[end - last + 1 : end + 1].mean(0) for end in ends))

    rates = [
        (flown["path"][end] - flown["path"][end - last]) / (last * rules.tick_length)
        for end in ends
    ]
    return Settled(
        stick, settled_value(flown["speed"]), aitken(*rates), settled_value(flown["angle"])
    )


def top_speed(rules: Rules, types: ShipTypes) -> torch.Tensor:
    """(B, types) the peak speed of a straight boost from cruise with a full store, within
    TOP_SPEED_LIMIT (stretched by the card's climb scale); the flight ends once every store
    is spent and every speed falls."""
    full = types.max_energy[..., None]
    cruise = cruise_speed(rules, types, full)
    zeros = torch.zeros_like(cruise)
    fade = rules.flight.store_fade_band * types.max_energy.repeat_interleave(full.shape[-1], -1)

    def past_peak(before, after) -> bool:
        """Every store is in its fade band and every speed falls: the boost is spent, and a
        ship coasting above cruise only slows."""
        return bool(((after.energy <= fade) & (after.proper_speed < before.proper_speed)).all())

    flown = trajectory(
        rules,
        types,
        cruise,
        full,
        zeros,
        zeros,
        zeros,
        zeros - 1.0,
        round(TOP_SPEED_LIMIT * card_stretch(rules.flight, types).climb / rules.tick_length),
        stop=past_peak,
    )
    return flown["speed"].amax(0)[..., 0]


ENDURANCE = dict(coasting=(0.0, 0.0), climbing=(0.0, 1.0), diving=(0.0, -1.0), hard_turn=(1.0, 0.0))


def endurance(rules: Rules, types: ShipTypes) -> dict[str, torch.Tensor]:
    """(B, types) per flight condition (ENDURANCE: stick, power): the time from cruise with a full
    store, firing at every chance (the game's fire rule), until a shot is refused for lack of
    energy; inf beyond ENDURANCE_LIMIT (stretched by the card's climb scale)."""
    B, type_count = types.max_energy.shape
    conditions = len(ENDURANCE)
    cruise = cruise_speed(rules, types, types.max_energy[..., None]).expand(
        B, type_count, conditions
    )
    full = types.max_energy[..., None].expand(B, type_count, conditions)
    stick = torch.tensor([v[0] for v in ENDURANCE.values()], dtype=full.dtype).expand(
        B, type_count, conditions
    )
    power = torch.tensor([v[1] for v in ENDURANCE.values()], dtype=full.dtype).expand(
        B, type_count, conditions
    )
    zeros = torch.zeros_like(full)
    ship_type, airframe, (speed, energy, stick, power, zeros) = _grid(
        types, cruise, full, stick, power, zeros
    )
    state = ships(speed, energy, zeros, zeros, ship_type)
    K = rules.bullet_slots
    empty = torch.zeros(speed.shape + (K,), dtype=speed.dtype)
    bullets = Bullets(
        x=empty,
        y=empty,
        direction_x=empty,
        direction_y=empty,
        proper_speed=empty,
        log_refractive_index=empty,
        log_index_gradient_x=empty,
        log_index_gradient_y=empty,
        moves_left=empty.int(),
    )
    cost = per_ship(types.bullet_energy_cost, ship_type)
    shoot = torch.ones_like(speed, dtype=torch.bool)
    spread = torch.zeros(speed.shape + (2,), dtype=speed.dtype)
    fields = empty_map(speed)
    refused = torch.full_like(speed, math.inf)
    limit = ENDURANCE_LIMIT * card_stretch(rules.flight, types).climb
    for tick in range(round(limit / rules.tick_length)):
        ready = state.gun_cooldown == 0
        state, bullets = fire(rules, types, state, bullets, shoot, spread)
        refused = torch.where(
            ready & ~state.fired & (state.energy < cost) & refused.isinf(),
            tick * rules.tick_length,
            refused,
        )
        state = flight_tick(rules.flight, airframe, rules.tick_length, state, stick, power, fields)
        state = replace(state, gun_cooldown=(state.gun_cooldown - 1).clamp_min(0))
        if refused.isfinite().all():
            break
    return dict(zip(ENDURANCE, refused.reshape(B, type_count, conditions).unbind(-1)))


class Shown(NamedTuple):
    """The shown quantities, (B, types) unless noted."""

    corner_speed: torch.Tensor
    peak_turn_rate: torch.Tensor  # the instantaneous peak over speed, full stick
    peak_turn_speed: torch.Tensor
    minimum_radius: torch.Tensor
    datum: Settled  # (B, types, 3) at the stick levels 0.4, 0.75, 1
    top_speed: torch.Tensor
    endurance: dict
    damage_per_second: torch.Tensor


def shown(rules: Rules, types: ShipTypes) -> Shown:
    """Every shown quantity of a card ((num_types,) or (B, num_types) fields), flown in its
    dtype at the rules' tick."""
    types = batched(types)
    constants = rules.flight
    corner = torch.sqrt(types.max_acceleration / types.max_lift_coefficient)
    stall = torch.sqrt(constants.gravity / types.max_lift_coefficient)
    fraction = torch.linspace(0.0, 1.0, SPEED_GRID, dtype=corner.dtype)
    speeds = (1.01 * stall)[..., None] + (2.0 * corner - 1.01 * stall)[..., None] * fraction
    hard = instant(rules, types, speeds, 1.0)
    peak, at = hard.turn_rate.max(-1)
    radius = (hard.speed / hard.turn_rate).amin(-1)
    return Shown(
        corner_speed=corner,
        peak_turn_rate=peak,
        peak_turn_speed=hard.speed.gather(-1, at[..., None])[..., 0],
        minimum_radius=radius,
        datum=settled(rules, types, STICKS),
        top_speed=top_speed(rules, types),
        endurance=endurance(rules, types),
        damage_per_second=types.hit_damage * types.fire_rate,
    )


# ------------------------------------------------------------------ similarity (spec 6)

# ship-card.md, "Similarity ratios": the P-51D inputs (m, s, m/s).
P51D = dict(
    top_speed=195.0,
    stall_speed=44.7,
    ship_radius=5.64,
    corner_speed=134.0,
    minimum_radius=231.0,
    sustained_circle=20.0,
    muzzle_speed=880.0,
    gun_range=250.0,
)


def similarity_inputs(rules: Rules, handles, view: Shown) -> dict[str, torch.Tensor]:
    """Ours in the P-51D table's terms, (B, types) each."""
    return dict(
        top_speed=view.top_speed,
        stall_speed=handles.stall_speed,
        ship_radius=handles.hit_radius,
        corner_speed=view.corner_speed,
        minimum_radius=view.minimum_radius,
        sustained_circle=2.0 * math.pi / handles.sustained_turn_rate,
        muzzle_speed=handles.muzzle_speed,
        gun_range=handles.gun_range,
        flight_time=whole_tick_life(
            rules,
            handles.gun_range,
            handles.cruise_speed + handles.muzzle_speed,
            handles.bullet_drag,
        ),
    )


def similarity_ratios(inputs: dict) -> dict:
    """The ship-card.md ratios from inputs (ours: similarity_inputs; the P-51D's: P51D, whose
    bullet flies its range without drag)."""
    flight_time = inputs.get("flight_time", inputs["gun_range"] / inputs["muzzle_speed"])
    instant_circle = 2.0 * math.pi * inputs["minimum_radius"] / inputs["corner_speed"]
    return {
        "top speed / stall speed": inputs["top_speed"] / inputs["stall_speed"],
        "corner speed / top speed": inputs["corner_speed"] / inputs["top_speed"],
        "sustained circle / instantaneous circle": inputs["sustained_circle"] / instant_circle,
        "ship radius / min turn radius": inputs["ship_radius"] / inputs["minimum_radius"],
        "bullet speed / top speed": inputs["muzzle_speed"] / inputs["top_speed"],
        "gun range / min turn radius": inputs["gun_range"] / inputs["minimum_radius"],
        "gun range / ship radius": inputs["gun_range"] / inputs["ship_radius"],
        "min turn radius / top speed (s)": inputs["minimum_radius"] / inputs["top_speed"],
        "bullet flight time / sustained circle": flight_time / inputs["sustained_circle"],
        "degrees a target turns during bullet flight": 360.0
        * flight_time
        / inputs["sustained_circle"],
    }
