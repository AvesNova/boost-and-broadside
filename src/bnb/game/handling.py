"""A ship type's characteristic quantities, measured by flying the game's own code.

The scripted agent states every threshold as a dimensionless number times one of these
(scripted spec A2), so it follows the ship card. Each flown quantity comes from a short
batched simulation through `flight_tick`, `launch` and `bullet_motion` at the standard
field index (n = 1, no field nearby); nothing here restates the flight equations. The
card values are read as they are. `characteristics` caches the measurement per card and
rules.
"""

import dataclasses
import math

import torch

from bnb.contracts.tree import contract

from .physics.bullets import bullet_motion, launch, launch_state
from .physics.flight import airframe_of, flight_tick
from .rules import STICK_LEVELS, Rules
from .ship_types import ShipTypes, per_ship
from .state import Fields, Ships, replace

COAST, BOOST = 0.0, -1.0  # power modes pi (rules.POWER_MODES)
TURN_LEVELS = (STICK_LEVELS[2], STICK_LEVELS[4], STICK_LEVELS[6])  # left, by level
SETTLED = 1e-3  # m/s: cruise has settled when a second changes it by less than this
CRUISE_SECONDS = 240  # at most this long to settle (the default card settles in about 60)
TOP_SECONDS = 20  # a straight boost from cruise empties the store well within this
TURN_SECONDS = 1.0  # the nose settles into a held turn well within this (omega_c ~ 20/s)
FAR = 1e7  # m: the measurement's one field sits this far away, so n = 1 at the ship


@contract
class Characteristics:
    """Per ship type, (num_types,) float32 (turn_rate (num_types, 3)).

    cruise_speed: the no-input terminal speed, coasting straight until speed and store
        settle (m/s, proper, as every speed here); top_speed: the peak speed of a straight
        boost from cruise with a full store (the store empties first); corner_speed:
        sqrt(max_acceleration / max_lift_coefficient), the speed above which the g limit
        caps a full-stick turn; turn_rate: the steady flight-path turn rate at cruise for
        stick levels 1-3 (rad/s); stall_angle (rad), the gun's offset scale; gun_range: how
        far a bullet fired along the flight path at cruise travels in its life (m);
        bullet_speed: that bullet's launch speed (m/s); bullet_drag (1/m); the card's
        max_energy, bullet_energy_cost (J), max_shield, hit_radius (m) and
        max_acceleration (m/s^2); vision_range (m).
    """

    cruise_speed: torch.Tensor
    top_speed: torch.Tensor
    corner_speed: torch.Tensor
    turn_rate: torch.Tensor
    stall_angle: torch.Tensor
    gun_range: torch.Tensor
    bullet_speed: torch.Tensor
    bullet_drag: torch.Tensor
    max_energy: torch.Tensor
    bullet_energy_cost: torch.Tensor
    max_shield: torch.Tensor
    hit_radius: torch.Tensor
    max_acceleration: torch.Tensor
    vision_range: torch.Tensor


def time_of_flight(characteristics: Characteristics, ship_type: torch.Tensor,
                   distance: torch.Tensor) -> torch.Tensor:
    """The time a bullet fired at cruise by a ship of this type takes to travel `distance`
    (s), from the quadratic drag law at n = 1: t = (exp(k s) - 1) / (k w0)."""
    drag = per_ship(characteristics.bullet_drag, ship_type)
    speed = per_ship(characteristics.bullet_speed, ship_type)
    return torch.expm1(drag * distance) / (drag * speed)


def _far_field() -> Fields:
    one = torch.ones((1, 1))
    return Fields(x=one * FAR, y=one * FAR, radius=one, band_width=one,
                  level=torch.ones((1, 1), dtype=torch.int32))


def _straight(types: ShipTypes, ship_type: torch.Tensor, speed: torch.Tensor,
              energy: torch.Tensor) -> Ships:
    """Ships (1, P) at the origin flying along +x with no angle of attack, at n = 1."""
    zeros, ones = torch.zeros_like(speed), torch.ones_like(speed)
    integer_zeros = torch.zeros_like(ship_type)
    flags = torch.ones_like(speed, dtype=torch.bool)
    return Ships(x=zeros, y=zeros, heading_cos=ones, heading_sin=zeros, proper_speed=speed,
                 angle_of_attack=zeros, angle_of_attack_cos=ones, angle_of_attack_sin=zeros,
                 nose_rate=zeros, energy=energy, log_refractive_index=zeros,
                 log_index_gradient_x=zeros, log_index_gradient_y=zeros,
                 shield=per_ship(types.max_shield, ship_type), shield_recharge_wait=zeros,
                 alive=flags, fired=~flags, spawned=~flags, respawn_wait=integer_zeros,
                 gun_cooldown=integer_zeros, next_slot=integer_zeros, ship_type=ship_type)


def _fly(rules: Rules, types: ShipTypes, ships: Ships, stick: float, power: float,
         ticks: int, hold_speed: torch.Tensor | None = None) -> tuple[Ships, torch.Tensor]:
    """Fly `ticks` ticks with the controls held; returns the ships and the speed after
    every tick (ticks, P). hold_speed: reset the speed to it after every tick."""
    airframe, fields = airframe_of(types, ships.ship_type), _far_field()
    stick_level = torch.full_like(ships.x, stick)
    power_mode = torch.full_like(ships.x, power)
    speeds = []
    for _ in range(ticks):
        ships = flight_tick(rules.flight, airframe, rules.tick_length, ships, stick_level,
                            power_mode, fields)
        if hold_speed is not None:
            ships = replace(ships, proper_speed=hold_speed)
        speeds.append(ships.proper_speed[0])
    return ships, torch.stack(speeds)


def _path_angle(ships: Ships) -> torch.Tensor:
    cos, sin = ships.heading_cos, ships.heading_sin
    angle_cos, angle_sin = ships.angle_of_attack_cos, ships.angle_of_attack_sin
    return torch.atan2(sin * angle_cos - cos * angle_sin, cos * angle_cos + sin * angle_sin)


@torch.no_grad()
def measure(types: ShipTypes, rules: Rules) -> Characteristics:
    """Every characteristic of every ship type of the card (one card: (num_types,) values),
    on the CPU in float32; the result is on the card's device."""
    device = types.max_energy.device
    if types.max_energy.dim() != 1:
        raise ValueError("measure one card at a time, not a card per environment")
    types = ShipTypes(**{field.name: getattr(types, field.name).cpu()
                         for field in dataclasses.fields(types)})
    num_types = types.max_energy.shape[0]
    ship_type = torch.arange(num_types, dtype=torch.int32)[None]
    card = {name: per_ship(getattr(types, name), ship_type)[0]
            for name in ("max_energy", "bullet_energy_cost", "max_shield", "hit_radius",
                         "max_acceleration")}
    corner = airframe_of(types, ship_type).corner_speed

    # Cruise: coast straight from corner speed with a full store until a second changes
    # the speed by less than SETTLED.
    second = round(1.0 / rules.tick_length)
    ships = _straight(types, ship_type, corner.clone(), card["max_energy"][None].clone())
    for _ in range(CRUISE_SECONDS):
        start = ships.proper_speed
        ships, _ = _fly(rules, types, ships, 0.0, COAST, second)
        if bool(((ships.proper_speed - start).abs() < SETTLED).all()):
            break
    else:
        raise RuntimeError(f"cruise did not settle within {CRUISE_SECONDS} s")
    cruising = ships
    cruise = cruising.proper_speed

    # Top speed: a straight boost from cruise, full store, until well after it empties.
    _, speeds = _fly(rules, types, cruising, 0.0, BOOST, TOP_SECONDS * second)
    top = speeds.amax(0)

    # Turn rates: hold each stick level at cruise speed until the nose settles, then read
    # the flight path's rate over the last tick.
    rates = []
    for level in TURN_LEVELS:
        settled, _ = _fly(rules, types, cruising, level, COAST, round(TURN_SECONDS * second),
                          hold_speed=cruise)
        after, _ = _fly(rules, types, settled, level, COAST, 1, hold_speed=cruise)
        turned = torch.remainder(_path_angle(after) - _path_angle(settled) + math.pi,
                                 2 * math.pi) - math.pi
        rates.append(turned[0] / rules.tick_length)

    # Gun range: a bullet fired at cruise, no spread, flown for its whole life.
    bullet = launch(rules, types, launch_state(cruising), torch.zeros(cruising.x.shape + (2,)))
    start_x, start_y, bullet_speed = bullet.x, bullet.y, bullet.proper_speed[0]
    for _ in range(rules.bullet_moves):
        bullet = bullet_motion(rules, bullet, _far_field())
    gun_range = torch.hypot(bullet.x - start_x, bullet.y - start_y)[0]

    def per_type(value: float) -> torch.Tensor:
        return torch.full((num_types,), value)

    measured = Characteristics(
        cruise_speed=cruise[0], top_speed=top, corner_speed=corner[0],
        turn_rate=torch.stack(rates, -1), stall_angle=per_type(rules.flight.stall_angle),
        gun_range=gun_range, bullet_speed=bullet_speed,
        bullet_drag=per_type(rules.bullet_drag), vision_range=per_type(rules.vision_range),
        **card)
    return Characteristics(**{field.name: getattr(measured, field.name).float().to(device)
                              for field in dataclasses.fields(measured)})


_CACHE: dict = {}


def characteristics(types: ShipTypes, rules: Rules) -> Characteristics:
    """`measure`, cached per card values, device and the rules that the measurement reads."""
    key = (tuple(tuple(getattr(types, field.name).flatten().tolist())
                 for field in dataclasses.fields(types)), str(types.max_energy.device),
           rules.flight, rules.tick_length, rules.bullet_drag, rules.bullet_moves,
           rules.bullet_spread, rules.vision_range)
    if key not in _CACHE:
        _CACHE[key] = measure(types, rules)
    return _CACHE[key]
