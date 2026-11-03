"""A ship type's characteristic quantities for the scripted agent (scripted spec A2), from the
ship card's own measurements: the handles (`measure`), the shown top speed and the coast
datum's turn rates, all flown in the game's code. The agent states every threshold as a
dimensionless number times one of these, so it follows the card.

`characteristics` caches them per card and rules.
"""

import dataclasses

import torch

from bnb.contracts.tree import contract

from ..physics.flight import airframe_of
from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from .manoeuvres import batched
from .shown import STICKS, settled, top_speed
from .solver import measure


@contract
class Characteristics:
    """Per ship type, (num_types,) float32 (turn_rate (num_types, 3)).

    cruise_speed: the terminal speed coasting straight with a full store (m/s, proper, as
        every speed here); top_speed: the peak of a straight boost from cruise with a full
        store; corner_speed: the game's `Airframe.corner_speed`, above which the g limit
        caps a full-stick turn; turn_rate: the coast datum's settled flight-path
        rate at stick levels 1-3 (rad/s); stall_angle (rad), the gun's offset scale;
        gun_range: how far a bullet fired along the flight path at cruise travels in its
        life (m); bullet_speed: that bullet's launch speed, cruise plus muzzle speed (m/s);
        bullet_drag (1/m); the card's max_energy, bullet_energy_cost (J), max_shield,
        hit_radius (m) and max_acceleration (m/s^2); vision_range (m).
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


def time_of_flight(
    characteristics: Characteristics, ship_type: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """The time a bullet fired at cruise by a ship of this type takes to travel `distance`
    (s), from the quadratic drag law at n = 1: t = (exp(k s) - 1) / (k w0)."""
    drag = per_ship(characteristics.bullet_drag, ship_type)
    speed = per_ship(characteristics.bullet_speed, ship_type)
    return torch.expm1(drag * distance) / (drag * speed)


@torch.no_grad()
def measure_characteristics(types: ShipTypes, rules: Rules) -> Characteristics:
    """Every characteristic of every type of one card ((num_types,) fields), measured on
    the CPU in float32; the result is on the card's device."""
    device = types.max_energy.device
    if types.max_energy.dim() != 1:
        raise ValueError("measure one card at a time, not a card per environment")
    types = ShipTypes(
        **{
            field.name: getattr(types, field.name).cpu().float()
            for field in dataclasses.fields(types)
        }
    )
    handles = measure(types, rules)
    card = batched(types)
    all_types = torch.arange(types.max_energy.shape[0])
    measured = Characteristics(
        cruise_speed=handles.cruise_speed,
        top_speed=top_speed(rules, card)[0],
        corner_speed=airframe_of(types, all_types).corner_speed,
        turn_rate=settled(rules, card, STICKS).turn_rate[0],
        stall_angle=handles.stall_angle,
        gun_range=handles.gun_range,
        bullet_speed=handles.cruise_speed + types.muzzle_speed,
        bullet_drag=rules.bullet_drag * types.bullet_drag_scale,
        max_energy=types.max_energy,
        bullet_energy_cost=types.bullet_energy_cost,
        max_shield=types.max_shield,
        hit_radius=types.hit_radius,
        max_acceleration=types.max_acceleration,
        vision_range=handles.vision_range,
    )
    return Characteristics(
        **{
            field.name: getattr(measured, field.name).float().to(device)
            for field in dataclasses.fields(measured)
        }
    )


_CACHE: dict = {}


def characteristics(types: ShipTypes, rules: Rules) -> Characteristics:
    """`measure_characteristics`, cached per card values, device and the rules it reads."""
    key = (
        tuple(
            tuple(getattr(types, field.name).flatten().tolist())
            for field in dataclasses.fields(types)
        ),
        str(types.max_energy.device),
        rules.flight,
        rules.tick_length,
        rules.bullet_drag,
        rules.bullet_moves,
        rules.bullet_spread,
        rules.vision_range,
        rules.shield_recharge_rate,
        rules.shield_recharge_delay,
        rules.head_on_fraction,
    )
    if key not in _CACHE:
        _CACHE[key] = measure_characteristics(types, rules)
    return _CACHE[key]
