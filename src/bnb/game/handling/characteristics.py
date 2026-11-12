"""A ship type's characteristic quantities for the scripted agent (scripted spec A2), from the
ship card's own measurements: the handles it reads (the cruise and the closed forms, as
`bnb.game.handling.measure` has them), the shown top speed and the coast
datum's turn rates, all flown in the game's code. The agent states every threshold as a
dimensionless number times one of these, so it follows the card.

`characteristics` caches them per card and rules, in the process and on disk
(`bnb.cache`); `measure_aside` measures them in a child process, for a caller (the play
shell) whose thread must not stall.
"""

import dataclasses
import multiprocessing
import os
import pathlib

import torch

from bnb import cache
from bnb.contracts.tree import contract

from ..physics.flight import airframe_of
from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from .manoeuvres import batched, cruise_speed, type_axis
from .shown import STICKS, settled, top_speed
from .weapons import combat_handles


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
        bullet_drag (1/m); climb_power: the store's charge rate coasting straight at
        cruise with the store below the engine knee (W, per unit mass g times the climb
        rate), in closed form: the mass times the cruise times the coast exchange; the
        card's max_energy, bullet_energy_cost (J), max_shield,
        hit_radius (m) and max_acceleration (m/s^2); vision_range (m); gravity (m/s^2), the
        rules' (the same for every type: the scripted agent tells a real-scale card by it).
    """

    cruise_speed: torch.Tensor
    top_speed: torch.Tensor
    corner_speed: torch.Tensor
    turn_rate: torch.Tensor
    stall_angle: torch.Tensor
    gun_range: torch.Tensor
    bullet_speed: torch.Tensor
    bullet_drag: torch.Tensor
    climb_power: torch.Tensor
    max_energy: torch.Tensor
    bullet_energy_cost: torch.Tensor
    max_shield: torch.Tensor
    hit_radius: torch.Tensor
    max_acceleration: torch.Tensor
    vision_range: torch.Tensor
    gravity: torch.Tensor


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
    # The handles it reads, without the flights only the solver needs (a real-scale climb
    # flies for an hour of game time): the cruise and the closed forms, as `measure` has them.
    card = batched(types)
    airframe = airframe_of(card, type_axis(card))
    cruise = cruise_speed(rules, card, airframe.max_energy[..., None])[..., 0]
    gun_range = combat_handles(rules, card, cruise)["gun_range"][0]
    all_types = torch.arange(types.max_energy.shape[0])
    measured = Characteristics(
        cruise_speed=cruise[0],
        top_speed=top_speed(rules, card)[0],
        corner_speed=airframe_of(types, all_types).corner_speed,
        turn_rate=settled(rules, card, STICKS).turn_rate[0],
        stall_angle=rules.flight.stall_angle * types.stall_angle_scale,
        gun_range=gun_range,
        bullet_speed=cruise[0] + types.muzzle_speed,
        bullet_drag=rules.bullet_drag * types.bullet_drag_scale,
        climb_power=rules.flight.mass
        * cruise[0]
        * rules.flight.coast_exchange
        * types.coast_exchange_scale,
        max_energy=types.max_energy,
        bullet_energy_cost=types.bullet_energy_cost,
        max_shield=types.max_shield,
        hit_radius=types.hit_radius,
        max_acceleration=types.max_acceleration,
        vision_range=rules.vision_range * types.vision_range_scale,
        gravity=torch.full_like(types.max_energy, rules.flight.gravity),
    )
    return Characteristics(
        **{
            field.name: getattr(measured, field.name).float().to(device)
            for field in dataclasses.fields(measured)
        }
    )


_CACHE: dict = {}
_CODE = ("game", "contracts")  # the code a measurement flies


def _key(types: ShipTypes, rules: Rules) -> tuple:
    """What a measurement depends on: the card's values, its device and the rules it
    reads."""
    return (
        tuple(
            tuple(getattr(types, field.name).flatten().tolist())
            for field in dataclasses.fields(types)
        ),
        str(types.max_energy.device),
        rules.flight,
        rules.tick_length,
        rules.bullet_drag,
        rules.bullet_spread,
        rules.vision_range,
        rules.shield_recharge_rate,
        rules.shield_recharge_delay,
        rules.head_on_fraction,
    )


def _disk_path(key: tuple) -> pathlib.Path:
    return cache.path("characteristics", key, packages=_CODE, suffix=".pt")


def cached(types: ShipTypes, rules: Rules) -> Characteristics | None:
    """The card's characteristics if this process or an earlier run measured them (on disk
    in `bnb.cache`), else None; never measures."""
    key = _key(types, rules)
    if key not in _CACHE:
        path = _disk_path(key)
        if not path.exists():
            return None
        try:
            stored = torch.load(path, map_location=types.max_energy.device, weights_only=True)
        except Exception:  # a damaged entry: measured again
            return None
        _CACHE[key] = Characteristics(**stored)
    return _CACHE[key]


def characteristics(types: ShipTypes, rules: Rules) -> Characteristics:
    """`measure_characteristics`, cached per card values, device and the rules it reads, in
    this process and on disk (`cached`)."""
    found = cached(types, rules)
    if found is not None:
        return found
    key = _key(types, rules)
    measured = _CACHE[key] = measure_characteristics(types, rules)
    path = _disk_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial")
    fields = {field.name: getattr(measured, field.name) for field in dataclasses.fields(measured)}
    torch.save(fields, partial)
    partial.replace(path)
    return measured


def _measure_child(types: ShipTypes, rules: Rules) -> None:
    torch.set_num_threads(1)
    characteristics(types, rules)


def measure_aside(types: ShipTypes, rules: Rules) -> None:
    """Measure the card's characteristics in a child process into the disk cache, and wait
    for it. Called from a thread, the wait holds no interpreter lock, so the process's
    other threads run on (measuring on a thread of this process would starve them)."""
    if cached(types, rules) is not None:
        return
    child = multiprocessing.get_context("spawn").Process(
        target=_measure_child, args=(types, rules), name="characteristics", daemon=True
    )
    child.start()
    child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"measuring the characteristics failed (exit code {child.exitcode})")
