"""The handles (ship-card spec 3): what a designer sets for each ship type.

Every field is a tensor shaped like the card's fields: (num_types,), or (B, num_types) for
a batch of cards. Units are SI with angles in radians; `UNITS` gives each handle's unit and
the factor that shows it in the chart tool's display units.
"""

import dataclasses
import math

import torch

from bnb.contracts.tree import contract


@contract
class Handles:
    """Flight handles first (FLIGHT, in the spec's order; ZERO_LIFT_DRAG_HANDLE is the one
    this section adds), then the combat handles (COMBAT)."""

    stall_speed: torch.Tensor  # m/s, where the lift reach is 1 g
    g_limit: torch.Tensor  # the load factor a_max / g
    cruise_speed: torch.Tensor  # m/s, the terminal speed coasting straight
    climb_time: torch.Tensor  # s, empty to (1 - beta) of the store, coasting from cruise
    ceiling: torch.Tensor  # J, the store size
    low_speed_acceleration: torch.Tensor  # m/s^2 at the stall speed, store empty
    sustained_turn_rate: torch.Tensor  # rad/s, settled full-stick coasting turn
    bleed_rate: torch.Tensor  # W/kg, specific energy rate at corner speed, full stick
    nose_reversal_time: torch.Tensor  # s, to 90% of the reversed angle of attack
    dive_acceleration: torch.Tensor  # m/s^2 at the start of a boost from cruise
    zoom_acceleration: torch.Tensor  # m/s^2 at the start of a reverse from cruise
    stall_angle: torch.Tensor  # rad, the angle-of-attack limit
    vision_range: torch.Tensor  # m
    zero_lift_drag: torch.Tensor  # 1/m, held by the solver
    hit_damage: torch.Tensor
    fire_interval: torch.Tensor  # s
    muzzle_speed: torch.Tensor  # m/s, proper
    bullet_spread: torch.Tensor  # m/s per axis, proper
    gun_range: torch.Tensor  # m, a bullet fired at cruise along the flight path, in its life
    shots_per_store: torch.Tensor  # shots a full store pays for
    max_shield: torch.Tensor
    shield_recharge_rate: torch.Tensor  # per s
    shield_recharge_delay: torch.Tensor  # s
    head_on_fraction: torch.Tensor  # damage taken from dead ahead
    hit_radius: torch.Tensor  # m
    spawn_energy: torch.Tensor  # J

    def replace(self, **values) -> Handles:
        return dataclasses.replace(self, **values)


NAMES = tuple(field.name for field in dataclasses.fields(Handles))
FLIGHT = NAMES[:14]
COMBAT = NAMES[14:]
ZERO_LIFT_DRAG_HANDLE = "zero_lift_drag"
# Measured by flying the game's flight tick; the rest follow from the card in closed form.
FLOWN = ("cruise_speed", "climb_time", "low_speed_acceleration", "sustained_turn_rate",
         "bleed_rate", "nose_reversal_time", "dive_acceleration", "zoom_acceleration")
DIRECT_FLIGHT = tuple(name for name in FLIGHT if name not in FLOWN)

DEGREES = 180.0 / math.pi
# (display unit, factor from SI) for the chart tool.
UNITS = dict(
    stall_speed=("m/s", 1.0), g_limit=("g", 1.0), cruise_speed=("m/s", 1.0),
    climb_time=("s", 1.0), ceiling=("J", 1.0), low_speed_acceleration=("m/s^2", 1.0),
    sustained_turn_rate=("deg/s", DEGREES), bleed_rate=("W/kg", 1.0),
    nose_reversal_time=("s", 1.0), dive_acceleration=("m/s^2", 1.0),
    zoom_acceleration=("m/s^2", 1.0), stall_angle=("deg", DEGREES), vision_range=("m", 1.0),
    zero_lift_drag=("1/m", 1.0), hit_damage=("", 1.0), fire_interval=("s", 1.0),
    muzzle_speed=("m/s", 1.0), bullet_spread=("m/s", 1.0), gun_range=("m", 1.0),
    shots_per_store=("", 1.0), max_shield=("", 1.0), shield_recharge_rate=("1/s", 1.0),
    shield_recharge_delay=("s", 1.0), head_on_fraction=("", 1.0), hit_radius=("m", 1.0),
    spawn_energy=("J", 1.0),
)


def stack(handles: list[Handles]) -> Handles:
    """Cards (num_types,) stacked into a batch (B, num_types)."""
    return Handles(**{name: torch.stack([getattr(h, name) for h in handles]) for name in NAMES})
