"""Similarity: the game in other units of length and time.

Every quantity of the game is a length to some power times a time to some power (per unit
mass: energies are J/kg, m^2/s^2; powers W/kg, m^2/s^3). A similarity multiplies every
length by `length` and every time by `time`; each quantity then scales by
length^a * time^b for its units (a, b). Flying a card scaled this way under rules scaled
this way is the same flight in the new units, up to the tick: the tick stays, so a scaled
game is the same flight sampled at another rate, and the quantities held to whole ticks
(fire rates, the bullet life, the shield recharge delay) are snapped by their callers.

Two uses: building a ship set in other units from the standard one (the real-scale set,
`bnb.game.real_scale`), and live tuning's time scale (length 1, time 1/k: the game k
times faster at fixed map lengths).

`UNITS` gives (a, b) by name: the handles, the card's fields, the flight constants and the
rule values (`Rules`' fields, and `ModeRules`' and the pools' by their `bnb.game.tuning`
key). The card's per-type scales are dimensionless, as the shared values they scale carry
the units; the induced drag factor k is the exception: k (C^2)^(e/2) is a drag per metre,
so k has units m^(e - 1), which depends on each type's bleed exponent e. The shared factor
scales as m^1 (e = 2) and each type's `induced_drag_scale` takes length^(e - 2).
"""

import dataclasses

import torch

from ..rules import FlightConstants
from ..ship_types import ShipTypes
from .handles import Handles

LENGTH = (1, 0)
TIME = (0, 1)
SPEED = (1, -1)
ACCELERATION = (1, -2)
RATE = (0, -1)
ENERGY = (2, -2)  # J/kg
POWER = (2, -3)  # W/kg
PER_LENGTH = (-1, 0)
NONE = (0, 0)

HANDLE_UNITS = dict(
    stall_speed=SPEED,
    g_limit=NONE,
    cruise_speed=SPEED,
    climb_time=TIME,
    ceiling=ENERGY,
    low_speed_acceleration=ACCELERATION,
    sustained_turn_rate=RATE,
    half_lift_drag=NONE,
    nose_reversal_time=TIME,
    dive_acceleration=ACCELERATION,
    zoom_acceleration=ACCELERATION,
    stall_angle=NONE,
    vision_range=LENGTH,
    zero_lift_drag=PER_LENGTH,
    hit_damage=NONE,
    explosive_share=NONE,
    fire_rate=RATE,
    muzzle_speed=SPEED,
    bullet_spread=SPEED,
    gun_range=LENGTH,
    bullet_drag=PER_LENGTH,
    shots_per_store=NONE,
    max_shield=NONE,
    shield_recharge_rate=RATE,
    shield_recharge_delay=TIME,
    head_on_fraction=NONE,
    head_on_angle=NONE,
    head_on_sharpness=NONE,
    arming_distance=LENGTH,
    hit_radius=LENGTH,
    spawn_energy=ENERGY,
)
CARD_UNITS = dict(
    max_lift_coefficient=PER_LENGTH,
    max_acceleration=ACCELERATION,
    zero_lift_drag=PER_LENGTH,
    engine_power=POWER,
    engine_power_full_store=POWER,
    static_thrust=ACCELERATION,  # N/kg
    max_energy=ENERGY,
    corner_nose_frequency=RATE,
    max_shield=NONE,
    hit_radius=LENGTH,
    hit_damage=NONE,
    muzzle_speed=SPEED,
    bullet_energy_cost=ENERGY,
    fire_rate=RATE,
    bullet_life=TIME,
    spawn_energy=ENERGY,
    explosive_share=NONE,
    reference_bullet_speed=SPEED,
    bleed_exponent=NONE,
    induced_drag_scale=NONE,  # times length^(e - 2): see the module
    stall_angle_scale=NONE,
    coast_exchange_scale=NONE,
    dive_exchange_scale=NONE,
    climb_exchange_scale=NONE,
    vision_range_scale=NONE,
    bullet_spread_scale=NONE,
    bullet_drag_scale=NONE,
    shield_recharge_rate_scale=NONE,
    shield_recharge_delay_scale=NONE,
    head_on_fraction_scale=NONE,
    head_on_angle_scale=NONE,
    head_on_sharpness=NONE,
    arming_distance=LENGTH,
    engine_knee_scale=NONE,
)
FLIGHT_UNITS = dict(
    mass=NONE,
    gravity=ACCELERATION,
    stall_angle=NONE,
    induced_drag_factor=LENGTH,  # m^(e - 1) at e = 2
    engine_knee=NONE,
    store_fade_band=NONE,
    coast_exchange=ACCELERATION,
    climb_exchange=ACCELERATION,
    dive_exchange=ACCELERATION,
    lift_softmin_exponent=NONE,
    nose_damping=NONE,
    nose_frequency_floor=RATE,
    speed_floor=SPEED,
    critical_speed=SPEED,
    drag_rise_exponent=NONE,
)
# Rule values by `bnb.game.tuning` key (and the Rules fields no setting holds).
RULE_UNITS = {
    "rules.bullet_drag": PER_LENGTH,
    "rules.damage_reference_distance": LENGTH,
    "rules.bullet_spread": SPEED,
    "rules.head_on_fraction": NONE,
    "rules.head_on_angle": NONE,
    "rules.shield_recharge_delay": TIME,
    "rules.shield_recharge_rate": RATE,
    "rules.boundary_damage_base": RATE,
    "rules.boundary_damage_per_metre": (-1, -1),
    "rules.spawn_speed": SPEED,
    "rules.spawn_offset": NONE,
    "rules.spawn_radius": LENGTH,
    "rules.spawn_turn_distance": LENGTH,
    "rules.field_band_width": LENGTH,
    "rules.field_radius_min": LENGTH,
    "rules.field_radius_max": LENGTH,
    "rules.field_radius_distribution": NONE,
    "rules.field_radius_median": LENGTH,
    "rules.field_radius_log_sigma": NONE,
    "rules.field_disc_fraction": NONE,
    "rules.field_high_share": NONE,
    "rules.field_strong_share": NONE,
    "rules.vision_range": LENGTH,
    "pool.arena_radius": LENGTH,
    "pool.episode_seconds": TIME,
    "pool.respawn_seconds": TIME,
    "mode.kills_per_ship": NONE,
    "mode.group_centre_fraction": NONE,
    "mode.group_radius": LENGTH,
    "mode.anchor_distance_min": LENGTH,
    "mode.anchor_distance_max": LENGTH,
    "mode.spawn_margin": LENGTH,
    "mode.zone_radius": LENGTH,
    "mode.zone_ring_fraction": NONE,
    "mode.capture_seconds": TIME,
    "mode.front_threshold": NONE,
    "mode.spawn_zone_fraction": NONE,
    "mode.lock_seconds_per_root_km": (-0.5, 1),  # s / sqrt(km)
    "mode.wave_seconds": TIME,
    "mode.side_area_fraction": NONE,
    "mode.side_heading_jitter": NONE,
    **{f"flight.{name}": units for name, units in FLIGHT_UNITS.items()},
}


def factor(units: tuple[int, int], length: float, time: float) -> float:
    """What a quantity of these units is multiplied by."""
    length_power, time_power = units
    return length**length_power * time**time_power


def scale_handles(handles: Handles, length: float, time: float) -> Handles:
    """Handles in the scaled units."""
    return Handles(
        **{
            name: getattr(handles, name) * factor(units, length, time)
            for name, units in HANDLE_UNITS.items()
        }
    )


def scale_card(types: ShipTypes, length: float, time: float) -> ShipTypes:
    """A card in the scaled units (its per-type induced drag scale times length^(e - 2))."""
    values = {
        name: getattr(types, name) * factor(units, length, time)
        for name, units in CARD_UNITS.items()
    }
    values["induced_drag_scale"] = types.induced_drag_scale * torch.pow(
        torch.as_tensor(length, dtype=types.bleed_exponent.dtype), types.bleed_exponent - 2.0
    )
    return ShipTypes(**values)


def scale_flight(flight: FlightConstants, length: float, time: float) -> FlightConstants:
    """The flight constants in the scaled units."""
    return dataclasses.replace(
        flight,
        **{
            name: getattr(flight, name) * factor(units, length, time)
            for name, units in FLIGHT_UNITS.items()
        },
    )


def scale_rules(values: dict[str, float], length: float, time: float) -> dict[str, float]:
    """Rule values by key (`RULE_UNITS`) in the scaled units; whole numbers stay."""
    return {key: value * factor(RULE_UNITS[key], length, time) for key, value in values.items()}
