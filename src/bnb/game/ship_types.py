"""The ship card: parameters that differ by ship type (game spec 1.1, ship-card spec 1), as
tensors.

Each field is a tensor with one entry per type, shape (num_types,), or (B, num_types) to
vary them per environment (a sweep). Code reads a ship's value with `per_ship`, which
fuses into the kernels that use it, so changing a value never recompiles.

The parameters that were shared before the ship card (`SCALED`) are stored as a per-type
scale of their shared value in `Rules` or `FlightConstants`: a type's stall angle is
`flight.stall_angle * stall_angle_scale`. The shared values keep their meaning for every
caller that reads them, and the default card (every scale 1) is exactly the shared game.
"""

import torch

from bnb.contracts.tree import contract

FIGHTER, HEAVY = 0, 1

# Game spec 1.1, by type: fighter, heavy (spec key in brackets).
DEFAULTS = dict(
    max_lift_coefficient=(0.027, 0.022),  # 1/m [cl_max]
    max_acceleration=(216.0, 160.0),  # m/s^2, the g limit [a_max]
    zero_lift_drag=(6.9612e-4, 8.5e-4),  # 1/m [cd0]
    engine_power=(1200.0, 1500.0),  # W, below the knee [power]
    engine_power_full_store=(863.2, 1100.0),  # W [p_min]
    static_thrust=(20.0, 24.0),  # N [f_static]
    max_energy=(10_000.0, 14_000.0),  # J [e_max]
    corner_nose_frequency=(22.0, 16.0),  # rad/s at corner speed [omega_c]
    max_shield=(100.0, 160.0),  # [shield_max]
    hit_radius=(8.0, 11.0),  # m
    hit_damage=(12.0, 16.0),  # [damage]
    muzzle_speed=(500.0, 450.0),  # m/s, proper
    bullet_energy_cost=(300.0, 360.0),  # J [bullet_cost]
    ticks_between_shots=(2, 3),  # [fire_interval]
    spawn_energy=(4000.0, 5600.0),  # J
    bleed_exponent=(2.0, 2.0),  # e: induced drag k (C^2 + (z/u^2)^2)^(e/2)
    # Per-type scales of shared values (SCALED), 1 by default.
    induced_drag_scale=(1.0, 1.0),  # of flight.induced_drag_factor [k]
    stall_angle_scale=(1.0, 1.0),  # of flight.stall_angle [alpha_s]
    coast_exchange_scale=(1.0, 1.0),  # of flight.coast_exchange [a_0]
    dive_exchange_scale=(1.0, 1.0),  # of flight.dive_exchange [a_d]
    climb_exchange_scale=(1.0, 1.0),  # of flight.climb_exchange [a_c]
    vision_range_scale=(1.0, 1.0),  # of rules.vision_range; the observer's [V]
    bullet_spread_scale=(1.0, 1.0),  # of rules.bullet_spread; the shooter's [sigma_b]
    bullet_drag_scale=(1.0, 1.0),  # of rules.bullet_drag; the shooter's [k_b]
    shield_recharge_rate_scale=(1.0, 1.0),  # of rules.shield_recharge_rate
    shield_recharge_delay_scale=(1.0, 1.0),  # of rules.shield_recharge_delay
    head_on_fraction_scale=(1.0, 1.0),  # of rules.head_on_fraction; the target's [f_0]
)
# Card field -> (owner in Rules, shared field) for the per-type scales.
SCALED = dict(
    induced_drag_scale=("flight", "induced_drag_factor"),
    stall_angle_scale=("flight", "stall_angle"),
    coast_exchange_scale=("flight", "coast_exchange"),
    dive_exchange_scale=("flight", "dive_exchange"),
    climb_exchange_scale=("flight", "climb_exchange"),
    vision_range_scale=("rules", "vision_range"),
    bullet_spread_scale=("rules", "bullet_spread"),
    bullet_drag_scale=("rules", "bullet_drag"),
    shield_recharge_rate_scale=("rules", "shield_recharge_rate"),
    shield_recharge_delay_scale=("rules", "shield_recharge_delay"),
    head_on_fraction_scale=("rules", "head_on_fraction"),
)


@contract
class ShipTypes:
    """Ship-card parameters by type; see DEFAULTS for units and SCALED for the scales.
    ticks_between_shots is int32, at the rules' tick."""

    max_lift_coefficient: torch.Tensor
    max_acceleration: torch.Tensor
    zero_lift_drag: torch.Tensor
    engine_power: torch.Tensor
    engine_power_full_store: torch.Tensor
    static_thrust: torch.Tensor
    max_energy: torch.Tensor
    corner_nose_frequency: torch.Tensor
    max_shield: torch.Tensor
    hit_radius: torch.Tensor
    hit_damage: torch.Tensor
    muzzle_speed: torch.Tensor
    bullet_energy_cost: torch.Tensor
    ticks_between_shots: torch.Tensor
    spawn_energy: torch.Tensor
    bleed_exponent: torch.Tensor
    induced_drag_scale: torch.Tensor
    stall_angle_scale: torch.Tensor
    coast_exchange_scale: torch.Tensor
    dive_exchange_scale: torch.Tensor
    climb_exchange_scale: torch.Tensor
    vision_range_scale: torch.Tensor
    bullet_spread_scale: torch.Tensor
    bullet_drag_scale: torch.Tensor
    shield_recharge_rate_scale: torch.Tensor
    shield_recharge_delay_scale: torch.Tensor
    head_on_fraction_scale: torch.Tensor


def ship_types(device=None, dtype=torch.float32, **overrides) -> ShipTypes:
    """The default card. An override is a per-type sequence or tensor ((num_types,) or
    (B, num_types)), or one number for every type. dtype is the floats' (offline tools use
    float64); ticks_between_shots is always int32."""
    values = {}
    for name, default in DEFAULTS.items():
        value = overrides.pop(name, default)
        kind = torch.int32 if name == "ticks_between_shots" else dtype
        value = torch.as_tensor(value, dtype=kind, device=device)
        values[name] = value.expand(len(default)).clone() if value.dim() == 0 else value
    if overrides:
        raise TypeError(f"unknown ship-card parameters {sorted(overrides)}")
    batched = [v.shape[0] for v in values.values() if v.dim() == 2]
    if batched:  # one card per environment: give every parameter the batch axis
        values = {k: v if v.dim() == 2 else v.expand(batched[0], -1).clone()
                  for k, v in values.items()}
    return ShipTypes(**values)


def per_ship(table: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
    """A card parameter for each ship, shaped like ship_type (B, ...).

    A select over the (few, statically known) types rather than a gather: it fuses into
    any kernel without an indirect load or bounds check, and needs no host copy, so it is
    also safe in eager CUDA-graph capture.
    """
    if table.dim() == 2:  # (B, num_types): each environment its own card
        table = table.reshape(table.shape[:1] + (1,) * (ship_type.dim() - 1) + table.shape[1:])
    out = table[..., 0].expand(ship_type.shape)
    for i in range(1, table.shape[-1]):
        out = torch.where(ship_type == i, table[..., i], out)
    return out
