"""The ship card: parameters that differ by ship type (game spec 1.1), as tensors.

Each field is a tensor with one entry per type, shape (num_types,), or (B, num_types) to
vary them per environment (a sweep). Code reads a ship's value with `per_ship`, a gather
that fuses into the kernels that use it, so changing a value never recompiles.
"""

import torch

from bnb.contracts.tree import contract

FIGHTER, HEAVY = 0, 1

# Game spec 1.1, by type: fighter, heavy.
DEFAULTS = dict(
    cl_max=(0.027, 0.022),  # maximum lift coefficient (1/m)
    a_max=(216.0, 160.0),  # g limit (m/s^2)
    cd0=(6.9612e-4, 8.5e-4),  # zero-lift drag (1/m)
    power=(1200.0, 1500.0),  # engine power below the knee (W)
    p_min=(863.2, 1100.0),  # engine power at a full store (W)
    f_static=(20.0, 24.0),  # static-thrust cap (N)
    e_max=(10_000.0, 14_000.0),  # energy store (J)
    omega_c=(22.0, 16.0),  # nose frequency at corner speed (rad/s)
    shield_max=(100.0, 160.0),
    hit_radius=(8.0, 11.0),  # m
    damage=(12.0, 16.0),  # per hit
    muzzle_speed=(500.0, 450.0),  # proper (m/s)
    bullet_cost=(300.0, 360.0),  # J
    fire_interval=(2, 3),  # ticks between shots
    spawn_energy=(4000.0, 5600.0),  # J
)


@contract
class ShipTypes:
    """Ship-card parameters by type; see DEFAULTS for units. fire_interval is int32."""

    cl_max: torch.Tensor
    a_max: torch.Tensor
    cd0: torch.Tensor
    power: torch.Tensor
    p_min: torch.Tensor
    f_static: torch.Tensor
    e_max: torch.Tensor
    omega_c: torch.Tensor
    shield_max: torch.Tensor
    hit_radius: torch.Tensor
    damage: torch.Tensor
    muzzle_speed: torch.Tensor
    bullet_cost: torch.Tensor
    fire_interval: torch.Tensor
    spawn_energy: torch.Tensor


def ship_types(device=None, **overrides) -> ShipTypes:
    """The default card. An override is a per-type sequence or tensor, or one number for
    every type."""
    values = {}
    for name, default in DEFAULTS.items():
        v = overrides.pop(name, default)
        dtype = torch.int32 if name == "fire_interval" else torch.float32
        v = torch.as_tensor(v, dtype=dtype, device=device)
        values[name] = v.expand(len(default)).clone() if v.dim() == 0 else v
    if overrides:
        raise TypeError(f"unknown ship-card parameters {sorted(overrides)}")
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

