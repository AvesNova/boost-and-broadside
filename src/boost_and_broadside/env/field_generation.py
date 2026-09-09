"""On-reset generation of independent, arbitrarily overlapping fields."""

import math

import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.field_physics import material_tensors


def generate_field_layout(
    batch_size: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    device: torch.device,
    *,
    map_center: torch.Tensor | None = None,
    playable_radius: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Generate one fresh field layout per environment.

    Combat maps sample centers uniformly over the toroid. Frontline maps pass a
    translated ``map_center`` and ``playable_radius``; centers are then sampled
    area-uniformly inside the practical battlefield, with each complete field
    kept inside its soft boundary. No pairwise placement test is needed because
    overlaps, coincident fields, and nesting are all valid.
    """

    count = env_config.num_fields
    shape = (batch_size, count)
    world_w, world_h = ship_config.world_size
    if count == 0:
        return (
            torch.empty(shape, dtype=torch.complex64, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.int8, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.int8, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
        )

    width = _uniform(
        shape,
        ship_config.field_transition_width_min,
        ship_config.field_transition_width_max,
        device,
        generator,
    )
    radius = _uniform(
        shape,
        ship_config.field_radius_min,
        ship_config.field_radius_max,
        device,
        generator,
    )
    if map_center is None:
        pos = torch.complex(
            torch.rand(shape, device=device, generator=generator) * world_w,
            torch.rand(shape, device=device, generator=generator) * world_h,
        )
    else:
        if playable_radius is None:
            raise ValueError("playable_radius is required with map_center")
        outer = radius + 0.5 * width
        center_limit = (playable_radius.unsqueeze(1) - outer).clamp(min=0.0)
        radial = center_limit * torch.rand(shape, device=device, generator=generator).sqrt()
        angle = torch.rand(shape, device=device, generator=generator) * (2.0 * math.pi)
        offset = torch.polar(radial, angle)
        translated = map_center.unsqueeze(1) + offset
        pos = torch.complex(translated.real % world_w, translated.imag % world_h)

    levels = torch.tensor([-2, -1, 1, 2], device=device, dtype=torch.int8)
    level_draw = torch.randint(0, 4, shape, device=device, generator=generator)
    index_level = levels[level_draw]
    damage_level = torch.randint(0, 3, shape, device=device, generator=generator).to(torch.int8)
    index, damage = material_tensors(index_level, damage_level, ship_config)
    return pos, radius, width, index_level, index, damage_level, damage


def _uniform(
    shape: tuple[int, int],
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    draw = torch.rand(shape, device=device, generator=generator)
    return low + draw * (high - low)
