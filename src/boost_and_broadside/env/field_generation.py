"""On-reset generation of independent, arbitrarily overlapping fields."""

import math

import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.field_physics import material_tensors

# The four log-symmetric index levels are a constant. Building them from a
# Python list copies from the host and drains the CUDA queue, and layout
# generation runs on every environment reset -- which, across 1280 training and
# 2560 evaluation environments, is most physics ticks.
_INDEX_LEVEL_CACHE: dict[str, torch.Tensor] = {}


def _index_levels(device: torch.device) -> torch.Tensor:
    """The sampled log-index levels, cached per device."""

    key = str(device)
    levels = _INDEX_LEVEL_CACHE.get(key)
    if levels is None:
        levels = torch.tensor([-2, -1, 1, 2], device=device, dtype=torch.int8)
        _INDEX_LEVEL_CACHE[key] = levels
    return levels


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
        unit_x, unit_y = _low_discrepancy_toroid(
            batch_size,
            count,
            device,
            generator,
        )
        pos = torch.complex(unit_x * world_w, unit_y * world_h)
    else:
        if playable_radius is None:
            raise ValueError("playable_radius is required with map_center")
        outer = radius + 0.5 * width
        center_limit = (playable_radius.unsqueeze(1) - outer).clamp(min=0.0)
        radial_fraction, angle = _low_discrepancy_disk(
            batch_size,
            count,
            device,
            generator,
        )
        radial = center_limit * radial_fraction
        offset = torch.polar(radial, angle)
        translated = map_center.unsqueeze(1) + offset
        pos = torch.complex(translated.real % world_w, translated.imag % world_h)

    levels = _index_levels(device)
    level_draw = torch.randint(0, 4, shape, device=device, generator=generator)
    index_level = levels[level_draw]
    damage_level = torch.randint(0, 3, shape, device=device, generator=generator).to(torch.int8)
    index, damage = material_tensors(index_level, damage_level, ship_config)
    return pos, radius, width, index_level, index, damage_level, damage


def _low_discrepancy_disk(
    batch_size: int,
    count: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return randomized sunflower samples with one point per equal-area stratum."""

    shape = (batch_size, count)
    rank = torch.arange(count, dtype=torch.float32, device=device).unsqueeze(0)
    radial_jitter = torch.rand(shape, device=device, generator=generator)
    radial_fraction = ((rank + radial_jitter) / count).sqrt()

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rotation = torch.rand((batch_size, 1), device=device, generator=generator) * (2.0 * math.pi)
    angular_jitter = (torch.rand(shape, device=device, generator=generator) - 0.5) * (
        math.pi / count
    )
    angle = rotation + rank * golden_angle + angular_jitter
    return radial_fraction, angle


def _low_discrepancy_toroid(
    batch_size: int,
    count: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return randomized R2-sequence samples on the unit torus."""

    shape = (batch_size, count)
    rank = torch.arange(count, dtype=torch.float32, device=device).unsqueeze(0)
    plastic = 1.324717957244746
    phase = torch.rand((batch_size, 2), device=device, generator=generator)
    jitter_scale = 0.25 / math.sqrt(count)
    jitter = (torch.rand((*shape, 2), device=device, generator=generator) - 0.5) * jitter_scale
    unit_x = (phase[:, :1] + rank / plastic + jitter[:, :, 0]) % 1.0
    unit_y = (phase[:, 1:] + rank / (plastic * plastic) + jitter[:, :, 1]) % 1.0
    return unit_x, unit_y


def _uniform(
    shape: tuple[int, int],
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    draw = torch.rand(shape, device=device, generator=generator)
    return low + draw * (high - low)
