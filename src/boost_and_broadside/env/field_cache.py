"""GPU-resident cache of valid static refractive-field maps."""

import math

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.field_physics import material_tensors, validate_field_layout


class FieldMapCache:
    """Fixed-shape bank of static laminar field maps.

    Maps are generated with bounded construction-time rejection sampling. Episode
    resets only gather a cached row and apply a shared toroidal translation, so
    parent relationships and all material tensors remain valid without host sync.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        radius: torch.Tensor,
        transition_width: torch.Tensor,
        index_level: torch.Tensor,
        damage_level: torch.Tensor,
        ship_config: ShipConfig,
    ) -> None:
        if pos.ndim != 2:
            raise ValueError("field cache tensors must have shape (cache_size, num_fields)")
        parent = validate_field_layout(
            pos,
            radius,
            transition_width,
            index_level,
            damage_level,
            ship_config.world_size,
        )
        index, damage, delta_index = material_tensors(
            index_level, damage_level, parent, ship_config
        )
        self._pos = pos.to(torch.complex64)
        self._radius = radius.to(torch.float32)
        self._transition_width = transition_width.to(torch.float32)
        self._index_level = index_level.to(torch.int8)
        self._index = index.to(torch.float32)
        self._damage_level = damage_level.to(torch.int8)
        self._damage = damage.to(torch.float32)
        self._parent = parent.to(torch.long)
        self._delta_index = delta_index.to(torch.float32)

    def __len__(self) -> int:
        return self._pos.shape[0]

    @property
    def num_fields(self) -> int:
        return self._pos.shape[1]

    @staticmethod
    def generate(
        ship_config: ShipConfig,
        env_config: EnvConfig,
        map_config: FieldMapConfig,
        device: torch.device,
        seed: int | None = None,
    ) -> "FieldMapCache":
        """Generate ``cache_size`` valid maps, failing after bounded attempts."""

        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        cache_pos: list[list[complex]] = []
        cache_radius: list[list[float]] = []
        cache_width: list[list[float]] = []
        cache_index_level: list[list[int]] = []
        cache_damage_level: list[list[int]] = []

        for map_idx in range(map_config.cache_size):
            generated = _generate_one_map(
                env_config.num_fields,
                ship_config,
                map_config,
                generator,
            )
            if generated is None:
                raise RuntimeError(
                    "Field map generation failed for map "
                    f"{map_idx + 1}/{map_config.cache_size} after "
                    f"{map_config.max_generation_attempts} attempts per field; "
                    "reduce num_fields/radii/widths or increase max_generation_attempts"
                )
            pos, radius, width, index_level, damage_level = generated
            cache_pos.append(pos)
            cache_radius.append(radius)
            cache_width.append(width)
            cache_index_level.append(index_level)
            cache_damage_level.append(damage_level)

        shape = (map_config.cache_size, env_config.num_fields)
        if env_config.num_fields == 0:
            pos_t = torch.empty(shape, dtype=torch.complex64, device=device)
            radius_t = torch.empty(shape, dtype=torch.float32, device=device)
            width_t = torch.empty(shape, dtype=torch.float32, device=device)
            index_t = torch.empty(shape, dtype=torch.int8, device=device)
            damage_t = torch.empty(shape, dtype=torch.int8, device=device)
        else:
            pos_t = torch.tensor(cache_pos, dtype=torch.complex64, device=device)
            radius_t = torch.tensor(cache_radius, dtype=torch.float32, device=device)
            width_t = torch.tensor(cache_width, dtype=torch.float32, device=device)
            index_t = torch.tensor(cache_index_level, dtype=torch.int8, device=device)
            damage_t = torch.tensor(cache_damage_level, dtype=torch.int8, device=device)
        return FieldMapCache(pos_t, radius_t, width_t, index_t, damage_t, ship_config)

    def sample(
        self,
        batch_size: int,
        world_size: tuple[float, float],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Gather maps and apply a shared per-environment toroidal translation."""

        world_w, world_h = world_size
        index = torch.randint(0, len(self), (batch_size,), device=device)
        pos = self._pos[index].to(device)

        offset_x = torch.rand(batch_size, device=device) * world_w
        offset_y = torch.rand(batch_size, device=device) * world_h
        pos = torch.complex(
            (pos.real + offset_x.unsqueeze(1)) % world_w,
            (pos.imag + offset_y.unsqueeze(1)) % world_h,
        )
        return (
            pos,
            self._radius[index].to(device),
            self._transition_width[index].to(device),
            self._index_level[index].to(device),
            self._index[index].to(device),
            self._damage_level[index].to(device),
            self._damage[index].to(device),
            self._parent[index].to(device),
            self._delta_index[index].to(device),
        )


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(low + torch.rand((), generator=generator).item() * (high - low))


def _rand_index(generator: torch.Generator, size: int) -> int:
    return int(torch.randint(size, (), generator=generator).item())


def _minimum_image_distance(a: complex, b: complex, world_size: tuple[float, float]) -> float:
    world_w, world_h = world_size
    dx = (a.real - b.real + world_w / 2.0) % world_w - world_w / 2.0
    dy = (a.imag - b.imag + world_h / 2.0) % world_h - world_h / 2.0
    return math.hypot(dx, dy)


def _pair_is_laminar(
    pos_a: complex,
    radius_a: float,
    width_a: float,
    pos_b: complex,
    radius_b: float,
    width_b: float,
    world_size: tuple[float, float],
) -> bool:
    distance = _minimum_image_distance(pos_a, pos_b, world_size)
    outer_a = radius_a + width_a / 2.0
    outer_b = radius_b + width_b / 2.0
    core_a = radius_a - width_a / 2.0
    core_b = radius_b - width_b / 2.0
    return (
        distance >= outer_a + outer_b
        or distance + outer_b <= core_a
        or distance + outer_a <= core_b
    )


def _generate_one_map(
    num_fields: int,
    ship_config: ShipConfig,
    map_config: FieldMapConfig,
    generator: torch.Generator,
) -> tuple[list[complex], list[float], list[float], list[int], list[int]] | None:
    """Generate one map sequentially; every field has a bounded attempt budget."""

    world_w, world_h = ship_config.world_size
    positions: list[complex] = []
    radii: list[float] = []
    widths: list[float] = []
    index_levels: list[int] = []
    damage_levels: list[int] = []

    for _ in range(num_fields):
        accepted = False
        for _attempt in range(map_config.max_generation_attempts):
            width = _uniform(
                generator,
                ship_config.field_transition_width_min,
                ship_config.field_transition_width_max,
            )
            radius = _uniform(
                generator,
                ship_config.field_radius_min,
                ship_config.field_radius_max,
            )

            parent_candidates = [
                i
                for i, (parent_radius, parent_width) in enumerate(zip(radii, widths, strict=True))
                if parent_radius - parent_width / 2.0 >= ship_config.field_radius_min + width / 2.0
            ]
            choose_nested = (
                bool(parent_candidates)
                and torch.rand((), generator=generator).item() < map_config.nesting_probability
            )
            if choose_nested:
                parent = parent_candidates[_rand_index(generator, len(parent_candidates))]
                max_radius = min(
                    ship_config.field_radius_max,
                    radii[parent] - widths[parent] / 2.0 - width / 2.0,
                )
                radius = _uniform(generator, ship_config.field_radius_min, max_radius)
                max_offset = radii[parent] - widths[parent] / 2.0 - radius - width / 2.0
                offset_radius = math.sqrt(torch.rand((), generator=generator).item()) * max_offset
                angle = torch.rand((), generator=generator).item() * 2.0 * math.pi
                candidate_pos = complex(
                    (positions[parent].real + offset_radius * math.cos(angle)) % world_w,
                    (positions[parent].imag + offset_radius * math.sin(angle)) % world_h,
                )
            else:
                candidate_pos = complex(
                    torch.rand((), generator=generator).item() * world_w,
                    torch.rand((), generator=generator).item() * world_h,
                )

            if all(
                _pair_is_laminar(
                    candidate_pos,
                    radius,
                    width,
                    old_pos,
                    old_radius,
                    old_width,
                    ship_config.world_size,
                )
                for old_pos, old_radius, old_width in zip(positions, radii, widths, strict=True)
            ):
                positions.append(candidate_pos)
                radii.append(radius)
                widths.append(width)
                index_levels.append((-2, -1, 1, 2)[_rand_index(generator, 4)])
                damage_levels.append(_rand_index(generator, 3))
                accepted = True
                break

        if not accepted:
            return None

    return positions, radii, widths, index_levels, damage_levels
