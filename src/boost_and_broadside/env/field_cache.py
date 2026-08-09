"""GPU-resident cache of valid static refractive-field maps."""

import math

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.field_physics import material_tensors, validate_field_layout


class FieldMapCache:
    """Fixed-shape bank of static laminar field maps.

    Maps are generated on device, valid by construction: candidate placements are
    rejected against already-placed fields before they are accepted, so no map
    ever has to be validated on the hot path. Episode resets gather a cached row
    and apply a shared toroidal translation, so parent relationships and all
    material tensors remain valid without host sync.

    ``refresh`` regenerates the whole bank in place. Training calls it once per
    rollout: a fixed bank is a small map distribution that a long run sees
    thousands of times over, whereas a bank replaced every rollout supplies
    roughly one distinct map per episode for the cost of a few hundred
    microseconds.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        radius: torch.Tensor,
        transition_width: torch.Tensor,
        index_level: torch.Tensor,
        damage_level: torch.Tensor,
        ship_config: ShipConfig,
        validate: bool = True,
    ) -> None:
        """Build a cache from field layout tensors.

        Args:
            validate: Run the full laminar-geometry check. It costs several
                device-to-host syncs and raises on failure, so the generator —
                whose output is valid by construction — skips it and reports a
                device-side failure count instead. Externally supplied layouts
                (tests, tooling) keep the loud check.
        """
        if pos.ndim != 2:
            raise ValueError("field cache tensors must have shape (cache_size, num_fields)")
        if validate:
            parent = validate_field_layout(
                pos,
                radius,
                transition_width,
                index_level,
                damage_level,
                ship_config.world_size,
            )
        else:
            parent = field_parents_only(pos, radius, transition_width, ship_config.world_size)
        index, damage, delta_index = material_tensors(
            index_level, damage_level, parent, ship_config
        )
        self._ship_config = ship_config
        self._pos = pos.to(torch.complex64)
        self._radius = radius.to(torch.float32)
        self._transition_width = transition_width.to(torch.float32)
        self._index_level = index_level.to(torch.int8)
        self._index = index.to(torch.float32)
        self._damage_level = damage_level.to(torch.int8)
        self._damage = damage.to(torch.float32)
        self._parent = parent.to(torch.long)
        self._delta_index = delta_index.to(torch.float32)
        # Maps the generator could not place within its proposal budget, kept on
        # device so reading it never forces a sync. Those rows retain whatever
        # map they previously held; a sustained non-zero reading means the
        # radius/width/count combination is too tight to place reliably.
        self.generation_failures = torch.zeros((), device=self._pos.device, dtype=torch.float32)

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
        """Generate ``cache_size`` valid maps, failing loudly on a bad config.

        The initial generation validates: a radius/width/field-count combination
        that cannot be placed is a configuration error and should stop the run
        rather than quietly thin the map distribution. ``refresh`` afterwards
        skips validation, having already proved the config workable.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        layout, valid = _generate_batch(
            map_config.cache_size, ship_config, env_config, map_config, device, generator
        )
        if env_config.num_fields > 0 and not bool(valid.all().item()):
            failed = int((~valid).sum().item())
            raise RuntimeError(
                f"Field map generation failed for {failed}/{map_config.cache_size} maps "
                f"after {map_config.max_generation_attempts} attempts per field; reduce "
                "num_fields/radii/widths, or raise max_generation_attempts"
            )
        cache = FieldMapCache(*layout, ship_config, validate=True)
        cache._map_config = map_config
        cache._env_config = env_config
        return cache

    def refresh(self, generator: torch.Generator | None = None) -> None:
        """Regenerate the whole bank in place, on device, without syncing.

        Maps that fail to place keep their previous row rather than raising —
        a mid-run exception would lose the run over a transient sampling miss,
        and ``generation_failures`` makes any sustained problem visible in the
        per-update metrics instead.
        """
        if self.num_fields == 0:
            return
        layout, valid = _generate_batch(
            len(self),
            self._ship_config,
            self._env_config,
            self._map_config,
            self._pos.device,
            generator,
        )
        pos, radius, width, index_level, damage_level = layout
        keep = valid.unsqueeze(1)
        pos = torch.where(keep, pos, self._pos)
        radius = torch.where(keep, radius, self._radius)
        width = torch.where(keep, width, self._transition_width)
        index_level = torch.where(keep, index_level, self._index_level)
        damage_level = torch.where(keep, damage_level, self._damage_level)

        parent = field_parents_only(pos, radius, width, self._ship_config.world_size)
        index, damage, delta_index = material_tensors(
            index_level, damage_level, parent, self._ship_config
        )
        self._pos = pos.to(torch.complex64)
        self._radius = radius.to(torch.float32)
        self._transition_width = width.to(torch.float32)
        self._index_level = index_level.to(torch.int8)
        self._index = index.to(torch.float32)
        self._damage_level = damage_level.to(torch.int8)
        self._damage = damage.to(torch.float32)
        self._parent = parent.to(torch.long)
        self._delta_index = delta_index.to(torch.float32)
        self.generation_failures = (~valid).float().sum()

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


def field_parents_only(
    centers: torch.Tensor,
    radii: torch.Tensor,
    transition_widths: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Direct parents for an already-laminar layout, without the validity check.

    ``validate_field_layout`` returns the same thing but spends eight
    device-to-host syncs proving the layout legal first. Generated maps are legal
    by construction, so the hot path takes this instead.
    """
    from boost_and_broadside.env.field_physics import field_parents

    return field_parents(centers, radii, transition_widths, world_size)


def _wrap(delta: torch.Tensor, world_size: tuple[float, float]) -> torch.Tensor:
    world_w, world_h = world_size
    return torch.complex(
        (delta.real + world_w / 2.0) % world_w - world_w / 2.0,
        (delta.imag + world_h / 2.0) % world_h - world_h / 2.0,
    )


def _rand(shape, device, generator, low=0.0, high=1.0) -> torch.Tensor:
    values = torch.rand(shape, device=device, generator=generator)
    return values if (low, high) == (0.0, 1.0) else low + values * (high - low)


def _generate_batch(
    count: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    map_config: FieldMapConfig,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Place ``count`` laminar maps at once, entirely on device.

    Loops over fields, not over maps or over retries: each field proposes
    ``max_generation_attempts`` candidate placements for every map simultaneously,
    checks them against the fields already placed in that map, and takes the
    first that fits. Cost is ``num_fields`` iterations of fixed-shape tensor
    work, with no data-dependent control flow and so no host synchronisation.
    ``FieldMapConfig.max_generation_attempts`` sets the proposal count.

    Returns:
        ((pos, radius, width, index_level, damage_level), valid) where ``valid``
        is (count,) bool — False for maps that ran out of proposals for some
        field. Callers decide whether that is fatal.
    """
    num_fields = env_config.num_fields
    world_w, world_h = ship_config.world_size
    shape = (count, num_fields)
    pos = torch.zeros(shape, dtype=torch.complex64, device=device)
    radius = torch.zeros(shape, dtype=torch.float32, device=device)
    width = torch.zeros(shape, dtype=torch.float32, device=device)
    index_level = torch.zeros(shape, dtype=torch.int8, device=device)
    damage_level = torch.zeros(shape, dtype=torch.int8, device=device)
    placed = torch.zeros(shape, dtype=torch.bool, device=device)
    if num_fields == 0:
        return (pos, radius, width, index_level, damage_level), torch.ones(
            count, dtype=torch.bool, device=device
        )

    # Attempts become a proposal axis rather than a retry loop: every map
    # evaluates all of them at once and takes the first that fits.
    proposals = map_config.max_generation_attempts
    for field in range(num_fields):
        candidate_width = _rand(
            (count, proposals),
            device,
            generator,
            ship_config.field_transition_width_min,
            ship_config.field_transition_width_max,
        )
        candidate_radius = _rand(
            (count, proposals),
            device,
            generator,
            ship_config.field_radius_min,
            ship_config.field_radius_max,
        )
        free_pos = torch.complex(
            _rand((count, proposals), device, generator) * world_w,
            _rand((count, proposals), device, generator) * world_h,
        )

        if field == 0:
            candidate_pos = free_pos
            fits = torch.ones((count, proposals), dtype=torch.bool, device=device)
        else:
            prior_pos = pos[:, :field].unsqueeze(1)  # (count, 1, field)
            prior_radius = radius[:, :field].unsqueeze(1)
            prior_width = width[:, :field].unsqueeze(1)
            prior_placed = placed[:, :field].unsqueeze(1)

            # A field can nest inside a prior one only if that one's flat core
            # has room for the candidate's whole transition band.
            room = prior_placed & (
                prior_radius - 0.5 * prior_width
                >= ship_config.field_radius_min + 0.5 * candidate_width.unsqueeze(2)
            )  # (count, proposals, field)
            has_parent = room.any(dim=2)
            # Uniform choice among the fields with room: randomise then take the
            # best-scoring available one.
            parent_index = (
                _rand((count, proposals, field), device, generator) * room
            ).argmax(dim=2, keepdim=True)
            parent_pos = prior_pos.expand(-1, proposals, -1).gather(2, parent_index).squeeze(2)
            parent_radius = (
                prior_radius.expand(-1, proposals, -1).gather(2, parent_index).squeeze(2)
            )
            parent_width = prior_width.expand(-1, proposals, -1).gather(2, parent_index).squeeze(2)

            inner_limit = parent_radius - 0.5 * parent_width - 0.5 * candidate_width
            nested_radius = _rand((count, proposals), device, generator) * (
                inner_limit.clamp(max=ship_config.field_radius_max)
                - ship_config.field_radius_min
            ).clamp(min=0.0) + ship_config.field_radius_min
            max_offset = (inner_limit - nested_radius).clamp(min=0.0)
            # sqrt of a uniform draw gives an area-uniform radial offset.
            offset = max_offset * _rand((count, proposals), device, generator).sqrt()
            angle = _rand((count, proposals), device, generator) * (2.0 * math.pi)
            nested_pos = torch.complex(
                (parent_pos.real + offset * angle.cos()) % world_w,
                (parent_pos.imag + offset * angle.sin()) % world_h,
            )

            nest = has_parent & (
                _rand((count, proposals), device, generator) < map_config.nesting_probability
            )
            candidate_pos = torch.where(nest, nested_pos, free_pos)
            candidate_radius = torch.where(nest, nested_radius, candidate_radius)

            # Laminar against every field already placed in the same map: either
            # disjoint including both transition bands, or strictly nested.
            distance = _wrap(
                candidate_pos.unsqueeze(2) - prior_pos, ship_config.world_size
            ).abs()  # (count, proposals, field)
            outer = (candidate_radius + 0.5 * candidate_width).unsqueeze(2)
            core = (candidate_radius - 0.5 * candidate_width).unsqueeze(2)
            prior_outer = prior_radius + 0.5 * prior_width
            prior_core = prior_radius - 0.5 * prior_width
            laminar = (
                (distance >= outer + prior_outer)
                | (distance + prior_outer <= core)
                | (distance + outer <= prior_core)
            )
            fits = (laminar | ~prior_placed).all(dim=2)

        # First proposal that fits; argmax on a bool returns the first True.
        chosen = fits.float().argmax(dim=1, keepdim=True)  # (count, 1)
        accepted = fits.any(dim=1)
        pos[:, field] = candidate_pos.gather(1, chosen).squeeze(1)
        radius[:, field] = candidate_radius.gather(1, chosen).squeeze(1)
        width[:, field] = candidate_width.gather(1, chosen).squeeze(1)
        # Ambient (level 0) is not a legal field material, hence {-2,-1,1,2}.
        levels = torch.tensor([-2, -1, 1, 2], device=device, dtype=torch.int8)
        draw = torch.randint(0, 4, (count,), device=device, generator=generator)
        index_level[:, field] = levels[draw]
        damage_level[:, field] = torch.randint(
            0, 3, (count,), device=device, generator=generator
        ).to(torch.int8)
        placed[:, field] = accepted

    valid = placed.all(dim=1)
    # A failed map still has to hold a legal layout for the material tensors, so
    # collapse its fields onto the first placed one (self-nesting is laminar).
    fallback = pos[:, :1].expand(-1, num_fields)
    pos = torch.where(placed, pos, fallback)
    radius = torch.where(placed, radius, radius[:, :1].expand(-1, num_fields))
    width = torch.where(placed, width, width[:, :1].expand(-1, num_fields))
    return (pos, radius, width, index_level, damage_level), valid
