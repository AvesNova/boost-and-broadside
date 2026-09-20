"""One-transfer immutable render packets for the experimental GPU frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from boost_and_broadside.env.perception import TeamVisibility
    from boost_and_broadside.env.state import TensorState


VisionPerspective = Literal["full", "team0", "team1"]


@dataclass(frozen=True, slots=True)
class SnapshotShip:
    previous_position: tuple[float, float]
    current_position: tuple[float, float]
    previous_heading: tuple[float, float]
    current_heading: tuple[float, float]
    team: int
    alive: bool
    health: float
    visibility_bits: int


@dataclass(frozen=True, slots=True)
class SnapshotProjectile:
    """A ring slot, retaining identity even while inactive."""

    owner_index: int
    slot_index: int
    previous_position: tuple[float, float]
    current_position: tuple[float, float]
    owner_team: int
    active: bool
    remaining_lifetime: float
    visibility_bits: int


@dataclass(frozen=True, slots=True)
class SnapshotCore:
    position: tuple[float, float]
    radius: float


@dataclass(frozen=True, slots=True)
class SnapshotZone:
    position: tuple[float, float]
    radius: float
    role: int
    capture_progress: float


@dataclass(frozen=True, slots=True)
class SnapshotFog:
    observer_positions: tuple[tuple[tuple[float, float], ...], ...]
    vision_range: float | None
    field_cores: tuple[SnapshotCore, ...]
    opaque_cores: tuple[SnapshotCore, ...]


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    world_size: tuple[float, float]
    step: int
    ships: tuple[SnapshotShip, ...]
    projectiles: tuple[SnapshotProjectile, ...]
    map_center: tuple[float, float]
    playable_boundary_radius: float
    zones: tuple[SnapshotZone, ...]
    fog: SnapshotFog

    def visible_to(self, perspective: VisionPerspective, visibility_bits: int) -> bool:
        return perspective == "full" or bool(
            visibility_bits & (1 << (0 if perspective == "team0" else 1))
        )


@dataclass(frozen=True, slots=True)
class PackedRenderSnapshot:
    """Renderer-fast CPU tensor packet, owned independently of live state."""

    packed: torch.Tensor
    world_size: tuple[float, float]
    vision_range: float | None
    num_ships: int
    max_bullets: int
    num_zones: int
    num_fields: int
    zones_occlude: bool

    @property
    def step(self) -> int:
        return int(self.packed[0])

    def ship_rows(self) -> torch.Tensor:
        return self.packed[1 : 1 + self.num_ships * 9].view(self.num_ships, 9)

    def bullet_rows(self) -> torch.Tensor:
        start = 1 + self.num_ships * 9
        return self.packed[start : start + self.num_ships * self.max_bullets * 6].view(
            self.num_ships * self.max_bullets, 6
        )

    def map_header(self) -> torch.Tensor:
        start = 1 + self.num_ships * 9 + self.num_ships * self.max_bullets * 6
        return self.packed[start : start + 3]

    def zone_rows(self) -> torch.Tensor:
        start = 1 + self.num_ships * 9 + self.num_ships * self.max_bullets * 6 + 3
        return self.packed[start : start + self.num_zones * 5].view(self.num_zones, 5)

    def field_rows(self) -> torch.Tensor:
        start = 1 + self.num_ships * 9 + self.num_ships * self.max_bullets * 6 + 3
        start += self.num_zones * 5
        return self.packed[start : start + self.num_fields * 3].view(self.num_fields, 3)

    def map_instances(self) -> torch.Tensor:
        """Return contiguous GPU map rows: ``x, y, radius, kind, role``.

        This is intentionally a tensor operation: the packed renderer writes
        its buffer directly from the returned CPU ndarray, rather than making
        one Python object for every map primitive.
        """

        fields = self.field_rows()
        zones = self.zone_rows()
        field_rows = torch.cat(
            (fields, torch.zeros((self.num_fields, 2), dtype=fields.dtype)), dim=1
        )
        zone_rows = torch.cat(
            (
                zones[:, :3],
                torch.ones((self.num_zones, 1), dtype=zones.dtype),
                zones[:, 3:4],
            ),
            dim=1,
        )
        header = self.map_header()
        boundary = torch.stack(
            (header[0], header[1], header[2], header.new_tensor(2.0), header.new_tensor(0.0))
        ).view(1, 5)
        return torch.cat((field_rows, zone_rows, boundary), dim=0).contiguous()

    def fog_rows(self, perspective: VisionPerspective) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return active observer positions and opaque cores for a team view."""

        if perspective == "full":
            return None
        team = 0 if perspective == "team0" else 1
        ships = self.ship_rows()
        observers = ships[(ships[:, 6].bool()) & (ships[:, 5] == team), :2][:50]
        cores = self.field_rows()
        if self.zones_occlude:
            cores = torch.cat((cores, self.zone_rows()[:, :3]), dim=0)
        return observers.contiguous(), cores[:128].contiguous()

    def ship_instances(self, previous: PackedRenderSnapshot | None) -> torch.Tensor:
        current = self.ship_rows()
        old = current if previous is None else previous.ship_rows()
        result = torch.stack(
            (
                old[:, 0],
                old[:, 1],
                current[:, 0],
                current[:, 1],
                current[:, 5],
                current[:, 7] + 2 * current[:, 8],
            ),
            dim=1,
        )
        return result[current[:, 6].bool()].contiguous()

    def projectile_instances(self, previous: PackedRenderSnapshot | None) -> torch.Tensor:
        current = self.bullet_rows()
        old = current if previous is None else previous.bullet_rows()
        reused = ~old[:, 2].bool() | (current[:, 3] > old[:, 3])
        previous_xy = torch.where(reused[:, None], current[:, :2], old[:, :2])
        owner_team = self.ship_rows()[:, 5].repeat_interleave(self.max_bullets)
        result = torch.cat(
            (
                previous_xy,
                current[:, :2],
                owner_team[:, None],
                (current[:, 4] + 2 * current[:, 5])[:, None],
            ),
            dim=1,
        )
        return result[current[:, 2].bool()].contiguous()


def make_packed_render_snapshot(
    current: TensorState,
    *,
    world_size: tuple[float, float],
    visibility: TeamVisibility | None = None,
    env_index: int = 0,
    zones_occlude: bool = False,
) -> PackedRenderSnapshot:
    """One D2H transfer without ``tolist`` or per-entity Python objects."""

    if not 0 <= env_index < current.num_envs:
        raise IndexError(f"environment index {env_index} is outside 0..{current.num_envs - 1}")
    packed = (
        _packed_current(current, visibility, env_index)
        .to(dtype=torch.float32)
        .detach()
        .cpu()
        .contiguous()
    )
    return PackedRenderSnapshot(
        packed,
        tuple(map(float, world_size)),
        None if visibility is None else visibility.vision_range,
        current.ship_pos.shape[1],
        current.bullet_pos.shape[2],
        current.zone_pos.shape[1],
        current.field_pos.shape[1],
        zones_occlude,
    )


def _visibility_columns(
    current: TensorState, visibility: TeamVisibility | None, env_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float GPU columns shaped ``(N, 2)`` and ``(N*K, 2)``."""

    ships, bullets = current.ship_pos.shape[1], current.bullet_pos.shape[2]
    if visibility is None:
        ship = current.ship_alive[env_index].float().unsqueeze(1).expand(ships, 2)
        bullet = current.bullet_active[env_index].float().reshape(-1, 1).expand(ships * bullets, 2)
        return ship, bullet
    if visibility.bullet is None:
        raise ValueError(
            "projectile rendering requires an authoritative projectile visibility mask"
        )
    return (
        visibility.ship[env_index].transpose(0, 1).float(),
        visibility.bullet[env_index].permute(1, 2, 0).reshape(ships * bullets, 2).float(),
    )


def _packed_current(
    current: TensorState, visibility: TeamVisibility | None, env_index: int
) -> torch.Tensor:
    """Build the sole source-device transfer buffer (all columns are float)."""

    ship_masks, bullet_masks = _visibility_columns(current, visibility, env_index)
    ships = torch.stack(
        (
            current.ship_pos[env_index].real,
            current.ship_pos[env_index].imag,
            current.ship_attitude[env_index].real,
            current.ship_attitude[env_index].imag,
            current.ship_health[env_index],
            current.ship_team_id[env_index].float(),
            current.ship_alive[env_index].float(),
            ship_masks[:, 0],
            ship_masks[:, 1],
        ),
        dim=1,
    ).flatten()
    bullets = torch.stack(
        (
            current.bullet_pos[env_index].real.flatten(),
            current.bullet_pos[env_index].imag.flatten(),
            current.bullet_active[env_index].float().flatten(),
            current.bullet_time[env_index].flatten(),
            bullet_masks[:, 0],
            bullet_masks[:, 1],
        ),
        dim=1,
    ).flatten()
    map_data = torch.cat(
        (
            torch.stack(
                (
                    current.map_center[env_index].real,
                    current.map_center[env_index].imag,
                    current.playable_boundary_radius[env_index],
                )
            ),
            torch.stack(
                (
                    current.zone_pos[env_index].real,
                    current.zone_pos[env_index].imag,
                    current.zone_radius[env_index],
                    current.zone_roles[env_index].float(),
                    current.zone_capture_progress[env_index],
                ),
                dim=1,
            ).flatten(),
            torch.stack(
                (
                    current.field_pos[env_index].real,
                    current.field_pos[env_index].imag,
                    (
                        current.field_radius[env_index]
                        - 0.5 * current.field_transition_width[env_index]
                    ).clamp(min=0),
                ),
                dim=1,
            ).flatten(),
        )
    )
    return torch.cat(
        (current.step_count[env_index : env_index + 1].float(), ships, bullets, map_data)
    )


def make_render_snapshot(
    current: TensorState,
    *,
    world_size: tuple[float, float],
    previous: RenderSnapshot | None = None,
    visibility: TeamVisibility | None = None,
    env_index: int = 0,
    zones_occlude: bool = False,
) -> RenderSnapshot:
    """Make a snapshot with exactly one ``detach().cpu()`` host transfer.

    ``previous`` is the prior immutable render snapshot.  This makes the
    production path independent of an old TensorState and avoids a second GPU
    synchronization.  The first frame snaps every transform to current.
    """

    if not 0 <= env_index < current.num_envs:
        raise IndexError(f"environment index {env_index} is outside 0..{current.num_envs - 1}")
    packed = _packed_current(current, visibility, env_index).detach().cpu().tolist()
    ships_count, slots, zones = (
        current.ship_pos.shape[1],
        current.bullet_pos.shape[2],
        current.zone_pos.shape[1],
    )
    cursor = 0
    step = int(packed[cursor])
    cursor += 1
    old_ships = previous.ships if previous is not None else ()
    ship_values, cursor = packed[cursor : cursor + ships_count * 9], cursor + ships_count * 9
    ships = []
    for index in range(ships_count):
        row = ship_values[index * 9 : (index + 1) * 9]
        old = old_ships[index] if index < len(old_ships) else None
        current_position, current_heading = (
            (float(row[0]), float(row[1])),
            (float(row[2]), float(row[3])),
        )
        ships.append(
            SnapshotShip(
                old.current_position if old else current_position,
                current_position,
                old.current_heading if old else current_heading,
                current_heading,
                int(row[5]),
                bool(row[6]),
                float(row[4]),
                int(row[7]) | (int(row[8]) << 1),
            )
        )
    old_slots = (
        {(item.owner_index, item.slot_index): item for item in previous.projectiles}
        if previous
        else {}
    )
    bullet_values, cursor = (
        packed[cursor : cursor + ships_count * slots * 6],
        cursor + ships_count * slots * 6,
    )
    projectiles = []
    for index in range(ships_count * slots):
        row = bullet_values[index * 6 : (index + 1) * 6]
        owner, slot = divmod(index, slots)
        old = old_slots.get((owner, slot))
        current_position, active, lifetime = (
            (float(row[0]), float(row[1])),
            bool(row[2]),
            float(row[3]),
        )
        reused = old is None or not old.active or lifetime > old.remaining_lifetime
        projectiles.append(
            SnapshotProjectile(
                owner,
                slot,
                current_position if reused else old.current_position,
                current_position,
                ships[owner].team,
                active,
                lifetime,
                int(row[4]) | (int(row[5]) << 1),
            )
        )
    center, boundary = (float(packed[cursor]), float(packed[cursor + 1])), float(packed[cursor + 2])
    cursor += 3
    zone_values, cursor = packed[cursor : cursor + zones * 5], cursor + zones * 5
    zone_rows = tuple(
        SnapshotZone(
            (float(zone_values[i]), float(zone_values[i + 1])),
            float(zone_values[i + 2]),
            int(zone_values[i + 3]),
            float(zone_values[i + 4]),
        )
        for i in range(0, len(zone_values), 5)
    )
    field_values = packed[cursor:]
    fields = tuple(
        SnapshotCore(
            (float(field_values[i]), float(field_values[i + 1])), float(field_values[i + 2])
        )
        for i in range(0, len(field_values), 3)
    )
    opaque = fields + (
        tuple(SnapshotCore(zone.position, zone.radius) for zone in zone_rows)
        if zones_occlude
        else ()
    )
    observers = tuple(
        tuple(ship.current_position for ship in ships if ship.alive and ship.team == team)
        for team in range(2)
    )
    return RenderSnapshot(
        tuple(map(float, world_size)),
        step,
        tuple(ships),
        tuple(projectiles),
        center,
        boundary,
        zone_rows,
        SnapshotFog(
            observers, None if visibility is None else visibility.vision_range, fields, opaque
        ),
    )
