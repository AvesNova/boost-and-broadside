from collections.abc import ItemsView
from dataclasses import dataclass
from enum import StrEnum

import torch

from boost_and_broadside.config.core import ShipConfig
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.state import TensorState


class ObsKey(StrEnum):
    POS = "pos"
    VEL = "vel"
    ATT = "att"
    ANG_VEL = "ang_vel"
    HEALTH = "health"
    POWER = "power"
    COOLDOWN = "cooldown"
    TEAM_ID = "team_id"
    ALIVE = "alive"
    RADIUS = "radius"
    PREVIOUS_ACTION = "previous_action"


@dataclass(frozen=True)
class MVPObservation:
    """Typed immutable observation for all entities.

    data: maps ObsKey → tensor of shape (B, N+M, ...) or (T, B, N+M, ...) etc.

    team_id:  (B, N+M)   int32 — 0/1 ships, 2 obstacles
    alive:    (B, N+M)   bool
    all others have a trailing feature dimension.
    """

    data: dict[ObsKey, torch.Tensor]

    # ------------------------------------------------------------------
    # Key access — supports ObsKey enum or str
    # ------------------------------------------------------------------

    def __getitem__(self, key: "ObsKey | str") -> torch.Tensor:
        if isinstance(key, ObsKey):
            return self.data[key]
        return self.data[ObsKey(key)]

    def __contains__(self, key: "ObsKey | str") -> bool:
        if isinstance(key, ObsKey):
            return key in self.data
        try:
            return ObsKey(key) in self.data
        except ValueError:
            return False

    def items(self) -> ItemsView[ObsKey, torch.Tensor]:
        return self.data.items()

    # ------------------------------------------------------------------
    # Typed property accessors
    # ------------------------------------------------------------------

    @property
    def pos(self) -> torch.Tensor:
        return self.data[ObsKey.POS]

    @property
    def vel(self) -> torch.Tensor:
        return self.data[ObsKey.VEL]

    @property
    def att(self) -> torch.Tensor:
        return self.data[ObsKey.ATT]

    @property
    def ang_vel(self) -> torch.Tensor:
        return self.data[ObsKey.ANG_VEL]

    @property
    def health(self) -> torch.Tensor:
        return self.data[ObsKey.HEALTH]

    @property
    def power(self) -> torch.Tensor:
        return self.data[ObsKey.POWER]

    @property
    def cooldown(self) -> torch.Tensor:
        return self.data[ObsKey.COOLDOWN]

    @property
    def team_id(self) -> torch.Tensor:
        return self.data[ObsKey.TEAM_ID]

    @property
    def alive(self) -> torch.Tensor:
        return self.data[ObsKey.ALIVE]

    @property
    def radius(self) -> torch.Tensor:
        return self.data[ObsKey.RADIUS]

    @property
    def previous_action(self) -> torch.Tensor:
        return self.data[ObsKey.PREVIOUS_ACTION]

    # ------------------------------------------------------------------
    # Immutable update / structural ops
    # ------------------------------------------------------------------

    def update(self, key: ObsKey, value: torch.Tensor) -> "MVPObservation":
        new_data = dict(self.data)
        new_data[key] = value
        return MVPObservation(data=new_data)

    def flip_team(self, num_ships: int) -> "MVPObservation":
        """Swap team IDs 0 and 1 for the first num_ships entity slots."""
        team_id = self.data[ObsKey.TEAM_ID].clone()
        ship_slice = team_id[..., :num_ships]
        flipped = torch.where(ship_slice == 0, 1, torch.where(ship_slice == 1, 0, ship_slice))
        team_id[..., :num_ships] = flipped
        return self.update(ObsKey.TEAM_ID, team_id)

    def slice_envs(self, idx: "slice | torch.Tensor") -> "MVPObservation":
        return MVPObservation(data={k: v[idx] for k, v in self.data.items()})

    def slice_time(self, start: int, end: int) -> "MVPObservation":
        return MVPObservation(data={k: v[start:end] for k, v in self.data.items()})

    def concat_batch(self, other: "MVPObservation") -> "MVPObservation":
        """Concatenate two observations along the batch (env) dimension (dim 0)."""
        return MVPObservation(
            data={k: torch.cat([v, other.data[k]], dim=0) for k, v in self.data.items()}
        )


@dataclass
class ObservationBuffers:
    """Reusable static tensors for constructing raw observations.

    Training allocates these once so the wrapper's step path stays allocation-free.
    Evaluation modes can omit them and let :func:`observation_from_state` create
    short-lived buffers instead.
    """

    ship_radius: torch.Tensor
    obstacle_ang_vel: torch.Tensor | None = None
    obstacle_health: torch.Tensor | None = None
    obstacle_power: torch.Tensor | None = None
    obstacle_cooldown: torch.Tensor | None = None
    obstacle_team_id: torch.Tensor | None = None
    obstacle_alive: torch.Tensor | None = None
    obstacle_prev_action: torch.Tensor | None = None
    obstacle_radius: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        num_ships: int,
        num_obstacles: int,
        ship_config: ShipConfig,
        device: torch.device,
    ) -> "ObservationBuffers":
        """Allocate reusable tensors for an environment configuration."""
        ship_radius = torch.full(
            (num_envs, num_ships, 1),
            ship_config.collision_radius,
            device=device,
            dtype=torch.float32,
        )
        if num_obstacles == 0:
            return cls(ship_radius=ship_radius)

        return cls(
            ship_radius=ship_radius,
            obstacle_ang_vel=torch.zeros(num_envs, num_obstacles, 1, device=device),
            obstacle_health=torch.full(
                (num_envs, num_obstacles, 1), ship_config.max_health, device=device
            ),
            obstacle_power=torch.zeros(num_envs, num_obstacles, 1, device=device),
            obstacle_cooldown=torch.zeros(num_envs, num_obstacles, 1, device=device),
            obstacle_team_id=torch.full(
                (num_envs, num_obstacles), 2, device=device, dtype=torch.int32
            ),
            obstacle_alive=torch.ones(num_envs, num_obstacles, device=device, dtype=torch.bool),
            obstacle_prev_action=torch.zeros(
                num_envs, num_obstacles, 3, device=device, dtype=torch.long
            ),
            obstacle_radius=torch.zeros(num_envs, num_obstacles, 1, device=device),
        )

    def refresh_obstacle_radius_all(self, state: TensorState) -> None:
        """Copy every obstacle radius from ``state`` into the reusable buffer."""
        if self.obstacle_radius is not None:
            self.obstacle_radius.copy_(state.obstacle_radius.unsqueeze(-1))

    def refresh_obstacle_radius(self, state: TensorState, mask: torch.Tensor) -> None:
        """Refresh obstacle radii for reset environments selected by ``mask``."""
        if self.obstacle_radius is not None:
            self.obstacle_radius.copy_(
                torch.where(
                    mask.view(-1, 1, 1),
                    state.obstacle_radius.unsqueeze(-1),
                    self.obstacle_radius,
                )
            )


def observation_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    buffers: ObservationBuffers | None = None,
) -> MVPObservation:
    """Build the raw policy observation for the supplied environment state.

    Obstacles are represented as always-alive team-2 tokens. Passing reusable
    ``buffers`` keeps the training step path allocation-free; callers outside
    training may omit them.
    """
    if buffers is None:
        buffers = ObservationBuffers.allocate(
            state.num_envs,
            state.max_ships,
            state.num_obstacles,
            ship_config,
            state.device,
        )
        buffers.refresh_obstacle_radius_all(state)

    ship_pos = torch.stack([state.ship_pos.real, state.ship_pos.imag], dim=-1)
    ship_vel = torch.stack([state.ship_vel.real, state.ship_vel.imag], dim=-1)
    ship_att = torch.stack([state.ship_attitude.real, state.ship_attitude.imag], dim=-1)
    ship_ang = state.ship_ang_vel.unsqueeze(-1)
    ship_health = state.ship_health.unsqueeze(-1)
    ship_power = state.ship_power.unsqueeze(-1)
    ship_cooldown = state.ship_cooldown.unsqueeze(-1)
    ship_prev_action = state.prev_action.long()

    if state.num_obstacles == 0:
        return MVPObservation(
            data={
                ObsKey.POS: ship_pos,
                ObsKey.VEL: ship_vel,
                ObsKey.ATT: ship_att,
                ObsKey.ANG_VEL: ship_ang,
                ObsKey.HEALTH: ship_health,
                ObsKey.POWER: ship_power,
                ObsKey.COOLDOWN: ship_cooldown,
                ObsKey.TEAM_ID: state.ship_team_id,
                ObsKey.ALIVE: state.ship_alive,
                ObsKey.PREVIOUS_ACTION: ship_prev_action,
                ObsKey.RADIUS: buffers.ship_radius,
            }
        )

    assert buffers.obstacle_ang_vel is not None
    assert buffers.obstacle_health is not None
    assert buffers.obstacle_power is not None
    assert buffers.obstacle_cooldown is not None
    assert buffers.obstacle_team_id is not None
    assert buffers.obstacle_alive is not None
    assert buffers.obstacle_prev_action is not None
    assert buffers.obstacle_radius is not None

    obstacle_pos = torch.stack([state.obstacle_pos.real, state.obstacle_pos.imag], dim=-1)
    obstacle_vel = torch.stack([state.obstacle_vel.real, state.obstacle_vel.imag], dim=-1)
    obstacle_speed = torch.norm(obstacle_vel, dim=-1, keepdim=True).clamp(min=EPS)
    obstacle_att = obstacle_vel / obstacle_speed
    return MVPObservation(
        data={
            ObsKey.POS: torch.cat([ship_pos, obstacle_pos], dim=1),
            ObsKey.VEL: torch.cat([ship_vel, obstacle_vel], dim=1),
            ObsKey.ATT: torch.cat([ship_att, obstacle_att], dim=1),
            ObsKey.ANG_VEL: torch.cat([ship_ang, buffers.obstacle_ang_vel], dim=1),
            ObsKey.HEALTH: torch.cat([ship_health, buffers.obstacle_health], dim=1),
            ObsKey.POWER: torch.cat([ship_power, buffers.obstacle_power], dim=1),
            ObsKey.COOLDOWN: torch.cat([ship_cooldown, buffers.obstacle_cooldown], dim=1),
            ObsKey.TEAM_ID: torch.cat([state.ship_team_id, buffers.obstacle_team_id], dim=1),
            ObsKey.ALIVE: torch.cat([state.ship_alive, buffers.obstacle_alive], dim=1),
            ObsKey.PREVIOUS_ACTION: torch.cat(
                [ship_prev_action, buffers.obstacle_prev_action], dim=1
            ),
            ObsKey.RADIUS: torch.cat([buffers.ship_radius, buffers.obstacle_radius], dim=1),
        }
    )
