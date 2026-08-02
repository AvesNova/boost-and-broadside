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
    LOCAL_LOG_INDEX = "local_log_index"
    LOCAL_INDEX_GRADIENT = "local_index_gradient"
    FIELD_TRANSITION_WIDTH = "field_transition_width"
    FIELD_INSIDE_LOG_INDEX = "field_inside_log_index"
    FIELD_OUTSIDE_LOG_INDEX = "field_outside_log_index"
    FIELD_LOG_INDEX_RATIO = "field_log_index_ratio"
    FIELD_DAMAGE = "field_damage"


# Channels whose last axis IS the token axis — everything else has a trailing
# feature dim. Used by YemongObservation.slice_tokens.
_TOKEN_LAST_KEYS = frozenset({ObsKey.TEAM_ID, ObsKey.ALIVE})


@dataclass(frozen=True)
class YemongObservation:
    """Typed immutable observation for all entities.

    data: maps ObsKey → tensor of shape (B, N+M, ...) or (T, B, N+M, ...) etc.

    team_id:  (B, N+M)   int32 — 0/1 ships, 2 fields
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

    @property
    def local_log_index(self) -> torch.Tensor:
        return self.data[ObsKey.LOCAL_LOG_INDEX]

    # ------------------------------------------------------------------
    # Immutable update / structural ops
    # ------------------------------------------------------------------

    def update(self, key: ObsKey, value: torch.Tensor) -> "YemongObservation":
        new_data = dict(self.data)
        new_data[key] = value
        return YemongObservation(data=new_data)

    def flip_team(self, num_ships: int) -> "YemongObservation":
        """Swap team IDs 0 and 1 for the first num_ships entity slots."""
        team_id = self.data[ObsKey.TEAM_ID].clone()
        ship_slice = team_id[..., :num_ships]
        flipped = torch.where(ship_slice == 0, 1, torch.where(ship_slice == 1, 0, ship_slice))
        team_id[..., :num_ships] = flipped
        return self.update(ObsKey.TEAM_ID, team_id)

    def slice_envs(self, idx: "slice | torch.Tensor") -> "YemongObservation":
        return YemongObservation(data={k: v[idx] for k, v in self.data.items()})

    def slice_time(self, start: int, end: int) -> "YemongObservation":
        return YemongObservation(data={k: v[start:end] for k, v in self.data.items()})

    def slice_tokens(self, start: int, end: int) -> "YemongObservation":
        """Slice the entity-token axis, which is always the last non-feature dim.

        ``team_id`` and ``alive`` end at the token axis while every other channel
        carries a trailing feature dim, so the axis is addressed from the right.
        """
        return YemongObservation(
            data={
                k: (v[..., start:end] if k in _TOKEN_LAST_KEYS else v[..., start:end, :])
                for k, v in self.data.items()
            }
        )

    def concat_batch(self, other: "YemongObservation") -> "YemongObservation":
        """Concatenate two observations along the batch (env) dimension (dim 0)."""
        return YemongObservation(
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
    field_zero_vec: torch.Tensor | None = None
    field_zero_scalar: torch.Tensor | None = None
    field_health: torch.Tensor | None = None
    field_team_id: torch.Tensor | None = None
    field_alive: torch.Tensor | None = None
    field_prev_action: torch.Tensor | None = None
    ship_field_feature_zeros: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        num_ships: int,
        num_fields: int,
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
        if num_fields == 0:
            return cls(ship_radius=ship_radius)

        return cls(
            ship_radius=ship_radius,
            field_zero_vec=torch.zeros(num_envs, num_fields, 2, device=device),
            field_zero_scalar=torch.zeros(num_envs, num_fields, 1, device=device),
            field_health=torch.full(
                (num_envs, num_fields, 1), ship_config.max_health, device=device
            ),
            field_team_id=torch.full((num_envs, num_fields), 2, device=device, dtype=torch.int32),
            field_alive=torch.ones(num_envs, num_fields, device=device, dtype=torch.bool),
            field_prev_action=torch.zeros(num_envs, num_fields, 3, device=device, dtype=torch.long),
            ship_field_feature_zeros=torch.zeros(num_envs, num_ships, 1, device=device),
        )

    def refresh_field_state_all(self, state: TensorState) -> None:
        """Compatibility no-op: field geometry is read directly from state."""

    def refresh_field_state(self, state: TensorState, mask: torch.Tensor) -> None:
        """Compatibility no-op: field geometry is read directly from state."""


def index_gradient_scale(ship_config: ShipConfig) -> float:
    """Normalising scale for grad(n), so the encoded channel lands near [-1, 1].

    The interface profile is the quintic smoothstep ``alpha = 6z^5 - 15z^4 + 10z^3``
    with ``z = clamp(0.5 - d/w, 0, 1)``. Its slope ``30z^2(z-1)^2`` peaks at 15/8
    when ``z = 1/2``, so ``|d alpha/d d| <= 1.875/w``. Composition telescopes as
    ``grad(n) = sum(delta_n_i grad(alpha_i))``, and the widest index span a single
    interface can carry is ``s^2 - s^-2``. The narrowest configured band therefore
    bounds the whole map's gradient.
    """
    step = ship_config.field_index_step
    max_delta_index = step**2 - step**-2
    return max(
        1.875 * max_delta_index / ship_config.field_transition_width_min,
        EPS,
    )


def observation_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    buffers: ObservationBuffers | None = None,
) -> YemongObservation:
    """Build the raw policy observation for the supplied environment state.

    Fields are represented as always-alive team-2 tokens. Passing reusable
    ``buffers`` keeps the training step path allocation-free; callers outside
    training may omit them.
    """
    if buffers is None:
        buffers = ObservationBuffers.allocate(
            state.num_envs,
            state.max_ships,
            state.num_fields,
            ship_config,
            state.device,
        )
        buffers.refresh_field_state_all(state)

    ship_pos = torch.stack([state.ship_pos.real, state.ship_pos.imag], dim=-1)
    ship_vel = torch.stack([state.ship_vel.real, state.ship_vel.imag], dim=-1)
    ship_att = torch.stack([state.ship_attitude.real, state.ship_attitude.imag], dim=-1)
    ship_ang = state.ship_ang_vel.unsqueeze(-1)
    ship_health = state.ship_health.unsqueeze(-1)
    ship_power = state.ship_power.unsqueeze(-1)
    ship_cooldown = state.ship_cooldown.unsqueeze(-1)
    ship_prev_action = state.prev_action.long()

    log_scale = 2.0 * torch.log(
        torch.tensor(ship_config.field_index_step, device=state.device, dtype=torch.float32)
    )
    ship_local_log_index = torch.log(state.ship_local_index).unsqueeze(-1) / log_scale

    # grad(n) at the ship. This is the direction the medium is changing, and it is
    # the force term in a = F/m + 0.5|v|^2 grad(log m) - (v.grad(log m))v — so
    # without it a ship feels an acceleration whose source it cannot see.
    ship_index_gradient = torch.stack(
        [state.ship_field_gradient.real, state.ship_field_gradient.imag],
        dim=-1,
    ) / index_gradient_scale(ship_config)

    if state.num_fields == 0:
        zeros = torch.zeros_like(ship_local_log_index)
        return YemongObservation(
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
                ObsKey.LOCAL_LOG_INDEX: ship_local_log_index,
                ObsKey.LOCAL_INDEX_GRADIENT: ship_index_gradient,
                ObsKey.FIELD_TRANSITION_WIDTH: zeros,
                ObsKey.FIELD_INSIDE_LOG_INDEX: zeros,
                ObsKey.FIELD_OUTSIDE_LOG_INDEX: zeros,
                ObsKey.FIELD_LOG_INDEX_RATIO: zeros,
                ObsKey.FIELD_DAMAGE: zeros,
            }
        )

    assert buffers.field_zero_vec is not None
    assert buffers.field_zero_scalar is not None
    assert buffers.field_health is not None
    assert buffers.field_team_id is not None
    assert buffers.field_alive is not None
    assert buffers.field_prev_action is not None
    assert buffers.ship_field_feature_zeros is not None

    field_pos = torch.stack([state.field_pos.real, state.field_pos.imag], dim=-1)
    field_inside = torch.log(state.field_index).unsqueeze(-1) / log_scale
    parent_index = state.field_index - state.field_delta_index
    field_outside = torch.log(parent_index).unsqueeze(-1) / log_scale
    # Absolute encodings use ±2 levels. A parent/child interface can span four
    # levels, so ratio encoding uses 4*log(step) to retain [-1, 1].
    ratio_scale = 4.0 * torch.log(
        torch.tensor(ship_config.field_index_step, device=state.device, dtype=torch.float32)
    )
    field_ratio = torch.log(state.field_index / parent_index).unsqueeze(-1) / ratio_scale
    max_damage = max(2.0 * ship_config.field_interface_damage, EPS)
    field_damage = state.field_damage.unsqueeze(-1) / max_damage
    ship_zero = buffers.ship_field_feature_zeros
    return YemongObservation(
        data={
            ObsKey.POS: torch.cat([ship_pos, field_pos], dim=1),
            ObsKey.VEL: torch.cat([ship_vel, buffers.field_zero_vec], dim=1),
            ObsKey.ATT: torch.cat([ship_att, buffers.field_zero_vec], dim=1),
            ObsKey.ANG_VEL: torch.cat([ship_ang, buffers.field_zero_scalar], dim=1),
            ObsKey.HEALTH: torch.cat([ship_health, buffers.field_health], dim=1),
            ObsKey.POWER: torch.cat([ship_power, buffers.field_zero_scalar], dim=1),
            ObsKey.COOLDOWN: torch.cat([ship_cooldown, buffers.field_zero_scalar], dim=1),
            ObsKey.TEAM_ID: torch.cat([state.ship_team_id, buffers.field_team_id], dim=1),
            ObsKey.ALIVE: torch.cat([state.ship_alive, buffers.field_alive], dim=1),
            ObsKey.PREVIOUS_ACTION: torch.cat([ship_prev_action, buffers.field_prev_action], dim=1),
            ObsKey.RADIUS: torch.cat(
                [buffers.ship_radius, state.field_radius.unsqueeze(-1)], dim=1
            ),
            ObsKey.LOCAL_LOG_INDEX: torch.cat(
                [ship_local_log_index, buffers.field_zero_scalar], dim=1
            ),
            ObsKey.LOCAL_INDEX_GRADIENT: torch.cat(
                [ship_index_gradient, buffers.field_zero_vec], dim=1
            ),
            ObsKey.FIELD_TRANSITION_WIDTH: torch.cat(
                [ship_zero, state.field_transition_width.unsqueeze(-1)], dim=1
            ),
            ObsKey.FIELD_INSIDE_LOG_INDEX: torch.cat([ship_zero, field_inside], dim=1),
            ObsKey.FIELD_OUTSIDE_LOG_INDEX: torch.cat([ship_zero, field_outside], dim=1),
            ObsKey.FIELD_LOG_INDEX_RATIO: torch.cat([ship_zero, field_ratio], dim=1),
            ObsKey.FIELD_DAMAGE: torch.cat([ship_zero, field_damage], dim=1),
        }
    )
