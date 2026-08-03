from collections.abc import Callable, ItemsView
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


class BulletObsKey(StrEnum):
    """Channels of the bullet observation.

    Bullets live on their own ``(B, N*K, ...)`` axis rather than the entity-token
    axis: they are key/value-only inputs to cross-attention, never queries, and
    never carry recurrent state.
    """

    POS = "bullet_pos"
    VEL = "bullet_vel"
    DAMAGE = "bullet_damage"
    LIFETIME = "bullet_lifetime"
    LOCAL_LOG_INDEX = "bullet_local_log_index"
    LOCAL_INDEX_GRADIENT = "bullet_local_index_gradient"
    TEAM_ID = "bullet_team_id"
    ACTIVE = "bullet_active"


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
    # Optional bullet channels on their own (B, N*K, ...) axis. Kept here rather
    # than passed alongside so every structural op (slice/concat/flip) carries
    # them automatically and cannot fall out of sync with the entity tokens.
    bullets: dict["BulletObsKey", torch.Tensor] | None = None

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
        return YemongObservation(data=new_data, bullets=self.bullets)

    def flip_team(
        self, num_ships: int, mask: "torch.Tensor | None" = None
    ) -> "YemongObservation":
        """Swap team IDs 0 and 1 for the first num_ships entity slots.

        Ship and bullet team IDs flip *together*. A bullet's team is its
        shooter's, so mirroring one without the other shows a policy its own
        fire as the enemy's. Inactive bullet slots flip too, which is harmless —
        they are masked out of attention. Field tokens keep their own team.

        ``mask`` selects which environments flip; None flips all of them. It must
        broadcast against the leading (batch) dims — an ``(B,)`` bool for a
        ``(B, tokens)`` observation. Per-env selection exists because an ego_pass
        policy plays team 1 in only some environments and carries one recurrent
        state, so it cannot be run twice to cover both perspectives.
        """
        selector = None if mask is None else mask.unsqueeze(-1)

        def swap(values: torch.Tensor) -> torch.Tensor:
            swapped = torch.where(values == 0, 1, torch.where(values == 1, 0, values))
            return swapped if selector is None else torch.where(selector, swapped, values)

        team_id = self.data[ObsKey.TEAM_ID].clone()
        team_id[..., :num_ships] = swap(team_id[..., :num_ships])
        flipped_obs = self.update(ObsKey.TEAM_ID, team_id)
        if self.bullets is None:
            return flipped_obs
        new_bullets = dict(self.bullets)
        new_bullets[BulletObsKey.TEAM_ID] = swap(self.bullets[BulletObsKey.TEAM_ID])
        return YemongObservation(data=flipped_obs.data, bullets=new_bullets)

    def map(self, fn: "Callable[[torch.Tensor], torch.Tensor]") -> "YemongObservation":
        """Apply ``fn`` to every channel tensor, entity and bullet alike.

        Transfer helpers (pin/to/slice) use this so a new channel group cannot be
        silently dropped by a call site that only knew about ``data``.
        """
        return YemongObservation(
            data={k: fn(v) for k, v in self.data.items()},
            bullets=None if self.bullets is None else {k: fn(v) for k, v in self.bullets.items()},
        )

    def slice_envs(self, idx: "slice | torch.Tensor") -> "YemongObservation":
        return YemongObservation(
            data={k: v[idx] for k, v in self.data.items()},
            bullets=None if self.bullets is None else {k: v[idx] for k, v in self.bullets.items()},
        )

    def slice_time(self, start: int, end: int) -> "YemongObservation":
        return YemongObservation(
            data={k: v[start:end] for k, v in self.data.items()},
            bullets=(
                None if self.bullets is None else {k: v[start:end] for k, v in self.bullets.items()}
            ),
        )

    def slice_tokens(self, start: int, end: int) -> "YemongObservation":
        """Slice the entity-token axis, which is always the last non-feature dim.

        ``team_id`` and ``alive`` end at the token axis while every other channel
        carries a trailing feature dim, so the axis is addressed from the right.
        """
        # Bullets are not on the entity-token axis, so they pass through unchanged.
        return YemongObservation(
            data={
                k: (v[..., start:end] if k in _TOKEN_LAST_KEYS else v[..., start:end, :])
                for k, v in self.data.items()
            },
            bullets=self.bullets,
        )

    def concat_batch(self, other: "YemongObservation") -> "YemongObservation":
        """Concatenate two observations along the batch (env) dimension (dim 0)."""
        bullets = None
        if self.bullets is not None and other.bullets is not None:
            bullets = {k: torch.cat([v, other.bullets[k]], dim=0) for k, v in self.bullets.items()}
        return YemongObservation(
            data={k: torch.cat([v, other.data[k]], dim=0) for k, v in self.data.items()},
            bullets=bullets,
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


def bullet_observation_from_state(
    state: TensorState,
    ship_config: ShipConfig,
) -> dict[BulletObsKey, torch.Tensor]:
    """Flatten the per-ship bullet ring buffers into one (B, N*K, ...) axis.

    Every slot is emitted, active or not; inactive slots are masked out of
    attention rather than compacted, keeping the shape static.

    Position and velocity are deliberately encoded exactly as ship position and
    velocity are (see ``build_standard_coordinator``). Attention computes relative
    geometry as a bilinear form over the two tokens' Fourier features, which only
    collapses into a function of displacement when both sides share one frequency
    basis — so matching the ship encoding is load-bearing, not cosmetic.
    """
    B, N, K = state.bullet_pos.shape
    flat = (B, N * K)

    bullet_pos = torch.stack(
        [state.bullet_pos.real.reshape(flat), state.bullet_pos.imag.reshape(flat)], dim=-1
    )
    bullet_vel = torch.stack(
        [state.bullet_vel.real.reshape(flat), state.bullet_vel.imag.reshape(flat)], dim=-1
    )
    gradient = state.bullet_field_gradient.reshape(flat)
    bullet_gradient = torch.stack([gradient.real, gradient.imag], dim=-1) / index_gradient_scale(
        ship_config
    )

    log_scale = 2.0 * torch.log(
        torch.tensor(ship_config.field_index_step, device=state.device, dtype=torch.float32)
    )
    # Inactive slots keep a stale index of 0 from reset, and log(0) is -inf.
    local_index = state.bullet_local_index.reshape(flat).clamp(min=EPS)
    bullet_log_index = torch.log(local_index).unsqueeze(-1) / log_scale

    # A bullet's team is its shooter's; the ring buffer's ship axis supplies it.
    shooter_team = state.ship_team_id.unsqueeze(-1).expand(B, N, K).reshape(flat)

    return {
        BulletObsKey.POS: bullet_pos,
        BulletObsKey.VEL: bullet_vel,
        BulletObsKey.DAMAGE: state.bullet_remaining_damage.reshape(flat).unsqueeze(-1)
        / max(ship_config.bullet_damage, EPS),
        BulletObsKey.LIFETIME: state.bullet_time.reshape(flat).unsqueeze(-1)
        / max(ship_config.bullet_lifetime, EPS),
        BulletObsKey.LOCAL_LOG_INDEX: bullet_log_index,
        BulletObsKey.LOCAL_INDEX_GRADIENT: bullet_gradient,
        BulletObsKey.TEAM_ID: shooter_team.to(torch.int32),
        BulletObsKey.ACTIVE: state.bullet_active.reshape(flat),
    }


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
    include_bullets: bool = False,
) -> YemongObservation:
    """Build the raw policy observation for the supplied environment state.

    Fields are represented as always-alive team-2 tokens. Passing reusable
    ``buffers`` keeps the training step path allocation-free; callers outside
    training may omit them.

    ``include_bullets`` attaches the bullet cross-attention channels. It is off by
    default so profiles that do not read bullets pay neither the reduction nor the
    rollout storage.
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

    bullets = (
        bullet_observation_from_state(state, ship_config)
        if include_bullets and state.max_bullets > 0
        else None
    )

    if state.num_fields == 0:
        zeros = torch.zeros_like(ship_local_log_index)
        return YemongObservation(
            bullets=bullets,
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
            },
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
        bullets=bullets,
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
        },
    )
