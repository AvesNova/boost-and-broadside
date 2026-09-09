from collections.abc import Callable, ItemsView
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import torch

from boost_and_broadside.config.core import EnvConfig, ShipConfig
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.perception import TeamVisibility, team_visibility_from_state
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
    VISIBLE = "visible"
    OBJECT_TYPE = "object_type"
    RADIUS = "radius"
    PREVIOUS_ACTION = "previous_action"
    LOCAL_LOG_INDEX = "local_log_index"
    LOCAL_INDEX_GRADIENT = "local_index_gradient"
    FIELD_TRANSITION_WIDTH = "field_transition_width"
    FIELD_TARGET_LOG_INDEX = "field_target_log_index"
    FIELD_DAMAGE = "field_damage"
    ZONE_ROLE = "zone_role"
    CAPTURE_PROGRESS = "capture_progress"
    CAPTURE_DIRECTION = "capture_direction"
    FRONT_POSITION = "front_position"
    FRONT_WIN_THRESHOLD = "front_win_threshold"
    TIME_REMAINING = "time_remaining"
    GAME_MODE = "game_mode"


class ObjectType(IntEnum):
    SHIP = 0
    FIELD = 1
    ZONE = 2
    BOUNDARY = 3


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
    VISIBLE = "bullet_visible"


# Channels whose last axis IS the token axis — everything else has a trailing
# feature dim. Used by YemongObservation.slice_tokens.
_TOKEN_LAST_KEYS = frozenset(
    {ObsKey.TEAM_ID, ObsKey.ALIVE, ObsKey.VISIBLE, ObsKey.OBJECT_TYPE, ObsKey.ZONE_ROLE}
)


@dataclass(frozen=True)
class YemongObservation:
    """Typed immutable observation for all entities.

    data: maps ObsKey → tensors whose token axis contains ships, fields, zones,
    and the combined boundary/global token.

    team_id:  (B, tokens) int32 — 0/1 owned objects, 2 neutral objects
    alive:    (B, tokens) bool
    all others have a trailing feature dimension.
    """

    data: dict[ObsKey, torch.Tensor]
    # Optional bullet channels on their own (B, N*K, ...) axis. Kept here rather
    # than passed alongside so every structural op (slice/concat/flip) carries
    # them automatically and cannot fall out of sync with the entity tokens.
    bullets: dict["BulletObsKey", torch.Tensor] | None = None
    # A root observation returned by an environment carries the independently
    # masked team-1 perspective here. Rollout storage intentionally stores only
    # the selected team-0 view used by ego-pass training. Keeping the alternate
    # attached during live action selection prevents callers from manufacturing
    # team-1 sight by swapping labels on team-0 truth.
    team1_data: dict[ObsKey, torch.Tensor] | None = None
    team1_bullets: dict["BulletObsKey", torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Key access — supports ObsKey enum or str
    # ------------------------------------------------------------------

    def __getitem__(self, key: "ObsKey | str") -> torch.Tensor:
        resolved = key if isinstance(key, ObsKey) else ObsKey(key)
        if resolved in self.data:
            return self.data[resolved]
        # Test fixtures and old in-process ship-only observations can omit new
        # map metadata. Their unambiguous defaults preserve the compact fixture
        # API; serialized checkpoints remain rejected by schema v6.
        team_id = self.data[ObsKey.TEAM_ID]
        if resolved == ObsKey.VISIBLE:
            return self.data[ObsKey.ALIVE]
        if resolved == ObsKey.OBJECT_TYPE:
            return torch.where(
                team_id == 2,
                torch.full_like(team_id, int(ObjectType.FIELD)),
                torch.full_like(team_id, int(ObjectType.SHIP)),
            )
        if resolved == ObsKey.ZONE_ROLE:
            return torch.full_like(team_id, 5)
        if resolved in {
            ObsKey.CAPTURE_PROGRESS,
            ObsKey.CAPTURE_DIRECTION,
            ObsKey.FRONT_POSITION,
            ObsKey.FRONT_WIN_THRESHOLD,
            ObsKey.TIME_REMAINING,
            ObsKey.GAME_MODE,
        }:
            return torch.zeros((*team_id.shape, 1), dtype=torch.float32, device=team_id.device)
        raise KeyError(resolved)

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
    def visible(self) -> torch.Tensor:
        return self.data[ObsKey.VISIBLE]

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
        return YemongObservation(
            data=new_data,
            bullets=self.bullets,
            team1_data=self.team1_data,
            team1_bullets=self.team1_bullets,
        )

    def for_team(self, team: int) -> "YemongObservation":
        """Return one independently masked team view with no alternate attached."""

        if team == 0:
            return YemongObservation(data=self.data, bullets=self.bullets)
        if team != 1:
            raise ValueError(f"team perspective must be 0 or 1, got {team}")
        if self.team1_data is None:
            raise ValueError("observation does not carry a team-1 perception")
        return YemongObservation(data=self.team1_data, bullets=self.team1_bullets)

    def select_team(self, as_team1: torch.Tensor) -> "YemongObservation":
        """Select team-0/team-1 perception independently for every environment."""

        if self.team1_data is None:
            # Omniscient/fixture observations predate attached team views; label
            # flipping remains a valid compatibility operation for those only.
            return self.for_team(0)

        def choose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            selector = as_team1
            while selector.dim() < a.dim():
                selector = selector.unsqueeze(-1)
            return torch.where(selector, b, a)

        bullets = None
        if self.bullets is not None and self.team1_bullets is not None:
            bullets = {k: choose(v, self.team1_bullets[k]) for k, v in self.bullets.items()}
        return YemongObservation(
            data={k: choose(v, self.team1_data[k]) for k, v in self.data.items()},
            bullets=bullets,
        )

    def flip_team(
        self, num_ships: int, mask: "torch.Tensor | None" = None
    ) -> "YemongObservation":
        """Swap team IDs 0 and 1 across ships and owned strategic tokens.

        Ship and bullet team IDs flip *together*. A bullet's team is its
        shooter's, so mirroring one without the other shows a policy its own
        fire as the enemy's. Inactive bullet slots flip too, which is harmless —
        they are masked out of attention. Field and boundary tokens use the
        neutral team ID and therefore remain unchanged. Zone ownership swaps
        with the ship teams so the canonical team-0 view stays self-relative.

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

        # ``num_ships`` remains in the public signature because callers and
        # checkpoint-era fixtures use it, but typed map tokens also carry team
        # ownership now and must be canonicalized with the ships.
        del num_ships
        team_id = swap(self.data[ObsKey.TEAM_ID])
        flipped_obs = self.update(ObsKey.TEAM_ID, team_id)
        flipped_data = dict(flipped_obs.data)

        def select_env(original: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
            if mask is None:
                return changed
            env_selector = mask
            while env_selector.dim() < original.dim():
                env_selector = env_selector.unsqueeze(-1)
            return torch.where(env_selector, changed, original)

        if ObsKey.ZONE_ROLE in flipped_data:
            roles = flipped_data[ObsKey.ZONE_ROLE]
            swapped_roles = torch.where(
                roles == 0,
                4,
                torch.where(
                    roles == 4,
                    0,
                    torch.where(roles == 1, 3, torch.where(roles == 3, 1, roles)),
                ),
            )
            flipped_data[ObsKey.ZONE_ROLE] = select_env(roles, swapped_roles)
        for key in (ObsKey.CAPTURE_DIRECTION, ObsKey.FRONT_POSITION):
            if key in flipped_data:
                flipped_data[key] = select_env(flipped_data[key], -flipped_data[key])
        flipped_obs = YemongObservation(data=flipped_data, bullets=flipped_obs.bullets)
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
            team1_data=(
                None if self.team1_data is None else {k: fn(v) for k, v in self.team1_data.items()}
            ),
            team1_bullets=(
                None
                if self.team1_bullets is None
                else {k: fn(v) for k, v in self.team1_bullets.items()}
            ),
        )

    def slice_envs(self, idx: "slice | torch.Tensor") -> "YemongObservation":
        return YemongObservation(
            data={k: v[idx] for k, v in self.data.items()},
            bullets=None if self.bullets is None else {k: v[idx] for k, v in self.bullets.items()},
            team1_data=(
                None if self.team1_data is None else {k: v[idx] for k, v in self.team1_data.items()}
            ),
            team1_bullets=(
                None
                if self.team1_bullets is None
                else {k: v[idx] for k, v in self.team1_bullets.items()}
            ),
        )

    def slice_time(self, start: int, end: int) -> "YemongObservation":
        return YemongObservation(
            data={k: v[start:end] for k, v in self.data.items()},
            bullets=(
                None if self.bullets is None else {k: v[start:end] for k, v in self.bullets.items()}
            ),
            team1_data=(
                None
                if self.team1_data is None
                else {k: v[start:end] for k, v in self.team1_data.items()}
            ),
            team1_bullets=(
                None
                if self.team1_bullets is None
                else {k: v[start:end] for k, v in self.team1_bullets.items()}
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
            team1_data=(
                None
                if self.team1_data is None
                else {
                    k: (v[..., start:end] if k in _TOKEN_LAST_KEYS else v[..., start:end, :])
                    for k, v in self.team1_data.items()
                }
            ),
            team1_bullets=self.team1_bullets,
        )

    def concat_batch(self, other: "YemongObservation") -> "YemongObservation":
        """Concatenate two observations along the batch (env) dimension (dim 0)."""
        bullets = None
        if self.bullets is not None and other.bullets is not None:
            bullets = {k: torch.cat([v, other.bullets[k]], dim=0) for k, v in self.bullets.items()}
        return YemongObservation(
            data={k: torch.cat([v, other.data[k]], dim=0) for k, v in self.data.items()},
            bullets=bullets,
            team1_data=(
                None
                if self.team1_data is None or other.team1_data is None
                else {
                    k: torch.cat([v, other.team1_data[k]], dim=0)
                    for k, v in self.team1_data.items()
                }
            ),
            team1_bullets=(
                None
                if self.team1_bullets is None or other.team1_bullets is None
                else {
                    k: torch.cat([v, other.team1_bullets[k]], dim=0)
                    for k, v in self.team1_bullets.items()
                }
            ),
        )


@dataclass
class ObservationBuffers:
    """Reusable static tensors for constructing raw observations.

    Training allocates these once so the wrapper's step path stays allocation-free.
    Evaluation modes can omit them and let :func:`observation_from_state` create
    short-lived buffers instead.
    """

    ship_radius: torch.Tensor
    object_zero_vec: torch.Tensor | None = None
    object_zero_scalar: torch.Tensor | None = None
    object_team_id: torch.Tensor | None = None
    object_alive: torch.Tensor | None = None
    object_prev_action: torch.Tensor | None = None
    ship_object_feature_zeros: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        num_ships: int,
        num_fields: int,
        num_zones: int,
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
        num_objects = num_fields + num_zones + (1 if num_zones > 0 else 0)
        if num_objects == 0:
            return cls(ship_radius=ship_radius)

        return cls(
            ship_radius=ship_radius,
            object_zero_vec=torch.zeros(num_envs, num_objects, 2, device=device),
            object_zero_scalar=torch.zeros(num_envs, num_objects, 1, device=device),
            object_team_id=torch.full(
                (num_envs, num_objects), 2, device=device, dtype=torch.int32
            ),
            object_alive=torch.ones(num_envs, num_objects, device=device, dtype=torch.bool),
            object_prev_action=torch.zeros(
                num_envs, num_objects, 3, device=device, dtype=torch.long
            ),
            ship_object_feature_zeros=torch.zeros(num_envs, num_ships, 1, device=device),
        )

    def refresh_field_state_all(self, state: TensorState) -> None:
        """Compatibility no-op: field geometry is read directly from state."""

    def refresh_field_state(self, state: TensorState, mask: torch.Tensor) -> None:
        """Compatibility no-op: field geometry is read directly from state."""


def bullet_observation_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    visibility: torch.Tensor | None = None,
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

    active = state.bullet_active.reshape(flat)
    visible = active if visibility is None else visibility.reshape(flat) & active
    result = {
        BulletObsKey.POS: bullet_pos,
        BulletObsKey.VEL: bullet_vel,
        BulletObsKey.DAMAGE: state.bullet_remaining_damage.reshape(flat).unsqueeze(-1)
        / max(ship_config.bullet_damage, EPS),
        BulletObsKey.LIFETIME: state.bullet_time.reshape(flat).unsqueeze(-1)
        / max(ship_config.bullet_lifetime, EPS),
        BulletObsKey.LOCAL_LOG_INDEX: bullet_log_index,
        BulletObsKey.LOCAL_INDEX_GRADIENT: bullet_gradient,
        BulletObsKey.TEAM_ID: shooter_team.to(torch.int32),
        BulletObsKey.ACTIVE: visible,
        BulletObsKey.VISIBLE: visible,
    }
    if visibility is None:
        return result

    # Ignored tokens carry literal zeros as well as an explicit false mask. This
    # defense in depth catches accidental consumers that forget the attention
    # mask, and prevents hidden projectile state from entering auxiliary paths.
    for key, value in tuple(result.items()):
        if key in (BulletObsKey.ACTIVE, BulletObsKey.VISIBLE):
            continue
        mask = visible
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        result[key] = torch.where(mask, value, torch.zeros_like(value))
    return result


def index_gradient_scale(ship_config: ShipConfig) -> float:
    """Normalising scale for grad(n), so the encoded channel lands near [-1, 1].

    The interface profile is the quintic smoothstep ``alpha = 6z^5 - 15z^4 + 10z^3``
    with ``z = clamp(0.5 - d/w, 0, 1)``. Its slope ``30z^2(z-1)^2`` peaks at 15/8
    when ``z = 1/2``, so ``|d alpha/d d| <= 1.875/w``. The full overlap gradient
    is bounded and smooth but depends on local material mixtures. The complete
    configured index span across the narrowest band supplies a stable reference
    scale shared by every field count.
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
    ship_visibility: torch.Tensor | None = None,
    bullet_visibility: torch.Tensor | None = None,
    perspective_team: int | None = None,
) -> YemongObservation:
    """Build the raw policy observation for the supplied environment state.

    Fields are represented as always-alive team-2 tokens. Passing reusable
    ``buffers`` keeps the training step path allocation-free; callers outside
    training may omit them.

    ``include_bullets`` attaches the bullet cross-attention channels. It is off by
    default so profiles that do not read bullets pay neither the reduction nor the
    rollout storage. ``perspective_team`` keeps allied pending actions while
    replacing enemy actions with zero, even when the enemy ship is visible.
    """
    if buffers is None:
        buffers = ObservationBuffers.allocate(
            state.num_envs,
            state.max_ships,
            state.num_fields,
            state.num_zones,
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
    if perspective_team is not None:
        if perspective_team not in (0, 1):
            raise ValueError(f"perspective_team must be 0 or 1, got {perspective_team}")
        own_ship = (state.ship_team_id == perspective_team).unsqueeze(-1)
        ship_prev_action = torch.where(
            own_ship, ship_prev_action, torch.zeros_like(ship_prev_action)
        )

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
        bullet_observation_from_state(state, ship_config, bullet_visibility)
        if include_bullets and state.max_bullets > 0
        else None
    )

    visible_ships = (
        torch.ones_like(state.ship_alive) if ship_visibility is None else ship_visibility
    )
    observed_alive = state.ship_alive & visible_ships

    batch = state.num_envs
    num_fields = state.num_fields
    num_zones = state.num_zones
    has_frontline = num_zones > 0
    num_objects = num_fields + num_zones + int(has_frontline)
    ship_zero = torch.zeros_like(ship_local_log_index)
    ship_type = torch.zeros_like(state.ship_team_id)
    ship_no_zone = torch.full_like(state.ship_team_id, 5)

    if num_objects == 0:
        object_pos = ship_pos[:, :0]
        object_radius = ship_zero[:, :0]
        object_type = state.ship_team_id[:, :0]
        object_zone_role = state.ship_team_id[:, :0]
        object_team = state.ship_team_id[:, :0]
        object_alive = state.ship_alive[:, :0]
        object_zero_vec = ship_pos[:, :0]
        object_zero_scalar = ship_zero[:, :0]
        object_prev_action = ship_prev_action[:, :0]
    else:
        assert buffers.object_zero_vec is not None
        assert buffers.object_zero_scalar is not None
        assert buffers.object_team_id is not None
        assert buffers.object_alive is not None
        assert buffers.object_prev_action is not None
        assert buffers.ship_object_feature_zeros is not None
        object_zero_vec = buffers.object_zero_vec
        object_zero_scalar = buffers.object_zero_scalar
        object_prev_action = buffers.object_prev_action

        field_pos = torch.stack([state.field_pos.real, state.field_pos.imag], dim=-1)
        position_parts = [
            field_pos,
            torch.stack([state.zone_pos.real, state.zone_pos.imag], dim=-1),
        ]
        radius_parts = [state.field_radius.unsqueeze(-1), state.zone_radius.unsqueeze(-1)]
        type_parts = [
            torch.full((batch, num_fields), 1, dtype=torch.int32, device=state.device),
            torch.full((batch, num_zones), 2, dtype=torch.int32, device=state.device),
        ]
        role_parts = [
            torch.full((batch, num_fields), 5, dtype=torch.int32, device=state.device),
            state.zone_roles.to(torch.int32),
        ]
        zone_team = torch.where(
            state.zone_roles <= 1,
            torch.zeros_like(state.zone_roles, dtype=torch.int32),
            torch.where(
                state.zone_roles >= 3,
                torch.ones_like(state.zone_roles, dtype=torch.int32),
                torch.full_like(state.zone_roles, 2, dtype=torch.int32),
            ),
        )
        team_parts = [
            torch.full((batch, num_fields), 2, dtype=torch.int32, device=state.device),
            zone_team,
        ]
        if has_frontline:
            position_parts.append(
                torch.stack([state.map_center.real, state.map_center.imag], dim=-1).unsqueeze(1)
            )
            radius_parts.append(state.playable_boundary_radius[:, None, None])
            type_parts.append(torch.full((batch, 1), 3, dtype=torch.int32, device=state.device))
            role_parts.append(torch.full((batch, 1), 5, dtype=torch.int32, device=state.device))
            team_parts.append(torch.full((batch, 1), 2, dtype=torch.int32, device=state.device))
        object_pos = torch.cat(position_parts, dim=1)
        object_radius = torch.cat(radius_parts, dim=1)
        object_type = torch.cat(type_parts, dim=1)
        object_zone_role = torch.cat(role_parts, dim=1)
        object_team = torch.cat(team_parts, dim=1)
        object_alive = buffers.object_alive

    def object_scalar(
        field: torch.Tensor | None = None,
        zone: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            object_zero_scalar[:, :num_fields] if field is None else field,
            object_zero_scalar[:, num_fields : num_fields + num_zones] if zone is None else zone,
        ]
        if has_frontline:
            parts.append(object_zero_scalar[:, -1:] if boundary is None else boundary)
        return torch.cat(parts, dim=1) if parts else object_zero_scalar

    field_target = torch.log(state.field_index).unsqueeze(-1) / log_scale
    max_damage = max(2.0 * ship_config.field_interface_damage, EPS)
    field_damage = state.field_damage.unsqueeze(-1) / max_damage
    remaining = torch.where(
        state.match_max_steps > 0,
        (state.match_max_steps - state.step_count).clamp(min=0).float()
        / state.match_max_steps.clamp(min=1).float(),
        0.0,
    )
    observation = YemongObservation(
        bullets=bullets,
        data={
            ObsKey.POS: torch.cat([ship_pos, object_pos], dim=1),
            ObsKey.VEL: torch.cat([ship_vel, object_zero_vec], dim=1),
            ObsKey.ATT: torch.cat([ship_att, object_zero_vec], dim=1),
            ObsKey.ANG_VEL: torch.cat([ship_ang, object_zero_scalar], dim=1),
            ObsKey.HEALTH: torch.cat([ship_health, object_zero_scalar], dim=1),
            ObsKey.POWER: torch.cat([ship_power, object_zero_scalar], dim=1),
            ObsKey.COOLDOWN: torch.cat([ship_cooldown, object_zero_scalar], dim=1),
            ObsKey.TEAM_ID: torch.cat([state.ship_team_id, object_team], dim=1),
            ObsKey.ALIVE: torch.cat([observed_alive, object_alive], dim=1),
            ObsKey.VISIBLE: torch.cat([visible_ships, object_alive], dim=1),
            ObsKey.OBJECT_TYPE: torch.cat([ship_type, object_type], dim=1),
            ObsKey.ZONE_ROLE: torch.cat([ship_no_zone, object_zone_role], dim=1),
            ObsKey.PREVIOUS_ACTION: torch.cat([ship_prev_action, object_prev_action], dim=1),
            ObsKey.RADIUS: torch.cat([buffers.ship_radius, object_radius], dim=1),
            ObsKey.LOCAL_LOG_INDEX: torch.cat([ship_local_log_index, object_zero_scalar], dim=1),
            ObsKey.LOCAL_INDEX_GRADIENT: torch.cat([ship_index_gradient, object_zero_vec], dim=1),
            ObsKey.FIELD_TRANSITION_WIDTH: torch.cat(
                [ship_zero, object_scalar(field=state.field_transition_width.unsqueeze(-1))],
                dim=1,
            ),
            ObsKey.FIELD_TARGET_LOG_INDEX: torch.cat(
                [ship_zero, object_scalar(field=field_target)], dim=1
            ),
            ObsKey.FIELD_DAMAGE: torch.cat(
                [ship_zero, object_scalar(field=field_damage)], dim=1
            ),
            ObsKey.CAPTURE_PROGRESS: torch.cat(
                [ship_zero, object_scalar(zone=state.zone_capture_progress.unsqueeze(-1))], dim=1
            ),
            ObsKey.CAPTURE_DIRECTION: torch.cat(
                [ship_zero, object_scalar(zone=state.zone_capture_direction.float().unsqueeze(-1))],
                dim=1,
            ),
            ObsKey.FRONT_POSITION: torch.cat(
                [ship_zero, object_scalar(boundary=state.front_position.float()[:, None, None])],
                dim=1,
            ),
            ObsKey.FRONT_WIN_THRESHOLD: torch.cat(
                [
                    ship_zero,
                    object_scalar(boundary=state.front_win_threshold.float()[:, None, None]),
                ],
                dim=1,
            ),
            ObsKey.TIME_REMAINING: torch.cat(
                [ship_zero, object_scalar(boundary=remaining[:, None, None])], dim=1
            ),
            ObsKey.GAME_MODE: torch.cat(
                [
                    ship_zero,
                    object_scalar(
                        boundary=torch.ones((batch, 1, 1), device=state.device)
                        if has_frontline
                        else None
                    ),
                ],
                dim=1,
            ),
        },
    )
    return _mask_hidden_ships(observation, visible_ships, state.max_ships)


def _mask_hidden_ships(
    observation: YemongObservation,
    visible_ships: torch.Tensor,
    num_ships: int,
) -> YemongObservation:
    """Zero every hidden ship channel while retaining explicit false masks."""

    data = dict(observation.data)
    for key, value in tuple(data.items()):
        if key == ObsKey.VISIBLE:
            continue
        ship_value = (
            value[..., :num_ships]
            if key in _TOKEN_LAST_KEYS
            else value[..., :num_ships, :]
        )
        mask = visible_ships
        while mask.dim() < ship_value.dim():
            mask = mask.unsqueeze(-1)
        masked = torch.where(mask, ship_value, torch.zeros_like(ship_value))
        value = value.clone()
        if key in _TOKEN_LAST_KEYS:
            value[..., :num_ships] = masked
        else:
            value[..., :num_ships, :] = masked
        data[key] = value
    return YemongObservation(data=data, bullets=observation.bullets)


def perceived_observation_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    buffers: ObservationBuffers | None = None,
    include_bullets: bool = False,
) -> tuple[YemongObservation, TeamVisibility]:
    """Build independently masked team observations and return team 0 as root."""

    visibility = team_visibility_from_state(state, ship_config, env_config)
    views = [
        observation_from_state(
            state,
            ship_config,
            buffers,
            include_bullets,
            ship_visibility=visibility.ship[:, team],
            bullet_visibility=visibility.bullet[:, team],
            perspective_team=team,
        )
        for team in (0, 1)
    ]
    return YemongObservation(
        data=views[0].data,
        bullets=views[0].bullets,
        team1_data=views[1].data,
        team1_bullets=views[1].bullets,
    ), visibility
