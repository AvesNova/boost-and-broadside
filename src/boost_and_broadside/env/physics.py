"""GPU-vectorized ship and bullet physics.

All functions operate on TensorState in-place and return the mutated state.
No Python loops over batch or ship dimensions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import EPS, PowerActions, ShootActions
from boost_and_broadside.env.state import TensorState


# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------

# Keyed by (config, device string). Building the tables allocates four tensors
# from Python lists (host→device copies), so they must not be rebuilt on the
# per-step hot path.
_LOOKUP_TABLE_CACHE: dict = {}


def _get_lookup_tables(
    config: ShipConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached per-action physics lookup tensors for (config, device)."""
    key = (config, str(device))
    tables = _LOOKUP_TABLE_CACHE.get(key)
    if tables is None:
        tables = _build_lookup_tables(config, device)
        _LOOKUP_TABLE_CACHE[key] = tables
    return tables


def _build_lookup_tables(
    config: ShipConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-action physics lookup tensors from config.

    Returns:
        (thrust, turn_offset, drag_coeff, lift_coeff) — all float32.
    """
    thrust_table = torch.tensor(
        [config.base_thrust, config.boost_thrust, config.reverse_thrust],
        device=device,
        dtype=torch.float32,
    )
    turn_offset_table = torch.tensor(
        [
            0.0,
            -config.normal_turn_angle,
            config.normal_turn_angle,
            -config.sharp_turn_angle,
            config.sharp_turn_angle,
            0.0,
            0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    drag_coeff_table = torch.tensor(
        [
            config.no_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
            config.normal_turn_drag_coeff,
            config.sharp_turn_drag_coeff,
        ],
        device=device,
        dtype=torch.float32,
    )
    lift_coeff_table = torch.tensor(
        [
            0.0,
            -config.normal_turn_lift_coeff,
            config.normal_turn_lift_coeff,
            -config.sharp_turn_lift_coeff,
            config.sharp_turn_lift_coeff,
            0.0,
            0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    return (
        thrust_table,
        turn_offset_table,
        drag_coeff_table,
        lift_coeff_table,
    )


# ---------------------------------------------------------------------------
# Kinematics update
# ---------------------------------------------------------------------------


def _update_kinematics(
    state: TensorState,
    actions: torch.Tensor,
    config: ShipConfig,
    tables: tuple[torch.Tensor, ...],
) -> TensorState:
    """Update power, attitude, velocity, and position for all ships.

    GPU kernel: kept together for performance — splitting would force extra tensor
    allocations and destroy cache locality.
    """
    device = state.device
    (
        thrust_table,
        turn_offset_table,
        drag_coeff_table,
        lift_coeff_table,
    ) = tables

    power_action = actions[..., 0].long()  # (B, N)
    turn_action = actions[..., 1].long()  # (B, N)

    thrust_mag = thrust_table[power_action]  # (B, N)
    turn_offset = turn_offset_table[turn_action]  # (B, N)
    drag_coeff = drag_coeff_table[turn_action]  # (B, N)
    lift_coeff = lift_coeff_table[turn_action]  # (B, N)

    # Speed from current velocity — used for power drain and forces below
    speed = state.ship_vel.abs()  # (B, N)

    # Stall: below min_speed, lose turning authority and reverse thrust
    stalled = speed < config.min_speed
    turn_offset = torch.where(stalled, torch.zeros_like(turn_offset), turn_offset)
    lift_coeff  = torch.where(stalled, torch.zeros_like(lift_coeff),  lift_coeff)


    # Ships with no power can't thrust
    thrust_mag = thrust_mag * (state.ship_power > 0).float()

    # Power exchange: forward thrust drains, reverse thrust gains (equal and opposite).
    # Passive regen added on top regardless of action.
    power_delta = (
        -(thrust_mag / config.power_speed_constant) * speed
        + config.passive_power_gain
    ) * config.dt
    state.ship_power = torch.clamp(
        state.ship_power + power_delta, 0.0, config.max_power
    )

    # Attitude — align with velocity direction then apply turn rotation
    speed_safe = torch.clamp(speed, min=EPS)
    vel_dir = state.ship_vel / speed_safe
    stopped = speed < config.min_speed
    base_att = torch.where(stopped, state.ship_attitude, vel_dir)

    rotation = torch.polar(torch.ones_like(turn_offset), turn_offset)  # (B, N)
    state.ship_attitude = base_att * rotation
    state.ship_ang_vel = turn_offset / config.dt

    # Forces
    thrust_force = thrust_mag * state.ship_attitude  # (B, N) complex
    drag_force = -drag_coeff * speed * state.ship_vel  # (B, N) complex
    lift_force = (
        lift_coeff * speed * (state.ship_vel * 1j)
    )  # (B, N) complex  — perpendicular

    # Pairwise gravity (attracts fast ships toward each other)
    _, num_ships = state.ship_pos.shape
    world_w, world_h = config.world_size

    if config.gravity_factor == 0.0:
        gravity = torch.zeros_like(thrust_force)
    else:
        # (B, N_i, N_j) complex — wrapped difference i→j
        diff = state.ship_pos.unsqueeze(1) - state.ship_pos.unsqueeze(2)
        diff.real = (diff.real + world_w / 2) % world_w - world_w / 2
        diff.imag = (diff.imag + world_h / 2) % world_h - world_h / 2

        dist_sq = diff.real**2 + diff.imag**2  # (B, N, N)
        dist = torch.sqrt(dist_sq)

        def _symlog(x: torch.Tensor) -> torch.Tensor:
            return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        speed_i = speed.unsqueeze(2)  # (B, N, 1)
        speed_j = speed.unsqueeze(1)  # (B, 1, N)
        force_mag = (
            config.gravity_factor
            * config.gravity_eps
            * _symlog(speed_i * speed_j)
            / (dist_sq + config.gravity_eps)
        )  # (B, N, N)
        force_dir = diff / torch.clamp(dist, min=EPS)  # (B, N, N)
        force_vec = force_mag * force_dir  # (B, N, N) complex

        alive_mask = state.ship_alive.unsqueeze(2) & state.ship_alive.unsqueeze(
            1
        )  # (B, N, N)
        self_mask = torch.eye(num_ships, device=device, dtype=torch.bool).unsqueeze(0)
        force_vec = torch.where(
            alive_mask & ~self_mask, force_vec, torch.zeros_like(force_vec)
        )
        gravity = force_vec.sum(dim=2)  # (B, N) complex

    # Integrate
    total_force = thrust_force + drag_force + lift_force + gravity
    state.ship_vel = state.ship_vel + total_force * config.dt
    state.ship_pos = state.ship_pos + state.ship_vel * config.dt

    # Toroidal wrap
    state.ship_pos.real = state.ship_pos.real % world_w
    state.ship_pos.imag = state.ship_pos.imag % world_h

    # Prevent exactly-zero velocity (would break direction computations)
    new_speed = state.ship_vel.abs()
    too_slow = new_speed < EPS
    min_vel = EPS * state.ship_attitude
    state.ship_vel = torch.where(too_slow, min_vel, state.ship_vel)

    return state


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------


def _handle_shooting(
    state: TensorState, shoot_action: torch.Tensor, config: ShipConfig
) -> TensorState:
    """Manage cooldowns and spawn bullets for ships that fire.

    Fully branchless: bullet spawns are written through a one-hot mask on the
    ring-buffer cursor instead of nonzero/gather, so no host-device sync occurs
    (this runs every step of every rollout).
    """
    if state.max_bullets == 0:
        state.ship_is_shooting = torch.zeros_like(state.ship_is_shooting)
        return state

    state.ship_cooldown = (state.ship_cooldown - config.dt).clamp(min=0.0)

    can_shoot = (
        (shoot_action == ShootActions.SHOOT)
        & (state.ship_cooldown <= 0)
        & (state.ship_power >= config.bullet_energy_cost)
        & state.ship_alive
    )  # (B, N) bool
    state.ship_is_shooting = can_shoot

    state.ship_power = torch.where(
        can_shoot, state.ship_power - config.bullet_energy_cost, state.ship_power
    )
    state.ship_cooldown = torch.where(
        can_shoot,
        config.firing_cooldown,
        state.ship_cooldown,
    )

    # One-hot write mask: each shooter's cursor slot, masked to shooters only.
    K = state.max_bullets
    slot_onehot = F.one_hot(state.bullet_cursor, K).bool() & can_shoot.unsqueeze(-1)  # (B, N, K)

    base_vel = state.ship_vel + config.bullet_speed * state.ship_attitude  # (B, N)
    noise = torch.complex(
        torch.randn_like(base_vel.real) * config.bullet_spread,
        torch.randn_like(base_vel.real) * config.bullet_spread,
    )
    spawn_vel = base_vel + noise

    state.bullet_pos = torch.where(slot_onehot, state.ship_pos.unsqueeze(-1), state.bullet_pos)
    state.bullet_vel = torch.where(slot_onehot, spawn_vel.unsqueeze(-1), state.bullet_vel)
    state.bullet_time = torch.where(slot_onehot, config.bullet_lifetime, state.bullet_time)
    state.bullet_active = state.bullet_active | slot_onehot
    state.bullet_cursor = torch.where(
        can_shoot, (state.bullet_cursor + 1) % K, state.bullet_cursor
    )

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_ships(
    state: TensorState, actions: torch.Tensor, config: ShipConfig
) -> TensorState:
    """Apply one physics timestep: kinematics + shooting.

    Args:
        state: Current environment state (mutated in-place).
        actions: (B, N, 3) int tensor — [power_action, turn_action, shoot_action].
        config: Physics configuration.

    Returns:
        The mutated state.
    """
    tables = _get_lookup_tables(config, state.device)
    state = _update_kinematics(state, actions, config, tables)
    state = _handle_shooting(state, actions[..., 2].long(), config)
    return state


def update_bullets(state: TensorState, config: ShipConfig) -> TensorState:
    """Advance all active bullets and expire those whose lifetime ran out.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.

    Returns:
        The mutated state.
    """
    world_w, world_h = config.world_size

    state.bullet_time = state.bullet_time - config.dt
    state.bullet_active = state.bullet_active & (state.bullet_time > 0)
    state.bullet_pos = state.bullet_pos + state.bullet_vel * config.dt

    state.bullet_pos.real = state.bullet_pos.real % world_w
    state.bullet_pos.imag = state.bullet_pos.imag % world_h

    return state


def resolve_collisions(
    state: TensorState, config: ShipConfig
) -> Tuple[TensorState, torch.Tensor]:
    """Detect bullet-ship collisions, apply damage, and check game-over.

    Args:
        state: Current state (mutated in-place).
        config: Physics configuration.

    Returns:
        (state, dones) where dones is a (B,) bool tensor.
    """
    state = _apply_combat_damage(state, config)
    dones = _check_game_over(state)
    return state, dones


def _apply_combat_damage(state: TensorState, config: ShipConfig) -> TensorState:
    """Vectorized bullet-ship hit detection and damage application.

    Dense over all bullet slots (active and inactive): inactive slots are
    masked out of the hit test rather than compacted with nonzero/gather.
    This keeps every tensor shape static and avoids the host-device syncs
    that dynamic-shape indexing forces on the per-step hot path.

    Also fills state.damage_matrix (B, N_shooter, N_target) for this step and
    accumulates into state.cumulative_damage_matrix for episode-level attribution.
    """
    batch_size, num_ships = state.ship_pos.shape
    num_bullets = state.max_bullets
    device = state.device
    world_w, world_h = config.world_size

    # Reset per-step attribution; cumulative is carried forward across steps.
    state.damage_matrix.zero_()

    if num_bullets == 0:
        return state

    NB = num_ships * num_bullets
    flat_bullet_active = state.bullet_active.view(batch_size, NB)  # (B, NB)
    flat_bullet_pos = state.bullet_pos.view(batch_size, NB)        # (B, NB)
    flat_bullet_vel = state.bullet_vel.view(batch_size, NB)        # (B, NB)

    # Wrapped vector from bullet to ship: (B, NB, N)
    diff_r = state.ship_pos.real.unsqueeze(1) - flat_bullet_pos.real.unsqueeze(2)
    diff_i = state.ship_pos.imag.unsqueeze(1) - flat_bullet_pos.imag.unsqueeze(2)
    diff_r = (diff_r + world_w / 2) % world_w - world_w / 2
    diff_i = (diff_i + world_h / 2) % world_h - world_h / 2

    dist_sq = diff_r**2 + diff_i**2  # (B, NB, N)

    # Exclude own bullets: bullet slot j belongs to ship j // num_bullets.
    owner_idx = torch.arange(NB, device=device) // num_bullets  # (NB,)
    target_idx = torch.arange(num_ships, device=device)          # (N,)
    not_own_bullet = owner_idx.unsqueeze(1) != target_idx.unsqueeze(0)  # (NB, N)

    valid_hit = (
        (dist_sq < config.collision_radius**2)
        & state.ship_alive.unsqueeze(1)
        & flat_bullet_active.unsqueeze(2)
        & not_own_bullet.unsqueeze(0)
    )  # (B, NB, N)

    # Angle-scaled damage: head-on hits deal full damage, side hits reduced.
    # Inactive slots produce garbage angles but are zeroed by valid_hit below.
    hit_angles = torch.angle(
        -flat_bullet_vel.unsqueeze(2) * torch.conj(state.ship_attitude.unsqueeze(1))
    )  # (B, NB, N)
    damage_scale = 1.0 - (1.0 - config.bullet_min_damage_frac) * torch.exp(
        -(hit_angles**2) * 4.0 / torch.pi
    )
    damage_per_hit = damage_scale * valid_hit.float() * config.bullet_damage  # (B, NB, N)

    # Total damage received by each ship, and per-shooter attribution
    total_damage = damage_per_hit.sum(dim=1)  # (B, N)
    per_shooter = damage_per_hit.view(batch_size, num_ships, num_bullets, num_ships).sum(
        dim=2
    )  # (B, N_shooter, N_target)
    state.damage_matrix.copy_(per_shooter)
    state.cumulative_damage_matrix += per_shooter

    # Apply health reduction. Death is monotone (alive &= health > 0) so ships
    # flagged dead by other systems are never resurrected by the alive refresh.
    state.ship_health = state.ship_health - total_damage
    state.ship_alive = state.ship_alive & (state.ship_health > 0)
    state.ship_health = torch.clamp(state.ship_health, min=0.0)

    # Deactivate bullets that connected
    hit_any_ship = valid_hit.any(dim=2)  # (B, NB)
    state.bullet_active = (flat_bullet_active & ~hit_any_ship).view(
        batch_size, num_ships, num_bullets
    )

    return state


def _check_game_over(state: TensorState) -> torch.Tensor:
    """Return (B,) done mask — True when a team that exists is fully eliminated."""
    team0_alive = ((state.ship_team_id == 0) & state.ship_alive).sum(dim=1)  # (B,)
    team1_alive = ((state.ship_team_id == 1) & state.ship_alive).sum(dim=1)  # (B,)
    team0_exists = (state.ship_team_id == 0).any(dim=1)  # (B,)
    team1_exists = (state.ship_team_id == 1).any(dim=1)  # (B,)
    return (team0_exists & (team0_alive == 0)) | (team1_exists & (team1_alive == 0))


def resolve_obstacle_collisions(state: TensorState, config: ShipConfig) -> TensorState:
    """Detect ship-obstacle and bullet-obstacle collisions.

    Ship-obstacle: instant kill (health → 0, alive → False, ship_hit_obstacle → True).
    Bullet-obstacle: bullet deactivated on contact.

    GPU kernel: no Python loops over ships, bullets, or obstacles.
    """
    batch_size, num_ships = state.ship_pos.shape
    num_obstacles = state.num_obstacles
    world_w, world_h = config.world_size

    if num_obstacles == 0:
        return state

    # ----- Ship-obstacle collision -----
    # Toroidal wrapped differences: (B, N, M)
    diff_r = state.ship_pos.real.unsqueeze(2) - state.obstacle_pos.real.unsqueeze(1)
    diff_i = state.ship_pos.imag.unsqueeze(2) - state.obstacle_pos.imag.unsqueeze(1)
    diff_r = (diff_r + world_w / 2) % world_w - world_w / 2
    diff_i = (diff_i + world_h / 2) % world_h - world_h / 2
    dist_sq = diff_r**2 + diff_i**2  # (B, N, M)

    # Combined hitbox: ship_collision_radius + obstacle_radius[m]
    hit_r = config.obstacle_collision_radius + state.obstacle_radius  # (B, M)
    hit = dist_sq < hit_r.unsqueeze(1) ** 2  # (B, N, M)
    hit = hit & state.ship_alive.unsqueeze(2)  # only alive ships
    any_hit = hit.any(dim=2)  # (B, N)

    state.ship_hit_obstacle = any_hit
    state.ship_health = torch.where(any_hit, torch.zeros_like(state.ship_health), state.ship_health)
    state.ship_alive = state.ship_alive & (state.ship_health > 0)

    # ----- Bullet-obstacle collision -----
    # Dense over all bullet slots — inactive slots are masked at the end,
    # keeping shapes static so no host-device sync occurs on the hot path.
    if state.max_bullets == 0:
        return state

    num_bullets = state.max_bullets
    NB = num_ships * num_bullets
    flat_bullet_pos = state.bullet_pos.view(batch_size, NB)      # (B, NB)
    flat_bullet_active = state.bullet_active.view(batch_size, NB)  # (B, NB)

    bdiff_r = flat_bullet_pos.real.unsqueeze(2) - state.obstacle_pos.real.unsqueeze(1)  # (B, NB, M)
    bdiff_i = flat_bullet_pos.imag.unsqueeze(2) - state.obstacle_pos.imag.unsqueeze(1)
    bdiff_r = (bdiff_r + world_w / 2) % world_w - world_w / 2
    bdiff_i = (bdiff_i + world_h / 2) % world_h - world_h / 2
    bdist_sq = bdiff_r**2 + bdiff_i**2  # (B, NB, M)

    bullet_hit_r = config.bullet_collision_radius + state.obstacle_radius  # (B, M)
    bullet_hit_obs = bdist_sq < bullet_hit_r.unsqueeze(1) ** 2  # (B, NB, M)
    bullet_hit_any = bullet_hit_obs.any(dim=2)  # (B, NB)

    state.bullet_active = (flat_bullet_active & ~bullet_hit_any).view(
        batch_size, num_ships, num_bullets
    )

    return state


