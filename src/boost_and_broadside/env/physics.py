"""GPU-vectorized ship and bullet physics.

All functions operate on TensorState in-place and return the mutated state.
No Python loops over batch or ship dimensions.
"""

import torch
from typing import Tuple

from boost_and_broadside.env.state import TensorState

_EPS = 1e-6  # division safety guard for direction normalization
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, ShootActions


# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------


def _get_lookup_tables(
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
    speed_safe = torch.clamp(speed, min=_EPS)
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
        force_dir = diff / torch.clamp(dist, min=_EPS)  # (B, N, N)
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
    too_slow = new_speed < _EPS
    min_vel = _EPS * state.ship_attitude
    state.ship_vel = torch.where(too_slow, min_vel, state.ship_vel)

    return state


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------


def _handle_shooting(
    state: TensorState, shoot_action: torch.Tensor, config: ShipConfig
) -> TensorState:
    """Manage cooldowns and spawn bullets for ships that fire."""
    if state.max_bullets == 0:
        state.ship_is_shooting = torch.zeros_like(state.ship_is_shooting)
        return state

    device = state.device

    state.ship_cooldown = (state.ship_cooldown - config.dt).clamp(min=0.0)

    can_shoot = (
        (shoot_action == ShootActions.SHOOT)
        & (state.ship_cooldown <= 0)
        & (state.ship_power >= config.bullet_energy_cost)
        & state.ship_alive
    )  # (B, N) bool
    state.ship_is_shooting = can_shoot

    if not can_shoot.any():
        return state

    state.ship_power = torch.where(
        can_shoot, state.ship_power - config.bullet_energy_cost, state.ship_power
    )
    state.ship_cooldown = torch.where(
        can_shoot,
        config.firing_cooldown,
        state.ship_cooldown,
    )

    batch_idx, ship_idx = torch.nonzero(can_shoot, as_tuple=True)
    slots = state.bullet_cursor[batch_idx, ship_idx]  # write positions
    spawn_pos = state.ship_pos[batch_idx, ship_idx]
    att = state.ship_attitude[batch_idx, ship_idx]
    vel = state.ship_vel[batch_idx, ship_idx]

    base_vel = vel + config.bullet_speed * att
    noise = torch.complex(
        torch.randn_like(base_vel.real) * config.bullet_spread,
        torch.randn_like(base_vel.real) * config.bullet_spread,
    )

    state.bullet_pos[batch_idx, ship_idx, slots] = spawn_pos
    state.bullet_vel[batch_idx, ship_idx, slots] = base_vel + noise
    state.bullet_time[batch_idx, ship_idx, slots] = config.bullet_lifetime
    state.bullet_active[batch_idx, ship_idx, slots] = True
    state.bullet_cursor[batch_idx, ship_idx] = (slots + 1) % state.max_bullets

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

    GPU kernel: kept together for performance.
    Also fills state.damage_matrix (B, N_shooter, N_target) for this step and
    accumulates into state.cumulative_damage_matrix for episode-level attribution.
    """
    batch_size, num_ships = state.ship_pos.shape
    num_bullets = state.max_bullets
    device = state.device
    world_w, world_h = config.world_size

    # Reset per-step attribution; cumulative is carried forward across steps.
    state.damage_matrix.zero_()

    # Flatten bullet arrays over the ship dimension for broadcasting
    flat_bullet_active = state.bullet_active.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)

    # 1. Fast-path: If no bullets are active in the entire batch, return immediately!
    if not flat_bullet_active.any():
        return state

    flat_bullet_pos = state.bullet_pos.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)
    flat_bullet_vel = state.bullet_vel.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)

    # Get indices of active bullets
    env_idx, flat_b_idx = torch.nonzero(flat_bullet_active, as_tuple=True)

    # Slice attributes of active bullets
    act_pos = flat_bullet_pos[flat_bullet_active]  # (num_active,) complex
    act_vel = flat_bullet_vel[flat_bullet_active]  # (num_active,) complex
    act_owner = flat_b_idx // num_bullets          # (num_active,)

    # Slice target ship positions and statuses in matching environments
    ships_pos = state.ship_pos[env_idx]            # (num_active, num_ships) complex
    ships_alive = state.ship_alive[env_idx]        # (num_active, num_ships) bool
    ships_att = state.ship_attitude[env_idx]        # (num_active, num_ships) complex

    # Wrapped vector from bullet to ship: (num_active, num_ships)
    diff_r = ships_pos.real - act_pos.real.unsqueeze(1)
    diff_i = ships_pos.imag - act_pos.imag.unsqueeze(1)
    diff_r = (diff_r + world_w / 2) % world_w - world_w / 2
    diff_i = (diff_i + world_h / 2) % world_h - world_h / 2

    dist_sq = diff_r**2 + diff_i**2  # (num_active, num_ships)

    # Combined hitbox radius and raw candidate hits
    hit_mask = (dist_sq < config.collision_radius**2) & ships_alive  # (num_active, num_ships)

    # Exclude own bullets
    target_idx = torch.arange(num_ships, device=device).unsqueeze(0)  # (1, num_ships)
    not_own_bullet = target_idx != act_owner.unsqueeze(1)             # (num_active, num_ships)
    valid_hit = hit_mask & not_own_bullet                              # (num_active, num_ships)

    if not valid_hit.any():
        return state

    # Angle-scaled damage: head-on hits deal full damage, side hits reduced
    hit_angles = torch.angle(
        -act_vel.unsqueeze(1) * torch.conj(ships_att)
    )  # (num_active, num_ships)
    damage_scale = 1.0 - (1.0 - config.bullet_min_damage_frac) * torch.exp(
        -(hit_angles**2) * 4.0 / torch.pi
    )
    damage_per_hit = damage_scale * valid_hit.float() * config.bullet_damage  # (num_active, num_ships)

    # Sum total damage received by each ship
    total_damage = torch.zeros((batch_size, num_ships), device=device)  # (B, N)
    total_damage.index_add_(0, env_idx, damage_per_hit)

    # Build per-shooter attribution and sum
    target_idx_expanded = target_idx.expand(len(env_idx), num_ships)  # (num_active, num_ships)
    flat_idx = (
        env_idx.unsqueeze(1) * (num_ships * num_ships)
        + act_owner.unsqueeze(1) * num_ships
        + target_idx_expanded
    )  # (num_active, num_ships)

    state.damage_matrix.view(-1).index_add_(0, flat_idx.view(-1), damage_per_hit.view(-1))
    state.cumulative_damage_matrix.view(-1).index_add_(0, flat_idx.view(-1), damage_per_hit.view(-1))

    # Apply health reduction
    state.ship_health = state.ship_health - total_damage
    state.ship_alive = state.ship_health > 0
    state.ship_health = torch.clamp(state.ship_health, min=0.0)

    # Deactivate bullets that connected
    hit_any_ship = valid_hit.any(dim=1)  # (num_active,)
    if hit_any_ship.any():
        hitting_env = env_idx[hit_any_ship]
        hitting_flat_b = flat_b_idx[hit_any_ship]
        flat_bullet_active[hitting_env, hitting_flat_b] = False

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
    state.ship_alive = state.ship_health > 0

    # ----- Bullet-obstacle collision -----
    if state.max_bullets == 0:
        return state

    num_bullets = state.max_bullets
    flat_bullet_pos = state.bullet_pos.view(batch_size, num_ships * num_bullets)   # (B, N*K)
    flat_bullet_active = state.bullet_active.view(batch_size, num_ships * num_bullets)  # (B, N*K)

    active_mask = flat_bullet_active
    if active_mask.any():
        active_indices = torch.nonzero(active_mask, as_tuple=True)
        env_idx, flat_b_idx = active_indices
        
        act_pos = flat_bullet_pos[active_mask]  # (num_active,)
        act_obs_pos = state.obstacle_pos[env_idx]  # (num_active, M)
        act_obs_radius = state.obstacle_radius[env_idx]  # (num_active, M)
        
        bdiff_r = act_pos.real.unsqueeze(1) - act_obs_pos.real
        bdiff_i = act_pos.imag.unsqueeze(1) - act_obs_pos.imag
        bdiff_r = (bdiff_r + world_w / 2) % world_w - world_w / 2
        bdiff_i = (bdiff_i + world_h / 2) % world_h - world_h / 2
        bdist_sq = bdiff_r**2 + bdiff_i**2  # (num_active, M)
        
        bullet_hit_r = config.bullet_collision_radius + act_obs_radius  # (num_active, M)
        bullet_hit_obs = bdist_sq < bullet_hit_r ** 2  # (num_active, M)
        bullet_hit_any = bullet_hit_obs.any(dim=1)  # (num_active,)
        
        if bullet_hit_any.any():
            hitting_env_idx = env_idx[bullet_hit_any]
            hitting_flat_b_idx = flat_b_idx[bullet_hit_any]
            flat_bullet_active[hitting_env_idx, hitting_flat_b_idx] = False

    return state


