"""GPU-vectorized ship and bullet physics.

All functions operate on TensorState in-place and return the mutated state.
No Python loops over batch or ship dimensions.
"""

import torch
from typing import Tuple

from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import ShootActions


# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------


def _get_lookup_tables(
    config: ShipConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-action physics lookup tensors from config.

    Returns:
        (thrust, power_gain, turn_offset, drag_coeff, lift_coeff) — all float32.
    """
    thrust_table = torch.tensor(
        [config.base_thrust, config.boost_thrust, config.reverse_thrust],
        device=device,
        dtype=torch.float32,
    )
    power_gain_table = torch.tensor(
        [config.base_power_gain, config.boost_power_gain, config.reverse_power_gain],
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
        power_gain_table,
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
        power_gain_table,
        turn_offset_table,
        drag_coeff_table,
        lift_coeff_table,
    ) = tables

    power_action = actions[..., 0].long()  # (B, N)
    turn_action = actions[..., 1].long()  # (B, N)

    thrust_mag = thrust_table[power_action]  # (B, N)
    power_gain = power_gain_table[power_action]  # (B, N)
    turn_offset = turn_offset_table[turn_action]  # (B, N)
    drag_coeff = drag_coeff_table[turn_action]  # (B, N)
    lift_coeff = lift_coeff_table[turn_action]  # (B, N)

    # Power update — clamp to [0, max]
    state.ship_power = torch.clamp(
        state.ship_power + power_gain * config.dt, 0.0, config.max_power
    )
    # Ships with no power can't thrust
    thrust_mag = thrust_mag * (state.ship_power > 0).float()

    # Attitude — align with velocity direction then apply turn rotation
    speed = state.ship_vel.abs()  # (B, N)
    speed_safe = torch.clamp(speed, min=1e-6)
    vel_dir = state.ship_vel / speed_safe
    stopped = speed < 1e-6
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
    force_dir = diff / torch.clamp(dist, min=1e-6)  # (B, N, N)
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
    too_slow = new_speed < 1e-6
    min_vel = (
        torch.tensor(1e-6, device=device, dtype=torch.complex64) * state.ship_attitude
    )
    state.ship_vel = torch.where(too_slow, min_vel, state.ship_vel)

    return state


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------


def _handle_shooting(
    state: TensorState, shoot_action: torch.Tensor, config: ShipConfig
) -> TensorState:
    """Manage cooldowns and spawn bullets for ships that fire."""
    device = state.device

    state.ship_cooldown = state.ship_cooldown - config.dt

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
        torch.tensor(config.firing_cooldown, device=device),
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
    flat_bullet_pos = state.bullet_pos.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)
    flat_bullet_active = state.bullet_active.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)

    # Wrapped vector from bullet to ship
    diff = state.ship_pos.unsqueeze(2) - flat_bullet_pos.unsqueeze(
        1
    )  # (B, N_ship, N*K)
    diff.real = (diff.real + world_w / 2) % world_w - world_w / 2
    diff.imag = (diff.imag + world_h / 2) % world_h - world_h / 2

    dist_sq = diff.real**2 + diff.imag**2  # (B, N_ship, N*K)

    # (B, N_ship, N*K) — raw hit candidates
    hit_mask = (
        (dist_sq < config.collision_radius**2)
        & flat_bullet_active.unsqueeze(1)
        & state.ship_alive.unsqueeze(2)
    )

    # Exclude own bullets — bullet j belongs to ship j // num_bullets
    bullet_owner = (
        torch.arange(num_ships * num_bullets, device=device) // num_bullets
    )  # (N*K,)
    ship_idx = torch.arange(num_ships, device=device)  # (N,)
    own_bullet = bullet_owner.view(1, 1, -1) == ship_idx.view(1, -1, 1)  # (1, N, N*K)
    valid_hit = hit_mask & ~own_bullet  # (B, N, N*K)

    if not valid_hit.any():
        return state

    # Angle-scaled damage: head-on hits deal full damage, side hits reduced
    flat_bullet_vel = state.bullet_vel.view(
        batch_size, num_ships * num_bullets
    )  # (B, N*K)
    hit_angles = torch.angle(
        -flat_bullet_vel.unsqueeze(1) * torch.conj(state.ship_attitude.unsqueeze(2))
    )  # (B, N, N*K)
    damage_scale = 1.0 - (1.0 - config.bullet_min_damage_frac) * torch.exp(
        -(hit_angles**2) * 4.0 / torch.pi
    )
    damage_per_hit = damage_scale * valid_hit.float() * config.bullet_damage
    total_damage = damage_per_hit.sum(dim=2)  # (B, N)

    # Build per-shooter attribution before summing.
    # damage_per_hit: (B, N_target, N_shooter*K) — bullets laid out as
    # [ship_0_slot_0, ..., ship_0_slot_K-1, ship_1_slot_0, ...], so reshaping
    # to (B, N_target, N_shooter, K) groups slots by shooter.
    dm = (
        damage_per_hit.view(batch_size, num_ships, num_ships, num_bullets)
        .sum(dim=3)  # (B, N_target, N_shooter)
        .permute(0, 2, 1)  # (B, N_shooter, N_target)
    )
    state.damage_matrix.copy_(dm)
    state.cumulative_damage_matrix.add_(dm)

    state.ship_health = state.ship_health - total_damage
    state.ship_alive = state.ship_health > 0
    state.ship_health = torch.clamp(state.ship_health, min=0.0)

    # Deactivate bullets that connected
    hit_any_ship = valid_hit.any(dim=1)  # (B, N*K)
    state.bullet_active.view(batch_size, -1)[hit_any_ship] = False

    return state


def _check_game_over(state: TensorState) -> torch.Tensor:
    """Return (B,) done mask — True when either team is fully eliminated."""
    team0_alive = ((state.ship_team_id == 0) & state.ship_alive).sum(dim=1)  # (B,)
    team1_alive = ((state.ship_team_id == 1) & state.ship_alive).sum(dim=1)  # (B,)
    return (team0_alive == 0) | (team1_alive == 0)
