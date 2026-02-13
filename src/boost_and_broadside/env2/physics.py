
import torch
from typing import Tuple

from .state import TensorState
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.core.constants import ShootActions, RewardConstants


def _get_lookup_tables(config: ShipConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates physics lookup tables based on the configuration.
    
    Args:
        config: Ship configuration.
        device: Torch device.
        
    Returns:
        A tuple of tensors: (thrust, power_gain, turn_offset, drag_coeff, lift_coeff).
    """
    thrust_table = torch.tensor([
        config.base_thrust,      # COAST
        config.boost_thrust,     # BOOST
        config.reverse_thrust    # REVERSE
    ], device=device, dtype=torch.float32)
    
    power_gain_table = torch.tensor([
        config.base_power_gain,     # COAST
        config.boost_power_gain,    # BOOST
        config.reverse_power_gain   # REVERSE
    ], device=device, dtype=torch.float32)

    turn_offset_table = torch.tensor([
        0.0,
        -config.normal_turn_angle,
        config.normal_turn_angle,
        -config.sharp_turn_angle,
        config.sharp_turn_angle,
        0.0,
        0.0
    ], device=device, dtype=torch.float32)
    
    drag_coeff_table = torch.tensor([
        config.no_turn_drag_coeff,      # STRAIGHT
        config.normal_turn_drag_coeff,  # LEFT
        config.normal_turn_drag_coeff,  # RIGHT
        config.sharp_turn_drag_coeff,   # S_LEFT
        config.sharp_turn_drag_coeff,   # S_RIGHT
        config.normal_turn_drag_coeff,  # AIR
        config.sharp_turn_drag_coeff    # S_AIR
    ], device=device, dtype=torch.float32)
    
    lift_coeff_table = torch.tensor([
        0.0,                            # STRAIGHT
        -config.normal_turn_lift_coeff, # LEFT
        config.normal_turn_lift_coeff,  # RIGHT
        -config.sharp_turn_lift_coeff,  # S_LEFT
        config.sharp_turn_lift_coeff,   # S_RIGHT
        0.0,                            # AIR
        0.0                             # S_AIR
    ], device=device, dtype=torch.float32)

    return thrust_table, power_gain_table, turn_offset_table, drag_coeff_table, lift_coeff_table


def _update_kinematics(state: TensorState, actions: torch.Tensor, config: ShipConfig, tables: Tuple[torch.Tensor, ...]) -> TensorState:
    """
    Updates ship power, attitude, and kinematics (position/velocity).
    
    Args:
        state: Current state.
        actions: Actions tensor.
        config: Ship configuration.
        tables: Physics lookup tables.
        
    Returns:
        Updated state.
    """
    device = state.device
    thrust_table, power_gain_table, turn_offset_table, drag_coeff_table, lift_coeff_table = tables

    power_action = actions[..., 0].long()
    turn_action = actions[..., 1].long()

    # Apply Actions lookup for ALIVE ships
    thrust_mag = thrust_table[power_action]
    power_gain = power_gain_table[power_action]
    
    turn_offset = turn_offset_table[turn_action]
    drag_coeff = drag_coeff_table[turn_action]
    lift_coeff = lift_coeff_table[turn_action]
    
    # Update Power
    new_power = state.ship_power + power_gain * config.dt
    state.ship_power = torch.clamp(new_power, 0.0, config.max_power)
    
    has_power = state.ship_power > 0
    thrust_mag = torch.where(has_power, thrust_mag, torch.zeros_like(thrust_mag))
    
    # Update Attitude
    speed = state.ship_vel.abs()
    speed_safe = torch.maximum(speed, torch.tensor(1e-6, device=device))
    
    stopped = speed < 1e-6
    vel_dir = state.ship_vel / speed_safe
    base_attitude = torch.where(stopped, state.ship_attitude, vel_dir)
    
    rotation = torch.polar(torch.ones_like(turn_offset), turn_offset)
    state.ship_attitude = base_attitude * rotation
    
    state.ship_ang_vel = turn_offset / config.dt
    
    # Calculate Forces
    thrust_force = thrust_mag * state.ship_attitude
    drag_force = -drag_coeff * speed * state.ship_vel
    
    lift_vector = state.ship_vel * 1j
    lift_force = lift_coeff * speed * lift_vector
    
    total_force = thrust_force + drag_force + lift_force
    
    # Update Kinematics
    accel = total_force
    state.ship_vel = state.ship_vel + accel * config.dt
    state.ship_pos = state.ship_pos + state.ship_vel * config.dt
    
    # Wrap World
    world_width, world_height = config.world_size
    state.ship_pos.real = state.ship_pos.real % world_width
    state.ship_pos.imag = state.ship_pos.imag % world_height
    
    # Clamp velocity if negligible
    new_speed = state.ship_vel.abs()
    too_slow = new_speed < 1e-6
    clamped_vel = torch.tensor(1e-6, device=device, dtype=torch.complex64) * state.ship_attitude
    state.ship_vel = torch.where(too_slow, clamped_vel, state.ship_vel)

    return state


def _handle_shooting(state: TensorState, shoot_action: torch.Tensor, config: ShipConfig) -> TensorState:
    """
    Handles shooting logic, including cooldown management and bullet spawning.
    
    Args:
        state: Current state.
        shoot_action: Shoot action tensor.
        config: Ship configuration.
        
    Returns:
        Updated state.
    """
    device = state.device
    
    state.ship_cooldown = state.ship_cooldown - config.dt
    
    can_shoot = (
        (shoot_action == ShootActions.SHOOT) & 
        (state.ship_cooldown <= 0) & 
        (state.ship_power >= config.bullet_energy_cost) & 
        state.ship_alive
    )
    
    state.ship_is_shooting = can_shoot

    if not can_shoot.any():
        return state

    # Deduct power and set cooldown
    state.ship_power = torch.where(
        can_shoot, 
        state.ship_power - config.bullet_energy_cost, 
        state.ship_power
    )
    state.ship_cooldown = torch.where(
        can_shoot, 
        torch.tensor(config.firing_cooldown, device=device), 
        state.ship_cooldown
    )
    
    batch_indices, ship_indices = torch.nonzero(can_shoot, as_tuple=True)
    
    bullet_slot_indices = state.bullet_cursor[batch_indices, ship_indices]
    spawn_pos = state.ship_pos[batch_indices, ship_indices]
    
    shooter_att = state.ship_attitude[batch_indices, ship_indices]
    shooter_vel = state.ship_vel[batch_indices, ship_indices]
    
    base_vel = shooter_vel + config.bullet_speed * shooter_att
    
    # Generate complex noise
    noise_real = torch.randn(base_vel.shape, device=device) * config.bullet_spread
    noise_imag = torch.randn(base_vel.shape, device=device) * config.bullet_spread
    noise = torch.complex(noise_real, noise_imag)
    
    final_vel = base_vel + noise
    
    state.bullet_pos[batch_indices, ship_indices, bullet_slot_indices] = spawn_pos
    state.bullet_vel[batch_indices, ship_indices, bullet_slot_indices] = final_vel
    state.bullet_time[batch_indices, ship_indices, bullet_slot_indices] = config.bullet_lifetime
    state.bullet_active[batch_indices, ship_indices, bullet_slot_indices] = True
    
    state.bullet_cursor[batch_indices, ship_indices] = (bullet_slot_indices + 1) % state.max_bullets
            
    return state


def update_ships(state: TensorState, actions: torch.Tensor, config: ShipConfig) -> TensorState:
    """
    Update ship physics based on actions.

    Args:
        state: The current state of the environment.
        actions: A tensor of shape (batch, num_ships, 3) containing [Power, Turn, Shoot] actions.
        config: The ship configuration.

    Returns:
        The updated state.
    """
    device = state.device
    tables = _get_lookup_tables(config, device)
    
    state = _update_kinematics(state, actions, config, tables)
    
    shoot_action = actions[..., 2].long()
    state = _handle_shooting(state, shoot_action, config)
    
    return state


def update_bullets(state: TensorState, config: ShipConfig) -> TensorState:
    """
    Updates bullet positions and lifetimes.
    
    Args:
        state: Current state.
        config: Configuration.
        
    Returns:
        Updated state.
    """
    dt = config.dt
    world_width, world_height = config.world_size
    
    state.bullet_time = state.bullet_time - dt
    state.bullet_active = state.bullet_active & (state.bullet_time > 0)
    
    # Update positions
    state.bullet_pos = state.bullet_pos + state.bullet_vel * dt
    
    # Wrap world
    state.bullet_pos.real = state.bullet_pos.real % world_width
    state.bullet_pos.imag = state.bullet_pos.imag % world_height
    
    return state


def _apply_combat_damage(state: TensorState, config: ShipConfig) -> Tuple[TensorState, torch.Tensor]:
    """
    Calculates collisions, applies damage.
    
    Args:
        state: Current state.
        config: Configuration.
        
    Returns:
        Tuple of (updated_state, hit_mask).
    """
    batch_size, num_ships = state.ship_pos.shape
    num_bullets_per_ship = state.max_bullets
    device = state.device
    world_width, world_height = config.world_size
    
    # Flatten bullets for easier broadcasting
    flat_bullets_pos = state.bullet_pos.view(batch_size, num_ships * num_bullets_per_ship)
    flat_bullets_active = state.bullet_active.view(batch_size, num_ships * num_bullets_per_ship)
    
    # Calculate wrapped difference between ships and bullets
    # diff: Vector from bullet to ship
    diff = state.ship_pos.unsqueeze(2) - flat_bullets_pos.unsqueeze(1) # (B, N_ships, N_total_bullets)
    diff.real = (diff.real + world_width/2) % world_width - world_width/2
    diff.imag = (diff.imag + world_height/2) % world_height - world_height/2
    
    dist_sq = diff.real**2 + diff.imag**2
    
    # Collision mask checks:
    # 1. Distance < Collision Radius
    # 2. Bullet is active
    # 3. Ship is alive
    hit_mask = (dist_sq < config.collision_radius**2) & flat_bullets_active.unsqueeze(1) & state.ship_alive.unsqueeze(2)
    
    # Ignore own bullets (ships cannot shoot themselves)
    bullet_source_idx = torch.arange(num_ships * num_bullets_per_ship, device=device) // num_bullets_per_ship
    own_bullet_mask = (bullet_source_idx.view(1, 1, -1) == torch.arange(num_ships, device=device).view(1, num_ships, 1))
    
    valid_hit = hit_mask & (~own_bullet_mask) # (B, N_target_ships, N_total_bullets)
    
    if not valid_hit.any():
        return state

    # Count hits per ship
    hits_per_ship = valid_hit.sum(dim=2) # (B, N_target)
    damage = hits_per_ship.float() * config.bullet_damage
    
    # Apply Damage
    state.ship_health = state.ship_health - damage
    
    # Update Alive Status
    state.ship_alive = state.ship_health > 0
    state.ship_health = torch.maximum(state.ship_health, torch.tensor(0.0, device=device))
    
    # Calculate Rewards based on whom hit whom - REMOVED (Handled by RewardFunction)
    
    # Deactivate bullets that hit
    bullet_hits_flat = valid_hit.any(dim=1) # (B, N_total_bullets) - True if bullet hit ANY ship
    flat_active_ref = state.bullet_active.view(batch_size, -1)
    flat_active_ref[bullet_hits_flat] = False
        
    return state


def _check_game_over(state: TensorState) -> torch.Tensor:
    """
    Determines if the game is over.
    
    Assumes standard 2-team setup (Team 0 vs Team 1).
    
    Args:
        state: Current state.
        
    Returns:
        dones tensor.
    """
    batch_size, num_ships = state.ship_pos.shape
    device = state.device
    
    team0_alive = (state.ship_team_id == 0) & state.ship_alive
    team1_alive = (state.ship_team_id == 1) & state.ship_alive
    
    team0_count = team0_alive.sum(dim=1)
    team1_count = team1_alive.sum(dim=1)
    
    # Game Over if either team is wiped out
    dones = (team0_count == 0) | (team1_count == 0)
    
    if not dones.any():
        return dones

    # Rewards handled by RewardFunction
    return dones


def resolve_collisions(state: TensorState, config: ShipConfig) -> Tuple[TensorState, torch.Tensor]:
    """
    Detects collisions, applies damage, and checks for game over.
    
    Args:
        state: Current state.
        config: Configuration.
        
    Returns:
        Tuple of (state, dones).
    """
    # 1. Apply Damage
    state = _apply_combat_damage(state, config)
    
    # 2. Check Game Over
    dones = _check_game_over(state)
    
    return state, dones
