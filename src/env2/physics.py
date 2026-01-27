
import torch
from typing import Tuple

# Use relative imports if possible, or assume src is in path.
# Trying relative to be safe within the package structure.
from .state import TensorState, ShipConfig
from core.constants import PowerActions, TurnActions, ShootActions, RewardConstants

def update_ships(state: TensorState, actions: torch.Tensor, config: ShipConfig) -> TensorState:
    """
    Update ship physics based on actions.
    actions: (B, N, 3) [Power, Turn, Shoot]
    """
    device = state.device
    
    # 1. Unpack actions
    power_action = actions[..., 0].long()
    turn_action = actions[..., 1].long()
    shoot_action = actions[..., 2].long()
    
    # 2. Physics Lookup Tables
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
    
    # 3. Apply Actions lookup for ALIVE ships
    thrust_mag = thrust_table[power_action]
    power_gain = power_gain_table[power_action]
    
    turn_offset = turn_offset_table[turn_action]
    drag_coeff = drag_coeff_table[turn_action]
    lift_coeff = lift_coeff_table[turn_action]
    
    # 4. Update Power
    new_power = state.ship_power + power_gain * config.dt
    state.ship_power = torch.clamp(new_power, 0.0, config.max_power)
    
    has_power = state.ship_power > 0
    thrust_mag = torch.where(has_power, thrust_mag, torch.zeros_like(thrust_mag))
    
    # 5. Update Attitude
    speed = state.ship_vel.abs()
    speed_safe = torch.maximum(speed, torch.tensor(1e-6, device=device))
    
    stopped = speed < 1e-6
    vel_dir = state.ship_vel / speed_safe
    base_attitude = torch.where(stopped, state.ship_attitude, vel_dir)
    
    rotation = torch.polar(torch.ones_like(turn_offset), turn_offset)
    state.ship_attitude = base_attitude * rotation
    
    state.ship_ang_vel = turn_offset / config.dt
    
    # 6. Calculate Forces
    thrust_force = thrust_mag * state.ship_attitude
    drag_force = -drag_coeff * speed * state.ship_vel
    
    lift_vector = state.ship_vel * 1j
    lift_force = lift_coeff * speed * lift_vector
    
    total_force = thrust_force + drag_force + lift_force
    
    # 7. Update Kinematics
    accel = total_force
    state.ship_vel = state.ship_vel + accel * config.dt
    state.ship_pos = state.ship_pos + state.ship_vel * config.dt
    
    # Wrap World
    w, h = config.world_size
    state.ship_pos.real = state.ship_pos.real % w
    state.ship_pos.imag = state.ship_pos.imag % h
    
    # Stopped clamp
    new_speed = state.ship_vel.abs()
    too_slow = new_speed < 1e-6
    clamped_vel = torch.tensor(1e-6, device=device, dtype=torch.complex64) * state.ship_attitude
    state.ship_vel = torch.where(too_slow, clamped_vel, state.ship_vel)
    
    # 8. Shooting Logic
    state.ship_cooldown = state.ship_cooldown - config.dt
    
    can_shoot = (
        (shoot_action == ShootActions.SHOOT) & 
        (state.ship_cooldown <= 0) & 
        (state.ship_power >= config.bullet_energy_cost) & 
        state.ship_alive
    )
    
    # Update is_shooting state for observation/rendering
    state.ship_is_shooting = can_shoot
    
    if can_shoot.any():
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
        
        b_idx, s_idx = torch.nonzero(can_shoot, as_tuple=True)
        
        if len(b_idx) > 0:
            k_idx = state.bullet_cursor[b_idx, s_idx]
            spawn_pos = state.ship_pos[b_idx, s_idx]
            
            shooter_att = state.ship_attitude[b_idx, s_idx]
            shooter_vel = state.ship_vel[b_idx, s_idx]
            
            base_vel = shooter_vel + config.bullet_speed * shooter_att
            
            # Generate complex noise
            noise_real = torch.randn(base_vel.shape, device=device) * config.bullet_spread
            noise_imag = torch.randn(base_vel.shape, device=device) * config.bullet_spread
            noise = torch.complex(noise_real, noise_imag)
            
            final_vel = base_vel + noise
            
            state.bullet_pos[b_idx, s_idx, k_idx] = spawn_pos
            state.bullet_vel[b_idx, s_idx, k_idx] = final_vel
            state.bullet_time[b_idx, s_idx, k_idx] = config.bullet_lifetime
            state.bullet_active[b_idx, s_idx, k_idx] = True
            
            state.bullet_cursor[b_idx, s_idx] = (k_idx + 1) % state.max_bullets
            
    return state

def update_bullets(state: TensorState, config: ShipConfig) -> TensorState:
    dt = config.dt
    w, h = config.world_size
    
    state.bullet_time = state.bullet_time - dt
    state.bullet_active = state.bullet_active & (state.bullet_time > 0)
    
    # Update all positions (masked by active not stricly needed for safety, but good for cleanliness)
    state.bullet_pos = state.bullet_pos + state.bullet_vel * dt
    
    state.bullet_pos.real = state.bullet_pos.real % w
    state.bullet_pos.imag = state.bullet_pos.imag % h
    
    return state

def resolve_collisions(state: TensorState, config: ShipConfig) -> Tuple[TensorState, torch.Tensor, torch.Tensor]:
    """
    Detect collisions and apply damage.
    Returns: (state, rewards, dones)
    """
    B, N = state.ship_pos.shape
    K = state.max_bullets
    device = state.device
    w, h = config.world_size
    
    rewards = torch.zeros((B, N), device=device, dtype=torch.float32)
    
    # Flatten bullets
    flat_bullets_pos = state.bullet_pos.view(B, N*K)
    flat_bullets_active = state.bullet_active.view(B, N*K)
    
    # Calculate difference (Wrap world)
    diff = state.ship_pos.unsqueeze(2) - flat_bullets_pos.unsqueeze(1) # (B, N, N*K)
    diff.real = (diff.real + w/2) % w - w/2
    diff.imag = (diff.imag + h/2) % h - h/2
    
    dist_sq = diff.real**2 + diff.imag**2
    
    # Collision mask
    # Ship must be alive, bullet must be active
    hit_mask = (dist_sq < config.collision_radius**2) & flat_bullets_active.unsqueeze(1) & state.ship_alive.unsqueeze(2)
    
    # Ignore own bullets
    # bullet m comes from ship m // K
    bullet_source_idx = torch.arange(N*K, device=device) // K
    own_bullet_mask = (bullet_source_idx.view(1, 1, N*K) == torch.arange(N, device=device).view(1, N, 1))
    
    valid_hit = hit_mask & (~own_bullet_mask) # (B, N_target, M_bullet)
    
    if valid_hit.any():
        # Damage Loop
        hits_per_ship = valid_hit.sum(dim=2) # (B, N_target)
        damage = hits_per_ship.float() * config.bullet_damage
        
        # Apply Damage
        state.ship_health = state.ship_health - damage
        
        # Update Alive Status
        state.ship_alive = state.ship_health > 0
        state.ship_health = torch.maximum(state.ship_health, torch.tensor(0.0, device=device))
        
        # Rewards Loop
        valid_hit_reshaped = valid_hit.view(B, N, N, K) # (B, N_t, N_s, K)
        hits_matrix = valid_hit_reshaped.sum(dim=3).float() # (B, N_t, N_s)
        
        target_teams = state.ship_team_id.unsqueeze(2) # (B, N_t, 1)
        source_teams = state.ship_team_id.unsqueeze(1) # (B, 1, N_s)
        
        is_enemy = target_teams != source_teams
        
        # Enemy Hit Reward (credited to source) -> sum over targets
        # hits_matrix[b, t, s] counts hits BY s ON t
        enemy_hits_by_source = (hits_matrix * is_enemy.float()).sum(dim=1) # (B, N_s)
        rewards = rewards + enemy_hits_by_source * RewardConstants.ENEMY_DAMAGE
        
        # Ally Hit Penalty
        ally_hits_by_source = (hits_matrix * (~is_enemy).float()).sum(dim=1) # (B, N_s)
        rewards = rewards + ally_hits_by_source * RewardConstants.ALLY_DAMAGE
        
        # Deactivate bullets that hit
        bullet_hits = valid_hit.any(dim=1) # (B, M)
        flat_active_ref = state.bullet_active.view(B, N*K)
        flat_active_ref[bullet_hits] = False
    
    # Check Dones (Game Over)
    # If a team has 0 alive ships, game over?
    # Or if only one team remains?
    # Typically: done if <= 1 team alive.
    
    # Count alive ships per team
    # We need max team id or just iterate unique teams?
    # Assume 2 teams for now? Or N teams?
    # Generalized:
    # Get mask of alive
    # Check if all alive ships belong to same team.
    
    # Optimization: count unique team IDs among alive ships.
    # If count <= 1, done.
    
    # Since we can't easily iterate per batch, we do:
    # 1. Mask team_ids with alive.
    # 2. Check if multiple teams present.
    # Actually, simpler:
    # Team 0 alive count, Team 1 alive count.
    # If Team 0 count == 0 OR Team 1 count == 0 -> Done.
    # This assumes 2 teams (0 and 1). Valid for 1v1, 2v2 etc.
    # What if NvM with > 2 teams?
    # Codebase implies 2 teams usually.
    # I'll implement generic check later if needed, assume 2 teams (0, 1) for now is safe for "env1" parity.
    
    team0_alive = (state.ship_team_id == 0) & state.ship_alive
    team1_alive = (state.ship_team_id == 1) & state.ship_alive
    
    team0_count = team0_alive.sum(dim=1)
    team1_count = team1_alive.sum(dim=1)
    
    dones = (team0_count == 0) | (team1_count == 0)
    
    # Add Victory/Defeat Rewards
    # If done:
    # Winner gets VICTORY, Loser gets DEFEAT.
    # If both 0 (Draw?), DRAW.
    
    # We can handle terminal rewards here or in Env wrapper.
    # Usually here is better for "raw physics returns rewards".
    
    # This logic is slightly complex to vectorize cleanly for arbitary N teams.
    # For 2 teams:
    team0_win = (team0_count > 0) & (team1_count == 0)
    team1_win = (team1_count > 0) & (team0_count == 0)
    draw = (team0_count == 0) & (team1_count == 0)
    
    # Apply to all ships of the team
    # Team 0 ships get VICTORY if team0_win
    # Team 1 ships get DEFEAT if team0_win
    
    mask_win = torch.zeros((B, N), device=device, dtype=torch.bool)
    mask_lose = torch.zeros((B, N), device=device, dtype=torch.bool)
    
    # If Team 0 wins:
    # T0 ships -> Win
    # T1 ships -> Lose
    
    t0_idx = (state.ship_team_id == 0)
    t1_idx = (state.ship_team_id == 1)
    
    # Expand done flags
    t0_win_expanded = team0_win.unsqueeze(1).expand(B, N)
    t1_win_expanded = team1_win.unsqueeze(1).expand(B, N)
    draw_expanded = draw.unsqueeze(1).expand(B, N)
    
    # T0 ships win if T0 wins
    mask_win = mask_win | (t0_idx & t0_win_expanded)
    mask_win = mask_win | (t1_idx & t1_win_expanded)
    
    mask_lose = mask_lose | (t0_idx & t1_win_expanded)
    mask_lose = mask_lose | (t1_idx & t0_win_expanded)
    
    rewards = torch.where(mask_win, rewards + RewardConstants.VICTORY, rewards)
    rewards = torch.where(mask_lose, rewards + RewardConstants.DEFEAT, rewards)
    rewards = torch.where(draw_expanded, rewards + RewardConstants.DRAW, rewards)
    
    return state, rewards, dones

