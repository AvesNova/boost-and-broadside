
"""
Stateless, vectorized physics logic for the GPU environment.
"""

import torch

def update_ships(
    ships_pos: torch.Tensor,
    ships_vel: torch.Tensor,
    ships_power: torch.Tensor,
    ships_cooldown: torch.Tensor,
    ships_team: torch.Tensor,
    ships_alive: torch.Tensor,
    actions_power: torch.Tensor,
    actions_turn: torch.Tensor,
    actions_shoot: torch.Tensor,
    dt: float,
    world_size: tuple[float, float],
    # Physics constants (can be passed as scalars or tensors if we want heterogeneity)
    base_thrust: float = 8.0,
    boost_thrust: float = 80.0,
    reverse_thrust: float = -10.0,
    base_power_gain: float = 10.0,
    boost_power_gain: float = -40.0,
    reverse_power_gain: float = 20.0,
    no_turn_drag_coeff: float = 8e-4,
    normal_turn_drag_coeff: float = 1.2e-3,
    normal_turn_lift_coeff: float = 15e-3,
    sharp_turn_drag_coeff: float = 5.0e-3,
    sharp_turn_lift_coeff: float = 27e-3,
    max_power: float = 100.0,
    bullet_energy_cost: float = 3.0,
    firing_cooldown: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update ship states based on actions and physics.
    
    Uses Semi-Implicit Euler:
    vel += acc * dt
    pos += vel * dt
    """
    
    # --- 1. Interpret Actions ---
    
    # Power Lookup Table: [COAST(0), BOOST(1), REVERSE(2)]
    real_dtype = ships_pos.real.dtype
    thrust_vals = torch.tensor([base_thrust, boost_thrust, reverse_thrust], device=ships_pos.device, dtype=real_dtype)
    power_gain_vals = torch.tensor([base_power_gain, boost_power_gain, reverse_power_gain], device=ships_pos.device, dtype=real_dtype)

    current_thrust = thrust_vals[actions_power]
    current_power_change = power_gain_vals[actions_power]

    # Turn Lookup Tables
    # [STRAIGHT(0), LEFT(1), RIGHT(2), HARD_LEFT(3), HARD_RIGHT(4), BRAKE(5), HARD_BRAKE(6)]
    # We need to map these to drag/lift coefficients and angle offsets.
    
    # Drag Constants
    drag_coeffs = torch.tensor([
        no_turn_drag_coeff,      # 0
        normal_turn_drag_coeff,  # 1
        normal_turn_drag_coeff,  # 2
        sharp_turn_drag_coeff,   # 3
        sharp_turn_drag_coeff,   # 4
        normal_turn_drag_coeff,  # 5
        sharp_turn_drag_coeff    # 6
    ], device=ships_pos.device, dtype=real_dtype)
    
    # Lift Constants 
    lift_coeffs = torch.tensor([
        0.0,                         # 0
        -normal_turn_lift_coeff,     # 1
        normal_turn_lift_coeff,      # 2
        -sharp_turn_lift_coeff,      # 3
        sharp_turn_lift_coeff,       # 4
        0.0,                         # 5
        0.0                          # 6
    ], device=ships_pos.device, dtype=real_dtype)
    
    # Turn Offsets (Radians)
    # LEFT is negative angle, RIGHT is positive angle.
    normal_turn_angle = 0.0872665 # 5 deg
    sharp_turn_angle = 0.261799 # 15 deg
    
    turn_offsets = torch.tensor([
        0.0,                # 0
        -normal_turn_angle, # 1
        normal_turn_angle,  # 2
        -sharp_turn_angle,  # 3
        sharp_turn_angle,   # 4
        0.0,                # 5
        0.0                 # 6
    ], device=ships_pos.device, dtype=real_dtype)

    current_drag = drag_coeffs[actions_turn]
    current_lift = lift_coeffs[actions_turn]
    current_turn_offset = turn_offsets[actions_turn]

    # --- 2. Calculate Forces ---
    
    # Current velocity state
    # Handle the complex numbers carefully. We assume pos/vel are complex tensors.
    # If they are (..., 2) float tensors, we need to convert logic.
    # Plan says complex or 2 floats. Let's stick to complex64/128 for consistency with env1 if easy, 
    # but PyTorch complex support is good now. Let's assume complex input.
    
    speed = ships_vel.abs()
    # Avoid div/0
    speed_safe = torch.maximum(speed, torch.tensor(1e-6, device=ships_pos.device, dtype=speed.dtype))
    
    # Attitude (Orientation)
    # Original: self.attitude = self.velocity / self.speed * np.exp(1j * self.turn_offset)
    # In original forward():
    # 1. _update_attitude (sets attitude for force calc)
    # 2. _update_kinematics (updates pos/vel)
    # 3. _update_attitude (re-aligns attitude to new velocity)
    
    velocity_dir = ships_vel / speed_safe
    
    # Apply turn offset to get current attitude for thrust direction
    rotator = torch.polar(torch.ones_like(current_turn_offset), current_turn_offset)
    attitude = velocity_dir * rotator
    
    # Thrust Force
    # Can only thrust if power > 0. But power update happens at end of step in original code?
    # Original: _update_kinematics uses current power. _update_power happens AFTER.
    # So we use current ships_power to check > 0.
    has_power = ships_power > 0
    thrust_magnitude = torch.where(has_power, current_thrust, torch.zeros_like(current_thrust))
    thrust_force = thrust_magnitude * attitude # thrust is along the attitude vector
    
    # Drag Force
    # drag_force = -drag_coeff * speed * velocity
    drag_force = -current_drag * speed * ships_vel
    
    # Lift Force
    # lift_vector = velocity * 1j
    lift_vector = ships_vel * 1j
    lift_force = current_lift * speed * lift_vector
    
    total_force = thrust_force + drag_force + lift_force
    
    # --- 3. Integration (Semi-Implicit Euler) ---
    
    # Acceleration (Mass = 1)
    acc = total_force
    
    # Update Velocity: vel += acc * dt
    new_vel = ships_vel + acc * dt
    
    # Update Position: pos += new_vel * dt
    new_pos = ships_pos + new_vel * dt
    
    # Wrap Position (Toroidal World)
    new_pos_real = new_pos.real % world_size[0]
    new_pos_imag = new_pos.imag % world_size[1]
    new_pos = torch.complex(new_pos_real, new_pos_imag)
    
    # Angular Velocity
    ang_vel = current_turn_offset / dt # rad/s
    
    # New Attitude (aligned with velocity)
    new_speed = new_vel.abs()
    safe_new_speed = torch.maximum(new_speed, torch.tensor(1e-6, device=ships_pos.device, dtype=new_speed.dtype))
    new_attitude = new_vel / safe_new_speed
    
    # --- 4. Update Power & Cooldown ---
    
    new_power = ships_power + current_power_change * dt
    # Clamp power [0, max_power]
    new_power = torch.clamp(new_power, 0.0, max_power)
    
    # Cooldown (for shooting)
    new_cooldown = ships_cooldown - dt
    
    should_shoot = (actions_shoot == 1) & (ships_cooldown <= 0) & (ships_power >= bullet_energy_cost)
    
    # consume power if shot
    new_power = torch.where(should_shoot, new_power - bullet_energy_cost, new_power)
    
    # reset cooldown if shot
    new_cooldown = torch.where(should_shoot, torch.tensor(firing_cooldown, device=ships_pos.device, dtype=new_cooldown.dtype), new_cooldown)
    
    # Masking for dead ships
    new_pos = torch.where(ships_alive, new_pos, ships_pos)
    new_vel = torch.where(ships_alive, new_vel, ships_vel)
    new_power = torch.where(ships_alive, new_power, ships_power)
    new_cooldown = torch.where(ships_alive, new_cooldown, ships_cooldown)
    acc = torch.where(ships_alive, acc, torch.zeros_like(acc))
    ang_vel = torch.where(ships_alive, ang_vel, torch.zeros_like(ang_vel))
    new_attitude = torch.where(ships_alive, new_attitude, velocity_dir) # If dead, keep old direction
    
    return new_pos, new_vel, new_power, new_cooldown, should_shoot, acc, ang_vel, new_attitude

def update_bullets(
    bullets_pos: torch.Tensor,
    bullets_vel: torch.Tensor,
    bullets_time: torch.Tensor,
    dt: float,
    world_size: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update bullet positions and lifetimes.
    
    Args:
        bullets_pos: (B, N, K) complex or (B, N, K, 2)
        bullets_vel: (B, N, K) complex
        bullets_time: (B, N, K) float
    """
    
    # Mask for active bullets? We just update everything and use time > 0 to check validity.
    
    # pos += vel * dt
    new_pos = bullets_pos + bullets_vel * dt
    
    # Wrap
    new_pos_real = new_pos.real % world_size[0]
    new_pos_imag = new_pos.imag % world_size[1]
    new_pos = torch.complex(new_pos_real, new_pos_imag)
    
    new_time = bullets_time - dt
    
    return new_pos, new_time

def check_collisions(
    ships_pos: torch.Tensor,
    ships_team: torch.Tensor,
    ships_alive: torch.Tensor,
    bullets_pos: torch.Tensor,
    bullets_team: torch.Tensor,
    bullets_time: torch.Tensor, # Used to check active
    ship_collision_radius: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Check collisions between ships and bullets.
    
    Returns:
        collision_matrix: (B, N_targets, M*K_sources) Boolean mask of hits.
        bullet_hits_mask: (B, N_sources, K) Boolean mask of bullets that hit something.
    """
    # ships_pos: (B, N)
    # bullets_pos: (B, M, K) where M is num_ships (sources)
    
    B, N = ships_pos.shape
    B, M, K = bullets_pos.shape # M should equal N usually
    
    # Flatten bullets to (B, M*K)
    flat_bullets_pos = bullets_pos.view(B, M * K)
    flat_bullets_team = bullets_team.view(B, M * K)
    flat_bullets_time = bullets_time.view(B, M * K)
    
    # Ships: (B, N, 1)
    # Bullets: (B, 1, MK)
    
    s_pos = ships_pos.unsqueeze(2) # (B, N, 1)
    b_pos = flat_bullets_pos.unsqueeze(1) # (B, 1, MK)
    
    # Distance Squared: |s - b|^2
    # complex abs sq
    diff = s_pos - b_pos
    dist_sq = diff.real.square() + diff.imag.square() # (B, N, MK)
    
    radius_sq = ship_collision_radius ** 2
    
    # Check conditions:
    # 1. Distance < Radius
    # 2. Teams are different (no friendly fire in original?)
    #    Original: hit_mask = ... & (bullet_ship_ids != ship.ship_id)
    #    Wait, original checks `bullet_ship_ids != ship.ship_id`.
    #    It allows friendly fire if it's not YOUR OWN bullet?
    #    Let's check `src/env/env.py`.
    #    `hit_mask = (distances_sq < ship.collision_radius_squared) & (bullet_ship_ids != ship.ship_id)`
    #    It DOES allow friendly fire, just not self-hit.
    
    # So check: dist < r^2 AND ship_id != bullet_source_id.
    # We need bullet source IDs. 
    # bullets_pos is (B, M, K). Index m implies source ship M.
    # Create tensor of source IDs?
    
    # Construct source IDs
    # source_ids = range(M). Repeat K times.
    source_ids = torch.arange(M, device=ships_pos.device).unsqueeze(1).repeat(1, K).view(M*K) # (MK,)
    source_ids = source_ids.unsqueeze(0).expand(B, -1) # (B, MK)
    
    # Ship IDs
    ship_ids = torch.arange(N, device=ships_pos.device).unsqueeze(0).expand(B, -1) # (B, N)
    
    # Expand for broadcast
    # s_ids: (B, N, 1)
    # b_src_ids: (B, 1, MK)
    s_ids = ship_ids.unsqueeze(2)
    b_src_ids = source_ids.unsqueeze(1)
    
    not_self = s_ids != b_src_ids
    in_range = dist_sq < radius_sq
    
    # Bullet must be active
    # b_active: (B, 1, MK)
    b_active = (flat_bullets_time > 0).unsqueeze(1)
    
    # Ship must be alive
    # s_alive: (B, N, 1)
    s_alive = ships_alive.unsqueeze(2)
    
    valid_hit = in_range & not_self & b_active & s_alive
    
    # Return the full collision matrix and the mask of bullets that hit active targets
    # collision_matrix: (B, N_targets, M*K_sources)
    
    # Identify bullets that hit (to remove them)
    # Any bullet that hit ANY ship is removed.
    bullet_hit_any = valid_hit.any(dim=1)
    bullet_hit_mask = bullet_hit_any.view(B, M, K)
    
    return valid_hit, bullet_hit_mask
