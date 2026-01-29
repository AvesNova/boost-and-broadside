
import torch
import numpy as np
from env.state import State
from env.ship import Ship, default_ship_config
from env.bullets import Bullets
from env2.state import TensorState

def tensor_state_to_cpu_state(tensor_state: TensorState, batch_idx: int = 0) -> State:
    """
    Convert a single environment's state from TensorState to legacy State.
    
    Used for rendering with the existing PyGame renderer.
    
    Args:
        tensor_state: The batch of environments state.
        batch_idx: The index of the specific environment to convert.
        
    Returns:
        A legacy State object containing ships and bullets for the specified environment.
    """
    # 1. Extract Ships
    ships = {}
    
    # Access and detach tensors for the specific batch index
    # We detach and move to CPU for numpy compatibility
    
    pos = tensor_state.ship_pos[batch_idx].detach().cpu().numpy()
    vel = tensor_state.ship_vel[batch_idx].detach().cpu().numpy()
    power = tensor_state.ship_power[batch_idx].detach().cpu().numpy()
    team = tensor_state.ship_team_id[batch_idx].detach().cpu().numpy()
    alive = tensor_state.ship_alive[batch_idx].detach().cpu().numpy()
    health = tensor_state.ship_health[batch_idx].detach().cpu().numpy()
    attitude = tensor_state.ship_attitude[batch_idx].detach().cpu().numpy()
    
    num_ships = len(pos)
    
    for i in range(num_ships):
        if not alive[i]:
            continue
            
        # Reconstruct Ship object with current state values
        # Note: Ship class computes some attributes in __init__ which we overwrite.
        s = Ship(
            ship_id=i,
            team_id=int(team[i]),
            ship_config=default_ship_config,
            initial_x=pos[i].real,
            initial_y=pos[i].imag,
            initial_vx=vel[i].real,
            initial_vy=vel[i].imag
        )
        
        s.alive = bool(alive[i])
        s.health = float(health[i])
        s.power = float(power[i])
        s.position = pos[i]
        s.velocity = vel[i]
        s.attitude = attitude[i]
        
        ships[i] = s
        
    # 2. Extract Bullets
    # Layout: (Batch, NumShips, NumBullets)
    
    b_pos = tensor_state.bullet_pos[batch_idx].detach().cpu().numpy()
    b_vel = tensor_state.bullet_vel[batch_idx].detach().cpu().numpy()
    b_time = tensor_state.bullet_time[batch_idx].detach().cpu().numpy()
    
    # Flatten
    flat_pos = b_pos.flatten()
    flat_vel = b_vel.flatten()
    flat_time = b_time.flatten()
    
    # Derive teams for bullets based on source ship
    # b_pos shape is (N, K). Row i corresponds to ship i.
    ship_teams = tensor_state.ship_team_id[batch_idx].detach().cpu().numpy()
    num_bullets_per_ship = b_pos.shape[1]
    flat_teams = np.repeat(ship_teams, num_bullets_per_ship)
    
    # Filter active bullets
    active_mask = flat_time > 0
    active_count = np.sum(active_mask)
    
    game_bullets = Bullets(max_bullets=len(flat_pos))
    game_bullets.num_active = active_count
    
    idx_active = np.where(active_mask)[0]
    count = len(idx_active)
    
    if count > 0:
        game_bullets.x[:count] = flat_pos[idx_active].real
        game_bullets.y[:count] = flat_pos[idx_active].imag
        game_bullets.vx[:count] = flat_vel[idx_active].real
        game_bullets.vy[:count] = flat_vel[idx_active].imag
        game_bullets.time_remaining[:count] = flat_time[idx_active]
        
        # Determine Render Color:
        # The legacy renderer often uses `ship_id` to determine color (0=Blue, else=Red).
        # To support proper team coloring in rendering without changing the renderer,
        # we populate `ship_id` with the `team_id` (0 or 1).
        game_bullets.ship_id[:count] = flat_teams[idx_active]
        
    # Calculate simulation time
    time_val = float(tensor_state.step_count[batch_idx]) * default_ship_config.dt
    
    state = State(ships=ships, time=time_val)
    state.bullets = game_bullets
    
    return state
