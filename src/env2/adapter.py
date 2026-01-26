
import torch
import numpy as np
from env.state import State
from env.ship import Ship, default_ship_config
from env.bullets import Bullets
from env2.state import TensorState

def tensor_state_to_cpu_state(t_state: TensorState, batch_idx: int = 0) -> State:
    """
    Convert a single environment's state from TensorState to legacy State.
    Used for rendering with the existing PyGame renderer.
    """
    # 1. Extract Ships
    ships = {}
    
    # We need to access tensors. Assume they are on GPU, so .cpu() is needed.
    # To optimize, caller might pass already CPU tensors, but we handle robustly.
    
    # Slice for the specific batch index
    b = batch_idx
    
    # Get arrays (numpy)
    pos = t_state.ships_pos[b].detach().cpu().numpy()
    vel = t_state.ships_vel[b].detach().cpu().numpy()
    power = t_state.ships_power[b].detach().cpu().numpy()
    team = t_state.ships_team[b].detach().cpu().numpy()
    alive = t_state.ships_alive[b].detach().cpu().numpy()
    health = t_state.ships_health[b].detach().cpu().numpy()
    
    # Attitude might be stored or derived.
    # In TensorEnv step, we store it.
    attitude = t_state.ships_attitude[b].detach().cpu().numpy()
    
    N = len(pos)
    
    for i in range(N):
        if not alive[i]:
            continue
            
        # Reconstruct Ship object
        # We don't need full simulation capabilities, just data for rendering.
        # Ship constructor requires init vals.
        
        # NOTE: Ship class in env/ship.py computes attributes in __init__.
        # We can construct it and then overwrite attributes.
        
        s = Ship(
            ship_id=i,
            team_id=int(team[i]),
            ship_config=default_ship_config,
            initial_x=pos[i].real,
            initial_y=pos[i].imag,
            initial_vx=vel[i].real,
            initial_vy=vel[i].imag
        )
        
        # Overwrite current state
        s.alive = bool(alive[i])
        s.health = float(health[i])
        s.power = float(power[i])
        s.position = pos[i]
        s.velocity = vel[i]
        s.attitude = attitude[i]
        
        ships[i] = s
        
    # 2. Extract Bullets
    # Layout: (B, N, K)
    # flattened for legacy Bullets class
    
    b_pos = t_state.bullets_pos[b].detach().cpu().numpy() # (N, K)
    b_vel = t_state.bullets_vel[b].detach().cpu().numpy()
    b_time = t_state.bullets_time[b].detach().cpu().numpy()
    b_team = t_state.bullets_team[b].detach().cpu().numpy()
    
    # Flatten
    flat_pos = b_pos.flatten()
    flat_vel = b_vel.flatten()
    flat_time = b_time.flatten()
    flat_team = b_team.flatten() # Wait, Bullets class stores ship_id, not team?
    
    # Renderers usually color by ship_id or team?
    # `_render_bullets` uses `bullets.ship_id`.
    # And then `if ship_id == 0: color ... else ...`
    # It assumes ship_id implies team? 
    # Actually in legacy env, id 0..mid is team 0.
    # But for NvM or shuffled, we define team explicitly.
    # Renderer uses `ship_id` to deduce color?
    # Line 252 in renderer.py: `if ship_id == 0: color ... `
    # This seems hardcoded to 1v1 or assumes ID 0 is blue.
    # If we have proper teams, this renderer logic is flawed for generic NvN.
    # But `TensorEnv` stores `bullets_team`. 
    # We should probably patch `ship_id` in bullets to be `team_id` if the renderer treats them as such?
    # Or strict ship_ids.
    
    # We know the source ship index for each bullet is the 'N' dim.
    # indices: i -> ship i.
    # So we can reconstruct ship_id array.
    
    # Construct ship_id array matching flat structure
    num_k = b_pos.shape[1]
    ship_ids_src = np.repeat(np.arange(N), num_k)
    
    # Filter active bullets (time > 0)
    active_mask = flat_time > 0
    
    active_count = np.sum(active_mask)
    
    game_bullets = Bullets(max_bullets=len(flat_pos))
    game_bullets.num_active = active_count
    
    # Fill active bullets at start of array (standard Bullets behavior)
    idx_active = np.where(active_mask)[0]
    
    count = len(idx_active)
    game_bullets.x[:count] = flat_pos[idx_active].real
    game_bullets.y[:count] = flat_pos[idx_active].imag
    game_bullets.vx[:count] = flat_vel[idx_active].real
    game_bullets.vy[:count] = flat_vel[idx_active].imag
    game_bullets.time_remaining[:count] = flat_time[idx_active]
    game_bullets.ship_id[:count] = ship_ids_src[idx_active] 
    
    # For rendering color:
    # Renderer line 252: `if ship_id == 0` -> Blue.
    # If ship_id > 0 -> Red.
    # This means legacy renderer assumes Ship 0 is Blue Team, everyone else is Red?
    # Or just 1v1 specific.
    # We should probably update Renderer to use `team_id` lookup if possible, 
    # but since we can't easily change Renderer signature without breaking things,
    # let's look at `_render_bullets`.
    # It reads `bullets.ship_id`.
    # If we want correct colors, we need `ship_id` roughly correlating to team for now?
    # Or we construct a dummy `ship_id` that is 0 for Team 0 and 1 for Team 1?
    # `ship_ids_src` gives actual ship ID.
    # If Ship 0 is Team 0, Ship 1 is Team 1.
    # If Ship 2 is Team 0... Renderer will paint it Red (since 2 != 0).
    # That's a problem for NvN.
    
    # FIX: We will fake `ship_id` in the Bullets object passed to renderer.
    # Set `ship_id` to `bullets_team` (0 or 1).
    # Then `if ship_id == 0` (Team 0) -> Blue. `else` (Team 1) -> Red.
    
    # Get team from tensor
    flat_teams = flat_team[idx_active]
    game_bullets.ship_id[:count] = flat_teams # 0 or 1
    
    state = State(ships=ships, time=float(t_state.time[b]))
    state.bullets = game_bullets
    return state
