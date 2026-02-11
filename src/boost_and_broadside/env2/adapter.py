import torch
import numpy as np
from boost_and_broadside.core.types import RenderState, RenderShip
from boost_and_broadside.env2.state import TensorState
from boost_and_broadside.core.config import ShipConfig

def tensor_state_to_render_state(tensor_state: TensorState, config: ShipConfig, batch_idx: int = 0) -> RenderState:
    """
    Convert a single environment's state from TensorState to RenderState.
    
    Used for rendering with the backend-agnostic renderer.
    
    Args:
        tensor_state: The batch of environments state.
        config: Ship configuration (for metadata like max health).
        batch_idx: The index of the specific environment to convert.
        
    Returns:
        A RenderState object containing ships and bullets for the specified environment.
    """
    # 1. Extract Ships
    ships = {}
    
    pos = tensor_state.ship_pos[batch_idx].detach().cpu().numpy()
    # vel = tensor_state.ship_vel[batch_idx].detach().cpu().numpy() # Not strictly needed for basic rendering
    power = tensor_state.ship_power[batch_idx].detach().cpu().numpy()
    team = tensor_state.ship_team_id[batch_idx].detach().cpu().numpy()
    alive = tensor_state.ship_alive[batch_idx].detach().cpu().numpy()
    health = tensor_state.ship_health[batch_idx].detach().cpu().numpy()
    attitude = tensor_state.ship_attitude[batch_idx].detach().cpu().numpy()
    
    num_ships = len(pos)
    
    for i in range(num_ships):
        # We render all ships, even dead ones might have effects? 
        # But typically we only care about alive ones or we render them as debris.
        # The renderer usually checks .alive flag.
        
        s = RenderShip(
            id=i,
            team_id=int(team[i]),
            position=pos[i],
            attitude=attitude[i],
            health=float(health[i]),
            max_health=config.max_health,
            power=float(power[i]),
            alive=bool(alive[i])
        )
        ships[i] = s
        
    # 2. Extract Bullets
    b_pos = tensor_state.bullet_pos[batch_idx].detach().cpu().numpy()
    b_time = tensor_state.bullet_time[batch_idx].detach().cpu().numpy()
    
    # Flatten
    flat_pos = b_pos.flatten()
    flat_time = b_time.flatten()
    
    # Derive teams/owners
    ship_ids_matrix = torch.arange(num_ships, device=tensor_state.device).unsqueeze(1).expand_as(tensor_state.bullet_pos[batch_idx])
    flat_owners = ship_ids_matrix.detach().cpu().numpy().flatten()
    
    # Filter active bullets
    active_mask = flat_time > 0
    
    active_pos = flat_pos[active_mask]
    active_owners = flat_owners[active_mask]
    
    render_bullets_x = active_pos.real.astype(np.float32)
    render_bullets_y = active_pos.imag.astype(np.float32)
    render_bullets_owner = active_owners.astype(np.int32)

    # 3. Time
    time_val = float(tensor_state.step_count[batch_idx]) * config.dt
    
    return RenderState(
        ships=ships,
        bullet_x=render_bullets_x,
        bullet_y=render_bullets_y,
        bullet_owner_id=render_bullets_owner,
        time=time_val
    )
