import torch

def compute_pairwise_features(states: torch.Tensor, world_size: tuple[float, float]) -> torch.Tensor:
    """
    Compute pairwise relative features (pos, vel) for all ships.
    
    Args:
        states: Tensor of shape (..., N, D) containing ship states.
                Expects indices 3,4 for Pos(x,y) and 5,6 for Vel(x,y).
        world_size: Tuple (width, height) for wrapping logic.
        
    Returns:
        Tensor of shape (..., N, N, 4) containing [rel_x, rel_y, rel_vx, rel_vy].
    """
    # states: (..., N, D)
    # Pos=[3,4], Vel=[5,6] (based on compile_tokens)
    
    pos = states[..., 3:5] # (..., N, 2)
    vel = states[..., 5:7] # (..., N, 2)
    
    # Broadcast for pairwise
    # Input has shape (Batch, Time, N, 2) or (Batch, N, 2)
    # leveraging unsqueeze relative to last dimmer (-2, -3) handles both cases
    
    # pos.unsqueeze(-2) -> (..., N, 1, 2)
    # pos.unsqueeze(-3) -> (..., 1, N, 2)
    # Result -> (..., N, N, 2)
    
    delta_pos = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    delta_vel = vel.unsqueeze(-2) - vel.unsqueeze(-3)
    
    # Wrap around
    W, H = float(world_size[0]), float(world_size[1])
    
    dx = delta_pos[..., 0]
    dy = delta_pos[..., 1]
    
    dx = dx - torch.round(dx / W) * W
    dy = dy - torch.round(dy / H) * H
    
    delta_pos_wrapped = torch.stack([dx, dy], dim=-1)
    
    return torch.cat([delta_pos_wrapped, delta_vel], dim=-1) # (..., N, N, 4)
