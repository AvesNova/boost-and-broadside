import torch

def compute_pairwise_features(
    tokens: torch.Tensor, world_size: tuple[float, float]
) -> torch.Tensor:
    """
    Precompute fundamental pairwise features (deltas) in local frame.
    
    Args:
        tokens: (..., N, D) tensor of ship state tokens.
        world_size: (W, H) tuple.
        
    Returns:
        (..., N, N, 4) tensor containing [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
        in the observing ship's local reference frame.
    """
    # ... dimensions handling ...
    # Support both (T, N, D) and (B, T, N, D)
    
    # Extract states
    pos_x = tokens[..., 3] * world_size[0]
    pos_y = tokens[..., 4] * world_size[1]
    vel_x = tokens[..., 5] * 180.0
    vel_y = tokens[..., 6] * 180.0
    # Attitude (orientation)
    att_x = tokens[..., 10]
    att_y = tokens[..., 11]
    
    # --- 1. Compute Deltas (Broadcasted) ---
    # Shape: (..., N, N)
    # Unsqueeze appropriate dimensions for pairwise calc
    # if tokens is (T, N, D), then pos_x is (T, N).
    # We want (T, observer, target).
    # observer: unsqueeze(-1) -> (T, N, 1)
    # target: unsqueeze(-2) -> (T, 1, N)
    
    dx = pos_x.unsqueeze(-1) - pos_x.unsqueeze(-2) # Target - Self
    dy = pos_y.unsqueeze(-1) - pos_y.unsqueeze(-2)
    
    dvx = vel_x.unsqueeze(-1) - vel_x.unsqueeze(-2)
    dvy = vel_y.unsqueeze(-1) - vel_y.unsqueeze(-2)
    
    # --- 2. Handle World Wrapping ---
    # Wrap dx, dy to min dist on torus
    dx = dx - torch.round(dx / world_size[0]) * world_size[0]
    dy = dy - torch.round(dy / world_size[1]) * world_size[1]
    
    # --- 3. Rotate to Local Frame ---
    # We need to rotate the delta vectors by the conjugate of the observing ship's attitude.
    # If Attitude = (Ax, Ay), then Conjugate = (Ax, -Ay).
    # Rotate (dx, dy) by (Ax, -Ay):
    # NewX = dx * Ax - dy * (-Ay) = dx*Ax + dy*Ay
    # NewY = dx * (-Ay) + dy * Ax = -dx*Ay + dy*Ax
    
    # Self Attitude: (..., N, 1)
    self_ax = att_x.unsqueeze(-1)
    self_ay = att_y.unsqueeze(-1)
    
    local_dx = dx * self_ax + dy * self_ay
    local_dy = -dx * self_ay + dy * self_ax
    
    local_dvx = dvx * self_ax + dvy * self_ay
    local_dvy = -dvx * self_ay + dvy * self_ax
    
    # Stack features
    # Shape: (..., N, N, 4)
    features = torch.stack([local_dx, local_dy, local_dvx, local_dvy], dim=-1)
    
    # Normalize to be consistent with token scale.
    features[..., 0] /= world_size[0]
    features[..., 1] /= world_size[1]
    features[..., 2] /= 180.0
    features[..., 3] /= 180.0
    
    return features
