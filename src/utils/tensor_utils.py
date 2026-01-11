import torch
import torch.nn.functional as F

def to_one_hot(actions: torch.Tensor) -> torch.Tensor:
    """
    Convert discrete actions to one-hot encoding.
    
    Args:
        actions: (..., 3) Tensor containing [power, turn, shoot] indices/values.
        
    Returns:
        (..., 12) Tensor containing concatenated one-hot vectors.
    """
    # Actions are likely floats, so convert to long
    actions = actions.long()
    
    power = actions[..., 0]
    turn = actions[..., 1]
    shoot = actions[..., 2]
    
    power_oh = F.one_hot(power, num_classes=3)
    turn_oh = F.one_hot(turn, num_classes=7)
    shoot_oh = F.one_hot(shoot, num_classes=2)
    
    return torch.cat([power_oh, turn_oh, shoot_oh], dim=-1).float()
