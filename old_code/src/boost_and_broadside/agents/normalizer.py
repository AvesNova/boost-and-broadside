
import torch
import torch.nn as nn

class InputNormalizer(nn.Module):
    """
    Normalizes raw physics inputs to neural network range (typically 0-1 or -1 to 1).
    Saved as part of the model checkpoint to ensure correct scaling during inference.
    """
    def __init__(self, world_size: tuple[float, float] = (1024.0, 1024.0), max_vel: float = 180.0):
        super().__init__()
        # Register as buffers so they are saved with state_dict but not trained
        self.register_buffer("world_size", torch.tensor(world_size, dtype=torch.float32))
        self.register_buffer("max_vel", torch.tensor(max_vel, dtype=torch.float32))
        self.register_buffer("max_power", torch.tensor(100.0, dtype=torch.float32))
        self.register_buffer("max_health", torch.tensor(100.0, dtype=torch.float32))

    def normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Normalize batch of tokens: (B, N, 15)
        Input: Raw Physics Values
        Output: Normalized Values (0-1 approx)
        """
        # Clone to avoid modifying input in-place if it matters
        norm = tokens.clone()
        
        # 0: Team ID (Keep as is, usually embedding index or 0/1)
        # 1: Health
        norm[..., 1] /= self.max_health
        # 2: Power
        norm[..., 2] /= self.max_power
        # 3,4: Pos
        norm[..., 3] /= self.world_size[0]
        norm[..., 4] /= self.world_size[1]
        # 5,6: Vel
        norm[..., 5] /= self.max_vel
        norm[..., 6] /= self.max_vel
        
        # 7,8: Acc (If used)
        
        # 9: Ang Vel (If raw is rad/s, maybe normalize by PI or MAX_TURN?)
        # Current collector saves raw ang_vel.
        # Let's assume input is rad/s.
        # Normalize by scalar? Or leave as is if small?
        # Usually ang_vel is small (< 5.0). Neural nets handle small floats fine.
        
        # 10,11: Attitude (Unit vector, already -1 to 1) -> No change.
        
        # 12: Shooting (0 or 1) -> No change.
        
        return norm

    def denormalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert back to raw physics values."""
        raw = tokens.clone()
        raw[..., 1] *= self.max_health
        raw[..., 2] *= self.max_power
        raw[..., 3] *= self.world_size[0]
        raw[..., 4] *= self.world_size[1]
        raw[..., 5] *= self.max_vel
        raw[..., 6] *= self.max_vel
        return raw
