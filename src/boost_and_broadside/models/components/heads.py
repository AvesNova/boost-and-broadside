import torch
import torch.nn as nn
from boost_and_broadside.models.components.layers.utils import RMSNorm

class ActorHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class WorldHead(nn.Module):
    def __init__(self, d_model: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, target_dim)
        )
    def forward(self, x):
        return self.net(x)

class ValueHead(nn.Module):
    """
    Predicts Value and Reward components.
    Often shared or closely related in the architecture.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # We reuse the TeamEvaluator logic/structure as it was in MambaBB
        # But we decouple it into a head here if possible, or we keep it as a component
        # For now, let's implement the head logic directly
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1) # Value scalar
        )
        # Assuming rewards are also scalar or small vector
        self.reward_net = nn.Sequential(
             nn.Linear(d_model, d_model),
             RMSNorm(d_model),
             nn.SiLU(),
             nn.Linear(d_model, 1) 
        )

    def forward(self, x):
        return self.net(x), self.reward_net(x)
