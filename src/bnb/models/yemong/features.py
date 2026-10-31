"""Small feature transforms of the ship token (spec 4)."""

import torch

from .critic import symlog
from .shots import time_features

RESPAWN_HORIZON = 16.0  # s
RESPAWN_HARMONICS = 4

__all__ = ["local_presence", "respawn_features", "symlog"]


def respawn_features(seconds: torch.Tensor, horizon: float = RESPAWN_HORIZON,
                     harmonics: int = RESPAWN_HARMONICS) -> torch.Tensor:
    """(...) respawn wait (s) to (..., 1 + 2 harmonics)."""
    return time_features(seconds, horizon, harmonics)


def local_presence(position: torch.Tensor, team: torch.Tensor, source: torch.Tensor,
                   radius: float) -> torch.Tensor:
    """(..., N, 2): ln(1 + Gaussian-weighted counts) of allies (same team, not itself) and
    enemies (the other ship team) around each token, from the tokens `source` allows.

    position (..., N, 2); team (..., N): 0 or 1 for ships, anything else for a token that is
    not a ship (its presence is zero).
    """
    difference = position[..., :, None, :] - position[..., None, :, :]
    weight = torch.exp(-(difference**2).sum(-1) / (2.0 * radius * radius))  # (..., N, N)
    ship = (team == 0) | (team == 1)
    contributes = (source & ship)[..., None, :]
    same = team[..., :, None] == team[..., None, :]
    itself = torch.eye(team.shape[-1], dtype=torch.bool, device=team.device)
    ally = torch.where(contributes & same & ~itself, weight, 0.0).sum(-1)
    enemy = torch.where(contributes & ~same, weight, 0.0).sum(-1)
    presence = torch.log1p(torch.stack([ally, enemy], -1))
    return torch.where(ship[..., None], presence, 0.0)
