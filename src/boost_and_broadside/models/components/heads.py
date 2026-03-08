import torch
import torch.nn as nn
from typing import List, TYPE_CHECKING

from boost_and_broadside.models.components.layers.utils import RMSNorm

if TYPE_CHECKING:
    from boost_and_broadside.models.components.soft_bins import SoftBinSpec


class TeamPoolingHead(nn.Module):
    """
    Pools per-ship token representations into a single team-level vector via
    cross-attention, then projects to an output.

    Used for both value (on state tokens) and reward (on action tokens).

    Architecture:
        team_token:  learnable query  (1, 1, d_model)
        norm:        RMSNorm applied to keys/values before attention
        pooler:      MultiheadAttention (team_token queries ship tokens)
        mlp:         Linear → SiLU → Linear  (d_model → out_dim)

    forward(x, key_padding_mask) → (B*T, out_dim)
        x:                (B*T, N, d_model)  — per-ship tokens
        key_padding_mask: (B*T, N) bool — True = ignore that ship

    Returns (B*T, out_dim). Callers reshape to (B, T, out_dim).
    """

    def __init__(self, d_model: int, out_dim: int, n_heads: int = 4):
        super().__init__()
        self.team_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = RMSNorm(d_model)
        self.pooler = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """
        x: (BT, N, d_model)
        key_padding_mask: (BT, N) bool, True = ignore
        returns: (BT, out_dim)
        """
        BT = x.shape[0]
        # Guard: if all ships are masked in a batch element, unblock one to avoid NaN softmax
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            key_padding_mask = key_padding_mask & ~all_masked

        q = self.team_token.expand(BT, -1, -1)          # (BT, 1, D)
        kv = self.norm(x)                               # (BT, N, D)
        team_vec, _ = self.pooler(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.mlp(team_vec.squeeze(1))             # (BT, out_dim)


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
        # We reuse the TeamEvaluator logic/structure as it was in the base architecture
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

class PairwiseRelationalHead(nn.Module):
    """
    Predicts pairwise relational deltas between ships i and j.
    Input: Concatenated tokens (Zi, Zj).
    Output: 4 targets (drel_x, drel_y, drel_vx, drel_vy).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4)
        )

    def forward(self, z):
        """
        z: (B, T, N, N, 2*D) - pairwise concatenated tokens
        returns: (B, T, N, N, 4)
        """
        return self.net(z)


class SoftBinnedWorldHead(nn.Module):
    """
    Produces soft-binned logits for each target field specified in `specs`.

    Architecture:
        Shared trunk: Linear(d_model, d_model) → RMSNorm → SiLU
        Per-spec head: Linear(d_model, spec.n_bins)   (no activation — raw logits)

    forward(x) returns List[Tensor], one (*, n_bins) per spec.
    """

    def __init__(self, d_model: int, specs: "List[SoftBinSpec]"):
        super().__init__()
        self.specs = specs

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(d_model, spec.n_bins) for spec in specs
        ])

    def forward(self, x: torch.Tensor) -> "List[torch.Tensor]":
        """
        x: (*, d_model)
        Returns: list of (*, n_bins_i) tensors in spec order.
        """
        z = self.trunk(x)
        return [head(z) for head in self.heads]
