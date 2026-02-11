import torch
import torch.nn as nn

from boost_and_broadside.agents.mamba_bb import RMSNorm

class TeamEvaluator(nn.Module):
    """
    Evaluates the value of a team's state from the perspective of the Team.
    
    Architecture:
    1. Attention Pooling: Aggregates ship latents into a single Team Vector using a learned query token.
    2. Value Head: Predicts discounted future returns (Value).
    3. Reward Head: Predicts immediate frame reward.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Learned Team Token (Query)
        # We use a single token to query the entire battlefield state.
        self.team_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention Pooling (Q=Token, K=V=ShipLatents)
        # Using Multi-Head Attention for robustness
        self.n_heads = 4
        self.attn = nn.MultiheadAttention(d_model, num_heads=self.n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)
        
        # Heads
        # Value: (256 -> 256 -> 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
        
        # Reward: (256 -> 256 -> 3) - 3 Components
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Ships, D) - Ship Latents (from Actor Stage 2).
            mask: (Batch, Ships) - Optional boolean mask (True=Alive/Valid, False=Padding/Dead).
                  Note: nn.MultiheadAttention expects key_padding_mask where True=Ignore.
                  So we invert our standard 'alive' mask.
                  
        Returns:
            value: (Batch, 1)
            reward: (Batch, 3) - 3 components of reward.
        """
        B, N, D = x.shape
        
        # Expand Team Token for Batch: (Batch, 1, D)
        query = self.team_token.expand(B, -1, -1)
        
        # Key Padding Mask
        # Pytorch MHA expects: (Batch, SrcLen) where True means IGNORE.
        # Our mask is usually 'alive' (True=Keep). So we invert it.
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask # True where we should ignore (dead)
            
        # Norm before Attention? MambaBB uses Pre-Norm.
        # The input 'x' is typically finding its way out of a block, so let's norm the input to pooler
        # similar to other blocks.
        x_norm = self.norm(x)
        
        # Note: MHA handles projection internally.
        team_vector, _ = self.attn(query, x_norm, x_norm, key_padding_mask=key_padding_mask)
        
        # Heads
        # Squeeze the sequence dim (1)
        tv = team_vector.squeeze(1) # (B, D)
        
        # Detach for Value Head so gradients don't flow back to backbone
        value = self.value_head(tv.detach())
        
        reward = self.reward_head(tv)
        
        return value, reward
