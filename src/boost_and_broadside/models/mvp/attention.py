"""Relational self-attention over the ship tokens at a single timestep.

Implements standard multi-head self-attention with an alive-ship mask and
optional pairwise relational features as per-head attention biases.

Input/output convention: (B, N, D) — no time dimension is handled here.
The caller reshapes (B*T, N, D) if multiple timesteps are needed at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.models.mvp.relational_features import RelationalFeatures


class RelationalSelfAttention(nn.Module):
    """Multi-head self-attention over N ship tokens with alive masking and
    pairwise geometric attention biases.

    Args:
        model_config: Must supply d_model and n_heads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.n_heads  = model_config.n_heads
        self.head_dim = model_config.d_model // model_config.n_heads
        self.d_model  = model_config.d_model

        self.qkv      = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.norm     = nn.LayerNorm(self.d_model)

        self.relational = RelationalFeatures(model_config.n_heads)

    def forward(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        obs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute relational self-attention.

        Args:
            x:          (B, N, D) ship token embeddings.
            alive_mask: (B, N) bool — True for ships that exist. Dead ships
                        are masked out of the key/value positions so they
                        cannot influence living ships.
            obs:        Optional obs dict with "pos", "vel", "att" keys
                        (B, N, ...). When provided, pairwise geometric biases
                        are added to the pre-softmax attention logits.

        Returns:
            (B, N, D) updated ship tokens (residual connection applied).
        """
        B, N, D = x.shape
        H, dh   = self.n_heads, self.head_dim

        qkv = self.qkv(x)                                          # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for SDPA: (B, H, N, dh)
        q = q.view(B, N, H, dh).permute(0, 2, 1, 3)
        k = k.view(B, N, H, dh).permute(0, 2, 1, 3)
        v = v.view(B, N, H, dh).permute(0, 2, 1, 3)

        attn_bias = None
        if alive_mask is not None:
            # Mask out dead ships as keys — they cannot emit information
            # Shape: (B, 1, 1, N) broadcastable over (B, H, N_q, N_k)
            key_mask  = alive_mask.view(B, 1, 1, N).float()       # 1.0 = alive
            large_neg = torch.finfo(x.dtype).min / 2
            attn_bias = (1.0 - key_mask) * large_neg               # (B, 1, 1, N)

        if obs is not None:
            # Pairwise relational bias: (B, H, N, N)
            rel_bias  = self.relational(obs)
            attn_bias = rel_bias if attn_bias is None else attn_bias + rel_bias

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )                                                           # (B, H, N, dh)

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)            # (B, N, D)
        out = self.out_proj(out)

        # Residual + layer-norm
        return self.norm(x + out)
