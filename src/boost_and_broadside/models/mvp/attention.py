"""Transformer block for relational reasoning over ship tokens.

Implements a single pre-norm transformer block:
    RMSNorm → Multi-Head Self-Attention → Residual
    RMSNorm → GatedMLP (SwiGLU, 4x expand) → Residual

Pre-norm placement follows modern best practices (GPT-style / LLaMA-style):
it stabilizes training by ensuring gradients flow cleanly through residuals.

Input/output convention: (B, N, D) — no time dimension is handled here.
The caller reshapes (B*T, N, D) if multiple timesteps are needed at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from boost_and_broadside.config import ModelConfig


class GatedMLP(nn.Module):
    """SwiGLU-style gated FFN: down_proj(gelu(gate_proj(x)) * up_proj(x)).

    Args:
        d_model: Input and output dimension D.
        expand:  Hidden expansion factor (default 4×).
    """

    def __init__(self, d_model: int, expand: int = 4) -> None:
        super().__init__()
        hidden = expand * d_model
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with full self-attention and FFN.

    Ordering: RMSNorm → MHSA → Residual → RMSNorm → FFN → Residual.
    Dead ships are masked out of key/value positions in attention so they
    cannot influence living ships.

    Args:
        model_config: Must supply d_model and n_heads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        D = model_config.d_model

        self.n_heads = model_config.n_heads
        self.head_dim = D // model_config.n_heads
        self.d_model = D

        self.norm1 = nn.RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        self.norm2 = nn.RMSNorm(D)
        self.ffn = GatedMLP(D)

    def forward(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one transformer block.

        Args:
            x:          (B, N, D) ship token embeddings.
            alive_mask: (B, N) bool — True for ships that exist. Dead ships
                        are masked out of key/value positions so they cannot
                        influence living ships.

        Returns:
            (B, N, D) updated ship tokens.
        """
        x = x + self._attn(self.norm1(x), alive_mask)  # pre-norm attn + residual
        x = x + self.ffn(self.norm2(x))  # pre-norm FFN + residual
        return x

    def _attn(self, x: torch.Tensor, alive_mask: torch.Tensor | None) -> torch.Tensor:
        """Multi-head self-attention with optional alive masking (no residual/norm).

        Args:
            x:          (B, N, D) pre-normed token embeddings.
            alive_mask: (B, N) bool or None.

        Returns:
            (B, N, D) attention output.
        """
        B, N, D = x.shape
        H, dh = self.n_heads, self.head_dim

        qkv = self.qkv(x)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, dh).permute(0, 2, 1, 3)  # (B, H, N, dh)
        k = k.view(B, N, H, dh).permute(0, 2, 1, 3)
        v = v.view(B, N, H, dh).permute(0, 2, 1, 3)

        attn_bias = None
        if alive_mask is not None:
            # Mask out dead ships as keys — they cannot emit information.
            # Shape: (B, 1, 1, N) broadcastable over (B, H, N_q, N_k).
            key_mask = alive_mask.view(B, 1, 1, N).float()  # 1.0 = alive
            large_neg = torch.finfo(x.dtype).min / 2
            attn_bias = (1.0 - key_mask) * large_neg  # (B, 1, 1, N)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )  # (B, H, N, dh)

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        return self.out_proj(out)
