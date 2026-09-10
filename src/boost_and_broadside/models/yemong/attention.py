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
    """Pre-norm transformer block with self-attention and optional read-only memories.

    Ordering: RMSNorm → MHSA → Residual → RMSNorm → FFN → Residual.
    Dead ships are masked out of key/value positions in attention so they
    cannot influence living ships.

    Args:
        model_config: Must supply d_model and n_heads.
        reads_bullets: Whether this layer reads projectile K/V.
        map_memory_dim: Width of map K/V inputs, or None to disable the read.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        reads_bullets: bool = False,
        map_memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        D = model_config.d_model

        self.n_heads = model_config.n_heads
        self.head_dim = D // model_config.n_heads
        self.d_model = D
        self.reads_bullets = reads_bullets
        self.reads_map_memory = map_memory_dim is not None

        self.norm1 = nn.RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        if map_memory_dim is not None:
            self.norm_map = nn.RMSNorm(map_memory_dim)
            self.kv_map = nn.Linear(map_memory_dim, 2 * D, bias=False)

        if reads_bullets:
            # Bullets are key/value only: no query, no output projection, no FFN.
            # That is 2*D^2 per bullet token against 16*D^2 for a full token, which
            # is what makes attending over every bullet affordable. Their own k/v
            # projection also means no adapter is needed to reconcile the bullet
            # encoder's latent space with the entity tokens' — this matrix absorbs it.
            self.norm_bullet = nn.RMSNorm(D)
            self.kv_bullet = nn.Linear(D, 2 * D, bias=False)

        self.norm2 = nn.RMSNorm(D)
        self.ffn = GatedMLP(D)

    def forward(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        bullets: torch.Tensor | None = None,
        bullet_mask: torch.Tensor | None = None,
        map_memory: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one transformer block.

        Args:
            x:          (B, N, D) entity token embeddings.
            alive_mask: (B, N) bool — True for entities that exist. Dead entities
                        are masked out of key/value positions so they cannot
                        influence living ones.
            bullets:    (B, NB, D) encoded bullet tokens, or None. Read as
                        key/value only; never updated and never queried.
            bullet_mask:(B, NB) bool — True for active ring-buffer slots.
            map_memory: (B, M, D_map) encoded map objects, read as K/V only.
            map_mask:   (B, M) bool — True for valid map-object slots.

        Returns:
            (B, N, D) updated entity tokens.
        """
        x = x + self._attn(self.norm1(x), alive_mask, bullets, bullet_mask, map_memory, map_mask)
        x = x + self.ffn(self.norm2(x))  # pre-norm FFN + residual
        return x

    def _key_bias(self, mask: torch.Tensor, like: torch.Tensor, batch: int) -> torch.Tensor:
        """Additive key-padding bias shaped (B, 1, 1, K), in the query's dtype.

        Build the bias in q's dtype, not x's: under bf16 autocast the qkv Linear
        emits bf16 while RMSNorm keeps x fp32, and an fp32 attn_mask on bf16
        q/k/v disqualifies the fused flash/mem-efficient SDPA kernels, silently
        falling back to the math kernel (materializes the full (B, H, N, N)
        scores — slower and a much larger activation peak).
        """
        key_mask = mask.view(batch, 1, 1, mask.shape[-1]).to(like.dtype)  # 1.0 = keep
        large_neg = torch.finfo(like.dtype).min / 2
        return (1.0 - key_mask) * large_neg

    def _attn(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None,
        bullets: torch.Tensor | None = None,
        bullet_mask: torch.Tensor | None = None,
        map_memory: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention plus optional cross-attention to read-only memories.

        The two attention outputs are summed into one residual and share this
        block's out_proj and FFN. Keeping them as separate softmaxes rather than
        one fused key set avoids letting a large auxiliary memory swamp the
        attention mass of ship keys and gives each memory its own learned K/V
        projection.

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
            # Mask out dead entities as keys — they cannot emit information.
            attn_bias = self._key_bias(alive_mask, q, B)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)

        if self.reads_bullets and bullets is not None:
            NB = bullets.shape[1]
            kb, vb = self.kv_bullet(self.norm_bullet(bullets)).chunk(2, dim=-1)
            kb = kb.view(B, NB, H, dh).permute(0, 2, 1, 3)  # (B, H, NB, dh)
            vb = vb.view(B, NB, H, dh).permute(0, 2, 1, 3)
            bullet_bias = self._key_bias(bullet_mask, q, B) if bullet_mask is not None else None
            bullet_out = F.scaled_dot_product_attention(
                q, kb, vb, attn_mask=bullet_bias, dropout_p=0.0
            )
            if bullet_mask is not None:
                # Softmax is shift-invariant, so masking *every* key does not zero
                # the output — it returns a uniform average over the dead slots.
                # An environment with nothing in flight is the common case (episode
                # start, or any lull), so the contribution is gated off explicitly
                # rather than left to leak an average of empty ring-buffer entries.
                any_active = bullet_mask.any(dim=-1).view(B, 1, 1, 1).to(bullet_out.dtype)
                bullet_out = bullet_out * any_active
            out = out + bullet_out

        if self.reads_map_memory and map_memory is not None and map_memory.shape[1]:
            NM = map_memory.shape[1]
            km, vm = self.kv_map(self.norm_map(map_memory)).chunk(2, dim=-1)
            km = km.view(B, NM, H, dh).permute(0, 2, 1, 3)
            vm = vm.view(B, NM, H, dh).permute(0, 2, 1, 3)
            map_bias = self._key_bias(map_mask, q, B) if map_mask is not None else None
            map_out = F.scaled_dot_product_attention(q, km, vm, attn_mask=map_bias, dropout_p=0.0)
            if map_mask is not None:
                any_map = map_mask.any(dim=-1).view(B, 1, 1, 1).to(map_out.dtype)
                map_out = map_out * any_map
            out = out + map_out

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        return self.out_proj(out)
