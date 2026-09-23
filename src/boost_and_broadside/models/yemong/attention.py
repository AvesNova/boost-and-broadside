"""Transformer block for relational reasoning over ship tokens.

Implements a single pre-norm transformer block:
    RMSNorm → Multi-Head Self-Attention → Residual
    RMSNorm → GatedMLP (SwiGLU, 4x expand) → Residual

Pre-norm placement follows modern best practices (GPT-style / LLaMA-style):
it stabilizes training by ensuring gradients flow cleanly through residuals.

Input/output convention: (B, N, D) — no time dimension is handled here.
The caller reshapes (B*T, N, D) if multiple timesteps are needed at once.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.models.yemong.relation import RelationalBias
from boost_and_broadside.models.yemong.rope import apply_rotary


@dataclass(frozen=True)
class SpatialGeometry:
    """Per-token geometry every spatial sublayer in one forward pass shares.

    Built once by the policy and passed down, because all of it is a function of
    the observation rather than of the layer: recomputing the rotary tables in
    each of the four spatial sublayers would pay for identical trigonometry four
    times and allocate four copies of it.

    Every tensor is laid out to broadcast against a head-major
    ``(B, tokens, heads, head_dim)`` query, i.e. ``(B, tokens, 1, pairs)``.

    Attributes:
        entity:  ``(cos, sin)`` for the entity tokens that carry queries.
        bullet:  ``(cos, sin)`` for bullet key/value tokens, or None.
        map_memory: ``(cos, sin)`` for K/V-only map tokens, or None.
        relation: ``(B, T, T, F)`` shared pairwise relational scalars for entity
            self-attention, or None. Built once and mapped to a per-head bias by
            each sublayer's own weights.
    """

    entity: tuple[torch.Tensor, torch.Tensor] | None = None
    bullet: tuple[torch.Tensor, torch.Tensor] | None = None
    map_memory: tuple[torch.Tensor, torch.Tensor] | None = None
    relation: torch.Tensor | None = None


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

    Entity and map keys carry no padding mask. Every ship is revealed to both
    teams on the decision it spawns and ``BeliefTracker.valid`` is sticky, so
    token validity is constant-true after spawn -- measured at 0 of 33,280 false
    over a Frontline rollout -- and map objects are static. Passing any
    ``attn_mask`` to ``scaled_dot_product_attention`` disqualifies the fused
    flash kernel, so a mask that encodes nothing is expensive rather than free.
    Bullets keep theirs: a ring-buffer slot genuinely can be empty.

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

        self.n_heads = model_config.spatial_heads
        self.head_dim = model_config.spatial_head_dim
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

        # Entity self-attention only. Bullets and K/V map memories are separate
        # softmaxes over tokens of a different kind; a shared relation function
        # over mixed kinds would have to mean the same thing for a ship pair and
        # a ship/bullet pair, and it does not.
        self.relational = RelationalBias(self.n_heads) if model_config.relational_bias else None

        self.norm2 = nn.RMSNorm(D)
        self.ffn = GatedMLP(D)

    def forward(
        self,
        x: torch.Tensor,
        bullets: torch.Tensor | None = None,
        bullet_mask: torch.Tensor | None = None,
        map_memory: torch.Tensor | None = None,
        geometry: SpatialGeometry | None = None,
    ) -> torch.Tensor:
        """Apply one transformer block.

        Args:
            x:          (B, N, D) entity token embeddings.
            bullets:    (B, NB, D) encoded bullet tokens, or None. Read as
                        key/value only; never updated and never queried.
            bullet_mask:(B, NB) bool — True for active ring-buffer slots.
            map_memory: (B, M, D_map) encoded map objects, read as K/V only.
            geometry:   Shared per-token rotary tables, or None to leave Q/K
                        unrotated.

        Returns:
            (B, N, D) updated entity tokens.
        """
        x = x + self._attn(self.norm1(x), bullets, bullet_mask, map_memory, geometry)
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
        bullets: torch.Tensor | None = None,
        bullet_mask: torch.Tensor | None = None,
        map_memory: torch.Tensor | None = None,
        geometry: SpatialGeometry | None = None,
    ) -> torch.Tensor:
        """Self-attention plus optional cross-attention to read-only memories.

        Map objects are folded into the *same* softmax as ships -- one
        non-square attention, N queries over N+M keys. Bullets keep a separate
        one: their ring buffer needs a key mask, and a masked key set would
        disqualify the fused SDPA kernel for the ship/map read it was merged into.

        Returns:
            (B, N, D) attention output.
        """
        B, N, D = x.shape
        H, dh = self.n_heads, self.head_dim

        qkv = self.qkv(x)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, dh)  # (B, N, H, dh) — heads last-but-one for rotation
        k = k.view(B, N, H, dh)
        if geometry is not None and geometry.entity is not None:
            cos, sin = geometry.entity
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, dh)
        k = k.permute(0, 2, 1, 3)
        v = v.view(B, N, H, dh).permute(0, 2, 1, 3)

        # Map objects join the ship key set rather than getting their own
        # softmax: one non-square attention of N queries over N+M keys. They are
        # keys and values only -- never queries, never updated, never through the
        # FFN or the temporal path -- so the map contributes geometry to a ship's
        # read without ever being carried by the trunk.
        #
        # Deliberately reversing the earlier two-softmax design, which summed an
        # independent map read into the same residual to stop a large auxiliary
        # memory swamping the attention mass of ship keys. Under one softmax they
        # do compete. That competition is the point: a ship near a field boundary
        # *should* be able to spend its attention there instead of on a distant
        # ally, and two separate softmaxes cannot express that trade at all.
        if self.reads_map_memory and map_memory is not None and map_memory.shape[1]:
            M = map_memory.shape[1]
            km, vm = self.kv_map(self.norm_map(map_memory)).chunk(2, dim=-1)
            km = km.view(B, M, H, dh)
            if geometry is not None and geometry.map_memory is not None:
                km = apply_rotary(km, *geometry.map_memory)
            k = torch.cat([k, km.permute(0, 2, 1, 3)], dim=2)  # (B, H, N+M, dh)
            v = torch.cat([v, vm.view(B, M, H, dh).permute(0, 2, 1, 3)], dim=2)

        # No key-padding bias: token validity is constant-true and map objects are
        # static geometry (see the class docstring), so attn_mask stays None over
        # the whole fused key set and SDPA can pick the flash kernel.
        attn_bias = None
        if self.relational is not None and geometry is not None and geometry.relation is not None:
            # The bias is built over every token pair, so its columns already
            # line up with the fused key set -- the map keys sit in the same
            # order behind the ships. Only the query rows need narrowing: ships
            # alone are queries here, where full-attention mode queries with all
            # of them. Ship/map pairs therefore get a learned relational term,
            # the same as they did in full-attention mode.
            #
            # Note this reintroduces a non-None attn_mask and so gives up the
            # flash kernel. That is the standing cost of relational bias, which
            # is off by default for exactly this reason.
            attn_bias = self.relational(geometry.relation).to(q.dtype)
            attn_bias = attn_bias[..., : q.shape[2], : k.shape[2]]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)

        if self.reads_bullets and bullets is not None:
            NB = bullets.shape[1]
            kb, vb = self.kv_bullet(self.norm_bullet(bullets)).chunk(2, dim=-1)
            kb = kb.view(B, NB, H, dh)
            if geometry is not None and geometry.bullet is not None:
                # A bullet is a world position on the same toroid, so its key is
                # rotated on the same x/y basis as the ship query it meets. It
                # carries no heading, so the attitude block's rotation is the
                # identity for it (see ``SpatialRotary``).
                kb = apply_rotary(kb, *geometry.bullet)
            kb = kb.permute(0, 2, 1, 3)  # (B, H, NB, dh)
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

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        return self.out_proj(out)
