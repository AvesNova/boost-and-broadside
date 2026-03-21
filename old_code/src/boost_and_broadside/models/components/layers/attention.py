import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Projection for Relational Features to Head Dim (if sizes mismatch)
        # Assuming bias_features comes in as D_model or similar. 
        # Based on encoder, trunk outputs 64-dim features or similar? 
        # Actually in scaffolds, 'trunk_out' dim is determined by RelationalEncoder adapter:
        # adapter = Linear(128, d_model). So bias is (B, N, N, d_model).
        # We need to reshape it to heads.

    def forward(self, x, bias_features, mask=None):
        """
        Args:
            x: (B, T, N, D) - Ship States
            bias_features: (B, T, N, N, D) - Projected Geometry from Adapter
            mask: (B, T, N) or (Batch, N) - Alive mask
        """
        B, T, N, D = x.shape
        Batch = B * T
        
        # 1. Flatten Time
        x_flat = x.view(Batch, N, D)
        bias_flat = bias_features.view(Batch, N, N, D)
        
        # 2. QKV Projection
        # Optimized: Single Matmul
        qkv = self.qkv(x_flat) # (Batch, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 3. Reshape for SDPA (Batch, Heads, Seq, Dim)
        # Note: SDPA expects (B, H, L, E)
        q = q.view(Batch, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (Batch, H, N, D_h)
        k = k.view(Batch, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(Batch, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 4. Compute Rich Relational Bias (Q * B)
        # bias_flat is (Batch, N, N, D). Reshape to (Batch, N, N, H, D_h)
        b_geo = bias_flat.view(Batch, N, N, self.n_heads, self.head_dim)
        b_geo = b_geo.permute(0, 3, 1, 2, 4) # (Batch, H(0), N(1), N(2), D_h(3))
        
        # We want: Q_i * B_{ij}. 
        # Q: (Batch, H, N_i, D_h)
        # B: (Batch, H, N_i, N_j, D_h)
        # Result: (Batch, H, N_i, N_j)
        #
        # Einsum is cleanest here, but sometimes slower than matmul.
        # Q is broadcasted over N_j axis.
        # Let's use matmul if possible? 
        # Q.unsqueeze(3) -> (Batch, H, N, 1, D)
        # B -> (Batch, H, N, N, D)
        # (Q * B).sum(-1) -> (Batch, H, N, N)
        
        # This interaction retains the "Rich Vector" property:
        # The query vector specifically interacts with the geometry vector
        rel_scores = (q.unsqueeze(3) * b_geo).sum(dim=-1)
        
        # Scale for stability (since SDPA also scales, we must be careful)
        # SDPA does: softmax((QK^T + mask)/sqrt(d))
        # So we should strictly provide the "unscaled" bias if it's an additive mask
        # But wait, Q*B is effectively a dot product like Q*K. 
        # It should probably be scaled by 1/sqrt(d) inherently to match magnitude of Q*K ??
        # SDPA scales the *entire* sum (QK + mask) if is_causal=False?
        # No, SDPA scales QK^T. It does NOT scale the mask.
        # So we must manually scale our rel_scores by 1/sqrt(d) so it matches the scale of the pre-softmax logits.
        scale = self.head_dim ** -0.5
        rel_scores = rel_scores * scale
        
        # 5. Masking
        # SDPA takes an attn_mask of shape (Batch, H, N, N) or broadcastable
        if mask is not None:
            # mask is (Batch, N) -> Need (Batch, 1, 1, N_j) (Key Mask)
            # PyTorch SDPA binary mask: True=Keep, False=Drop? Or Boolean?
            # Standard SDPA mask: Float (-inf/0) or Bool (True=Kill).
            # Let's use Float mask for additive composition
            mask_flat = mask.view(Batch, 1, 1, N) # (Batch, 1, 1, N_j)
            
            # 0 for alive, -inf for dead
            # mask is True for Alive.
            # We want to ADD 0 where True, -inf where False
            # Check dtypes: half precision safety
            large_neg = -1e4 if x.dtype == torch.half else -1e9
            
            # Combine geometric scores with existence mask
            # We add rel_scores (Geometry) to the "base" mask (Existence)
            # But wait, rel_scores IS the bias.
            
            # masked_fill_ on the tensor
            # We clone to avoid in-place issues if any
            attn_bias = rel_scores.masked_fill(~mask_flat, large_neg)
        else:
            attn_bias = rel_scores
            
        # 6. SDPA
        # Output: (Batch, H, N, D_h)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.1 if self.training else 0.0)
        
        # 7. Reassemble
        out = out.permute(0, 2, 1, 3).reshape(Batch, N, D)
        out = self.proj(out)
        
        # Unflatten time
        out = out.view(B, T, N, D)
        
        return out
