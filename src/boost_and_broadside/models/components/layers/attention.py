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

    def forward(self, x, bias_features, mask=None):
        """
        Args:
            x: (B, T, N, D) - Ship States
            bias_features: (B, T, N, N, D) - Projected Geometry from Adapter
            mask: (B, T, N) or (Batch, N) - Alive mask
        """
        B, T, N, D = x.shape
        Batch = B * T
        
        # Flatten time into batch
        x_flat = x.view(Batch, N, D)
        bias_flat = bias_features.view(Batch, N, N, D)
        
        # Project QKV
        qkv = self.qkv(x_flat) # (Batch, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to Heads: (Batch, N, H, D_head)
        q = q.view(Batch, N, self.n_heads, self.head_dim)
        k = k.view(Batch, N, self.n_heads, self.head_dim)
        v = v.view(Batch, N, self.n_heads, self.head_dim)
        
        # Reshape Bias to Heads: (Batch, N, N, H, D_head)
        b_geo = bias_flat.view(Batch, N, N, self.n_heads, self.head_dim)
        
        # Injection
        k_pairwise = k.unsqueeze(1) + b_geo # (Batch, N(i), N(j), H, D_head)
        v_pairwise = v.unsqueeze(1) + b_geo # (Batch, N(i), N(j), H, D_head)
        
        # Attention Scores
        scores = (q.unsqueeze(2) * k_pairwise).sum(dim=-1) # (Batch, N(i), N(j), H)
        scores = scores.permute(0, 3, 1, 2) # (Batch, H, N, N)
        scores = scores * (self.head_dim ** -0.5)
        
        # Mask
        if mask is not None:
            mask_flat = mask.view(Batch, 1, 1, N) # (Batch, 1, 1, N_j)
            scores = scores.masked_fill(~mask_flat, float('-inf'))
        
        # Softmax Stability
        max_scores, _ = scores.max(dim=-1, keepdim=True)
        is_nan_row = (max_scores == float('-inf'))
        scores_safe = torch.where(is_nan_row, torch.zeros_like(scores), scores)
        attn = F.softmax(scores_safe, dim=-1)
        attn = attn.masked_fill(is_nan_row, 0.0)
        
        # Weighted Sum
        out = torch.einsum('bhij, bijhd -> bihd', attn, v_pairwise)
        
        # Concatenate heads
        out = out.reshape(Batch, N, D)
        out = out.view(B, T, N, D)
        
        return self.proj(out)
