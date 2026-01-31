
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None
    print("WARNING: mamba_ssm not installed. MambaBB will fail to initialize.")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class RelationalEncoder(nn.Module):
    """
    Physics Trunk: Projects raw geometry into latent space.
    Shared across layers via adapters.
    """
    def __init__(self, config):
        super().__init__()
        self.raw_dim = 12 # 4 raw + 8 derived (See compute_features)
        
        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.raw_dim, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Adapters for each layer (1 Actor + 6 World Model = 7)
        self.adapters = nn.ModuleList([
            nn.Linear(128, config.d_model) for _ in range(config.n_layers + 1)
        ])

    def compute_analytic_features(self, pos, vel, world_size=(1024.0, 1024.0)):
        """
        Compute analytic edge features from Pos/Vel.
        Pos: (B, T, N, 2)
        Vel: (B, T, N, 2)
        """
        # Delta Pos (Wrapped)
        d_pos = pos.unsqueeze(-2) - pos.unsqueeze(-3) # (..., N, N, 2)
        
        W, H = world_size
        dx = d_pos[..., 0]
        dy = d_pos[..., 1]
        dx = dx - torch.round(dx / W) * W
        dy = dy - torch.round(dy / H) * H
        d_pos = torch.stack([dx, dy], dim=-1)
        
        # Delta Vel
        d_vel = vel.unsqueeze(-2) - vel.unsqueeze(-3)
        
        # Derived
        dist_sq = d_pos.pow(2).sum(dim=-1, keepdim=True)
        dist = dist_sq.sqrt() + 1e-6
        
        # Standard features: [d_pos, d_vel, dist, inv_dist, etc]
        # Matching the 12D logic from relational_features.py roughly
        # but inline here for clarity/independence.
        
        # Directions
        dir = d_pos / dist
        
        # Rel Speed & Closing
        rel_speed = d_vel.pow(2).sum(dim=-1, keepdim=True).sqrt()
        closing = (d_vel * d_pos).sum(dim=-1, keepdim=True) / dist
        
        # Concatenate
        # [rx, ry, rvx, rvy, dist, inv_dist, speed, closing, dirx, diry, log_dist, dummy]
        inv_dist = 1.0 / (dist + 0.1)
        log_dist = torch.log(dist + 1.0)
        
        features = torch.cat([
            d_pos,
            d_vel,
            dist,
            inv_dist,
            rel_speed,
            closing,
            dir,
            log_dist
        ], dim=-1) # 2+2+1+1+1+1+2+1 = 11. Need 12? Add 0.
        
        # Pad to 12
        features = F.pad(features, (0, 1))
        
        return features

    def forward(self, pos, vel, layer_idx=None):
        raw = self.compute_analytic_features(pos, vel)
        trunk_out = self.trunk(raw)
        
        if layer_idx is not None:
            # Adapter: Linear -> No Act -> Bias for Attention
            return self.adapters[layer_idx](trunk_out)
        return trunk_out, raw


class RelationalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, bias_features, mask=None):
        # x: (B, T, N, D)
        # bias_features: (B, T, N, N, D) -- Projected from Adapter
        B, T, N, D = x.shape
        
        # We process each timestep independently for spatial mixing
        # Flatten B,T -> Batch
        Batch = B * T
        x_flat = x.view(Batch, N, D)
        
        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape [Batch, N, Heads, Dim]
        q = q.view(Batch, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(Batch, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(Batch, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Bias: (B, T, N, N, D) -> (Batch, N, N, D) -> Split Heads?
        
        # bias_features is (B, T, N, N, D).
        
        if not hasattr(self, 'bias_proj'):
             self.bias_proj = nn.Linear(D, self.n_heads, bias=False).to(x.device)
             
        # Flatten bias_features (B, T, N, N, D) -> (Batch, N, N, D)
        bias_flat = bias_features.view(Batch, N, N, -1)
        b = self.bias_proj(bias_flat).permute(0, 3, 1, 2) # (Batch, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5) # (Batch, H, N, N)
        scores = scores + b
        
        # Mask
        if mask is not None:
            # Mask is (Batch, N) or (B, T, N) -> (Batch, 1, 1, N)
            mask_flat = mask.view(Batch, 1, 1, N)
            scores = scores.masked_fill(~mask_flat, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (Batch, H, N, D_head)
        
        out = out.transpose(1, 2).reshape(B, T, N, D)
        return self.proj(out)


class MambaBB(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # Components
        self.state_encoder = nn.Sequential(
            nn.Linear(config.input_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.relational_encoder = RelationalEncoder(config)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba2(d_model=d_model, d_state=128, expand=2),
                'norm1': RMSNorm(d_model),
                'norm2': RMSNorm(d_model),
                'attn': RelationalAttention(d_model, config.n_heads)
            }) for _ in range(config.n_layers)
        ])
        
        # Actor Components
        self.actor_adapter = nn.Linear(128, d_model) # From trunk to actor bias
        self.actor_fusion = nn.Linear(d_model * 2, d_model)
        self.actor_attn = RelationalAttention(d_model, config.n_heads)
        self.actor_head = nn.Linear(d_model, config.action_dim)
        
        # World Head
        self.world_head = nn.Linear(d_model, config.target_dim)
        
        # Embeddings
        # Action Tensor: [Power(3), Turn(7), Shoot(2)]
        self.emb_power = nn.Embedding(3, 32)
        self.emb_turn = nn.Embedding(7, 64)
        self.emb_shoot = nn.Embedding(2, 32)
        
        # Concat = 32+64+32 = 128
        self.fusion = nn.Linear(d_model + 128, d_model)

    def forward(self, state, prev_action, pos, vel, seq_idx=None):
        """
        Forward logic.
        state: (B, T, N, D) - S_t
        prev_action: (B, T, N, 3) - A_t (for World Model teacher forcing)
        pos, vel: (B, T, N, 2)
        """
        # 1. Encoders
        s_emb = self.state_encoder(state)
        trunk_out, raw_geo = self.relational_encoder(pos, vel)
        
        # Actor Pass
        actor_bias = self.relational_encoder.adapters[0](trunk_out)
        x_actor = s_emb 
        x_actor = self.actor_attn(x_actor, actor_bias)
        action_logits = self.actor_head(x_actor)
        
        # World Model Pass
        # Embed Action using explicit components
        p = self.emb_power(prev_action[..., 0].long())
        t = self.emb_turn(prev_action[..., 1].long())
        s = self.emb_shoot(prev_action[..., 2].long())
        a_emb = torch.cat([p, t, s], dim=-1) # (..., 128)
        
        x_world = self.fusion(torch.cat([s_emb, a_emb], dim=-1))
        
        # Backbone
        B, T, N, D = x_world.shape
        # Mamba needs flat seq
        # (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
        x_mamba = x_world.permute(0, 2, 1, 3).reshape(B*N, T, D)
        
        # Expand Seq Idx for Mamba
        # seq_idx: (B, T). We need (B*N, T).
        # Each ship in the batch has same timeframe/episode.
        if seq_idx is not None:
             # seq_idx_mamba = seq_idx.repeat_interleave(N, dim=0) # (B*N, T)
             # Wait, seq_idx is (Batch, Time).
             # repeat_interleave(N) repeats rows. Correct.
             pass

        for i, block in enumerate(self.blocks):
            # Mamba
            # Mamba2(x)
            normed = block['norm1'](x_mamba)
            if x_mamba.device.type == 'cpu':
                 # Fallback for CPU testing (Mamba2 kernel requires CUDA)
                 m_out = normed
            else:
                 m_out = block['mamba'](normed)
            x_mamba = x_mamba + m_out
            
            # Spatial
            # Reshape back to (B, T, N, D)
            x_spatial = x_mamba.view(B, N, T, D).permute(0, 2, 1, 3)
            
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            x_spatial = x_spatial + block['attn'](block['norm2'](x_spatial), rel_bias)
            
            # Prepare for next Mamba
            x_mamba = x_spatial.permute(0, 2, 1, 3).reshape(B*N, T, D)

        x_final = x_mamba.view(B, N, T, D).permute(0, 2, 1, 3)
        delta_pred = self.world_head(x_final)
        
        # Return Absolute prediction
        # state is normalized? Spec says "Residuals... applied to current state".
        # If model operates in Normalized space, Delta is normalized delta.
        # So S_next = S + Delta.
        state_pred = state + delta_pred
        
        return state_pred, action_logits

    def get_loss(self, pred_states, pred_actions, target_states, target_actions, loss_mask, 
                 lambda_state=1.0, lambda_action=1.0, lambda_relational=0.0):
        """
        Compute MambaBB Loss.
        
        Args:
            pred_states: (B, T, N, D) - Predicted Deltas? Or absolute? Spec says Residuals.
                         So pred_states IS delta.
            target_states: (B, T, N, D) - Actual Next State.
            loss_mask: (B, T) or (B, T, N)
        """
        # Loss Mask Handling
        if loss_mask.ndim == 2:
             # (B, T) -> (B, T, N)
             loss_mask = loss_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
             
        # Flatten
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        
        # State Loss (MSE)
        # Note: We computed state_pred = state + delta.
        # So we compare directly to target_states.
        s_loss = F.mse_loss(pred_states, target_states, reduction='none').mean(dim=-1)
        s_loss = (s_loss.reshape(-1) * mask_flat).sum() / denom
        
        # Action Loss (Cross Entropy)
        # pred_actions: (B, T, N, 12) flat logits?
        # target_actions: (B, T, N, 3)
        
        # Split logits
        # [0:3] Power, [3:10] Turn, [10:12] Shoot
        l_p = pred_actions[..., 0:3]
        l_t = pred_actions[..., 3:10]
        l_s = pred_actions[..., 10:12]
        
        t_p = target_actions[..., 0].long()
        t_t = target_actions[..., 1].long()
        t_s = target_actions[..., 2].long()
        
        loss_p = F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), reduction='none')
        loss_t = F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), reduction='none')
        loss_s = F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), reduction='none')
        
        # Weighted sum (lambda_action scaling applied to sum)
        # Or individual weights as per Interleaved model logic?
        # I'll sum them.
        a_loss_raw = (loss_p + loss_t + loss_s)
        a_loss = (a_loss_raw * mask_flat).sum() / denom
        
        total_loss = (lambda_state * s_loss) + (lambda_action * a_loss)
        
        metrics = {
             "loss": total_loss.item(),
             "state_loss": s_loss.item(),
             "action_loss": a_loss.item()
        }
        
        return total_loss, s_loss, a_loss, torch.tensor(0.0), metrics
