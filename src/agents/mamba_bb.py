
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
        # Features: [d_pos(2), d_vel(2), dist(1), inv(1), speed(1), close(1), dir(2), log(1)] = 11
        # + Fourier: d_pos(2) * 8 bands * 2(sin/cos) = 32
        # Total ~ 43 + padding -> 64?
        # Let's use 8 bands.
        self.num_bands = 8
        self.raw_dim = 11 + (2 * self.num_bands * 2) # 11 + 32 = 43.
        # Pad to 64 for nice numbers
        self.padded_dim = 64
        
        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.padded_dim, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Adapters for each layer (1 Actor + 6 World Model = 7)
        self.adapters = nn.ModuleList([
            nn.Linear(128, config.d_model) for _ in range(config.n_layers + 1)
        ])

    def compute_analytic_features(self, pos, vel, att=None, world_size=(1024.0, 1024.0)):
        """
        Compute analytic edge features from Pos/Vel/Attitude.
        Pos: (B, T, N, 2)
        Vel: (B, T, N, 2)
        Att: (B, T, N, 2) [cos, sin]
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
        
        # Standard features
        dir = d_pos / dist
        rel_speed = d_vel.pow(2).sum(dim=-1, keepdim=True).sqrt()
        closing = (d_vel * d_pos).sum(dim=-1, keepdim=True) / dist
        inv_dist = 1.0 / (dist + 0.1)
        log_dist = torch.log(dist + 1.0)
        tti = dist / (closing.clamp(min=1e-3))
        
        # Relational Geometry (ATA, AA, HCA)
        if att is not None:
             # att is (..., N, 2) [cos, sin]
             att_i = att.unsqueeze(-2) # (..., N, 1, 2)
             att_j = att.unsqueeze(-3) # (..., 1, N, 2)
             
             # ATA: Angle between my heading and target
             cos_ata = (dir * att_i).sum(dim=-1, keepdim=True)
             sin_ata = (dir[..., 0] * att_i[..., 1] - dir[..., 1] * att_i[..., 0]).unsqueeze(-1)
             
             # AA: Angle between target's heading and me (from target's perspective)
             # Line of sight from target to me is -dir
             cos_aa = (-dir * att_j).sum(dim=-1, keepdim=True)
             sin_aa = (-dir[..., 0] * att_j[..., 1] + dir[..., 1] * att_j[..., 0]).unsqueeze(-1)
             
             # HCA: Angle between our headings
             cos_hca = (att_i * att_j).sum(dim=-1, keepdim=True)
             sin_hca = (att_i[..., 0] * att_j[..., 1] - att_i[..., 1] * att_j[..., 0]).unsqueeze(-1)
        else:
             # Fallback if no attitude provided (e.g. initial steps of dreaming)
             cos_ata = sin_ata = cos_aa = sin_aa = cos_hca = sin_hca = torch.zeros_like(dist)

        # Fourier Encoding of d_pos (Normalized roughly by world size/scale)
        # Scale d_pos to [-PI, PI] range roughly for Fourier
        # World is 1024. 
        scale = 2 * math.pi / 1024.0
        scaled_pos = d_pos * scale
        
        fourier_feats = []
        for i in range(self.num_bands):
            freq = 2 ** i
            fourier_feats.append(torch.sin(scaled_pos * freq))
            fourier_feats.append(torch.cos(scaled_pos * freq))
        fourier = torch.cat(fourier_feats, dim=-1) # (..., 32)
        
        # Concatenate: 2(pos)+2(vel)+1(dist)+1(inv)+1(speed)+1(close)+2(dir)+1(log)+1(tti) + 6(geom) = 18 Base
        base_features = torch.cat([
            d_pos, d_vel, dist, inv_dist, rel_speed, closing, dir, log_dist, tti,
            cos_ata, sin_ata, cos_aa, sin_aa, cos_hca, sin_hca
        ], dim=-1)
        
        features = torch.cat([base_features, fourier], dim=-1) # 18 + 32 = 50
        
        # Pad to 64
        pad_size = self.padded_dim - features.shape[-1]
        if pad_size > 0:
            features = F.pad(features, (0, pad_size))
        
        return features

    def forward(self, pos, vel, att=None, layer_idx=None):
        raw = self.compute_analytic_features(pos, vel, att=att)
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
        # B_ij is the edge feature from i to j
        b_geo = bias_flat.view(Batch, N, N, self.n_heads, self.head_dim)
        
        # --- Spec 3.D.2: Injection ---
        # "Geometry biases the Keys" (K = K + B)
        # We need pairwise Keys K_ij = K_j + B_ij
        # K unsqueeze: (Batch, 1, N, H, D_head) -> Broadcaster over i
        k_pairwise = k.unsqueeze(1) + b_geo # (Batch, N(i), N(j), H, D_head)
        
        # "Geometry biases the Values" (V = V + B)
        # V_ij = V_j + B_ij
        v_pairwise = v.unsqueeze(1) + b_geo # (Batch, N(i), N(j), H, D_head)
        
        # --- Attention Scores ---
        # Score_ij = Q_i . K_ij^T
        # Q unsqueeze: (Batch, N, 1, H, D_head)
        # Einsum: b n(i) h d, b n(i) n(j) h d -> b h n(i) n(j)
        # Or manual sum over D_head
        
        # Let's align dimensions for flexible matmul or einsum
        # q: (B, N, 1, H, D)
        # k_p: (B, N, N, H, D)
        scores = (q.unsqueeze(2) * k_pairwise).sum(dim=-1) # (Batch, N(i), N(j), H)
        scores = scores.permute(0, 3, 1, 2) # (Batch, H, N, N)
        
        scores = scores * (self.head_dim ** -0.5)
        
        # Mask
        if mask is not None:
            # Mask is (Batch, N). We want to mask columns (j) that are dead.
            mask_flat = mask.view(Batch, 1, 1, N) # (Batch, 1, 1, N(j))
            scores = scores.masked_fill(~mask_flat, float('-inf'))

        attn = F.softmax(scores, dim=-1) # (Batch, H, N(i), N(j))
        
        # --- Weighted Sum ---
        # Out_i = Sum_j (Attn_ij * V_ij)
        # Attn: (B, H, N, N)
        # V_p: (B, N, N, H, D) -> Permute to (B, H, N, N, D) for matmul?
        
        # Let's use einsum for clarity
        # b: Batch, h: Head, i: Rows, j: Cols, d: Head_Dim
        # attn: b h i j
        # v_p: b i j h d
        
        out = torch.einsum('bhij, bijhd -> bihd', attn, v_pairwise)
        
        # Concatenate heads
        out = out.reshape(Batch, N, D)
        out = out.view(B, T, N, D)
        
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
        self.actor_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        self.actor_attn = RelationalAttention(d_model, config.n_heads)
        
        # Actor Head (Trunk + Projection)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, config.action_dim)
        )
        
        # World Head
        self.world_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, config.target_dim)
        )
        
        # Embeddings
        # Action Tensor: [Power(3), Turn(7), Shoot(2)]
        self.emb_power = nn.Embedding(3, 32)
        self.emb_turn = nn.Embedding(7, 64)
        self.emb_shoot = nn.Embedding(2, 32)
        
        # Concat = 32+64+32 = 128
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        # Special Embeddings
        self.dead_embedding = nn.Parameter(torch.zeros(config.input_dim))
        self.reset_embedding = nn.Parameter(torch.zeros(d_model))
        
        # ID and Team Embeddings (Spec 2.A.2)
        # 8 Ships max, 2 Teams
        self.ship_id_embed = nn.Embedding(8, d_model)
        self.team_id_embed = nn.Embedding(2, d_model)

    def forward(self, state, prev_action, pos, vel, att=None, team_ids=None, seq_idx=None, alive=None, reset_mask=None):
        """
        Forward pass of the MambaBB World Model.

        Args:
            state: (B, T, N, D) Intrinsic State Tensor.
            prev_action: (B, T, N, 3) Previous Action (Teacher Forcing).
            pos: (B, T, N, 2) Position.
            vel: (B, T, N, 2) Velocity.
            att: (B, T, N, 2) Optional Attitude [cos, sin].
            team_ids: (B, T, N) Optional Team IDs (0 or 1).
            seq_idx: (B, T) Sequence Index for Mamba Kernel Resets (Int32).
            alive: (B, T, N) Optional boolean mask for alive ships.
            reset_mask: (B, T) Optional boolean mask for episode resets.

        Returns:
            pred_states: (B, T, N, D) Predicted Next State (Deltas).
            pred_actions: (B, T, N, 12) Predicted Action Logits.
        """
        batch_size, seq_len, num_ships, _ = state.shape

        # 1. Dead Ship Masking (state replacement)
        if alive is None:
             alive = state[..., 1] > 0
             
        dead_mask = ~alive
        dead_mask = ~alive
        # Avoid graph break from .any() and boolean indexing
        dead_mask_bc = dead_mask.unsqueeze(-1)
        dead_embedding_bc = self.dead_embedding.to(state.dtype).view(1, 1, 1, -1)
        state = torch.where(dead_mask_bc, dead_embedding_bc, state)

        # 2. Reset Logic (Semantic)
        # Use provided reset_mask (from View) or fallback to inferring from seq_idx
        if reset_mask is None and seq_idx is not None:
             diff = torch.zeros_like(seq_idx, dtype=torch.bool)
             diff[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
             reset_mask = diff # (B, T)
        
        # Expand for broadcasting (B, T, 1, 1)
        reset_mask_bc = None
        if reset_mask is not None:
             if reset_mask.ndim == 2:
                  reset_mask_bc = reset_mask.unsqueeze(-1).unsqueeze(-1)
             else:
                  reset_mask_bc = reset_mask

        # 3. Encoders
        # Mask Absolute Position (Indices 3,4) in State for Intrinsic Encoding
        # Spec 2.A: "Absolute position is explicitly excluded"
        state_no_pos = state.clone()
        state_no_pos[..., 3:5] = 0.0
        
        s_emb = self.state_encoder(state_no_pos)
        
        # Add Identity and Affiliation Embeddings (Spec 2.A.2)
        if team_ids is not None:
             t_emb = self.team_id_embed(team_ids.long())
             if t_emb.ndim == 4: # (B, T, N, D)
                  s_emb = s_emb + t_emb
             else: # (B, T, D) -> Broadcast to N
                  s_emb = s_emb + t_emb.unsqueeze(-2)
             
        # Add Ship ID Embed (Identity)
        ship_ids = torch.arange(num_ships, device=state.device).view(1, 1, num_ships).expand(batch_size, seq_len, -1)
        s_emb = s_emb + self.ship_id_embed(ship_ids)

        trunk_out, raw_geo = self.relational_encoder(pos, vel, att=att)
        
        # Apply Reset Embedding to State Input
        if reset_mask_bc is not None:
             s_emb = s_emb + (reset_mask_bc * self.reset_embedding)

        # --- World Model Pass (Backbone) ---
        # Runs first to generate History for Actor
        
        # Embed Action (Teacher Forcing)
        p = self.emb_power(prev_action[..., 0].long().clamp(0, 2))
        t = self.emb_turn(prev_action[..., 1].long().clamp(0, 6))
        s = self.emb_shoot(prev_action[..., 2].long().clamp(0, 1))
        a_emb = torch.cat([p, t, s], dim=-1) # (..., 128)
        
        x_world = self.fusion(torch.cat([s_emb, a_emb], dim=-1))
        
        # Backbone (Time Mixing)
        # Reshape for Mamba (B*N, T, D)
        x_mamba = x_world.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, self.config.d_model)
        
        # Expand Seq Idx for Kernel
        mamba_seq_idx = None
        if seq_idx is not None:
             mamba_seq_idx = seq_idx.unsqueeze(1).expand(-1, num_ships, -1).reshape(batch_size * num_ships, seq_len)

        for i, block in enumerate(self.blocks):
            # Mamba
            normed = block['norm1'](x_mamba)
            if x_mamba.device.type == 'cpu':
                 m_out = normed
            else:
                 try:
                    m_out = block['mamba'](normed, seq_idx=mamba_seq_idx)
                 except TypeError:
                    m_out = block['mamba'](normed)
            x_mamba = x_mamba + m_out
            
            # Spatial (Attention)
            x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            x_spatial = x_spatial + block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
            
            x_mamba = x_spatial.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)

        x_final = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3) # (B, T, N, D)
        
        # World Model Predictions
        delta_pred = self.world_head(x_final)
        state_pred = state + delta_pred # Residual

        # --- Actor Pass ---
        # Input: State_t + History_{t-1}
        # History_{t-1} is Backbone Output shifted by 1.
        
        history = torch.zeros_like(x_final)
        history[:, 1:] = x_final[:, :-1]
        
        # Mask History at Resets (Prevent bleed from prev episode)
        if reset_mask_bc is not None:
             history = history * (~reset_mask_bc)
             
        # Fusion: Concat State + History -> Fusion -> Actor Flow
        # s_emb (B,T,N,D), history (B,T,N,D) -> (B,T,N, 2D)
        x_actor_input = torch.cat([s_emb, history], dim=-1)
        x_actor = self.actor_fusion(x_actor_input)
        
        # Spatial Mixing (Before Heads)
        actor_bias = self.relational_encoder.adapters[0](trunk_out)
        x_actor = self.actor_attn(x_actor, actor_bias, mask=alive)
        
        action_logits = self.actor_head(x_actor)
        
        return state_pred, action_logits

    def get_loss(self, pred_states, pred_actions, target_states, target_actions, loss_mask, 
                 lambda_state=1.0, 
                 lambda_power=1.0, lambda_turn=1.0, lambda_shoot=1.0,
                 weights_power=None, weights_turn=None, weights_shoot=None,
                 target_alive=None):
        """
        Compute MambaBB Loss.
        Masks out transition frames AND dead entities.
        """
        # Loss Mask Handling
        if loss_mask.ndim == 2:
             loss_mask = loss_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
             
        # Dead Entity Masking (Spec 5.B)
        if target_alive is not None:
             # target_alive: (B, T, N)
             if target_alive.ndim == 3:
                  loss_mask = loss_mask & target_alive
             
        # Flatten
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        
        # State Loss (MSE)
        # (B, T, N, D)
        mse = F.mse_loss(pred_states, target_states, reduction='none')
        
        # Feature Masking (Spec Section 4 targets only)
        # Exclude: Team(0), Acc(7,8), Attitude(10,11 - Integrated elsewhere)
        D = mse.shape[-1]
        feature_mask = torch.zeros(D, device=mse.device)
        # Targets are: Pos(3,4), Vel(5,6), Health(1), Power(2), AngVel(9), Shooting(12)
        feature_mask[1:7] = 1.0
        feature_mask[9] = 1.0
        feature_mask[12] = 1.0
        
        # Apply mask
        s_feature_loss = (mse * feature_mask).sum(dim=-1) / (feature_mask.sum() + 1e-6)
        
        # Apply Loss Mask (Sequence/Dead)
        # s_loss scalar
        s_loss = (s_feature_loss.reshape(-1) * mask_flat).sum() / denom
        
        # Action Loss (Cross Entropy)
        l_p = pred_actions[..., 0:3]
        l_t = pred_actions[..., 3:10]
        l_s = pred_actions[..., 10:12]
        
        t_p = target_actions[..., 0].long().clamp(0, 2)
        t_t = target_actions[..., 1].long().clamp(0, 6)
        t_s = target_actions[..., 2].long().clamp(0, 1)
        
        loss_p = F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), weight=weights_power, reduction='none')
        loss_t = F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), weight=weights_turn, reduction='none')
        loss_s = F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), weight=weights_shoot, reduction='none')
        
        a_loss_p = (loss_p * mask_flat).sum() / denom
        a_loss_t = (loss_t * mask_flat).sum() / denom
        a_loss_s = (loss_s * mask_flat).sum() / denom
        
        total_loss = (lambda_state * s_loss) + \
                     (lambda_power * a_loss_p) + \
                     (lambda_turn * a_loss_t) + \
                     (lambda_shoot * a_loss_s)
        
        metrics = {
             "loss": total_loss.item(),
             "state_loss": s_loss.item(),
             "action_loss_power": a_loss_p.item(),
             "action_loss_turn": a_loss_t.item(),
             "action_loss_shoot": a_loss_s.item()
        }
        
        return total_loss, s_loss, (a_loss_p + a_loss_t + a_loss_s), torch.tensor(0.0), metrics
