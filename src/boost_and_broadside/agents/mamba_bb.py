
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    Mamba2 = None
    InferenceParams = None
    print("WARNING: mamba_ssm not installed. MambaBB will fail to initialize.")

class MambaConfig:
    """Configuration for MambaBB World Model."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Cast weight to x.dtype for fused kernel support (especially in BF16)
        if hasattr(F, "rms_norm"):
             return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)
        else:
             # Manual fallback
             dims = x.shape[-1]
             var = x.pow(2).mean(-1, keepdim=True)
             x_normed = x * torch.rsqrt(var + self.eps)
             return self.weight.to(x.dtype) * x_normed

class RelationalEncoder(nn.Module):
    """
    Physics Trunk: Projects raw geometry into latent space.
    Shared across layers via adapters.
    """
    def __init__(self, config):
        super().__init__()
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
        pos = pos.to(torch.float32)
        
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
        eps = 1e-6
        dir = d_pos / (dist + eps)
        rel_speed = d_vel.pow(2).sum(dim=-1, keepdim=True).sqrt()
        closing = (d_vel * d_pos).sum(dim=-1, keepdim=True) / (dist + eps)
        inv_dist = 1.0 / (dist + 0.1) # 0.1 is safe enough
        log_dist = torch.log(dist + 1.0)
        tti = dist / (closing.clamp(min=1e-3)) # Time to impact can be large, clamp denom
        
        # Relational Geometry (ATA, AA, HCA)
        if att is not None:
             # att is (..., N, 2) [cos, sin]
             att_i = att.unsqueeze(-2) # (..., N, 1, 2)
             att_j = att.unsqueeze(-3) # (..., 1, N, 2)
             
             # ATA: Angle between my heading and target
             cos_ata = (dir * att_i).sum(dim=-1, keepdim=True)
             sin_ata = (dir[..., 0] * att_i[..., 1] - dir[..., 1] * att_i[..., 0]).unsqueeze(-1)
             
             # AA: Angle between target's heading and me (from target's perspective)
             cos_aa = (-dir * att_j).sum(dim=-1, keepdim=True)
             sin_aa = (-dir[..., 0] * att_j[..., 1] + dir[..., 1] * att_j[..., 0]).unsqueeze(-1)
             
             # HCA: Angle between our headings
             cos_hca = (att_i * att_j).sum(dim=-1, keepdim=True)
             sin_hca = (att_i[..., 0] * att_j[..., 1] - att_i[..., 1] * att_j[..., 0]).unsqueeze(-1)
        else:
             # Fallback if no attitude provided (e.g. initial steps of dreaming)
             cos_ata = sin_ata = cos_aa = sin_aa = cos_hca = sin_hca = torch.zeros_like(dist)

        # Fourier Encoding of d_pos (Normalized roughly by world size/scale)
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
            d_pos, d_vel, dist, inv_dist, 
            rel_speed, closing, dir, log_dist, tti,
            cos_ata, sin_ata, cos_aa, sin_aa, cos_hca, sin_hca
        ], dim=-1)
        
        # Final feature assembly
        fourier = fourier.to(torch.bfloat16)
        features = torch.cat([base_features.to(fourier.dtype), fourier], dim=-1) # 18 + 32 = 50
        
        # Pad to 64
        pad_size = self.padded_dim - features.shape[-1]
        if pad_size > 0:
            features = F.pad(features, (0, pad_size))
        
        # Robust cast to ensure compatibility with Master Weights (Float) when not in Autocast
        target_dtype = self.trunk[0].weight.dtype
        return features.to(target_dtype)

    def forward(self, pos, vel, att=None, layer_idx=None, world_size=(1024.0, 1024.0)):
        raw = self.compute_analytic_features(pos, vel, att=att, world_size=world_size)
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
        
        def create_mamba_block(layer_idx):
            if Mamba2 is None: return nn.Identity()
            try: return Mamba2(d_model=d_model, d_state=128, expand=2, layer_idx=layer_idx)
            except: return nn.Identity()

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'mamba': create_mamba_block(i),
                'norm1': RMSNorm(d_model),
                'norm2': RMSNorm(d_model),
                'attn': RelationalAttention(d_model, config.n_heads)
            }) for i in range(config.n_layers)
        ])
        
        # Actor Components
        self.actor_spatial_attn = RelationalAttention(d_model, config.n_heads)
        self.actor_spatial_norm = RMSNorm(d_model)
        self.actor_temporal_attn = nn.MultiheadAttention(d_model, num_heads=config.n_heads, batch_first=True)
        self.actor_temporal_norm = RMSNorm(d_model)
        
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, config.action_dim)
        )
        
        # Team Evaluator
        from boost_and_broadside.agents.components.team_evaluator import TeamEvaluator
        self.team_evaluator = TeamEvaluator(d_model)
        
        # World Head
        self.world_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, config.target_dim)
        )
        
        # Embeddings
        self.emb_power = nn.Embedding(3, 32)
        self.emb_turn = nn.Embedding(7, 64)
        self.emb_shoot = nn.Embedding(2, 32)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        # Special Embeddings
        self.dead_embedding = nn.Parameter(torch.zeros(config.input_dim))
        self.reset_embedding = nn.Parameter(torch.zeros(d_model))
        
        self.ship_id_embed = nn.Embedding(8, d_model)
        self.team_id_embed = nn.Embedding(2, d_model)

        # Uncertainty Weighting
        self.log_vars = None
        if getattr(config, "loss_type", "fixed") == "uncertainty":
             self.log_vars = nn.ParameterDict({
                 "state": nn.Parameter(torch.tensor(0.0)),
                 "actions": nn.Parameter(torch.tensor(0.0)),
                 "value": nn.Parameter(torch.tensor(0.0)),
                 "reward": nn.Parameter(torch.tensor(0.0))
             })

    def forward(self, state, prev_action, pos, vel, att=None, team_ids=None, seq_idx=None, alive=None, reset_mask=None,
                inference_params=None, actor_cache=None, world_size=(1024.0, 1024.0)):
        
        batch_size, seq_len, num_ships, _ = state.shape

        # Dead Masking
        if alive is None: alive = state[..., 1] > 0
        state = torch.where(alive.unsqueeze(-1), state.to(self.dead_embedding.dtype), self.dead_embedding.view(1, 1, 1, -1))

        # Reset Logic
        if reset_mask is None and seq_idx is not None:
             diff = torch.zeros_like(seq_idx, dtype=torch.bool)
             diff[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
             reset_mask = diff
        
        w_dtype = self.state_encoder[0].weight.dtype
        s_emb = self.state_encoder(state.to(w_dtype))
        
        if team_ids is not None:
             t_emb = self.team_id_embed(team_ids.long())
             s_emb = s_emb + (t_emb if t_emb.ndim == 4 else t_emb.unsqueeze(-2))
             
        ship_ids = torch.arange(num_ships, device=state.device).view(1, 1, num_ships).expand(batch_size, seq_len, -1)
        s_emb = s_emb + self.ship_id_embed(ship_ids)

        trunk_out, _ = self.relational_encoder(pos, vel, att=att, world_size=world_size)
        
        if reset_mask is not None:
             s_emb = s_emb + (reset_mask.unsqueeze(-1).unsqueeze(-1) * self.reset_embedding)

        # Actions
        p = self.emb_power(prev_action[..., 0].long().clamp(0, 2))
        t = self.emb_turn(prev_action[..., 1].long().clamp(0, 6))
        s = self.emb_shoot(prev_action[..., 2].long().clamp(0, 1))
        a_emb = torch.cat([p, t, s], dim=-1)
        
        x_world = self.fusion(torch.cat([s_emb, a_emb], dim=-1))
        
        # Backbone
        x_mamba = x_world.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, self.config.d_model)
        mamba_seq_idx = seq_idx.unsqueeze(1).expand(-1, num_ships, -1).reshape(batch_size * num_ships, seq_len) if seq_idx is not None else None

        for i, block in enumerate(self.blocks):
            normed = block['norm1'](x_mamba)
            if x_mamba.device.type == 'cpu' or Mamba2 is None:
                 m_out = normed # Mamba blocks don't support CPU
            else:
                 try:
                      m_out = block['mamba'](normed, seq_idx=mamba_seq_idx, inference_params=inference_params)
                 except Exception:
                      # print(f"DEBUG: Mamba forward failed with inference_params: {e}")
                      m_out = block['mamba'](normed)
            x_mamba = x_mamba + m_out
            
            x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            x_spatial = x_spatial + block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
            x_mamba = x_spatial.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)

        x_final = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
        delta_pred = self.world_head(x_final)
        state_pred = state + delta_pred

        # Actor
        if actor_cache is not None: history = actor_cache
        else:
             history = torch.zeros_like(x_final)
             history[:, 1:] = x_final[:, :-1]
        
        if reset_mask is not None: history = history * (~reset_mask.unsqueeze(-1).unsqueeze(-1))

        actor_bias = self.relational_encoder.adapters[0](trunk_out)
        x_actor_spatial = s_emb + self.actor_spatial_attn(self.actor_spatial_norm(s_emb), actor_bias, mask=alive)
        
        Batch_Time = batch_size * seq_len
        q_temp = self.actor_temporal_norm(x_actor_spatial).reshape(Batch_Time, num_ships, -1)
        kv_temp = history.reshape(Batch_Time, num_ships, -1)
        key_padding_mask = ~alive.reshape(Batch_Time, num_ships) if alive is not None else None
        x_actor_temporal, _ = self.actor_temporal_attn(q_temp, kv_temp, kv_temp, key_padding_mask=key_padding_mask)
        x_actor = x_actor_spatial + x_actor_temporal.reshape(batch_size, seq_len, num_ships, -1)
        
        action_logits = self.actor_head(x_actor)
        x_eval_input = x_actor.reshape(Batch_Time, num_ships, -1)
        eval_mask = alive.reshape(Batch_Time, num_ships) if alive is not None else None
        value_pred, reward_components = self.team_evaluator(x_eval_input, mask=eval_mask)
        
        return state_pred, action_logits, value_pred.reshape(batch_size, seq_len, 1), reward_components.sum(dim=-1, keepdim=True).reshape(batch_size, seq_len, 1), x_final

    def hex_loss(self, *args, **kwargs): return self.get_loss(*args, **kwargs) # Alias

    def get_loss(self, pred_states, pred_actions, target_states, target_actions, loss_mask, 
                 lambda_state=1.0, lambda_power=1.0, lambda_turn=1.0, lambda_shoot=1.0,
                 pred_values=None, pred_rewards=None, target_returns=None, target_rewards=None,
                 lambda_value=1.0, lambda_reward=1.0, weights_power=None, weights_turn=None, weights_shoot=None,
                 target_alive=None, min_sigma=0.1):
        
        target_states = target_states.to(pred_states.dtype)
        if loss_mask.ndim == 2: loss_mask = loss_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
        if target_alive is not None: loss_mask = loss_mask & target_alive
             
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        mse = F.mse_loss(pred_states, target_states, reduction='none')
        
        D = mse.shape[-1]
        feature_mask = torch.zeros(D, device=mse.device)
        feature_mask[1:5] = 1.0 # Health, Power, VelX, VelY
        feature_mask[7] = 1.0   # Shoot
        feature_mask[8] = 1.0   # AngVel
        
        s_loss = ((mse * feature_mask).sum(dim=-1) / (feature_mask.sum() + 1e-6)).reshape(-1).mul(mask_flat).sum() / denom
        
        l_p, l_t, l_s = pred_actions[..., 0:3], pred_actions[..., 3:10], pred_actions[..., 10:12]
        t_p, t_t, t_s = target_actions[..., 0].long().clamp(0, 2), target_actions[..., 1].long().clamp(0, 6), target_actions[..., 2].long().clamp(0, 1)
        
        a_loss_p = (F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), weight=weights_power, reduction='none') * mask_flat).sum() / denom / math.log(3)
        a_loss_t = (F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), weight=weights_turn, reduction='none') * mask_flat).sum() / denom / math.log(7)
        a_loss_s = (F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), weight=weights_shoot, reduction='none') * mask_flat).sum() / denom / math.log(2)
        
        v_loss = r_loss = torch.tensor(0.0, device=pred_states.device)
        if pred_values is not None and target_returns is not None:
             valid_cnt = target_alive.sum(dim=-1, keepdim=True).clamp(min=1.0) if target_alive is not None else 1.0
             team_ret = (target_returns * target_alive).sum(dim=-1, keepdim=True) / valid_cnt if target_alive is not None else target_returns.mean(dim=-1, keepdim=True)
             team_rew = (target_rewards * target_alive).sum(dim=-1, keepdim=True) / valid_cnt if target_alive is not None else target_rewards.mean(dim=-1, keepdim=True)
             m_glob = loss_mask.any(dim=-1, keepdim=True).float()
             d_glob = m_glob.sum() + 1e-6
             v_loss = (F.mse_loss(pred_values, team_ret, reduction='none') * m_glob).sum() / d_glob
             r_loss = (F.mse_loss(pred_rewards, team_rew, reduction='none') * m_glob).sum() / d_glob

        loss_type = getattr(self.config, "loss_type", "fixed")
        if loss_type == "uncertainty" and self.log_vars is not None:
             clamped_sigmas = {}
             def apply_u(loss, name):
                  s = torch.clamp(self.log_vars[name], min=2.0 * math.log(min_sigma))
                  clamped_sigmas[name] = torch.exp(0.5 * s).item()
                  return 0.5 * torch.exp(-s) * loss + 0.5 * s
             l_state_w = apply_u(s_loss, "state")
             l_actions_w = apply_u(a_loss_p + a_loss_t + a_loss_s, "actions")
             l_value_w = apply_u(v_loss, "value")
             l_reward_w = apply_u(r_loss, "reward")
             total_loss = l_state_w + l_actions_w + l_value_w + l_reward_w
        else:
             total_loss = (lambda_state * s_loss) + (lambda_power * a_loss_p) + (lambda_turn * a_loss_t) + (lambda_shoot * a_loss_s) + (lambda_value * v_loss) + (lambda_reward * r_loss)
        
        with torch.no_grad():
             eps = 1e-8
             ent_p = -(F.softmax(l_p.reshape(-1, 3), dim=-1) * torch.log(F.softmax(l_p.reshape(-1, 3), dim=-1) + eps)).sum(-1).mean()
             ent_t = -(F.softmax(l_t.reshape(-1, 7), dim=-1) * torch.log(F.softmax(l_t.reshape(-1, 7), dim=-1) + eps)).sum(-1).mean()
             ent_s = -(F.softmax(l_s.reshape(-1, 2), dim=-1) * torch.log(F.softmax(l_s.reshape(-1, 2), dim=-1) + eps)).sum(-1).mean()

        metrics = {"loss": total_loss.item(), "loss_sub/state_mse": s_loss.item(), "loss_sub/action_power": a_loss_p.item(), "loss_sub/action_turn": a_loss_t.item(), "loss_sub/action_shoot": a_loss_s.item(), "loss_sub/value_mse": v_loss.item(), "loss_sub/reward_mse": r_loss.item(), "entropy/power": ent_p.item(), "entropy/turn": ent_t.item(), "entropy/shoot": ent_s.item()}
        if loss_type == "uncertainty":
             for n, s in clamped_sigmas.items(): metrics[f"loss_sigma/{n}"] = s
        return total_loss, s_loss, (a_loss_p + a_loss_t + a_loss_s), torch.tensor(0.0), metrics

    @torch.no_grad()
    def generate(self, initial_state: torch.Tensor, initial_action: torch.Tensor, initial_pos: torch.Tensor = None, steps: int = 10, n_ships: int = 1):
        """
        Autoregressive generation of future states and actions.
        """
        if initial_state.ndim == 3: initial_state = initial_state.unsqueeze(1)
        if initial_action.ndim == 3: initial_action = initial_action.unsqueeze(1)
        if initial_pos is not None and initial_pos.ndim == 3: initial_pos = initial_pos.unsqueeze(1)
            
        B, _, N, D = initial_state.shape
        device = initial_state.device
        
        current_state = initial_state
        if initial_action.shape[-1] == 12:
            p_idx = initial_action[..., 0:3].argmax(dim=-1)
            t_idx = initial_action[..., 3:10].argmax(dim=-1)
            s_idx = initial_action[..., 10:12].argmax(dim=-1)
            curr_a_idx = torch.stack([p_idx, t_idx, s_idx], dim=-1).float()
        else:
            curr_a_idx = initial_action
            
        from boost_and_broadside.core.constants import NORM_VELOCITY
        curr_pos = initial_pos if initial_pos is not None else torch.zeros(B, 1, N, 2, device=device)
        curr_vel = current_state[..., 3:5] * NORM_VELOCITY
        
        gen_s, gen_a = [], []
        actor_cache = None
        
        inference_params = InferenceParams(max_batch_size=B*N, max_seqlen=steps+10) if InferenceParams else None

        for _ in range(steps):
            with torch.no_grad():
                pred_s, pred_a_logits, _, _, nc = self.forward(
                    state=current_state,
                    prev_action=curr_a_idx,
                    pos=curr_pos,
                    vel=curr_vel, 
                    actor_cache=actor_cache,
                    inference_params=inference_params
                )
            
            actor_cache = nc
            p_idx = pred_a_logits[..., 0:3].argmax(dim=-1)
            t_idx = pred_a_logits[..., 3:10].argmax(dim=-1)
            s_idx = pred_a_logits[..., 10:12].argmax(dim=-1)
            
            next_a_idx = torch.stack([p_idx, t_idx, s_idx], dim=-1).float()
            next_a_oh = torch.cat([F.one_hot(p_idx, 3), F.one_hot(t_idx, 7), F.one_hot(s_idx, 2)], dim=-1)
            
            gen_s.append(pred_s)
            gen_a.append(next_a_oh)
            
            current_state = pred_s
            curr_a_idx = next_a_idx
            curr_vel = pred_s[..., 3:5] * NORM_VELOCITY
            curr_pos = curr_pos + curr_vel * 0.1 # Simple dt integration
            
        return torch.cat(gen_s, dim=1), torch.cat(gen_a, dim=1)
