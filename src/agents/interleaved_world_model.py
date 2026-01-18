
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.rope import RotaryPositionEmbedding

class DyadicFourierFeatureExtractor(nn.Module):
    """
    Maps continuous inputs to higher dimensional space using Dyadic Fourier Features.
    Features: [sin(2^k * pi * x), cos(2^k * pi * x)] for k in frequencies.
    """
    def __init__(self, input_dim: int, embed_dim: int, num_freqs: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.out_dim = input_dim * num_freqs * 2
        
        # Project to embed_dim (to input into FFN correctly)
        self.projection = nn.Linear(self.out_dim, embed_dim)
        
        # Frequencies: 2^k * PI.
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        # x: (..., input_dim)
        scaled = x.unsqueeze(-1) * self.freqs * torch.pi
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        feats = torch.stack([sin_feat, cos_feat], dim=-1)
        feats = feats.view(*x.shape[:-1], -1)
        return self.projection(feats)

class UnifiedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout

    def forward(self, x, rope, position_ids, mask=None, past_kv=None, use_cache=False):
        B, L, E = x.size()
        head_dim = E // self.n_heads
        
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        
        # Reshape to (B, L, H, D)
        k = k.view(B, L, self.n_heads, head_dim)
        q = q.view(B, L, self.n_heads, head_dim)
        v = v.view(B, L, self.n_heads, head_dim)
        
        # Prepare for RoPE: Flatten Batch and Heads -> (B*H, L, D)
        # We need to expand position_ids to match (B*H, L)
        # position_ids: (B, L)
        pos_ids_flat = position_ids.repeat_interleave(self.n_heads, dim=0) # (B*H, L)
        
        # (B, L, H, D) -> (B, H, L, D) -> (B*H, L, D)
        q_flat = q.transpose(1, 2).reshape(B * self.n_heads, L, head_dim)
        k_flat = k.transpose(1, 2).reshape(B * self.n_heads, L, head_dim)
        
        q_rot, k_rot = rope(q_flat, k_flat, pos_ids_flat)
        
        # Reshape back to (B, H, L, D)
        q = q_rot.view(B, self.n_heads, L, head_dim)
        k = k_rot.view(B, self.n_heads, L, head_dim)
        
        # v also needs to be (B, H, L, D)
        v = v.transpose(1, 2)
        
        if past_kv is not None:
             pk, pv = past_kv
             k = torch.cat([pk, k], dim=2)
             v = torch.cat([pv, v], dim=2)
             
        current_kv = (k, v) if use_cache else None
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0
        )
        
        y = y.transpose(1, 2).contiguous().view(B, L, E)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, current_kv

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = UnifiedSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, rope, position_ids, mask=None, past_kv=None, use_cache=False):
        attn_out, current_kv = self.attn(
            self.ln_1(x), rope, position_ids, mask, past_kv, use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, current_kv

class InterleavedWorldModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class InterleavedWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 12, # Discrete 3+7+2
        embed_dim: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        max_ships: int = 8,
        max_context_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = InterleavedWorldModelConfig(
            state_dim=state_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_ships=max_ships,
            dropout=dropout
        )
        
        self.num_binary = 4
        self.num_continuous = state_dim - self.num_binary
        
        # State Encoders
        self.fourier = DyadicFourierFeatureExtractor(self.num_continuous, embed_dim)
        self.bin_emb_list = nn.ModuleList([nn.Embedding(2, embed_dim) for _ in range(self.num_binary)])
        
        self.state_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Action Encoders
        self.emb_power = nn.Embedding(3, embed_dim)
        self.emb_turn = nn.Embedding(7, embed_dim)
        self.emb_shoot = nn.Embedding(2, embed_dim)
        
        # Positional/Type
        self.rope = RotaryPositionEmbedding(
            dim=embed_dim // n_heads,
            max_seq_len=max_context_len * 2,
        )
        self.ship_embed = nn.Embedding(max_ships, embed_dim)
        self.team_embed = nn.Embedding(2, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)
        
        # Transformer
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Heads
        self.action_head = nn.Linear(embed_dim, 3 + 7 + 2)
        self.state_head = nn.Linear(embed_dim, state_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_block_causal_mask(self, B, T, N, device):
        L = T * 2 * N
        # Base Causal
        mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        
        # Add N*N blocks on diagonal for each T, Type
        # There are 2*T blocks of size N
        for k in range(2 * T):
            start = k * N
            end = start + N
            mask[start:end, start:end] = True
            
        return mask

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        team_ids: torch.Tensor,
        noise_scale: float = 0.0,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        states: (B, T, N, D)
        actions: (B, T, N, 3)
        team_ids: (B, N) or (B, T, N)
        """
        B, T, N, D = states.shape
        device = states.device
        
        # 1. State Encoding
        if self.training and noise_scale > 0:
             continuous = states[..., :self.num_continuous]
             binary = states[..., self.num_continuous:]
             noise = torch.randn_like(continuous) * noise_scale
             continuous = continuous + noise
        else:
             continuous = states[..., :self.num_continuous]
             binary = states[..., self.num_continuous:]

        cont_emb = self.fourier(continuous)
        
        binary_long = binary.long().clamp(0, 1)
        bin_emb = torch.zeros_like(cont_emb)
        for i in range(self.num_binary):
            bin_emb = bin_emb + self.bin_emb_list[i](binary_long[..., i])
            
        state_tokens = self.state_ffn(cont_emb + bin_emb)
        
        # 2. Action Encoding
        p_emb = self.emb_power(actions[..., 0].long())
        t_emb = self.emb_turn(actions[..., 1].long())
        s_emb = self.emb_shoot(actions[..., 2].long())
        action_tokens = p_emb + t_emb + s_emb 
        
        
        # 3. Add Embeddings
        ship_ids = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
        ship_emb = self.ship_embed(ship_ids)
        
        if team_ids.ndim == 2: # (B, N)
            team_ids = team_ids.unsqueeze(1).expand(-1, T, -1)
        team_emb = self.team_embed(team_ids.long()) 
        
        state_tokens = state_tokens + ship_emb + team_emb + self.type_embed(torch.tensor(0, device=device))
        action_tokens = action_tokens + ship_emb + team_emb + self.type_embed(torch.tensor(1, device=device))
        
        # 4. Interleave & Flatten
        # Stack dim 2 -> (B, T, 2, N, E)
        combined = torch.stack([state_tokens, action_tokens], dim=2)
        L = T * 2 * N
        tokens = combined.view(B, L, self.config.embed_dim)
        
        # 5. RoPE Indices
        # T indices logic
        t_indices = torch.arange(T, device=device).view(1, T, 1, 1).expand(B, T, 2, N)
        position_ids = t_indices.reshape(B, L)
        
        # 6. Mask
        mask = self.create_block_causal_mask(B, T, N, device)
        
        # 7. Transformer
        x = tokens
        current_key_values = []
        
        for i, block in enumerate(self.blocks):
            pk = past_key_values[i] if past_key_values is not None else None
            x, kv = block(x, self.rope, position_ids, mask, pk, use_cache)
            if use_cache:
                current_key_values.append(kv)
                
        x = self.ln_f(x)
        
        # 8. Heads - Mapping back to logical outputs
        # x is (B, L, E).
        # S_tokens (Predict Actions) are at 0, 2, 4... (Even blocks of N)
        # A_tokens (Predict States) are at 1, 3, 5... (Odd blocks of N)
        # However, flattening was T, 2, N.
        # Order: T0 [S0..SN, A0..AN], T1...
        # So S_blocks are at indices corresponding to t*2*N ... t*2*N + N.
        # A_blocks are at indices t*2*N + N ... (t+1)*2*N.
        
        # Reshape back to (B, T, 2, N, E)
        x_reshaped = x.view(B, T, 2, N, self.config.embed_dim)
        
        # S tokens -> Action Head
        s_out = x_reshaped[:, :, 0, :, :] # (B, T, N, E)
        pred_actions = self.action_head(s_out) # (B, T, N, A_dim)
        
        # A tokens -> State Head
        a_out = x_reshaped[:, :, 1, :, :] # (B, T, N, E)
        pred_states = self.state_head(a_out) # (B, T, N, S_dim)
        
        return pred_states, pred_actions, current_key_values

    def get_loss(
        self,
        pred_states,
        pred_actions,
        target_states,
        target_actions,
        loss_mask, 
        lambda_state=1.0,
        lambda_action=0.1
    ):
        """
        pred_states: (B, T, N, D) - Output from A_t (should predict S_{t+1})
        pred_actions: (B, T, N, A) - Output from S_t (should predict A_t)
        
        target_states: (B, T, N, D) - Should be S_{t+1}
        target_actions: (B, T, N, 3) - Should be A_t
        """
        valid_mask = loss_mask.bool() # (B, T)
        
        # Action Loss
        # pred_actions corresponds to predicting A_t from S_t.
        # Target is A_t (target_actions).
        # We assume target_actions is already aligned (A_0, A_1... A_{T-1})
        
        valid_pred_act = pred_actions[valid_mask] # (M, N, A_dim)
        valid_target_act = target_actions[valid_mask] # (M, N, 3)
        
        p_logits = valid_pred_act[..., 0:3].reshape(-1, 3)
        t_logits = valid_pred_act[..., 3:10].reshape(-1, 7)
        s_logits = valid_pred_act[..., 10:12].reshape(-1, 2)
        
        p_target = valid_target_act[..., 0].long().reshape(-1)
        t_target = valid_target_act[..., 1].long().reshape(-1)
        s_target = valid_target_act[..., 2].long().reshape(-1)
        
        loss_p = F.cross_entropy(p_logits, p_target)
        loss_t = F.cross_entropy(t_logits, t_target)
        loss_s = F.cross_entropy(s_logits, s_target)
        
        action_loss = loss_p + loss_t + loss_s
        
        # State Loss
        # pred_states corresponds to predicting S_{t+1} from A_t.
        # Target is S_{t+1} (target_states).
        
        valid_pred_state = pred_states[valid_mask]
        valid_target_state = target_states[valid_mask]
        
        state_loss = F.mse_loss(valid_pred_state, valid_target_state)
        
        total_loss = lambda_state * state_loss + lambda_action * action_loss
        
        return total_loss, state_loss, action_loss

