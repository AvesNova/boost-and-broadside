
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.rope import RotaryPositionEmbedding
from agents.relational_features import RelationalFeatureExtractor
from agents.layers import DyadicFourierFeatureExtractor, GatedSwiGLU



class GatedStateEncoder(nn.Module):
    """
    Encodes continuous state vector with Fourier features and Gated SwiGLU.
    Structure: Input -> Fourier(128) -> SwiGLU(256) -> + Residual -> LN -> Out
    """
    def __init__(self, input_dim: int, embed_dim: int = 128, num_freqs: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fourier = DyadicFourierFeatureExtractor(input_dim, embed_dim, num_freqs=num_freqs)
        
        # SwiGLU: Map 128 -> 256 (internal) -> 128
        # "Map the 128-dim Fourier features to 256-dim using two parallel linear layers"
        # So hidden_features=256
        self.swiglu = GatedSwiGLU(
            in_features=embed_dim, 
            hidden_features=256, 
            out_features=embed_dim,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (..., input_dim)
        fourier_feats = self.fourier(x)
        
        # SwiGLU + Residual
        out = self.swiglu(fourier_feats)
        out = out + fourier_feats
        
        return self.norm(out)

class TemporalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        assert self.embed_dim % self.n_heads == 0
        self.head_dim = self.embed_dim // self.n_heads
        
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE for Temporal Sequence
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_context_len * 2 # 2 tokens per timestep
        )

    def forward(self, x, position_ids, past_kv=None, use_cache=False):
        # x: (B, L, N, E)
        B, L, N, E = x.shape
        
        # Reshape for per-ship causal attention: (B*N, L, E)
        # We treat each ship as an independent sequence for temporal perception.
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L, E)
        
        qkv = self.c_attn(x_flat)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        # (B*N, L, H, D)
        q = q.view(B * N, L, self.n_heads, self.head_dim)
        k = k.view(B * N, L, self.n_heads, self.head_dim)
        v = v.view(B * N, L, self.n_heads, self.head_dim)
        
        # RoPE
        # position_ids: (B, L). 
        # We need to broadcast position_ids to (B*N, L).
        # q, k shape roughly (B*N, L, H, D).
        # RoPE expects (Batch, Seq, Dim).
        # So we treat independent sequences as (B * N * H).
        
        # 1. Expand position_ids -> (B*N, L)
        pos_ids_flat = position_ids.repeat_interleave(N, dim=0) # (B*N, L)
        # 2. Further expand for Heads -> (B*N*H, L)
        pos_ids_final = pos_ids_flat.repeat_interleave(self.n_heads, dim=0)

        # 3. Reshape Q, K -> (B*N*H, L, D)
        q_flat = q.transpose(1, 2).reshape(B * N * self.n_heads, L, self.head_dim)
        k_flat = k.transpose(1, 2).reshape(B * N * self.n_heads, L, self.head_dim)
        
        # 4. Apply RoPE
        q_rot, k_rot = self.rope(q_flat, k_flat, pos_ids_final)
        
        # 5. Reshape back -> (B*N, L, H, D) -> Then transpose to (B*N, H, L, D)
        # Wait, q_rot is (B*N*H, L, D). 
        # view(B*N, H, L, D) produces (B*N, H, L, D).
        # This IS the correct shape for SDPA (Batch, Heads, Seq, Dim).
        # We DO NOT want to transpose(1, 2) which would swap H and L giving (B*N, L, H, D)
        
        q = q_rot.view(B * N, self.n_heads, L, self.head_dim)
        k = k_rot.view(B * N, self.n_heads, L, self.head_dim)
        
        # V needs transpose to (B*N, H, L, D)
        # v start: (B*N, L, H, D). Transpose makes it (B*N, H, L, D).
        v = v.transpose(1, 2)

        if past_kv is not None:
             pk, pv = past_kv
             # pk: (B*N, H, L_past, D)
             k = torch.cat([pk, k], dim=2)
             v = torch.cat([pv, v], dim=2)
        
        if use_cache:
            current_kv = (k, v)
        else:
            current_kv = None
            
        # Flash Attention Causal
        # Input to SDPA is (Batch, Heads, Seq, Dim)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # (B*N, H, L, D) -> (B*N, L, E)
        y = y.transpose(1, 2).contiguous().view(B * N, L, E)
        
        # Back to (B, L, N, E)
        y = y.view(B, N, L, E).permute(0, 2, 1, 3)
        
        y = self.resid_dropout(self.c_proj(y))
        return y, current_kv

class SpatialSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = self.embed_dim // self.n_heads
        
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None, use_cache=False, relational_bias=None):
        # x: (B, L, N, E)
        # Goal: Each timestep L attends to itself (N) and previous timestep (N window).
        # Sequence L is S0, A0, S1, A1...
        # S0 attends to S0 (N) and nothing else (pad).
        # A0 attends to A0 (N) and S0 (N).
        # S1 attends to S1 (N) and A0 (N).
        
        B, L, N, E = x.shape
        
        # Projects
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        
        # Split Heads: (B, L, N, H, D)
        q = q.view(B, L, N, self.n_heads, self.head_dim)
        k = k.view(B, L, N, self.n_heads, self.head_dim)
        v = v.view(B, L, N, self.n_heads, self.head_dim)

        # Prepare Windowed Keys/Values
        # If we have cache (Generation L=1) output k,v of Previous step
        
        if use_cache:
            # Save CURRENT step's k, v for next step
            current_kv = (k, v)
        else:
            current_kv = None
            
        # Get Previous Step K, V
        if past_kv is not None and past_kv[0].shape[1] > 0:
              pk, pv = past_kv
              # pk: (B, L_past, N, H, D)
              # Take last step from past
              last_k = pk[:, -1:] # (B, 1, N, H, D)
              last_v = pv[:, -1:]
        else:
              # No history, use zeros
              last_k = torch.zeros_like(k[:, :1])
              last_v = torch.zeros_like(v[:, :1])
              
        # Construct Previous Window inputs
        # k_prev[i] should be the token *before* k[i].
        # For i=0, it is last_k.
        # For i>0, it is k[i-1].
        
        # Shift current k right by 1, and prepend last_k
        k_prev = torch.cat([last_k, k[:, :-1]], dim=1) # (B, L, N, H, D)
        v_prev = torch.cat([last_v, v[:, :-1]], dim=1)
        
        # Concatenate on N dimension: (B, L, 2N, H, D)
        # Use simple concat. Current N followed by Prev N? Or Prev then Current?
        # Let's do Prev then Current for logical time ordering, though non-causal attention doesn't care.
        # But we must ensure proper shapes.
        
        k_window = torch.cat([k_prev, k], dim=2)
        v_window = torch.cat([v_prev, v], dim=2)
        
        # Flatten for Attention
        # (B, L, N, H, D) -> (B*L, H, N, D)
        q_flat = q.permute(0, 1, 3, 2, 4).reshape(B*L, self.n_heads, N, self.head_dim)
        
        # (B, L, 2N, H, D) -> (B*L, H, 2N, D)
        k_flat = k_window.permute(0, 1, 3, 2, 4).reshape(B*L, self.n_heads, 2*N, self.head_dim)
        v_flat = v_window.permute(0, 1, 3, 2, 4).reshape(B*L, self.n_heads, 2*N, self.head_dim)
        
        # Attention - NON CAUSAL (because we manually windowed)
        # y = SDPA(q, k, v)
        # q: (batch, heads, N, D)
        # k: (batch, heads, 2N, D)
        # bias: (batch, heads, N, 2N)
        
        y = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat,
            attn_mask=relational_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        # (B*L, H, N, D) -> (B, L, N, E)
        y = y.view(B, L, self.n_heads, N, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, L, N, E)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, current_kv

class Block(nn.Module):
    def __init__(self, config, attn_type="temporal"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn_type = attn_type
        if attn_type == "spatial":
            self.attn = SpatialSelfAttention(config)
        else:
            self.attn = TemporalSelfAttention(config)
            
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, position_ids=None, past_kv=None, use_cache=False, relational_bias=None):
        # Determine KV for this block type
        if self.attn_type == "temporal":
             attn_out, current_kv = self.attn(self.ln_1(x), position_ids, past_kv, use_cache)
        else:
             attn_out, current_kv = self.attn(self.ln_1(x), past_kv, use_cache, relational_bias=relational_bias)
             
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
            max_context_len=max_context_len,
            dropout=dropout
        )
        
        self.num_binary = 4
        self.num_continuous = state_dim - self.num_binary
        
        # State Encoders
        self.state_encoder = GatedStateEncoder(
            input_dim=self.num_continuous,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Relational Bias Extractor
        self.relational_extractor = RelationalFeatureExtractor(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout
        )
        
        # Relational Prediction Head (12D target)
        # Input: Pair of embeddings (2 * embed_dim)
        # Output: 12 (features)
        self.relational_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 12)
        )
        
        self.bin_emb_list = nn.ModuleList([nn.Embedding(2, embed_dim) for _ in range(self.num_binary)])
        
        # Action Encoders
        self.emb_power = nn.Embedding(3, 48)
        self.emb_turn = nn.Embedding(7, 48)
        self.emb_shoot = nn.Embedding(2, 32)
        
        # Projection Layer: Concatenated (128) -> Embed Dim (128)
        self.action_proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.SiLU()
        )
        
        # Positional/Type
        self.ship_embed = nn.Embedding(max_ships, embed_dim)
        self.team_embed = nn.Embedding(2, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)
        
        # Transformer Blocks (Alternating T -> S)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            block_type = "temporal" if i % 2 == 0 else "spatial"
            self.blocks.append(Block(self.config, attn_type=block_type))

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

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        team_ids: torch.Tensor,
        noise_scale: float = 0.0,
        past_key_values=None,
        use_cache: bool = False,
        return_embeddings: bool = False,
        relational_features: torch.Tensor = None
    ):
        """
        states: (B, T, N, D)
        actions: (B, T, N, 3)
        team_ids: (B, N) or (B, T, N)
        relational_features: (B, T, N, N, 4) - Precomputed features
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

        cont_emb = self.state_encoder(continuous)
        
        binary_long = binary.long().clamp(0, 1)
        bin_emb = torch.zeros_like(cont_emb)
        for i in range(self.num_binary):
            bin_emb = bin_emb + self.bin_emb_list[i](binary_long[..., i])
            
        state_tokens = cont_emb + bin_emb
        
        # 2. Action Encoding
        p_emb = self.emb_power(actions[..., 0].long()) # (..., 48)
        t_emb = self.emb_turn(actions[..., 1].long())  # (..., 48)
        s_emb = self.emb_shoot(actions[..., 2].long()) # (..., 32)
        
        action_concat = torch.cat([p_emb, t_emb, s_emb], dim=-1)
        action_tokens = self.action_proj(action_concat) 
        
        # 3. Add Embeddings
        ship_ids = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
        ship_emb = self.ship_embed(ship_ids)
        
        if team_ids.ndim == 2: # (B, N)
            team_ids = team_ids.unsqueeze(1).expand(-1, T, -1)
        team_emb = self.team_embed(team_ids.long()) 
        
        state_tokens = state_tokens + ship_emb + team_emb + self.type_embed(torch.tensor(0, device=device))
        action_tokens = action_tokens + ship_emb + team_emb + self.type_embed(torch.tensor(1, device=device))
        
        # 4. Interleave & Flatten
        combined = torch.stack([state_tokens, action_tokens], dim=2)
        L = T * 2
        tokens = combined.view(B, L, N, self.config.embed_dim)
        
        # 5. Position IDs
        past_len = 0
        if past_key_values is not None:
             if len(past_key_values) > 0 and past_key_values[0] is not None:
                 past_len = past_key_values[0][0].shape[2] 
        
        position_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0).expand(B, L)
        
        # 6. Relational Bias Computation
        # -------------------------------
        rel_bias_flat = None
        
        if relational_features is not None:
            # (B, T, N, N, 4)
            # Flatten to (B*T, N, N, 4)
            feats_flat = relational_features.reshape(B * T, N, N, 4)
            
            # The previous logic had bias_self and bias_prev.
            # Here we approximate bias_prev using current relative features.
            # bias_self = bias_prev = features
            
            # Compute 12D features
            features_12d = self.relational_extractor.compute_features(feats_flat) # (B*T, N, N, 12)
            
            bias_self = self.relational_extractor.project_features(features_12d) # (B*T, H, N, N)
            
            # Use same bias for 'prev' (S_t -> S_{t-1})
            # This ignores relative velocity changes? 
            # If feats has RelVel, then S_t -> S_{t-1} should logically use -RelVel?
            # Or just assume frames are close enough.
            # Given we only precompute for current frame, we use it for both.
            bias_prev = bias_self
            
            # Unflatten back to (B, T, H, N, N)
            bias_self = bias_self.view(B, T, self.config.n_heads, N, N)
            bias_prev = bias_prev.view(B, T, self.config.n_heads, N, N)
            
            # Assembly
            row_s = torch.cat([bias_prev, bias_self], dim=-1)
            row_a = torch.cat([bias_self, bias_self], dim=-1)
            
            combined_bias = torch.stack([row_s, row_a], dim=2) # (B, T, 2, H, N, 2N)
            rel_bias_flat = combined_bias.view(B * L, self.config.n_heads, N, 2 * N)
        
        elif past_key_values is None and not use_cache:
            # Training mode but NO features passed?
            pass

        # 7. Transformer
        x = tokens
        current_key_values = []
        
        for i, block in enumerate(self.blocks):
            pk = past_key_values[i] if past_key_values is not None else None
            x, kv = block(x, position_ids=position_ids, past_kv=pk, use_cache=use_cache, relational_bias=rel_bias_flat)
            if use_cache:
                current_key_values.append(kv)
                
        x = self.ln_f(x)
        
        # 8. Heads
        x_reshaped = x.view(B, T, 2, N, self.config.embed_dim)
        
        s_out = x_reshaped[:, :, 0, :, :] 
        pred_actions = self.action_head(s_out)
        
        a_out = x_reshaped[:, :, 1, :, :]
        pred_states = self.state_head(a_out)
        
        # Predict Relational Features (S_{t+1} relations) from A_t (Action tokens)
        # a_out corresponds to Action tokens which predict next state.
        pred_relational = self.predict_relational(a_out, B, T, N)
        
        if return_embeddings:
             if features_12d is not None and len(features_12d.shape) == 4 and features_12d.shape[0] == B * T:
                 features_12d = features_12d.view(B, T, N, N, 12)
                 
             return pred_states, pred_actions, current_key_values, x, features_12d, pred_relational
             
        return pred_states, pred_actions, current_key_values

    def predict_relational(self, embeddings: torch.Tensor, B: int, T: int, N: int):
        """
        Predict relational features for next state from current embeddings.
        Args:
            embeddings: (..., N, E) corresponding to valid predictive tokens (Action Tokens)
                        Action tokens at t predict State at t+1.
            B, T, N: Dimensions to reshape for pairing.
        Returns:
            pred_rel: (B, T, N, N, 12)
        """
        # Embeddings Input: x_reshaped[:, :, 1] which is Action Tokens -> State_{t+1}
        # Shape: (B, T, N, E)
        
        # We need pairs (N, N)
        # For each ship i, and ship j, we concatenate emb_i, emb_j.
        # (B, T, N, 1, E) expand (B, T, N, N, E)
        # (B, T, 1, N, E) expand (B, T, N, N, E)
        
        src = embeddings.unsqueeze(3).expand(-1, -1, -1, N, -1)
        tgt = embeddings.unsqueeze(2).expand(-1, -1, N, -1, -1)
        
        pairs = torch.cat([src, tgt], dim=-1) # (B, T, N, N, 2E)
        
        return self.relational_head(pairs)

    def get_loss(
        self,
        pred_states,
        pred_actions,
        target_states,
        target_actions,
        loss_mask, 
        latents=None,
        target_features_12d=None,
        pred_relational=None,
        lambda_state=1.0,
        lambda_action=0.01,
        lambda_relational=0.1
    ):
        """
        pred_states: (B, T, N, D)
        pred_actions: (B, T, N, A)
        target_states: (B, T, N, D)
        target_actions: (B, T, N, 3)
        loss_mask: (B, T)
        latents: (B, 2T, N, E) - Optional
        target_features_12d: (B, T, N, N, 12) - Ground truth (from compute_features)
        pred_relational: (B, T, N, N, 12) - Predicted
        """
        valid_mask = loss_mask.bool() 
        
        valid_pred_act = pred_actions[valid_mask] 
        valid_target_act = target_actions[valid_mask] 
        
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
        
        # Metrics: Entropy & Confidence
        with torch.no_grad():
            # Power
            p_probs = F.softmax(p_logits, dim=-1)
            entropy_p = -torch.sum(p_probs * torch.log(p_probs + 1e-8), dim=-1).mean()
            prob_p = p_probs.gather(1, p_target.unsqueeze(1)).mean()

            # Turn
            t_probs = F.softmax(t_logits, dim=-1)
            entropy_t = -torch.sum(t_probs * torch.log(t_probs + 1e-8), dim=-1).mean()
            prob_t = t_probs.gather(1, t_target.unsqueeze(1)).mean()

            # Shoot
            s_probs = F.softmax(s_logits, dim=-1)
            entropy_s = -torch.sum(s_probs * torch.log(s_probs + 1e-8), dim=-1).mean()
            prob_s = s_probs.gather(1, s_target.unsqueeze(1)).mean()

            # Latent Norms
            norm_latent = torch.tensor(0.0, device=pred_states.device)
            if latents is not None:
                norm_latent = latents.norm(dim=-1).mean()

            # Classification Errors (Hard)
            # Compare argmax logits to target
            error_p = (p_logits.argmax(dim=-1) != p_target).float().mean()
            error_t = (t_logits.argmax(dim=-1) != t_target).float().mean()
            error_s = (s_logits.argmax(dim=-1) != s_target).float().mean()

        valid_pred_state = pred_states[valid_mask]
        valid_target_state = target_states[valid_mask]
        
        state_loss = F.mse_loss(valid_pred_state, valid_target_state)
        
        # Relational Loss
        relational_loss = torch.tensor(0.0, device=pred_states.device)
        if target_features_12d is not None and pred_relational is not None:
             # Mask: (B, T) -> (B, T, N, N)
             # Expand mask to ships and pairs
             B, T, N = pred_states.shape[:3]
             # (B, T, 1, 1)
             rel_mask = loss_mask.view(B, T, 1, 1).expand(B, T, N, N).bool()
             
             # Also mask diagonal (i == j)
             eye_mask = ~torch.eye(N, device=pred_states.device).bool().view(1, 1, N, N).expand(B, T, N, N)
             
             final_mask = rel_mask & eye_mask
             
             valid_pred_rel = pred_relational[final_mask]
             valid_target_rel = target_features_12d[final_mask]
             
             if valid_target_rel.numel() > 0:
                 relational_loss = F.mse_loss(valid_pred_rel, valid_target_rel)
        
        total_loss = lambda_state * state_loss + lambda_action * action_loss + lambda_relational * relational_loss
        
        metrics = {
             "entropy_power": entropy_p,
             "entropy_turn": entropy_t,
             "entropy_shoot": entropy_s,
             "prob_power": prob_p,
             "prob_turn": prob_t,
             "prob_shoot": prob_s,
             "error_power": error_p,
             "error_turn": error_t,
             "error_shoot": error_s,
             "norm_latent": norm_latent,
        }
        
        return total_loss, state_loss, action_loss, relational_loss, metrics

    @torch.no_grad()
    def generate(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        steps: int,
        n_ships: int,
        temperature: float = 1.0,
        team_ids: torch.Tensor = None
    ):
        """
        Autoregressive generation of state and action trajectories.

        Args:
            initial_state: (1, N, D)
            initial_action: (1, N, 12) - Typically ignored/dummy as we predict A_0 from S_0?
                            Actually S_0 -> A_0. So we don't need initial_action input really.
                            But signature matches old world model.
            steps: Number of steps to generate.
            n_ships: Number of ships (unused if inferred from state).
            team_ids: (1, N) Optional team IDs.

        Returns:
            dream_states: (1, Steps, N, D)
            gen_actions: (1, Steps, N, 12) - One Hot
        """
        B, N, D = initial_state.shape
        device = initial_state.device
        
        curr_state = initial_state.unsqueeze(1) # (B, 1, N, D)
        
        # Cache
        past_key_values = None
        
        all_states = []
        all_actions = []
        
        # Team IDs: Default to 0 if not provided
        if team_ids is None:
            team_ids = torch.zeros((B, N), dtype=torch.long, device=device)

        for _ in range(steps):
            # 1. Predict Action A_t from S_t
            # We pass dummy action. use_cache=False to peek.
            dummy_action_idx = torch.zeros((B, 1, N, 3), device=device)
            
            _, pred_actions_logits, _ = self.forward(
                states=curr_state,
                actions=dummy_action_idx,
                team_ids=team_ids,
                past_key_values=past_key_values,
                use_cache=False 
            )
            
            # Extract logits for the single step
            logits = pred_actions_logits[:, -1, :, :] # (B, N, 12)
            
            # Greedy decoding
            p_idx = logits[..., 0:3].argmax(dim=-1)
            t_idx = logits[..., 3:10].argmax(dim=-1)
            s_idx = logits[..., 10:12].argmax(dim=-1)
            
            # Indices for Next Pass Input
            action_indices = torch.stack([p_idx, t_idx, s_idx], dim=-1).unsqueeze(1).float()
            
            # One Hot for Output
            p_oh = F.one_hot(p_idx, num_classes=3)
            t_oh = F.one_hot(t_idx, num_classes=7)
            s_oh = F.one_hot(s_idx, num_classes=2)
            action_oh = torch.cat([p_oh, t_oh, s_oh], dim=-1).unsqueeze(1) # (B, 1, N, 12)
            
            # 2. Predict State S_{t+1} from (S_t, A_t)
            # This time use_cache=True to advance
            pred_s, _, new_kv = self.forward(
                 states=curr_state,
                 actions=action_indices,
                 team_ids=team_ids,
                 past_key_values=past_key_values,
                 use_cache=True
            )
            
            past_key_values = new_kv
            next_state = pred_s[:, -1:, :, :] # (B, 1, N, D)
            
            all_states.append(next_state)
            all_actions.append(action_oh)
            
            curr_state = next_state
            
        # Concat
        dream_states = torch.cat(all_states, dim=1)
        gen_actions = torch.cat(all_actions, dim=1).float()
        
        return dream_states, gen_actions
