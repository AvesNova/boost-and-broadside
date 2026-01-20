
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
        # Get Previous Step K, V
        if past_kv is not None:
             # Generation Mode: L=1, past_kv holds ALL past steps (B, 1, N, H, D) ??? 
             # NO. In standard cache, K is cat'd along time.
             # So pk shape is (B, L_past, N, H, D).
             # We want the LAST step of the past.
             
             pk, pv = past_kv
             
             # Slice the last step from past
             # pk: (B, L_past, N, H, D)
             if pk.shape[1] > 0:
                 k_prev = pk[:, -1:] # (B, 1, N, H, D)
                 v_prev = pv[:, -1:] # (B, 1, N, H, D)
             else:
                 # No history? Should not happen if L=1 generation starts after t=0?
                 # If t=0 generation (cold start), past_kv might be empty or None.
                 # But if None, we fall to 'else' block which creates zeros.
                 # If past_kv is not None but empty??
                 k_prev = torch.zeros_like(k)
                 v_prev = torch.zeros_like(v)
                 
        else:
             # Training Mode: Shift current K, V sequences
             # Pad with zeros at start
             # k: (B, L, N, H, D)
             zeros_k = torch.zeros_like(k[:, :1])
             zeros_v = torch.zeros_like(v[:, :1])
             k_prev = torch.cat([zeros_k, k[:, :-1]], dim=1)
             v_prev = torch.cat([zeros_v, v[:, :-1]], dim=1)
        
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
        # Attention - NON CAUSAL (because we manually windowed)
        # y = SDPA(q, k, v)
        # q: (batch, heads, N, D)
        # k: (batch, heads, 2N, D)
        # bias: (batch, heads, N, 2N)
        
        if relational_bias is not None:
             # relational_bias is (B*L, Heads, N, 2N)
             # SDPA supports attn_mask.
             # but strictly speaking SDPA with mask is usually boolean or additive float.
             # To apply additive bias, we might need to manually do attention if SDPA doesn't support 'bias' directly (it supports mask).
             # PyTorch 2.0 SDPA: attn_mask can be a float mask (added to scores).
             pass
             
        # Manual Attention for Bias (if bias present) or rely on SDPA support
        # SDPA docs: "If attn_mask is a Tensor, it must be ... or a float tensor of shape compatible with (batch, heads, target_seq, source_seq) ... added to the attention scores"
        
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
        
        self.bin_emb_list = nn.ModuleList([nn.Embedding(2, embed_dim) for _ in range(self.num_binary)])
        
        # Action Encoders
        # Action Encoders
        # Total concatenated width: 48 + 48 + 32 = 128
        self.emb_power = nn.Embedding(3, 48)
        self.emb_turn = nn.Embedding(7, 48)
        self.emb_shoot = nn.Embedding(2, 32)
        
        # Projection Layer: Concatenated (128) -> Embed Dim (128)
        # Allows learning nonlinear interactions between control inputs (e.g. "Full Power + Hard Turn")
        self.action_proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.SiLU()
        )
        
        # Positional/Type
        # Temporal RoPE is inside TemporalSelfAttention
        
        # Spatial ID Embedding
        self.ship_embed = nn.Embedding(max_ships, embed_dim)
        self.team_embed = nn.Embedding(2, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)
        
        # Transformer Blocks (Alternating T -> S)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            # We want alternating. If n_layers = 6, does that mean 6 T and 6 S? Or 3 T and 3 S?
            # Standard convention: n_layers is total blocks.
            # Layer 0: Temporal
            # Layer 1: Spatial
            # ...
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
        
        # Concatenate: (..., 48+48+32=128)
        action_concat = torch.cat([p_emb, t_emb, s_emb], dim=-1)
        
        # Project: (..., 128) -> (..., embed_dim)
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
        # Stack dim 2 -> (B, T, 2, N, E)
        combined = torch.stack([state_tokens, action_tokens], dim=2)
        L = T * 2
        # Flatten Time: (B, L, N, E)
        tokens = combined.view(B, L, N, self.config.embed_dim)
        
        # 5. Position IDs (for Temporal RoPE)
        # We process (S0, A0), (S1, A1)...
        # Indices should be 0, 1, 2, 3...
        # If passed cache, we need offset.
        past_len = 0
        # Check cache structure: list of tuples. First layer (Tempo) cache.
        # Temp cache shape is (B*N, H, L, D).
        if past_key_values is not None:
             # Look at first temporal layer cache to get length
             # First layer is index 0
             if len(past_key_values) > 0 and past_key_values[0] is not None:
                 past_len = past_key_values[0][0].shape[2] 
        
        position_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0).expand(B, L)
        
        # 6. Relational Bias Computation
        # -------------------------------
        # Structural Deduplication Strategy:
        # Instead of computing features for all 2*T tokens (S0, A0, S1, A1...) linearly,
        # we recognize that S_t and A_t share the same physical state (from S_t).
        # We compute two fundamental matrices per timestep T:
        # 1. bias_self: Rel(S_t, S_t)
        # 2. bias_prev: Rel(S_t, S_{t-1})
        #
        # Then we assemble the full sequence:
        # S_t (Index 2t):   Window [Prev, Curr] -> [S_{t-1}, S_t] -> [bias_prev, bias_self]
        # A_t (Index 2t+1): Window [Prev, Curr] -> [S_t, S_t]     -> [bias_self, bias_self]
        
        # 1. Prepare Physical States (B, T, N, D)
        curr_phys = states
        
        # Prev Physical States (Shifted right, padded with zeros)
        zeros_phys = torch.zeros_like(curr_phys[:, :1])
        prev_phys = torch.cat([zeros_phys, curr_phys[:, :-1]], dim=1)
        
        # 2. Compute Biases (Vectorized over T)
        # Input to extractor: (B*T, N, D)
        # We verify if we are in generation mode (past_kv present)
        if past_key_values is not None and len(past_key_values) > 0:
             # GENERATION MODE (Simplified fallback for now)
             # Logic is tricky because 'states' input might be just length 1.
             # Fallback to the naive approach for generation safety or handle explicitly.
             # For now, let's use the explicit robust logic:
             phys_seq = states.repeat_interleave(2, dim=1)
             q_phys = phys_seq
             zeros_gen = torch.zeros_like(q_phys[:, :1])
             prev_gen = torch.cat([zeros_gen, q_phys[:, :-1]], dim=1)
             
             # If we have history, we might need actual previous state from history?
             # Since this is "Spatial" attention, the window is always [Prev, Curr].
             # If L=1 (generating A_t), Prev is S_t.
             # If L=1 (generating S_{t+1}), Prev is A_t (which has pos S_t).
             # So actually, for generation step, reusing the naive flattening is safer 
             # and performance matters less (batch size small).
             
             q_flat_gen = q_phys.view(B*L, N, -1)
             k_flat_gen = torch.cat([prev_gen, q_phys], dim=2).view(B*L, 2*N, -1)
             rel_bias_flat = self.relational_extractor(q_flat_gen, k_flat_gen)
             
        else:
            # TRAINING MODE (Full Sequence Optimization)
            
            # Reshape for Extractor (B*T, N, D)
            curr_flat = curr_phys.reshape(B*T, N, -1)
            prev_flat = prev_phys.reshape(B*T, N, -1)
            
            # Compute Matrices
            # bias_self: (B*T, H, N, N)
            bias_self = self.relational_extractor(curr_flat, curr_flat)
            
            # bias_prev: (B*T, H, N, N)
            bias_prev = self.relational_extractor(curr_flat, prev_flat)
            
            # Unflatten back to (B, T, H, N, N)
            bias_self = bias_self.view(B, T, self.config.n_heads, N, N)
            bias_prev = bias_prev.view(B, T, self.config.n_heads, N, N)
            
            # 3. Assembly
            # S_t Rows: Cat(bias_prev, bias_self) -> (B, T, H, N, 2N)
            row_s = torch.cat([bias_prev, bias_self], dim=-1)
            
            # A_t Rows: Cat(bias_self, bias_self) -> (B, T, H, N, 2N)
            row_a = torch.cat([bias_self, bias_self], dim=-1)
            
            # Interleave Rows: Stack (B, T, 2, ...) -> Flatten T*2
            # Stack dim 2
            combined_bias = torch.stack([row_s, row_a], dim=2) # (B, T, 2, H, N, 2N)
            
            # Flatten to (B*L, H, N, 2N) where L = 2*T
            rel_bias_flat = combined_bias.view(B * L, self.config.n_heads, N, 2 * N)
        
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
        # x is (B, L, N, E)
        # S tokens at 0, 2, 4 -> Predict Actions
        # A tokens at 1, 3, 5 -> Predict States
        
        # Reshape back to (B, T, 2, N, E)
        x_reshaped = x.view(B, T, 2, N, self.config.embed_dim)
        
        # S tokens (index 0) -> Action Head (Predict A_t)
        s_out = x_reshaped[:, :, 0, :, :] 
        pred_actions = self.action_head(s_out)
        
        # A tokens (index 1) -> State Head (Predict S_{t+1})
        a_out = x_reshaped[:, :, 1, :, :]
        pred_states = self.state_head(a_out)
        
        if return_embeddings:
             return pred_states, pred_actions, current_key_values, x
             
        return pred_states, pred_actions, current_key_values

    def get_loss(
        self,
        pred_states,
        pred_actions,
        target_states,
        target_actions,
        loss_mask, 
        lambda_state=1.0,
        lambda_action=0.01
    ):
        """
        pred_states: (B, T, N, D)
        pred_actions: (B, T, N, A)
        target_states: (B, T, N, D)
        target_actions: (B, T, N, 3)
        loss_mask: (B, T)
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
        
        valid_pred_state = pred_states[valid_mask]
        valid_target_state = target_states[valid_mask]
        
        state_loss = F.mse_loss(valid_pred_state, valid_target_state)
        
        total_loss = lambda_state * state_loss + lambda_action * action_loss
        
        return total_loss, state_loss, action_loss

