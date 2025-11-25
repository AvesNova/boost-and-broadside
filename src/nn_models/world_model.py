import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialSelfAttention(nn.Module):
    """
    Standard self-attention over the spatial dimension (N ships).
    Input: (B, T, N, E)
    Reshapes to (B*T, N, E) for attention.
    """

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

    def forward(self, input_tensor, past_kv=None, use_cache=False):
        # input_tensor: (batch_size, time_steps, num_ships, embed_dim)
        batch_size, time_steps, num_ships, embed_dim = input_tensor.size()

        # Merge batch_size and time_steps for spatial attention
        input_flat = input_tensor.view(
            batch_size * time_steps, num_ships, embed_dim
        )  # (Batch, SeqLen, Emb) where Batch=batch_size*time_steps, SeqLen=num_ships

        query, key, value = self.c_attn(input_flat).split(self.embed_dim, dim=2)

        # (Batch, num_ships, n_heads, head_dim) -> (Batch, n_heads, num_ships, head_dim)
        key = key.view(
            batch_size * time_steps, num_ships, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)
        query = query.view(
            batch_size * time_steps, num_ships, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)
        value = value.view(
            batch_size * time_steps, num_ships, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)

        # Spatial attention is not causal and doesn't use past_kv (usually)
        # Because we attend to all ships at the SAME timestep.
        # So past_kv is ignored here.

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size * time_steps, num_ships, embed_dim)
        )
        attention_output = self.resid_dropout(self.c_proj(attention_output))

        # Reshape back to (batch_size, time_steps, num_ships, embed_dim)
        attention_output = attention_output.view(
            batch_size, time_steps, num_ships, embed_dim
        )

        return attention_output, None  # No KV cache for spatial attention


class TemporalSelfAttention(nn.Module):
    """
    Causal self-attention over the temporal dimension (T timesteps).
    Input: (B, T, N, E)
    Reshapes to (B*N, T, E) for attention.
    """

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

    def forward(self, input_tensor, past_kv=None, use_cache=False):
        # input_tensor: (batch_size, time_steps, num_ships, embed_dim)
        batch_size, time_steps, num_ships, embed_dim = input_tensor.size()

        # Merge batch_size and num_ships for temporal attention
        # We want to preserve time_steps as the sequence dimension
        input_flat = (
            input_tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size * num_ships, time_steps, embed_dim)
        )  # (batch_size*num_ships, time_steps, embed_dim)

        query, key, value = self.c_attn(input_flat).split(self.embed_dim, dim=2)

        # (Batch, time_steps, n_heads, head_dim) -> (Batch, n_heads, time_steps, head_dim)
        key = key.view(
            batch_size * num_ships, time_steps, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)
        query = query.view(
            batch_size * num_ships, time_steps, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)
        value = value.view(
            batch_size * num_ships, time_steps, self.n_heads, embed_dim // self.n_heads
        ).transpose(1, 2)

        if past_kv is not None:
            past_key, past_value = past_kv
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            current_kv = (key, value)
        else:
            current_kv = None

        # Causal attention
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True if past_kv is None else False,
        )

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size * num_ships, time_steps, embed_dim)
        )
        attention_output = self.resid_dropout(self.c_proj(attention_output))

        # Reshape back to (batch_size, time_steps, num_ships, embed_dim)
        # (batch_size*num_ships, time_steps, embed_dim) -> (batch_size, num_ships, time_steps, embed_dim) -> (batch_size, time_steps, num_ships, embed_dim)
        attention_output = (
            attention_output.view(batch_size, num_ships, time_steps, embed_dim)
            .transpose(1, 2)
            .contiguous()
        )

        return attention_output, current_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_tensor):
        hidden = self.c_fc(input_tensor)
        hidden = self.gelu(hidden)
        hidden = self.c_proj(hidden)
        hidden = self.dropout(hidden)
        return hidden


class Block(nn.Module):
    def __init__(self, config, attn_type="spatial"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn_type = attn_type
        if attn_type == "spatial":
            self.attn = SpatialSelfAttention(config)
        elif attn_type == "temporal":
            self.attn = TemporalSelfAttention(config)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, input_tensor, past_kv=None, use_cache=False):
        attention_output, current_kv = self.attn(
            self.ln_1(input_tensor), past_kv, use_cache
        )
        residual = input_tensor + attention_output
        output = residual + self.mlp(self.ln_2(residual))
        return output, current_kv


class WorldModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class WorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_ships: int = 8,
        max_context_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = WorldModelConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_ships=max_ships,
            max_context_len=max_context_len,
            dropout=dropout,
        )

        # Embeddings
        self.input_proj = nn.Linear(state_dim + action_dim, embed_dim)
        self.ship_embed = nn.Parameter(torch.randn(max_ships, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(max_context_len, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(embed_dim))

        # Transformer Backbone
        # Schedule: Spatial, Spatial, Spatial, Temporal, Repeat
        self.blocks = nn.ModuleList()
        pattern = ["spatial", "spatial", "spatial", "temporal"]

        for i in range(n_layers):
            attn_type = pattern[i % len(pattern)]
            self.blocks.append(Block(self.config, attn_type=attn_type))

        self.ln_f = nn.LayerNorm(embed_dim)

        # Heads
        self.state_head = nn.Linear(embed_dim, state_dim)
        self.action_head = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask_ratio: float = 0.0,
        noise_scale: float = 0.0,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Args:
            states: (B, T, N, F)
            actions: (B, T, N, A)
        """
        # Ensure 4D input
        if states.ndim == 3:
            # (batch_size, seq_len, features) -> Assume flattened time_steps*num_ships
            # We need to unflatten.
            batch_size, seq_len, features = states.shape
            num_ships = self.config.max_ships
            if seq_len % num_ships != 0:
                raise ValueError(
                    f"Sequence length {seq_len} not divisible by max_ships {num_ships}"
                )
            time_steps = seq_len // num_ships
            states = states.view(batch_size, time_steps, num_ships, features)
            actions = actions.view(batch_size, time_steps, num_ships, actions.shape[-1])

        batch_size, time_steps, num_ships, _ = states.shape
        device = states.device

        # 1. Construct Tokens
        embeddings = self.input_proj(
            torch.cat([states, actions], dim=-1)
        )  # (batch_size, time_steps, num_ships, embed_dim)

        # Add structural embeddings
        # We need to broadcast ship_embed and time_embed
        # ship_embed: (num_ships, embed_dim) -> (1, 1, num_ships, embed_dim)
        # time_embed: (time_steps, embed_dim) -> (1, time_steps, 1, embed_dim)

        # If we are generating step-by-step, time_steps might be small, but time_ids should be correct.
        # But here forward assumes we start from 0 or we need to handle offsets.
        # For training, we assume full sequence 0..time_steps-1.
        # For generation, we usually pass one step but we need the time index.
        # The current signature doesn't accept time_offset.
        # We'll assume for now forward is used for training (full seq) or we handle it via past_kv logic?
        # Actually, if past_key_values is present, we are appending.
        # But we need to know the current time index.
        # Let's assume standard forward is 0..time_steps.

        # If using cache, we assume we are at step time_steps_past.
        time_offset = 0
        if past_key_values is not None:
            # We need to infer time_offset.
            # Temporal blocks have KV cache.
            # Find the first temporal block's KV cache to get length.
            for i, block in enumerate(self.blocks):
                if block.attn_type == "temporal" and past_key_values[i] is not None:
                    time_offset = past_key_values[i][0].shape[
                        2
                    ]  # (batch_size*num_ships, n_heads, time_steps_past, head_dim)
                    break

        ship_ids = torch.arange(num_ships, device=device).view(1, 1, num_ships)
        time_ids = torch.arange(
            time_offset, time_offset + time_steps, device=device
        ).view(1, time_steps, 1)

        embeddings = embeddings + self.ship_embed[ship_ids] + self.time_embed[time_ids]

        # 2. Masking & Denoising (Training only)
        mask = None
        if mask_ratio > 0 and past_key_values is None:
            # Create mask over (batch_size, time_steps, num_ships)
            mask = (
                torch.rand(batch_size, time_steps, num_ships, device=device)
                < mask_ratio
            )

            # Denoise
            if noise_scale > 0:
                # Noise on embeddings before adding positional? Or after?
                # Original code: noise on content_embed, then add pos.
                # Let's re-calculate content_embed for noise purpose if needed,
                # or just add noise to embeddings (which includes pos now).
                # Original: content_embed[~mask] += noise
                # Here embeddings includes pos.
                # Let's generate noise on the projected input
                content_input = self.input_proj(torch.cat([states, actions], dim=-1))
                tau = torch.rand(batch_size, 1, 1, 1, device=device).pow(2)
                noise_std = (1 - tau).sqrt() * noise_scale
                noise = torch.randn_like(content_input) * noise_std

                # We need to apply noise to the unmasked parts of embeddings
                # But embeddings already has pos embeddings.
                # embeddings = content + pos.
                # We want embeddings' = (content + noise) + pos = embeddings + noise.
                embeddings[~mask] = embeddings[~mask] + noise[~mask]

            # Apply mask token
            embeddings[mask] = self.mask_token

        # 3. Transformer Pass
        current_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            embeddings, kv = block(embeddings, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                current_key_values.append(kv)

        embeddings = self.ln_f(embeddings)

        # 4. Predictions
        pred_states = self.state_head(embeddings)
        pred_actions = self.action_head(embeddings)

        return pred_states, pred_actions, mask, current_key_values

    def get_loss(self, states, actions, pred_states, pred_actions, mask):
        # states: (B, T, N, F)
        # mask: (B, T, N)

        if states.ndim == 3:
            # Unflatten if needed, but usually passed same as forward input
            batch_size, seq_len, features = states.shape
            num_ships = self.config.max_ships
            time_steps = seq_len // num_ships
            states = states.view(batch_size, time_steps, num_ships, features)
            actions = actions.view(batch_size, time_steps, num_ships, actions.shape[-1])
            if mask is not None:
                mask = mask.view(batch_size, time_steps, num_ships)

        recon_loss = 0
        if mask is not None and mask.any():
            recon_loss = F.mse_loss(pred_states[mask], states[mask])
            recon_loss += F.mse_loss(pred_actions[mask], actions[mask])

        denoise_loss = 0
        if mask is not None and (~mask).any():
            denoise_loss = F.mse_loss(pred_states[~mask], states[~mask])
            denoise_loss += F.mse_loss(pred_actions[~mask], actions[~mask])

        return recon_loss, denoise_loss

    @torch.no_grad()
    def generate(self, initial_state, initial_action, steps: int, n_ships: int):
        """
        Autoregressive generation.
        Args:
            initial_state: (B, F) - state of first ship at t=0?
                           No, usually initial_state is (B, N, F) for all ships at t=0?
                           Or just (B, F) if we only have 1 ship?
                           The original code had `initial_state.view(B, 1, 1, -1)`.
                           Let's assume initial_state is (B, N, F) or (B, F) broadcasted.
            initial_action: (B, N, A)
            steps: number of timesteps to generate
            n_ships: number of ships
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device

        # Ensure inputs are (batch_size, num_ships, features)
        if initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(1).repeat(
                1, n_ships, 1
            )  # (batch_size, num_ships, features)
        if initial_action.ndim == 2:
            initial_action = initial_action.unsqueeze(1).repeat(
                1, n_ships, 1
            )  # (batch_size, num_ships, action_dim)

        # Current input for step t=0
        current_state = initial_state.unsqueeze(
            1
        )  # (batch_size, 1, num_ships, features)
        current_action = initial_action.unsqueeze(
            1
        )  # (batch_size, 1, num_ships, action_dim)

        past_key_values = None
        all_states = []
        all_actions = []

        for timestep in range(steps):
            # Forward pass for current step
            pred_states, pred_actions, _, current_key_values = self.forward(
                current_state,
                current_action,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Update past_key_values
            past_key_values = current_key_values

            # Predictions are for the NEXT step
            # pred_states: (batch_size, 1, num_ships, features)
            next_state = pred_states
            next_action_logits = pred_actions

            # Sample action (deterministic for now)
            next_action_probs = torch.sigmoid(next_action_logits)
            next_action = (next_action_probs > 0.5).float()

            all_states.append(next_state)
            all_actions.append(next_action)

            # Prepare for next step
            current_state = next_state
            current_action = next_action

        # Concatenate
        gen_states = torch.cat(
            all_states, dim=1
        )  # (batch_size, time_steps, num_ships, features)
        gen_actions = torch.cat(
            all_actions, dim=1
        )  # (batch_size, time_steps, num_ships, action_dim)

        return gen_states, gen_actions
