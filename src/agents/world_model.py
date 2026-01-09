"""
World model architecture for multi-ship dynamics prediction.

Implements a transformer-based world model with factorized spatial-temporal
attention, masked reconstruction, and denoising objectives for learning
multi-agent dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.rope import RotaryPositionEmbedding


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

        # RoPE for temporal position encoding
        head_dim = config.embed_dim // config.n_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=config.max_context_len,
            base=getattr(config, "rope_base", 10000.0),
        )

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

        # Apply RoPE to query and key
        # Compute position offset from past_kv cache
        time_offset = 0
        if past_kv is not None:
            past_key, past_value = past_kv
            time_offset = past_key.shape[
                2
            ]  # (batch*num_ships, n_heads, past_time, head_dim)

        # Position IDs for current timesteps
        position_ids = torch.arange(
            time_offset, time_offset + time_steps, device=query.device
        )

        # Apply RoPE per head
        # query/key shape: (batch*num_ships, n_heads, time_steps, head_dim)
        # RoPE expects: (batch, seq_len, dim)
        # Reshape to (batch*num_ships*n_heads, time_steps, head_dim)
        batch_heads = batch_size * num_ships * self.n_heads
        head_dim = embed_dim // self.n_heads

        query_rope = (
            query.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_heads, time_steps, head_dim)
        )
        key_rope = (
            key.permute(0, 2, 1, 3).contiguous().view(batch_heads, time_steps, head_dim)
        )

        query_rope, key_rope = self.rope(query_rope, key_rope, position_ids)

        # Reshape back to (batch*num_ships, n_heads, time_steps, head_dim)
        query = query_rope.view(
            batch_size * num_ships, time_steps, self.n_heads, head_dim
        ).permute(0, 2, 1, 3)
        key = key_rope.view(
            batch_size * num_ships, time_steps, self.n_heads, head_dim
        ).permute(0, 2, 1, 3)

        if past_kv is not None:
            past_key, past_value = past_kv
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            current_kv = (key, value)
        else:
            current_kv = None

        # CRITICAL: Temporal attention must ALWAYS be causal
        # Even with KV cache, new tokens can only attend to themselves and past
        # We use is_causal=True which creates a causal mask for the query length
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,  # Always causal for temporal attention
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
        self.mask_token = nn.Parameter(torch.randn(embed_dim))

        # Transformer Backbone
        # Schedule: Temporal, Spatial, Spatial, Spatial, Repeat
        self.blocks = nn.ModuleList()
        pattern = ["temporal", "spatial", "spatial", "spatial"]

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
        mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass with correct masking and denoising.

        Token structure: Each token = [ship_state, previous_action] concatenated
        Embedding structure: final_embed = content + ship_id + time

        Critical:
        - Noise is applied to content BEFORE adding structural embeddings
        - Masking replaces content only, preserving ship_id and time
        - Masked tokens NEVER receive noise

        Args:
            states: (B, T, N, F) - ship states
            actions: (B, T, N, A) - previous actions
            mask_ratio: Fraction of tokens to mask (MAE-style)
            noise_scale: Scale of Gaussian noise for denoising
            mask: Optional boolean mask (True = masked). If provided, overrides mask_ratio.
            past_key_values: KV cache for autoregressive generation
            use_cache: Whether to return KV cache
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

        # 1. Project content (state + action concatenated)
        content_embed = self.input_proj(
            torch.cat([states, actions], dim=-1)
        )  # (batch_size, time_steps, num_ships, embed_dim)

        # 2. Masking & Denoising (Training only)
        # CRITICAL: Apply noise and masking to content BEFORE adding structural embeddings

        # Determine mask
        if mask is None:
            if mask_ratio > 0 and past_key_values is None:
                # Create mask over (batch_size, time_steps, num_ships)
                mask = (
                    torch.rand(batch_size, time_steps, num_ships, device=device)
                    < mask_ratio
                )

        if mask is not None:
            # Apply noise to UNMASKED tokens only, BEFORE adding positional embeddings
            if noise_scale > 0:
                # Flow-matching noise with τ ∈ [0, 1] sampled per batch
                tau = torch.rand(batch_size, 1, 1, 1, device=device).pow(2)
                noise_std = (1 - tau).sqrt() * noise_scale
                noise = torch.randn_like(content_embed) * noise_std

                # Apply noise only to unmasked tokens
                content_embed = torch.where(
                    mask.unsqueeze(-1),
                    content_embed,  # Keep masked tokens unchanged (will be replaced below)
                    content_embed + noise,  # Add noise to unmasked tokens
                )

            # Replace masked token content with learned mask_token
            # CRITICAL: This only replaces content, structural embeddings added later
            content_embed = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.view(1, 1, 1, -1).expand_as(content_embed),
                content_embed,
            )

        # 3. Add structural embeddings (ship_id only, RoPE handles temporal position)
        # This is done AFTER masking and denoising to preserve structural information

        # Broadcast ship embeddings
        ship_ids = torch.arange(num_ships, device=device).view(1, 1, num_ships)

        # Final embedding = content + ship_id (RoPE applied in temporal attention)
        embeddings = content_embed + self.ship_embed[ship_ids]

        # 4. Transformer Pass
        current_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            embeddings, kv = block(embeddings, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                current_key_values.append(kv)

        embeddings = self.ln_f(embeddings)

        # 5. Predictions
        pred_states = self.state_head(embeddings)
        pred_actions = self.action_head(embeddings)

        return pred_states, pred_actions, mask, current_key_values

    def get_loss(
        self, states, actions, pred_states, pred_actions, mask, loss_mask=None
    ):
        # states: (B, T, N, F)
        # actions: (B, T, N, 12) - One-hot encoded actions
        # mask: (B, T, N) - True = masked (reconstruction target), False = visible (denoising target)
        # loss_mask: (B, T, N) - True = compute loss, False = ignore (warm-up/padding)

        if states.ndim == 3:
            # Unflatten if needed, but usually passed same as forward input
            batch_size, seq_len, features = states.shape
            num_ships = self.config.max_ships
            time_steps = seq_len // num_ships
            states = states.view(batch_size, time_steps, num_ships, features)
            actions = actions.view(batch_size, time_steps, num_ships, actions.shape[-1])
            if mask is not None:
                mask = mask.view(batch_size, time_steps, num_ships)
            if loss_mask is not None:
                loss_mask = loss_mask.view(batch_size, time_steps, num_ships)

        # If no loss_mask provided, compute loss for all tokens
        if loss_mask is None:
            loss_mask = torch.ones_like(mask, dtype=torch.bool)
        elif loss_mask.ndim == 2:
            # (B, T) -> (B, T, N)
            loss_mask = loss_mask.unsqueeze(-1).expand_as(mask)

        recon_loss = torch.tensor(0.0, device=states.device)
        denoise_loss = torch.tensor(0.0, device=states.device)

        # Helper to compute categorical action loss
        def compute_action_loss(preds, targets):
            # preds: (K, 12)
            # targets: (K, 12) (one-hot)

            # Split heads
            pred_power = preds[:, 0:3]
            pred_turn = preds[:, 3:10]
            pred_shoot = preds[:, 10:12]

            target_power = targets[:, 0:3].argmax(dim=-1)
            target_turn = targets[:, 3:10].argmax(dim=-1)
            target_shoot = targets[:, 10:12].argmax(dim=-1)

            loss = F.cross_entropy(pred_power, target_power)
            loss += F.cross_entropy(pred_turn, target_turn)
            loss += F.cross_entropy(pred_shoot, target_shoot)
            return loss

        # Reconstruction Loss (for masked tokens)
        recon_target_mask = mask & loss_mask
        if recon_target_mask.any():
            # State loss: MSE
            recon_loss += F.mse_loss(
                pred_states[recon_target_mask], states[recon_target_mask]
            )
            # Action loss: Categorical
            recon_loss += compute_action_loss(
                pred_actions[recon_target_mask], actions[recon_target_mask]
            )

        # Denoising Loss (for unmasked tokens)
        denoise_target_mask = (~mask) & loss_mask
        if denoise_target_mask.any():
            # State loss: MSE
            denoise_loss += F.mse_loss(
                pred_states[denoise_target_mask], states[denoise_target_mask]
            )
            # Action loss: Categorical
            denoise_loss += compute_action_loss(
                pred_actions[denoise_target_mask], actions[denoise_target_mask]
            )

        return recon_loss, denoise_loss

    @torch.no_grad()
    def generate(self, initial_state, initial_action, steps: int, n_ships: int):
        """
        Autoregressive generation.

        Args:
            initial_state: (B, N, F) - Initial state of all ships.
            initial_action: (B, N, A) - Initial action (one-hot) of all ships.
            steps: Number of timesteps to generate.
            n_ships: Number of ships.
        """
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
            next_state = pred_states
            next_action_logits = pred_actions  # (B, 1, N, 12)

            # Sample action (categorical)
            # Deterministic argmax for generation/viz usually best, or temperature sampling
            # Using argmax for stability in this context

            # Split logits
            power_logits = next_action_logits[..., 0:3]
            turn_logits = next_action_logits[..., 3:10]
            shoot_logits = next_action_logits[..., 10:12]

            power_idx = power_logits.argmax(dim=-1)
            turn_idx = turn_logits.argmax(dim=-1)
            shoot_idx = shoot_logits.argmax(dim=-1)

            # Convert back to one-hot (B, 1, N, 12)
            power_oh = F.one_hot(power_idx, num_classes=3)
            turn_oh = F.one_hot(turn_idx, num_classes=7)
            shoot_oh = F.one_hot(shoot_idx, num_classes=2)

            next_action = torch.cat([power_oh, turn_oh, shoot_oh], dim=-1).float()

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
