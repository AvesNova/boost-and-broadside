import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from boost_and_broadside.models.components.layers.utils import RMSNorm, MambaBlock
from boost_and_broadside.models.components.encoders import StateEncoder, ActionEncoder, RelationalEncoder
from boost_and_broadside.models.components.heads import ActorHead, WorldHead, ValueHead, PairwiseRelationalHead
from boost_and_broadside.models.components.team_evaluator import TeamEvaluator 
from boost_and_broadside.models.components.layers.attention import RelationalAttention
from boost_and_broadside.models.components.normalizer import FeatureNormalizer
from boost_and_broadside.models.components.losses import CompositeLoss, LossModule
from boost_and_broadside.core.constants import StateFeature, TargetFeature, STATE_DIM, TARGET_DIM, TOTAL_ACTION_LOGITS


class BaseScaffold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load Normalizer if stats_path is provided
        stats_path = config.get("stats_path", None)
        self.normalizer = FeatureNormalizer(stats_path) if stats_path else None

        self.loss_fn = self.build_loss_fn(config)
    
    def build_loss_fn(self, config):
        if "loss" not in config:
             raise ValueError("Loss configuration missing! Please add a 'loss' section to your model config.")
        return hydra.utils.instantiate(config.loss)

    def get_loss(self, pred_states=None, pred_actions=None, target_states=None, target_actions=None, loss_mask=None, 
                 pred_values=None, pred_rewards=None, target_returns=None, target_rewards=None,
                 target_alive=None, pairwise_pred=None, target_pairwise=None, reward_components=None, **kwargs):
        
        # Adapter to map arguments to generic structure
        preds = {
            "states": pred_states,
            "actions": pred_actions,
            "value": pred_values,
            "reward": pred_rewards,
            "pairwise": pairwise_pred,
            "reward_components": reward_components
        }
        
        targets = {
            "states": target_states,
            "actions": target_actions,
            "returns": target_returns,
            "rewards": target_rewards,
            "pairwise": target_pairwise
        }

        # Pre-process Targets (Normalization)
        if self.normalizer:
            if preds["states"] is not None:
                preds["states"] = self.normalizer.normalize_target(preds["states"])
            if targets["states"] is not None:
                targets["states"] = self.normalizer.normalize_target(targets["states"])
        
        # Combined Mask Logic
        # loss_mask is typically (B, T, N) or (B, T)
        # target_alive is (B, T, N)
        # We merge them here for the generic mask
        final_mask = loss_mask
        
        # Ensure loss_mask is expanded if needed for state/action (B, T, N)
        # But if it's already generic mask, we pass it.
        # Loss modules expect to handle their own dimensional logic, but usually we want
        # a mask that matches the finest granularity (Ship Level).
        
        if final_mask.ndim == 2 and pred_states is not None:
             final_mask = final_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
             
        if target_alive is not None:
             # Ensure shapes match for broadcast
             if final_mask.shape != target_alive.shape:
                  # Try to work with it or assume caller handles it?
                  # trainer.py handles loss_mask slicing.
                  pass
             final_mask = final_mask & target_alive

        return self.loss_fn(preds, targets, final_mask, **kwargs)

class YemongFull(BaseScaffold):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = OmegaConf.create(kwargs)
        super().__init__(config)
        d_model = config.d_model
        
        # Encoders
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model, normalizer=self.normalizer)
        self.relational_encoder = RelationalEncoder(d_model, config.n_layers, normalizer=self.normalizer)
        
        # Action Encoder (Embeddings)
        self.action_encoder = ActionEncoder()
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        self.special_embeddings = nn.ModuleDict({
            'ship_id': nn.Embedding(config.get("max_ships", 8), d_model),
            'team_id': nn.Embedding(2, d_model)
        })
        
        self.special_params = nn.ParameterDict({
            'dead': nn.Parameter(torch.zeros(config.get("input_dim", STATE_DIM))),
            'reset': nn.Parameter(torch.zeros(d_model))
        })

        # Backbone (Interleaved)
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # We instantiate layers from config if provided, else default
            # For strict plug-and-play we would use hydra.utils.instantiate inside a loop
            # But here we hardcode defaults for the "Full" version matching the original architecture
            # We can allow overrides via config later
            
            block = nn.ModuleDict({
                'mamba': MambaBlock(d_model, layer_idx=i),
                'norm1': RMSNorm(d_model),
                'norm2': RMSNorm(d_model),
                # If we wanted to inject spatial layer via config:
                # 'attn': hydra.utils.instantiate(config.spatial_layer) 
                # But for now, we use the specific implementation
                'attn': hydra.utils.instantiate(config.spatial_layer) if 'spatial_layer' in config else \
                        RelationalAttention(d_model, config.n_heads) 
            })
            self.blocks.append(block)

        # Actor Components
        self.actor_spatial_attn = hydra.utils.instantiate(config.spatial_layer) if 'spatial_layer' in config else \
                                  RelationalAttention(d_model, config.n_heads)
        self.actor_spatial_norm = RMSNorm(d_model)
        self.actor_temporal_attn = nn.MultiheadAttention(d_model, num_heads=config.n_heads, batch_first=True)
        self.actor_temporal_norm = RMSNorm(d_model)
        
        self.actor_head = ActorHead(d_model, config.get("action_dim", TOTAL_ACTION_LOGITS))
        
        # World Components
        self.world_head = WorldHead(d_model, config.get("target_dim", TARGET_DIM))
        
        # Value/Reward
        self.team_evaluator = TeamEvaluator(d_model)

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
                inference_params=None, actor_cache=None, world_size=(1024.0, 1024.0), **kwargs):
        
        batch_size, seq_len, num_ships, _ = state.shape
        d_model = self.config.d_model

        # Dead Masking
        if alive is None: alive = state[..., StateFeature.HEALTH] > 0
        state = torch.where(alive.unsqueeze(-1), state.to(self.special_params['dead'].dtype), self.special_params['dead'].view(1, 1, 1, -1))

        # Reset Logic
        if reset_mask is None and seq_idx is not None:
             diff = torch.zeros_like(seq_idx, dtype=torch.bool)
             diff[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
             reset_mask = diff
        
        w_dtype = self.state_encoder.net[0].weight.dtype
        s_emb = self.state_encoder(state.to(w_dtype))
        
        if team_ids is not None:
             t_emb = self.special_embeddings['team_id'](team_ids.long())
             s_emb = s_emb + (t_emb if t_emb.ndim == 4 else t_emb.unsqueeze(1))
             
        ship_ids = torch.arange(num_ships, device=state.device).view(1, 1, num_ships).expand(batch_size, seq_len, -1)
        s_emb = s_emb + self.special_embeddings['ship_id'](ship_ids)

        trunk_out, _ = self.relational_encoder(pos, vel, att=att, world_size=world_size)
        
        if reset_mask is not None:
             s_emb = s_emb + (reset_mask.unsqueeze(-1).unsqueeze(-1) * self.special_params['reset'])

        # Actions
        a_emb = self.action_encoder(prev_action)
        
        x_world = self.fusion(torch.cat([s_emb, a_emb], dim=-1))
        
        # Backbone
        x_mamba = x_world.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, d_model)
        mamba_seq_idx = seq_idx.unsqueeze(1).expand(-1, num_ships, -1).reshape(batch_size * num_ships, seq_len) if seq_idx is not None else None

        for i, block in enumerate(self.blocks):
            normed = block['norm1'](x_mamba)
            m_out = block['mamba'](normed, seq_idx=mamba_seq_idx, inference_params=inference_params)
            x_mamba = x_mamba + m_out
            
            # Wrapper for Relational Attention taking (Batch*Ships, Time, Dim)
            # We perform the layout change inside the forward pass or here cleanly
            # Current Optimization: We inline the permute to keep implicit B*N structure
            # But wait, Mamba is (BN, T, D). Attention needs (B, T, N, D).
            # Let's perform the permute explicitly but efficiently.
            
            # (BN, T, D) -> (B, N, T, D) -> (B, T, N, D)
            x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
            
            # Calculate Attention
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            attn_out = block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
            
            # (B, T, N, D) -> (B, N, T, D) -> (BN, T, D)
            # Add Residual
            x_mamba = x_mamba + attn_out.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)

        x_final = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
        state_pred = self.world_head(x_final)

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




class YemongSpatial(BaseScaffold):
    """
    Spatial-Only Scaffold.
    Focuses on Action Prediction from State + Spatial Context.
    No temporal history, no Mamba blocks.
    Input: Single timestep (or sequence treated as batch).
    """
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = OmegaConf.create(kwargs)
        super().__init__(config)
        d_model = config.d_model
        
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model, normalizer=self.normalizer)
        self.relational_encoder = RelationalEncoder(d_model, config.n_layers, normalizer=self.normalizer) # n_layers here might range over stack
        
        self.special_embeddings = nn.ModuleDict({
            'ship_id': nn.Embedding(config.get("max_ships", 8), d_model),
            'team_id': nn.Embedding(2, d_model)
        })
        
        self.special_params = nn.ParameterDict({
            'dead': nn.Parameter(torch.zeros(config.get("input_dim", STATE_DIM)))
        })

        # Stack of Relational Attention Layers
        self.layers = nn.ModuleList([
             nn.ModuleDict({
                 'norm': RMSNorm(d_model),
                 'attn': hydra.utils.instantiate(config.spatial_layer) if 'spatial_layer' in config else \
                         RelationalAttention(d_model, config.n_heads)
             }) for _ in range(config.n_layers)
        ])
        
        self.actor_head = ActorHead(d_model, config.get("action_dim", TOTAL_ACTION_LOGITS))
        
    def forward(self, state, pos, vel, att=None, team_ids=None, alive=None, world_size=(1024.0, 1024.0), **kwargs):
        # Flatten time dim if present, or treat as batch
        # We expect (B, T, N, D) or (B, N, D)
        added_time = False
        if state.ndim == 4:
            B, T, N, D = state.shape
        else:
            B, N, D = state.shape
            T = 1
            added_time = True
            state = state.unsqueeze(1)
            pos = pos.unsqueeze(1)
            vel = vel.unsqueeze(1)
            if att is not None: att = att.unsqueeze(1)
            if team_ids is not None: team_ids = team_ids.unsqueeze(1)
            if alive is not None: alive = alive.unsqueeze(1)

        if alive is None: alive = state[..., StateFeature.HEALTH] > 0
        state = torch.where(alive.unsqueeze(-1), state.to(self.special_params['dead'].dtype), self.special_params['dead'].view(1, 1, 1, -1))

        s_emb = self.state_encoder(state)
        
        if team_ids is not None:
             t_emb = self.special_embeddings['team_id'](team_ids.long())
             s_emb = s_emb + (t_emb if t_emb.ndim == 4 else t_emb.unsqueeze(1))
             
        ship_ids = torch.arange(N, device=state.device).view(1, 1, N).expand(B, T, -1)
        s_emb = s_emb + self.special_embeddings['ship_id'](ship_ids)

        trunk_out, _ = self.relational_encoder(pos, vel, att=att, world_size=world_size)
        
        x = s_emb
        for i, layer in enumerate(self.layers):
            normed = layer['norm'](x)
            rel_bias = self.relational_encoder.adapters[i](trunk_out)
            x = x + layer['attn'](normed, rel_bias, mask=alive)
            
        action_logits = self.actor_head(x)
        
        if added_time:
             action_logits = action_logits.squeeze(1)
             x = x.squeeze(1)
        
        return None, action_logits, None, None, x




class YemongTemporal(BaseScaffold):
    """
    Temporal-Only Scaffold.
    Focuses on Next State Prediction from State + Action History.
    No spatial interaction (N=1 or Batch of independent ships).
    """
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = OmegaConf.create(kwargs)
        super().__init__(config)
        d_model = config.d_model
        
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model, normalizer=self.normalizer)
        self.action_encoder = ActionEncoder()
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mamba': MambaBlock(d_model, layer_idx=i),
                'norm': RMSNorm(d_model)
            }) for i in range(config.n_layers)
        ])
        
        self.world_head = WorldHead(d_model, config.get("target_dim", TARGET_DIM))
        
    def forward(self, state, prev_action, seq_idx=None, inference_params=None, **kwargs):
        # We assume input is (B, T, N, D) but we process as (B*N, T, D)
        state_is_4d = False
        if state.ndim == 4:
            state_is_4d = True
            B, T, N, D = state.shape
            # Use reshape instead of view to handle non-contiguous tensors
            state = state.reshape(B*N, T, D)
            prev_action = prev_action.reshape(B*N, T, -1)
            if seq_idx is not None:
                seq_idx = seq_idx.unsqueeze(1).expand(B, N, T).reshape(B*N, T)
        
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(prev_action)
        
        x = self.fusion(torch.cat([s_emb, a_emb], dim=-1))
        
        for block in self.mamba_blocks:
            normed = block['norm'](x)
            x = x + block['mamba'](normed, seq_idx=seq_idx, inference_params=inference_params)
            
        state_pred = self.world_head(x)
        
        # Reshape back to (B, T, N, D) if input was 4D
        if state_is_4d:
             # state_pred is (B*N, T, D) -> (B, T, N, D)
             # x is (B*N, T, D) -> (B, T, N, D)
             state_pred = state_pred.view(B, N, T, -1).permute(0, 2, 1, 3)
             x = x.view(B, N, T, -1).permute(0, 2, 1, 3)
        
        # Reshape back if needed, but for loss we can keep flattened usually
        if 'num_ships' in kwargs: 
             # ... explicit reshape logic if needed by trainer
             pass
             
        return state_pred, None, None, None, x




from boost_and_broadside.models.components.encoders import StateEncoder, ActionEncoder, RelationalEncoder, SeparatedActionEncoder

class YemongDynamics(BaseScaffold):
    """
    Dynamics Scaffold (Model-Based).
    Structure:
    1. Encoder(State, PrevAction) -> Backbone -> Z
    2. Z -> ActorHead -> ActionLogits
    3. Z -> AttentionPooling -> ValueHead -> Value (Team)
    4. Z + Action(Sampled/Target) -> FFN -> Z'
    5. Z' -> WorldHead -> NextState
    6. Z' -> AttentionPooling -> RewardHead -> Reward (Team)
    Action Embeddings:
    - Uses Modular 'SeparatedActionEncoder' for both stages.
    """
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = OmegaConf.create(kwargs)
        super().__init__(config)
        d_model = config.d_model
        
        # Encoders
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model, normalizer=self.normalizer)
        self.relational_encoder = RelationalEncoder(d_model, config.n_layers, normalizer=self.normalizer)
        
        # Action Embeddings Component
        embed_dim = config.get("action_embed_dim", 16)
        self.action_encoder_input = SeparatedActionEncoder(embed_dim)
        self.action_encoder_dynamics = SeparatedActionEncoder(embed_dim)
        
        total_action_emb = self.action_encoder_input.output_dim
        
        # Fusion for Backbone Input
        self.fusion = nn.Sequential(
            nn.Linear(d_model + total_action_emb, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        self.special_embeddings = nn.ModuleDict({
            'ship_id': nn.Embedding(config.get("max_ships", 8), d_model),
            'team_id': nn.Embedding(2, d_model)
        })
        
        self.special_params = nn.ParameterDict({
            'dead': nn.Parameter(torch.zeros(config.get("input_dim", STATE_DIM))),
            'reset': nn.Parameter(torch.zeros(d_model))
        })
        
        # Backbone (Shared Trunk)
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            block = nn.ModuleDict({
                'mamba': MambaBlock(d_model, layer_idx=i),
                'norm1': RMSNorm(d_model),
                'norm2': RMSNorm(d_model),
                'attn': hydra.utils.instantiate(config.spatial_layer) if 'spatial_layer' in config else \
                        RelationalAttention(d_model, config.n_heads) 
            })
            self.blocks.append(block)
            
        # Heads on Z
        self.actor_head = nn.Linear(d_model, 3 + 7 + 2) # [Power(3), Turn(7), Shoot(2)]
        
        # Value Head (Team Level)
        self.team_token_value = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm_value = nn.LayerNorm(d_model)
        self.pooler_value = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

        # Dynamics Heads (Auxiliary)
        num_reward_components = config.get("num_reward_components", 1)
        self.num_reward_components = num_reward_components
        self.reward_head = nn.Linear(d_model, num_reward_components)
        
        # Dynamics Module
        # Fusion Z + Action(t)
        self.dynamics_fusion = nn.Sequential(
            nn.Linear(d_model + total_action_emb, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        self.dynamics_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), # SwiGLU style expansion often good, but simple FFN here
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            RMSNorm(d_model)
        )
        
        # Heads on Z'
        self.world_head = WorldHead(d_model, config.get("target_dim", TARGET_DIM))
        
        # Optional Pairwise Relational Head
        self.use_pairwise = config.get("use_pairwise_targets", False)
        if self.use_pairwise:
            self.pairwise_head = PairwiseRelationalHead(d_model)
        
        # Reward w/ Team Pooling
        self.team_token_reward = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pooler_reward = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm_reward = RMSNorm(d_model)
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_reward_components) # Total reward scalar or components
        )
        
        # Loss Params
        self.log_vars = None
        if getattr(config, "loss_type", "fixed") == "uncertainty":
             self.log_vars = nn.ParameterDict({
                 "state": nn.Parameter(torch.tensor(0.0)),
                 "actions": nn.Parameter(torch.tensor(0.0)),
                 "value": nn.Parameter(torch.tensor(0.0)),
                 "reward": nn.Parameter(torch.tensor(0.0))
             })
             
    def forward(self, state, prev_action, pos, vel, att=None, team_ids=None, seq_idx=None, alive=None, reset_mask=None,
                target_actions=None, # (B, T, N, 3) - raw actions to be embedded
                inference_params=None, world_size=(1024.0, 1024.0), **kwargs):
        
        batch_size, seq_len, num_ships, _ = state.shape
        d_model = self.config.d_model
        
        if alive is None: alive = state[..., StateFeature.HEALTH] > 0
        state = torch.where(alive.unsqueeze(-1), state.to(self.special_params['dead'].dtype), self.special_params['dead'].view(1, 1, 1, -1))
        
        if reset_mask is None and seq_idx is not None:
             diff = torch.zeros_like(seq_idx, dtype=torch.bool)
             diff[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
             reset_mask = diff

        # 1. Encode Inputs
        w_dtype = self.state_encoder.net[0].weight.dtype
        s_emb = self.state_encoder(state.to(w_dtype))
        
        if team_ids is not None:
             t_emb = self.special_embeddings['team_id'](team_ids.long())
             s_emb = s_emb + (t_emb if t_emb.ndim == 4 else t_emb.unsqueeze(1))
        
        ship_ids = torch.arange(num_ships, device=state.device).view(1, 1, num_ships).expand(batch_size, seq_len, -1)
        s_emb = s_emb + self.special_embeddings['ship_id'](ship_ids)
        
        trunk_out, _ = self.relational_encoder(pos, vel, att=att, world_size=world_size)
        
        if reset_mask is not None:
             s_emb = s_emb + (reset_mask.unsqueeze(-1).unsqueeze(-1) * self.special_params['reset'])
             
        # Embed Prev Action using Component
        a_prev_emb = self.action_encoder_input(prev_action)
        
        x = self.fusion(torch.cat([s_emb, a_prev_emb], dim=-1))
        
        # 2. Backbone -> Z
        x_mamba = x.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, d_model)
        mamba_seq_idx = seq_idx.unsqueeze(1).expand(-1, num_ships, -1).reshape(batch_size * num_ships, seq_len) if seq_idx is not None else None
        
        for i, block in enumerate(self.blocks):
            normed = block['norm1'](x_mamba)
            m_out = block['mamba'](normed, seq_idx=mamba_seq_idx, inference_params=inference_params)
            x_mamba = x_mamba + m_out
            
            # (BN, T, D) -> (B, T, N, D)
            x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
            
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            attn_out = block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
            
            # (B, T, N, D) -> (BN, T, D)
            x_mamba = x_mamba + attn_out.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)
            
        Z = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3) # (B, T, N, D)
        
        # 3. Heads on Z (Action + Value)
        action_logits = self.actor_head(Z)
        
        # Value w/ Team Pooling
        Batch_Time = batch_size * seq_len
        Z_flat = Z.reshape(Batch_Time, num_ships, d_model)
        # Note: alive mask logic for pooling. True=Keep. key_padding_mask True=Ignore.
        # alive shape: (B, T, N)
        if alive is not None:
             key_padding_mask = ~alive.reshape(Batch_Time, num_ships)
        else:
             key_padding_mask = None
             
        # Value Pooling
        q_val = self.team_token_value.expand(Batch_Time, -1, -1)
        Z_norm_val = self.norm_value(Z_flat)
        team_vec_val, _ = self.pooler_value(q_val, Z_norm_val, Z_norm_val, key_padding_mask=key_padding_mask)
        value_pred = self.value_head(team_vec_val.squeeze(1)).reshape(batch_size, seq_len, 1)

        # 4. Action for Dynamics
        if target_actions is not None:
             # Teacher Forcing: Ground Truth Indices
             actions_to_embed = target_actions
        else:
             # Sampling: Argmax Indices
             l_p, l_t, l_s = action_logits.split([3, 7, 2], dim=-1)
             p = l_p.argmax(dim=-1)
             t = l_t.argmax(dim=-1)
             s_ = l_s.argmax(dim=-1)
             actions_to_embed = torch.stack([p, t, s_], dim=-1)
             
        # Embed using Dynamics Component
        a_curr_emb = self.action_encoder_dynamics(actions_to_embed)

        # 5. Dynamics Fusion -> Z'
        x_dyn = self.dynamics_fusion(torch.cat([Z, a_curr_emb], dim=-1))
        Z_prime = self.dynamics_ffn(x_dyn) + x_dyn 
        
        # 6. Heads on Z' (NextState + Reward)
        next_state_pred = self.world_head(Z_prime)
        
        # Pairwise Relational (Optional)
        if self.use_pairwise:
            zi = Z_prime.unsqueeze(3).expand(-1, -1, -1, num_ships, -1)
            zj = Z_prime.unsqueeze(2).expand(-1, -1, num_ships, -1, -1)
            z_pair = torch.cat([zi, zj], dim=-1)
            pairwise_pred = self.pairwise_head(z_pair)
        else:
            pairwise_pred = None
        
        # Reward w/ Team Pooling
        # Need to re-flatten Z_prime and re-apply mask logic (same mask)
        Z_prime_flat = Z_prime.reshape(Batch_Time, num_ships, d_model)
        q_rew = self.team_token_reward.expand(Batch_Time, -1, -1)
        Z_prime_norm_rew = self.norm_reward(Z_prime_flat)
        team_vec_rew, _ = self.pooler_reward(q_rew, Z_prime_norm_rew, Z_prime_norm_rew, key_padding_mask=key_padding_mask)
        reward_components = self.reward_head(team_vec_rew.squeeze(1)).reshape(batch_size, seq_len, self.num_reward_components)
        reward_pred = reward_components.sum(dim=-1, keepdim=True)
        
        # Return Tuple
        # 1.action_logits, 2.logprob, 3.entropy, 4.value, 5.state, 6.pred_states, 7.pred_rewards, 8.pred_pairwise, 9.reward_components
        return action_logits, None, None, value_pred, None, next_state_pred, reward_pred, pairwise_pred, reward_components
        

        
        if pred_values is not None and target_returns is not None:
             # Check if target_returns is per-node
             if target_returns.ndim == 3 and target_returns.shape[-1] == 1:
                  # Average over ships? Or assume targets are already team-level broadcasted?
                  # If targets are per-ship, we should average them to get team target if we predict team value.
                  # But usually we compute team return by summing/averaging.
                  # Let's assume target_returns is provided as Team Return.
                  # If it is (B, T, N, 1), we take mean? Or slice?
                  if target_returns.shape[2] > 1:
                      # It has N dimension > 1. Likely replicated team reward per ship. Take mean/slice.
                      target_returns = target_returns.mean(dim=2)
                  else:
                      target_returns = target_returns.squeeze(2) # (B, T, 1) -> (B, T)
             elif target_returns.ndim == 3:
                  # (B, T, N) case
                  target_returns = target_returns.mean(dim=2)
             
             # Re-shape to (B, T, 1) for MSE
             if target_returns.ndim == 2: target_returns = target_returns.unsqueeze(-1)
             
             # Similarity for rewards
             if target_rewards.ndim > 2:
                  target_rewards = target_rewards.mean(dim=2) if target_rewards.shape[2] > 1 else target_rewards.squeeze(2)
             if target_rewards.ndim == 2: target_rewards = target_rewards.unsqueeze(-1)

             # Mask for team level? A team is valid if ANY ship is valid?
             # loss_mask (B, T, N).
             m_team = loss_mask.any(dim=-1).float().unsqueeze(-1) # (B, T, 1)
             d_team = m_team.sum() + 1e-6
             
             v_loss = (F.mse_loss(pred_values, target_returns.to(pred_values.dtype), reduction='none') * m_team).sum() / d_team
             r_loss = (F.mse_loss(pred_rewards, target_rewards.to(pred_rewards.dtype), reduction='none') * m_team).sum() / d_team

        # Pairwise Loss
        p_loss = torch.tensor(0.0, device=pred_states.device)
        if pairwise_pred is not None and target_pairwise is not None:
            # target_pairwise: (B, T, N, N, 4)
            # loss_mask: (B, T, N). Pair (i, j) is valid if both are alive.
            # mask_2d: (B, T, N, N)
            m_2d = loss_mask.unsqueeze(3) & loss_mask.unsqueeze(2)
            m_2d_flat = m_2d.reshape(-1).float()
            d_2d = m_2d_flat.sum() + 1e-6
            
            # Predict only for j != i? Or all? Spec says "pairwise comparisons... (ship_0, ship_1) is different from (ship_1, ship_0)".
            # Usually i=j is just 0. Let's include all and let the model learn the diagonals or mask them.
            # Masking diagonal:
            num_ships = loss_mask.shape[2]
            diag_mask = ~torch.eye(num_ships, device=pred_states.device).bool().view(1, 1, num_ships, num_ships)
            m_2d = m_2d & diag_mask
            m_2d_flat = m_2d.reshape(-1).float()
            d_2d = m_2d_flat.sum() + 1e-6
            
            p_mse = F.mse_loss(pairwise_pred, target_pairwise.to(pairwise_pred.dtype), reduction='none')
            p_loss = p_mse.mean(dim=-1).reshape(-1).mul(m_2d_flat).sum() / d_2d

        loss = s_loss * lambda_state + \
               a_loss * lambda_actions + \
               v_loss * lambda_value + \
               r_loss * lambda_reward + \
               p_loss * lambda_pairwise

               
        metrics = {
            "loss": loss,
            "state_loss": s_loss.detach(),
            "action_loss": a_loss.detach(),
            "value_loss": v_loss.detach(),
            "reward_loss": r_loss.detach(),
            "pairwise_loss": p_loss.detach()
        }
        metrics.update(state_losses)
        metrics.update(action_losses)
        
        # Log Reward Components Mean
        if reward_components is not None:
             with torch.no_grad():
                  # shape: (B, T, K)
                  # mask: (B, T, N) -> m_team (B, T, 1)
                  # We want mean of valid steps?
                  # Yes, using m_team
                  m_team = loss_mask.any(dim=-1).float().unsqueeze(-1) # (B, T, 1)
                  d_team = m_team.sum() + 1e-6
                  
                  # Mask reward components
                  # (B, T, K) * (B, T, 1)
                  rc_masked = reward_components * m_team
                  rc_sum = rc_masked.sum(dim=(0, 1)) # Sum over B, T -> (K,)
                  rc_mean = rc_sum / d_team
                  
                  for i in range(self.num_reward_components):
                       metrics[f"val/reward_component_{i}"] = rc_mean[i].item()

        return metrics

        loss_type = getattr(self.config, "loss_type", "fixed")
        if loss_type == "uncertainty" and self.log_vars is not None:
             clamped_sigmas = {}
             def apply_u(loss, name):
                  s = torch.clamp(self.log_vars[name], min=2.0 * math.log(min_sigma))
                  clamped_sigmas[name] = torch.exp(0.5 * s).item()
                  return 0.5 * torch.exp(-s) * loss + 0.5 * s
             l_state_w = apply_u(s_loss, "state")
             l_actions_w = apply_u(a_loss, "actions")
             l_value_w = apply_u(v_loss, "value")
             l_reward_w = apply_u(r_loss, "reward")
             total_loss = l_state_w + l_actions_w + l_value_w + l_reward_w
        else:
             total_loss = (lambda_state * s_loss) + (lambda_actions * a_loss) + (lambda_value * v_loss) + (lambda_reward * r_loss)

        metrics = {"loss": total_loss.item(), 
                   "loss_sub/state_mse": s_loss.item(), 
                   "loss_sub/action_all": a_loss.item(), 
                   "loss_sub/value_mse": v_loss.item(), 
                   "loss_sub/reward_mse": r_loss.item()}
        return total_loss, s_loss, a_loss, v_loss + r_loss, metrics

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocates Mamba state for all layers."""
        cache = {}
        for i, block in enumerate(self.blocks):
            # MambaBlock.allocate_inference_cache returns (conv_state, ssm_state)
            cache[i] = block['mamba'].allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return cache

    def get_action_and_value(self, x, mamba_state=None, action=None, seq_idx=None):
        """
        RL Interface for PPO.
        x: Dict of tensors (state, prev_action, pos, vel, team_ids, alive, etc.)
        mamba_state: Dict of states {layer_idx: (conv, ssm)}
        action: Optional action sequences for update phase.
        seq_idx: Optional sequence indices for Mamba2 (B, T)
        """
        
        # Unpack Obs
        state = x['state']
        pos = x['pos']
        vel = x['vel']
        prev_action = x['prev_action']
        alive = x.get('alive', None)
        team_ids = x.get('team_ids', None)
        
        batch_size, seq_len, num_ships, _ = state.shape
        d_model = self.config.d_model
        
        # 1. Encode Inputs (Common)
        if alive is None: 
            alive = state[..., StateFeature.HEALTH] > 0
        else:
            alive = alive.bool()

        state = torch.where(alive.unsqueeze(-1), state.to(self.special_params['dead'].dtype), self.special_params['dead'].view(1, 1, 1, -1))
        
        w_dtype = self.state_encoder.net[0].weight.dtype
        s_emb = self.state_encoder(state.to(w_dtype))
        
        if team_ids is not None:
             t_emb = self.special_embeddings['team_id'](team_ids.long())
             s_emb = s_emb + (t_emb if t_emb.ndim == 4 else t_emb.unsqueeze(1))
        
        ship_ids = torch.arange(num_ships, device=state.device).view(1, 1, num_ships).expand(batch_size, seq_len, -1)
        s_emb = s_emb + self.special_embeddings['ship_id'](ship_ids)
        
        trunk_out, _ = self.relational_encoder(pos, vel, att=x.get('att', None), world_size=(1024.0, 1024.0))
        
        a_prev_emb = self.action_encoder_input(prev_action)
        x_in = self.fusion(torch.cat([s_emb, a_prev_emb], dim=-1))
        
        # 2. Backbone
        # Reshape for Mamba: (BN, T, D)
        x_mamba = x_in.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, d_model)
        
        if action is None: # RECURRENT MODE (Step-by-Step)
            # We assume seq_len == 1 for rollout step
            # Iterate layers using .step()
            
            new_mamba_state = {}
            
            for i, block in enumerate(self.blocks):
                normed = block['norm1'](x_mamba) # (BN, 1, D)
                
                # Mamba Step
                layer_state = mamba_state[i]
                
                # Check for step method
                if hasattr(block['mamba'], 'step'):
                     m_out, new_conv, new_ssm = block['mamba'].step(normed, layer_state[0], layer_state[1])
                     new_mamba_state[i] = (new_conv, new_ssm)
                else:
                     # Fallback if specific wrapper
                     raise NotImplementedError("MambaBlock .step() not found")

                x_mamba = x_mamba + m_out
                
                # Spatial Mixing
                x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3) # (B, 1, N, D)
                rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
                attn_out = block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
                x_mamba = x_mamba + attn_out.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)
            
            Z = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3) 
            
            # Heads
            action_logits = self.actor_head(Z)
            
            # Value
            Batch_Time = batch_size * seq_len
            Z_flat = Z.reshape(Batch_Time, num_ships, d_model)
            if alive is not None:
                 key_padding_mask = ~alive.reshape(Batch_Time, num_ships)
            else:
                 key_padding_mask = None
                 
            q_val = self.team_token_value.expand(Batch_Time, -1, -1)
            Z_norm_val = self.norm_value(Z_flat)
            team_vec_val, _ = self.pooler_value(q_val, Z_norm_val, Z_norm_val, key_padding_mask=key_padding_mask)
            value_pred = self.value_head(team_vec_val.squeeze(1)).reshape(batch_size, seq_len, 1)

            # Sampling Action
            l_p, l_t, l_s = action_logits.split([3, 7, 2], dim=-1)
            
            # Multi-Categorical Sampling
            probs_p = Categorical(logits=l_p)
            probs_t = Categorical(logits=l_t)
            probs_s = Categorical(logits=l_s)
            
            action_p = probs_p.sample()
            action_t = probs_t.sample()
            action_s = probs_s.sample()
            
            action_sampled = torch.stack([action_p, action_t, action_s], dim=-1) # (B, 1, N, 3)
            
            logprob_p = probs_p.log_prob(action_p)
            logprob_t = probs_t.log_prob(action_t)
            logprob_s = probs_s.log_prob(action_s)
            
            # Sum logprobs across action components
            logprob = logprob_p + logprob_t + logprob_s 
            
            # Sum entropy
            entropy = probs_p.entropy() + probs_t.entropy() + probs_s.entropy()
            
            return action_sampled, logprob, entropy, value_pred, new_mamba_state

        else: # PARALLEL MODE (Update)
            
            # Prepare seq_idx for Mamba (B, T) -> (BN, T)
            if seq_idx is not None:
                mamba_seq_idx = seq_idx.unsqueeze(1).expand(-1, num_ships, -1).reshape(batch_size * num_ships, seq_len)
            else:
                mamba_seq_idx = None
            
            # Loop blocks
            for i, block in enumerate(self.blocks):
                normed = block['norm1'](x_mamba)
                
                # Setup inference params for this block
                if mamba_state is not None:
                     ip = SimpleNamespace(
                         key_value_memory_dict = {i: mamba_state[i]},
                         seqlen_offset = 0 
                     )
                else:
                     ip = None

                m_out = block['mamba'](normed, inference_params=ip, seq_idx=mamba_seq_idx)
                x_mamba = x_mamba + m_out

                # Spatial
                x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
                rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
                attn_out = block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
                x_mamba = x_mamba + attn_out.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)
                
            Z = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3) 
            
            # Heads
            action_logits = self.actor_head(Z)
            
            # Value
            Batch_Time = batch_size * seq_len
            Z_flat = Z.reshape(Batch_Time, num_ships, d_model)
            if alive is not None:
                 key_padding_mask = ~alive.reshape(Batch_Time, num_ships)
            else:
                 key_padding_mask = None
            
            q_val = self.team_token_value.expand(Batch_Time, -1, -1)
            Z_norm_val = self.norm_value(Z_flat)
            team_vec_val, _ = self.pooler_value(q_val, Z_norm_val, Z_norm_val, key_padding_mask=key_padding_mask)
            value_pred = self.value_head(team_vec_val.squeeze(1)).reshape(batch_size, seq_len, 1)

            # Evaluation for Loss
            # action is provided (B, T, N, 3)
            l_p, l_t, l_s = action_logits.split([3, 7, 2], dim=-1)
            
            probs_p = Categorical(logits=l_p)
            probs_t = Categorical(logits=l_t)
            probs_s = Categorical(logits=l_s)
            
            a_p = action[..., 0]
            a_t = action[..., 1]
            a_s = action[..., 2]
            
            new_logprob = probs_p.log_prob(a_p) + probs_t.log_prob(a_t) + probs_s.log_prob(a_s)
            entropy = probs_p.entropy() + probs_t.entropy() + probs_s.entropy()
            
            # --- DYNAMICS PREDICTION (Auxiliary) ---
            # 1. Embed Action (B, T, N, D)
            a_curr_emb = self.action_encoder_dynamics(action.long())
            
            # 2. Fusion (Z + Action) -> Dynamics Content
            # Z is (B, T, N, D)
            x_dyn = self.dynamics_fusion(torch.cat([Z, a_curr_emb], dim=-1))
            
            # 3. Process
            Z_prime = self.dynamics_ffn(x_dyn) + x_dyn
            
            # 4. Heads
            next_state_pred = self.world_head(Z_prime)
            next_state_pred = self.world_head(Z_prime)
            
            # Reward w/ Team Pooling (Match forward)
            Z_prime_flat = Z_prime.reshape(Batch_Time, num_ships, d_model)
            q_rew = self.team_token_reward.expand(Batch_Time, -1, -1)
            Z_prime_norm_rew = self.norm_reward(Z_prime_flat)
            team_vec_rew, _ = self.pooler_reward(q_rew, Z_prime_norm_rew, Z_prime_norm_rew, key_padding_mask=key_padding_mask)
            reward_components = self.reward_head(team_vec_rew.squeeze(1)).reshape(batch_size, seq_len, self.num_reward_components)
            reward_pred = reward_components.sum(dim=-1, keepdim=True) # (B, T, 1)
            
            # 5. Pairwise Relational Head (Optional)
            if self.use_pairwise:
                # (B, T, N, D) -> (B, T, N, N, 2*D)
                zi = Z_prime.unsqueeze(3).expand(-1, -1, -1, num_ships, -1)
                zj = Z_prime.unsqueeze(2).expand(-1, -1, num_ships, -1, -1)
                z_pair = torch.cat([zi, zj], dim=-1)
                pairwise_pred = self.pairwise_head(z_pair)
            else:
                pairwise_pred = None
            
            return None, new_logprob, entropy, value_pred, None, next_state_pred, reward_pred, pairwise_pred


