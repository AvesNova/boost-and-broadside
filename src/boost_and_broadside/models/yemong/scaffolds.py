import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

from boost_and_broadside.models.components.layers.utils import RMSNorm, MambaBlock
from boost_and_broadside.models.components.encoders import StateEncoder, ActionEncoder, RelationalEncoder
from boost_and_broadside.models.components.heads import ActorHead, WorldHead, ValueHead
from boost_and_broadside.models.components.team_evaluator import TeamEvaluator 
from boost_and_broadside.models.components.layers.attention import RelationalAttention
from boost_and_broadside.core.constants import StateFeature, STATE_DIM, TARGET_DIM, TOTAL_ACTION_LOGITS


class BaseScaffold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def get_loss(self, *args, **kwargs):
        raise NotImplementedError

class YemongFull(BaseScaffold):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = OmegaConf.create(kwargs)
        super().__init__(config)
        d_model = config.d_model
        
        # Encoders
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model)
        self.relational_encoder = RelationalEncoder(d_model, config.n_layers)
        
        # Action Encoder (Embeddings)
        self.action_encoder = ActionEncoder()
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            RMSNorm(d_model),
            nn.SiLU()
        )
        
        self.special_embeddings = nn.ModuleDict({
            'ship_id': nn.Embedding(8, d_model),
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
            # But here we hardcode defaults for the "Full" version matching MambaBB
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
                inference_params=None, actor_cache=None, world_size=(1024.0, 1024.0)):
        
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
            
            x_spatial = x_mamba.view(batch_size, num_ships, seq_len, -1).permute(0, 2, 1, 3)
            rel_bias = self.relational_encoder.adapters[i+1](trunk_out)
            x_spatial = x_spatial + block['attn'](block['norm2'](x_spatial), rel_bias, mask=alive)
            x_mamba = x_spatial.permute(0, 2, 1, 3).reshape(batch_size * num_ships, seq_len, -1)

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

    def get_loss(self, pred_states, pred_actions, target_states, target_actions, loss_mask, 
                 lambda_state=1.0, lambda_actions=1.0,
                 pred_values=None, pred_rewards=None, target_returns=None, target_rewards=None,
                 lambda_value=1.0, lambda_reward=1.0, weights_power=None, weights_turn=None, weights_shoot=None,
                 target_alive=None, min_sigma=0.1):
        
        target_states = target_states.to(pred_states.dtype)
        if loss_mask.ndim == 2: loss_mask = loss_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
        if target_alive is not None: loss_mask = loss_mask & target_alive
             
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        mse = F.mse_loss(pred_states, target_states, reduction='none')
        
        s_loss = mse.mean(dim=-1).reshape(-1).mul(mask_flat).sum() / denom
        
        l_p, l_t, l_s = pred_actions[..., 0:3], pred_actions[..., 3:10], pred_actions[..., 10:12]
        t_p, t_t, t_s = target_actions[..., 0].long().clamp(0, 2), target_actions[..., 1].long().clamp(0, 6), target_actions[..., 2].long().clamp(0, 1)
        
        a_loss_p = (F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), weight=weights_power, reduction='none') * mask_flat).sum() / denom / math.log(3)
        a_loss_t = (F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), weight=weights_turn, reduction='none') * mask_flat).sum() / denom / math.log(7)
        a_loss_s = (F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), weight=weights_shoot, reduction='none') * mask_flat).sum() / denom / math.log(2)
        
        a_loss = a_loss_p + a_loss_t + a_loss_s

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
             l_actions_w = apply_u(a_loss, "actions")
             l_value_w = apply_u(v_loss, "value")
             l_reward_w = apply_u(r_loss, "reward")
             total_loss = l_state_w + l_actions_w + l_value_w + l_reward_w
        else:
             total_loss = (lambda_state * s_loss) + (lambda_actions * a_loss) + (lambda_value * v_loss) + (lambda_reward * r_loss)

        metrics = {"loss": total_loss.item(), "loss_sub/state_mse": s_loss.item(), "loss_sub/action_all": a_loss.item(), "loss_sub/action_power": a_loss_p.item(), "loss_sub/action_turn": a_loss_t.item(), "loss_sub/action_shoot": a_loss_s.item(), "loss_sub/value_mse": v_loss.item(), "loss_sub/reward_mse": r_loss.item()}
        return total_loss, s_loss, a_loss, torch.tensor(0.0), metrics


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
        
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model)
        self.relational_encoder = RelationalEncoder(d_model, config.n_layers) # n_layers here might range over stack
        
        self.special_embeddings = nn.ModuleDict({
            'ship_id': nn.Embedding(8, d_model),
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

    def get_loss(self, pred_actions, target_actions, loss_mask, 
                 lambda_actions=1.0,
                 weights_power=None, weights_turn=None, weights_shoot=None,
                 **kwargs):
        
        # Expand loss_mask to match pred_actions shape (B, T, N)
        if loss_mask.ndim == 2:
            # loss_mask is (B, T), need to expand to (B, T, N)
            loss_mask = loss_mask.unsqueeze(-1).expand_as(pred_actions[..., 0])
        
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        
        l_p, l_t, l_s = pred_actions[..., 0:3], pred_actions[..., 3:10], pred_actions[..., 10:12]
        t_p, t_t, t_s = target_actions[..., 0].long().clamp(0, 2), target_actions[..., 1].long().clamp(0, 6), target_actions[..., 2].long().clamp(0, 1)
        
        a_loss_p = (F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), weight=weights_power, reduction='none') * mask_flat).sum() / denom / math.log(3)
        a_loss_t = (F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), weight=weights_turn, reduction='none') * mask_flat).sum() / denom / math.log(7)
        a_loss_s = (F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), weight=weights_shoot, reduction='none') * mask_flat).sum() / denom / math.log(2)
        
        a_loss = a_loss_p + a_loss_t + a_loss_s
        total_loss = lambda_actions * a_loss
        
        metrics = {"loss": total_loss.item(), "loss_sub/action_all": a_loss.item(), "loss_sub/action_power": a_loss_p.item(), "loss_sub/action_turn": a_loss_t.item(), "loss_sub/action_shoot": a_loss_s.item()}
        return total_loss, torch.tensor(0.0), total_loss, torch.tensor(0.0), metrics


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
        
        self.state_encoder = StateEncoder(config.get("input_dim", STATE_DIM), d_model)
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
        if state.ndim == 4:
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
        
        # Reshape back if needed, but for loss we can keep flattened usually
        if 'num_ships' in kwargs: 
             # ... explicit reshape logic if needed by trainer
             pass
             
        return state_pred, None, None, None, x

    def get_loss(self, pred_states, target_states, loss_mask, lambda_state=1.0, **kwargs):
        
        if pred_states.ndim == 3 and target_states.ndim == 4:
            # Flatten target
            B, T, N, D = target_states.shape
            target_states = target_states.reshape(B*N, T, D)
            # loss_mask can be (B, T) or (B, T, N)
            if loss_mask.ndim == 2:
                loss_mask = loss_mask.unsqueeze(2).expand(B, T, N) # (B, T, N)
            loss_mask = loss_mask.permute(0, 2, 1).reshape(B*N, T)
            
        mask_flat = loss_mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        mse = F.mse_loss(pred_states, target_states, reduction='none')
        
        s_loss = mse.mean(dim=-1).reshape(-1).mul(mask_flat).sum() / denom
        
        total_loss = lambda_state * s_loss
        
        metrics = {"loss": total_loss.item(), "loss_sub/state_mse": s_loss.item()}
        return total_loss, s_loss, torch.tensor(0.0), torch.tensor(0.0), metrics

