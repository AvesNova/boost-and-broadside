import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongFull, YemongSpatial, YemongTemporal, YemongDynamics
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature

@pytest.mark.parametrize("scaffold_cls, config_overrides", [
    (YemongFull, {"loss_type": "fixed", "spatial_layer": {"_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention", "d_model": 128, "n_heads": 4}}),
    (YemongSpatial, {"loss_type": "fixed"}),
    (YemongTemporal, {"loss_type": "fixed"}),
    (YemongDynamics, {"loss_type": "fixed", "action_embed_dim": 16, "spatial_layer": {"_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention", "d_model": 128, "n_heads": 4}})
])
def test_pretraining_loop(scaffold_cls, config_overrides):
    # Setup
    d_model = 128
    base_config = {
        "d_model": d_model,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": 12,
    }
    base_config.update(config_overrides)
    config = OmegaConf.create(base_config)
    
    model = scaffold_cls(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy Data
    B, T, N = 2, 4, 3
    
    # Run 2 Epochs, 8 Steps per epoch, Accumulate every 2 steps
    num_epochs = 2
    steps_per_epoch = 8
    accum_steps = 2
    
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # Inputs
            state = torch.randn(B, T, N, STATE_DIM)
            state[..., StateFeature.HEALTH] = 1.0
            
            prev_action = torch.randint(0, 2, (B, T, N, 3))
            pos = torch.randn(B, T, N, 2)
            vel = torch.randn(B, T, N, 2)
            att = torch.randn(B, T, N, 2)
            team_ids = torch.randint(0, 2, (B, N))
            seq_idx = torch.zeros(B, T, dtype=torch.long)
            
            # Targets
            target_states = torch.randn(B, T, N, TARGET_DIM)
            target_actions = torch.randint(0, 2, (B, T, N, 3))
            loss_mask = torch.ones(B, T, N)
            target_returns = torch.randn(B, T, 1) # scalar team
            target_rewards = torch.randn(B, T, 1) # scalar team
            
            # Forward
            if scaffold_cls == YemongTemporal:
                 out = model(state, prev_action)
            elif scaffold_cls == YemongSpatial:
                 out = model(state=state, pos=pos, vel=vel, att=att)
            elif scaffold_cls == YemongDynamics:
                 out = model(state, prev_action, pos, vel, att, team_ids, seq_idx, target_actions=target_actions)
            else: # YemongFull
                 out = model(state, prev_action, pos, vel, att, team_ids, seq_idx)
                 
            # Unpack output
            pred_states = out[0] if scaffold_cls in [YemongFull, YemongTemporal, YemongDynamics] else None
            pred_actions = out[1] if scaffold_cls in [YemongFull, YemongSpatial, YemongDynamics] else None
            pred_values = out[2] if scaffold_cls in [YemongFull, YemongDynamics] else None
            pred_rewards = out[3] if scaffold_cls in [YemongFull, YemongDynamics] else None
            
            # Loss Call
            if scaffold_cls == YemongDynamics:
                 loss, _, _, _, _ = model.get_loss(
                     pred_states=pred_states,
                     pred_actions=pred_actions,
                     target_states=target_states,
                     target_actions=target_actions,
                     loss_mask=loss_mask,
                     pred_values=pred_values,
                     pred_rewards=pred_rewards,
                     target_returns=target_returns,
                     target_rewards=target_rewards
                 )
            elif scaffold_cls == YemongFull:
                 loss, _, _, _, _ = model.get_loss(
                     pred_states=pred_states,
                     pred_actions=pred_actions,
                     target_states=target_states,
                     target_actions=target_actions,
                     loss_mask=loss_mask,
                     pred_values=pred_values,
                     pred_rewards=pred_rewards,
                     target_returns=target_returns,
                     target_rewards=target_rewards
                 )
            elif scaffold_cls == YemongSpatial:
                 loss, _, _, _, _ = model.get_loss(
                     pred_actions=pred_actions,
                     target_actions=target_actions,
                     loss_mask=loss_mask
                 )
            elif scaffold_cls == YemongTemporal:
                 loss, _, _, _, _ = model.get_loss(
                     pred_states=pred_states,
                     target_states=target_states,
                     loss_mask=loss_mask
                 )
                 
            # Gradient Accumulation
            loss = loss / accum_steps
            loss.backward()
            
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            assert not torch.isnan(loss)
            assert loss.item() > 0
