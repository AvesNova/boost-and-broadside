
import pytest
import torch
import math
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongFull, YemongSpatial, YemongTemporal

@pytest.mark.parametrize("scaffold_class", [YemongFull, YemongSpatial, YemongTemporal])
def test_scaffold_instantiation(scaffold_class):
    config = OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": 9,
        "target_dim": 9,
        "action_dim": 12,
        "loss_type": "fixed",
        "spatial_layer": {
            "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
            "d_model": 128,
            "n_heads": 4
        }
    })
    model = scaffold_class(config)
    assert isinstance(model, scaffold_class)

def test_yemong_full_forward_and_loss():
    d_model = 128
    config = OmegaConf.create({
        "d_model": d_model,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": 9,
        "target_dim": 9,
        "action_dim": 12,
        "loss_type": "fixed"
    })
    model = YemongFull(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, 9)
    prev_action = torch.zeros(B, T, N, 12)
    pos = torch.randn(B, T, N, 2)
    vel = torch.randn(B, T, N, 2)
    att = torch.randn(B, T, N, 2)
    team_ids = torch.randint(0, 2, (B, N))
    seq_idx = torch.zeros(B, T, dtype=torch.long)
    
    # Forward
    state_pred, action_logits, value_pred, reward_pred, latent = model(
        state=state,
        prev_action=prev_action,
        pos=pos,
        vel=vel,
        att=att,
        team_ids=team_ids,
        seq_idx=seq_idx
    )
    
    assert state_pred.shape == (B, T, N, 9)
    assert action_logits.shape == (B, T, N, 12)
    assert value_pred.shape == (B, T, 1)
    assert reward_pred.shape == (B, T, 1)
    
    # Loss
    target_states = torch.randn(B, T, N, 9)
    target_actions = torch.randint(0, 2, (B, T, N, 3))
    loss_mask = torch.ones(B, T, N)
    
    loss, s_loss, a_loss, r_loss, metrics = model.get_loss(
        pred_states=state_pred,
        pred_actions=action_logits,
        target_states=target_states,
        target_actions=target_actions,
        loss_mask=loss_mask,
        lambda_state=1.0,
        lambda_actions=1.0
    )
    
    assert loss > 0
    assert "loss_sub/action_all" in metrics

def test_yemong_spatial_forward_and_loss():
    config = OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": 9,
        "action_dim": 12
    })
    model = YemongSpatial(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, 9)
    pos = torch.randn(B, T, N, 2)
    vel = torch.randn(B, T, N, 2)
    att = torch.randn(B, T, N, 2)
    
    # Forward (with time dim)
    _, action_logits, _, _, _ = model(
        state=state,
        pos=pos,
        vel=vel,
        att=att
    )
    
    assert action_logits.shape == (B, T, N, 12)
    
    # Loss
    target_actions = torch.randint(0, 2, (B, T, N, 3))
    loss_mask = torch.ones(B, T, N)
    
    loss, _, _, _, metrics = model.get_loss(
        pred_actions=action_logits,
        target_actions=target_actions,
        loss_mask=loss_mask,
        lambda_actions=1.0
    )
    
    assert loss > 0
    assert "loss_sub/action_all" in metrics

def test_yemong_temporal_forward_and_loss():
    config = OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "input_dim": 9,
        "target_dim": 9
    })
    model = YemongTemporal(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, 9)
    prev_action = torch.zeros(B, T, N, 12)
    
    # Forward
    state_pred, _, _, _, _ = model(
        state=state,
        prev_action=prev_action
    )
    
    # Temporal flattens internally if state is 4D
    assert state_pred.shape == (B*N, T, 9)
    
    # Loss
    target_states = torch.randn(B, T, N, 9)
    loss_mask = torch.ones(B, T, N)
    
    loss, s_loss, _, _, metrics = model.get_loss(
        pred_states=state_pred,
        target_states=target_states,
        loss_mask=loss_mask,
        lambda_state=1.0
    )
    
    assert loss > 0
    assert "loss_sub/state_mse" in metrics
