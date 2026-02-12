import pytest
import torch
import math
from pathlib import Path
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongFull, YemongSpatial, YemongTemporal
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature

# Get path to configs relative to this test file
CONFIG_DIR = Path(__file__).parent.parent / "configs" / "model"

@pytest.mark.parametrize("config_name, expected_class", [
    ("yemong_full.yaml", YemongFull),
    ("yemong_spatial.yaml", YemongSpatial),
    ("yemong_temporal.yaml", YemongTemporal),
])
def test_instantiate_from_yaml(config_name, expected_class):
    yaml_path = CONFIG_DIR / config_name
    assert yaml_path.exists(), f"Config file not found: {yaml_path}"
    
    model_config = OmegaConf.load(yaml_path)
    # Wrap in root to resolve ${model.d_model} interpolations
    root = OmegaConf.create({"model": model_config})
    
    model = expected_class(root.model)
    assert isinstance(model, expected_class)

@pytest.mark.parametrize("scaffold_class", [YemongFull, YemongSpatial, YemongTemporal])
def test_scaffold_instantiation(scaffold_class):
    config = OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
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
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": 12,
        "loss_type": "fixed"
    })
    model = YemongFull(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, STATE_DIM) * 0.1
    state[..., StateFeature.HEALTH] = 1.0 # Health > 0
    prev_action = torch.zeros(B, T, N, 12)
    pos = torch.randn(B, T, N, 2) * 0.1
    vel = torch.randn(B, T, N, 2) * 0.1
    att = torch.randn(B, T, N, 2) * 0.1
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
    
    # Debug NaNs
    if torch.isnan(state_pred).any(): print("DEBUG: state_pred has NaNs")
    if torch.isnan(action_logits).any(): print("DEBUG: action_logits has NaNs")
    if torch.isnan(value_pred).any(): print("DEBUG: value_pred has NaNs")
    
    assert state_pred.shape == (B, T, N, TARGET_DIM)
    assert action_logits.shape == (B, T, N, 12)
    assert value_pred.shape == (B, T, 1)
    assert reward_pred.shape == (B, T, 1)
    
    # Loss
    target_states = torch.randn(B, T, N, TARGET_DIM) # Ground truth deltas
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
        "input_dim": STATE_DIM,
        "action_dim": 12
    })
    model = YemongSpatial(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, STATE_DIM)
    state[..., StateFeature.HEALTH] = 1.0
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
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM
    })
    model = YemongTemporal(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, STATE_DIM)
    state[..., StateFeature.HEALTH] = 1.0
    prev_action = torch.zeros(B, T, N, 12)
    
    # Forward
    state_pred, _, _, _, _ = model(
        state=state,
        prev_action=prev_action
    )
    
    # Temporal flattens internally if state is 4D
    assert state_pred.shape == (B * N, T, TARGET_DIM)
    
    # Loss
    target_states = torch.randn(B, T, N, TARGET_DIM)
    loss_mask = torch.ones(B, T, N)
    
    loss, s_loss, _, _, metrics = model.get_loss(
        pred_states=state_pred,
        target_states=target_states,
        loss_mask=loss_mask,
        lambda_state=1.0
    )
    
    assert loss > 0
    assert "loss_sub/state_mse" in metrics
