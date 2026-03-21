import pytest
import torch
import torch.nn.functional as F
import math
from pathlib import Path
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongDynamics
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "model"

def test_instantiate_yemong_dynamics_from_yaml():
    yaml_path = CONFIG_DIR / "yemong_dynamics.yaml"
    assert yaml_path.exists(), f"Config file not found: {yaml_path}"
    
    model_config = OmegaConf.load(yaml_path)
    # Wrap in root to resolve ${model.d_model} interpolations
    root = OmegaConf.create({"model": model_config})
    
    model = YemongDynamics(root.model)
    assert isinstance(model, YemongDynamics)

def test_yemong_dynamics_forward_and_loss():
    d_model = 128
    config = OmegaConf.create({
        "d_model": d_model,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": 12,
        "loss_type": "fixed",
        "spatial_layer": {
            "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
            "d_model": d_model,
            "n_heads": 4
        },
        "loss": {
             "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
             "losses": [
                  {"_target_": "boost_and_broadside.models.components.losses.StateLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ActionLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ValueLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.RewardLoss", "weight": 1.0}
             ]
        }
    })
    model = YemongDynamics(config)
    
    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, STATE_DIM)
    state[..., StateFeature.HEALTH] = 1.0 # Health > 0
    
    # Prev Action now needs to be Indices (Long) for the separate embeddings
    # Shape (B, T, N, 3)
    prev_action = torch.randint(0, 2, (B, T, N, 3))
    
    pos = torch.randn(B, T, N, 2)
    vel = torch.randn(B, T, N, 2)
    att = torch.randn(B, T, N, 2)
    team_ids = torch.randint(0, 2, (B, N))
    seq_idx = torch.zeros(B, T, dtype=torch.long)
    
    # 1. Forward with Teacher Forcing (target_actions provided)
    target_actions = torch.randint(0, 2, (B, T, N, 3))
    
    # returns: action_logits, logprob, entropy, value_pred, mamba_state, next_state_pred, reward_pred, pairwise_pred, reward_components
    action_logits, _, _, value_pred, _, next_state_pred, reward_pred, _, _ = model(
        state=state,
        prev_action=prev_action,
        pos=pos,
        vel=vel,
        att=att,
        team_ids=team_ids,
        seq_idx=seq_idx,
        target_actions=target_actions
    )
    
    assert next_state_pred.shape == (B, T, N, TARGET_DIM)
    assert action_logits.shape == (B, T, N, 12)
    # Value and Reward should be Per-Batch (B, T, 1) due to pooling
    assert value_pred.shape == (B, T, 1) 
    assert reward_pred.shape == (B, T, 1)
    
    # Verify Embeddings Used
    # Check if we can differentiate input vs dynamics embeddings?
    # Hard to check without gradient hook, but shape checks confirm logic runs.
    
    # 2. Forward with Sampling (no target_actions)
    # This exercises the sampling logic
    action_logits_s, _, _, _, _, next_state_pred_s, _, _, _ = model(
        state=state,
        prev_action=prev_action,
        pos=pos,
        vel=vel,
        att=att,
        team_ids=team_ids,
        seq_idx=seq_idx,
        target_actions=None
    )
    assert next_state_pred_s.shape == (B, T, N, TARGET_DIM)
    
    # 3. Loss
    target_states = torch.randn(B, T, N, TARGET_DIM) # Ground truth Next State
    loss_mask = torch.ones(B, T, N)
    target_returns = torch.randn(B, T, 1) # Team level targets
    target_rewards = torch.randn(B, T, 1) # Team level targets
    
    metrics = model.get_loss(
        pred_states=next_state_pred,
        pred_actions=action_logits,
        target_states=target_states,
        target_actions=target_actions,
        loss_mask=loss_mask,
        pred_values=value_pred,
        pred_rewards=reward_pred,
        target_returns=target_returns,
        target_rewards=target_rewards,
        lambda_state=1.0,
        lambda_actions=1.0,
        lambda_value=1.0,
        lambda_reward=1.0
    )
    
    loss = metrics["loss"]
    assert "action_loss" in metrics
    assert "state_loss" in metrics
    assert "value_loss" in metrics
    assert "reward_loss" in metrics
    assert "loss_sub/action_power" in metrics
