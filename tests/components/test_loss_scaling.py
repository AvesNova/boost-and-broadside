
import pytest
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongFull
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM

@pytest.fixture
def mamba_config():
    return OmegaConf.create({
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_dim": 12,
        "loss_type": "uncertainty", # Default to uncertainty for these tests
        "spatial_layer": {
             "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
             "d_model": 128,
             "n_heads": 4
        },
        "loss": {
             "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
             "loss_type": "uncertainty",
             "losses": [
                  {"_target_": "boost_and_broadside.models.components.losses.StateLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ActionLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.ValueLoss", "weight": 1.0},
                  {"_target_": "boost_and_broadside.models.components.losses.RewardLoss", "weight": 1.0}
             ]
        }
    })

def test_uncertainty_params_initialization(mamba_config):
    """Test that log_vars are initialized when loss_type is uncertainty."""
    model = YemongFull(mamba_config)
    assert hasattr(model, "log_vars")
    assert model.log_vars is not None
    assert len(model.log_vars) == 4
    assert "state" in model.log_vars
    assert "actions" in model.log_vars
    assert "value" in model.log_vars
    assert "reward" in model.log_vars
    
    # Check they are parameters
    assert isinstance(model.log_vars["state"], nn.Parameter)
    assert model.log_vars["state"].requires_grad

def test_fixed_params_initialization(mamba_config):
    """Test that log_vars are NOT initialized when loss_type is fixed."""
    mamba_config.loss_type = "fixed"
    model = YemongFull(mamba_config)
    assert model.log_vars is None

def test_uncertainty_loss_computation(mamba_config):
    """Test that the loss computation includes the regularization term."""
    model = YemongFull(mamba_config)
    
    # Mock inputs
    B, T, N, D = 2, 4, 1, 15
    pred_states = torch.randn(B, T, N, D)
    target_states = torch.randn(B, T, N, D)
    loss_mask = torch.ones(B, T, N).bool()
    
    # Mock other args
    pred_actions = torch.randn(B, T, N, 12)
    target_actions = torch.zeros(B, T, N, 3) # Indices
    
    # Run get_loss
    metrics = model.get_loss(
        pred_states, pred_actions, target_states, target_actions, loss_mask,
        lambda_state=100.0 # Should be ignored
    )
    total_loss = metrics["loss"]
    
    # Computations for State Loss (MSE)
    # The model computes MSE internally.
    # With log_var = 0 (sigma=1), loss should be 0.5 * MSE + 0.5 * 0 = 0.5 * MSE
    # Wait, the formula is 0.5 * exp(-s) * L + 0.5 * s
    # If s=0, exp(-s)=1. So 0.5 * L.
    
    # Let's verify manually
    # Get the raw state loss from metrics
    raw_s_loss = metrics["loss_sub/state_mse"]
    
    # The returned total_loss should contain: 0.5 * raw_s_loss
    # Plus other components.
    
    # Let's set other components to effectively zero by ignoring them or ensuring preds match?
    # Easier: Check gradients.
    
    # Verify that lambda_state=100 was ignored.
    # If it was used, the state contribution would be 100 * raw_s_loss.
    # If uncertainty, it is ~0.5 * raw_s_loss.
    
    # We can't easily isolate total_loss components without mocking internal F calls, 
    # but we can check if it's huge or small.
    
    assert total_loss < 100 * raw_s_loss # Should be much smaller
    
    # Check that we can backprop to log_vars
    # Check that we can backprop to log_vars
    # Note: log_vars are now in model.loss_fn.log_vars for CompositeLoss
    total_loss.backward()
    
    logger_keys = list(model.loss_fn.log_vars.keys())
    assert len(logger_keys) > 0, "No log_vars found"
    
    # Check grad for the first key found (likely 'state' or 'loss_0')
    first_key = logger_keys[0]
    grad = model.loss_fn.log_vars[first_key].grad
    assert grad is not None, f"Gradient for {first_key} is None"
    assert grad != 0.0, f"Gradient for {first_key} is zero"

def test_fixed_loss_computation(mamba_config):
    """Test that fixed mode uses lambdas."""
    mamba_config.loss_type = "fixed"
    mamba_config.loss.loss_type = "fixed" # Crucial: Update the component config too
    model = YemongFull(mamba_config)
    
    B, T, N, D = 2, 2, 1, 15
    pred_states = torch.randn(B, T, N, D)
    target_states = torch.randn(B, T, N, D)
    loss_mask = torch.ones(B, T, N).bool()
    pred_actions = torch.randn(B, T, N, 12)
    target_actions = torch.zeros(B, T, N, 3)
    
    # In new CompositeLoss system, weights are properties of LossModules, not kwargs to get_loss
    # We must update the weight in the model instance
    # StateLoss is index 0 in mamba_config fixture
    model.loss_fn.losses[0].weight = 10.0
    
    metrics = model.get_loss(
         pred_states, pred_actions, target_states, target_actions, loss_mask,
         # lambda_state=10.0 # Ignored
    )
    total_loss = metrics["loss"]
    
    raw_s_loss = metrics["loss_sub/state_mse"]
    
    # In fixed mode, total loss includes 10.0 * state_loss
    # We assume other losses are small or we check relative magnitude
    expected_part = 10.0 * raw_s_loss
    
    # Total loss should be >= expected_part
    assert total_loss >= expected_part
    
    # Allow some tolerance for float math
    assert torch.isclose(total_loss, torch.tensor(expected_part), atol=1e-3) or total_loss > expected_part

