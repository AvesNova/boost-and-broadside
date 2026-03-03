import pytest
import torch
import math
from boost_and_broadside.models.components.encoders import FlattenedActionEncoder
from boost_and_broadside.models.components.losses import FlattenedActionLoss
from boost_and_broadside.core.constants import NUM_FLATTENED_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS

def test_flattened_action_encoder():
    embed_dim = 16
    encoder = FlattenedActionEncoder(embed_dim=embed_dim)
    
    # Test 3D input (B, T, N, 3)
    # [Power=1, Turn=2, Shoot=1]
    # Expected index = 1 * (7 * 2) + 2 * 2 + 1 = 14 + 4 + 1 = 19
    action_3d = torch.tensor([[[[1, 2, 1]]]], dtype=torch.long)
    emb_3d = encoder(action_3d)
    
    assert emb_3d.shape == (1, 1, 1, embed_dim * 3)
    
    # Test 1D flat input
    action_1d = torch.tensor([[[[19]]]], dtype=torch.long)
    emb_1d = encoder(action_1d)
    
    assert torch.allclose(emb_3d, emb_1d)

def test_flattened_action_loss():
    loss_module = FlattenedActionLoss(weight=1.0)
    
    batch_size = 2
    seq_len = 3
    num_ships = 4
    
    # Mock predictions: (B, T, N, 42)
    pred_actions = torch.randn(batch_size, seq_len, num_ships, NUM_FLATTENED_ACTIONS)
    
    # Mock targets: (B, T, N, 3)
    target_actions = torch.zeros(batch_size, seq_len, num_ships, 3, dtype=torch.long)
    target_actions[..., 0] = 1 # Power (0-2)
    target_actions[..., 1] = 3 # Turn (0-6)
    target_actions[..., 2] = 0 # Shoot (0-1)
    
    mask = torch.ones(batch_size, seq_len, num_ships)
    
    preds = {"actions": pred_actions}
    targets = {"actions": target_actions}
    
    weights_flat = torch.ones(NUM_FLATTENED_ACTIONS)
    
    loss_out = loss_module(preds, targets, mask, weights_flat=weights_flat)
    
    assert "loss" in loss_out
    assert "action_loss" in loss_out
    assert loss_out["loss"] > 0
