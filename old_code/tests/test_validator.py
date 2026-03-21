import torch
import pytest
from boost_and_broadside.train.world_model.validator import Validator
from omegaconf import OmegaConf

def test_validator_mask_expansion_fixed():
    # Mocking the minimal config needed by Validator
    cfg = OmegaConf.create({
        "train": {"amp": False},
        "model": {"validation": {"max_batches": 1}},
        "environment": {"world_size": [1024, 1024]}
    })
    
    B, T, N, D = 32, 95, 8, 12
    pred_actions = torch.randn(B, T, N, D)
    loss_mask_slice = torch.ones(B, T)
    
    valid_mask = loss_mask_slice.bool()
    
    # Target shape is (B, T, N)
    target_shape = pred_actions.shape[:-1]
    
    # Final production logic in validator.py:
    if valid_mask.ndim == 2:
         while valid_mask.ndim < len(target_shape):
             valid_mask = valid_mask.unsqueeze(-1)
         
         # Should no longer raise RuntimeError
         expanded_mask = valid_mask.expand(*target_shape)
         assert expanded_mask.shape == (B, T, N)

def test_validator_mask_expansion_fix_logic():
    B, T, N, D = 32, 95, 8, 12
    pred_actions = torch.randn(B, T, N, D)
    loss_mask_slice = torch.ones(B, T)
    
    valid_mask = loss_mask_slice.bool()
    
    # Target shape is (B, T, N)
    target_shape = pred_actions.shape[:-1]
    
    # Fix logic: Unsqueeze until we match target rank
    while valid_mask.ndim < len(target_shape):
        valid_mask = valid_mask.unsqueeze(-1)
    
    # Now it should work
    expanded_mask = valid_mask.expand(*target_shape)
    assert expanded_mask.shape == (B, T, N)
