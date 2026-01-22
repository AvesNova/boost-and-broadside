
import os
import h5py
import numpy as np
import torch
import pytest
from utils.dataset_stats import calculate_action_stats

@pytest.fixture
def mock_h5_file(tmp_path):
    d = tmp_path / "mock_data.h5"
    
    # Create imbalance:
    # Power: 0: 100, 1: 10, 2: 1
    # Turn: 0: 100, others: 1
    # Shoot: 0: 100, 1: 1
    
    actions = []
    masks = []
    
    # Add samples
    for _ in range(100):
        actions.append([0, 0, 0])
        masks.append([True]) # Just True for the ship
    for _ in range(10):
        actions.append([1, 1, 0])
        masks.append([True])
    for _ in range(1):
        actions.append([2, 2, 1])
        masks.append([True])
    
    # Flatten to make it simple logic for (N, 3) 
    actions = np.array(actions, dtype=np.float32) # (111, 3)
    # The code expects (N, S, 3) or flat.
    # Let's give it (N, 1, 3)
    actions = actions.reshape(111, 1, 3)
    
    # Masks (N, 1)
    masks = np.array(masks, dtype=bool).reshape(111, 1)

    with h5py.File(d, "w") as f:
        f.create_dataset("actions", data=actions)
        f.create_dataset("action_masks", data=masks)
        
    return str(d)

def test_calculate_action_stats(mock_h5_file):
    weights = calculate_action_stats(mock_h5_file, batch_size=10)
    
    w_p = weights["power"]
    w_t = weights["turn"]
    w_s = weights["shoot"]
    
    # Validate Output Types
    assert isinstance(w_p, torch.Tensor)
    assert len(w_p) == 3
    
    # Validate Logic: Less frequent class should have higher weight
    # Power 2 has 1 sample, Power 0 has 100 samples.
    # Old Weight ~ 1/Count. New Weight ~ 1/sqrt(Count).
    # w[2] should be > w[0]. 
    assert w_p[2] > w_p[0]
    
    # Check specific sqrt relationship roughly
    # Count 1 vs Count 100 -> Weight ratio should be approx sqrt(100)/sqrt(1) = 10
    # Allow some margin for epsilon and n_classes scaling
    ratio = w_p[2] / w_p[0]
    # 100 / 1 in counts -> sqrt(100)/sqrt(1) = 10 in weights
    assert torch.abs(ratio - 10.0) < 1.0
    
    # Check caching key
    with h5py.File(mock_h5_file, "r") as f:
        assert "action_weights_sqrt" in f.attrs
