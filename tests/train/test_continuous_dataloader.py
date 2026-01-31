
import pytest
import h5py
import numpy as np
import torch
import os
from pathlib import Path

from train.unified_dataset import UnifiedEpisodeDataset
from train.continuous_view import ContinuousView
from train.data_loader import create_continuous_data_loader

@pytest.fixture
def dummy_h5_data(tmp_path):
    """Create a dummy HDF5 file with known episode structure."""
    file_path = tmp_path / "test_data.h5"
    
    # 2 Episodes:
    # Ep 1: 10 steps
    # Ep 2: 20 steps
    # Total: 30 steps
    
    ep1_len = 10
    ep2_len = 20
    
    # Tokens (D=16)
    tokens_ep1 = np.ones((ep1_len, 16), dtype=np.float32) * 1.0
    tokens_ep2 = np.ones((ep2_len, 16), dtype=np.float32) * 2.0
    tokens = np.concatenate([tokens_ep1, tokens_ep2], axis=0) # (30, 16)
    
    # Actions (D=3)
    actions = np.zeros((30, 3), dtype=np.int32)
    
    # Episode IDs
    # Ep1 -> ID=1
    # Ep2 -> ID=2
    ep_ids = np.concatenate([
        np.full((ep1_len,), 1, dtype=np.int32),
        np.full((ep2_len,), 2, dtype=np.int32)
    ])
    
    episode_lengths = np.array([ep1_len, ep2_len], dtype=np.int32)
    
    with h5py.File(file_path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("actions", data=actions)
        f.create_dataset("episode_ids", data=ep_ids)
        f.create_dataset("episode_lengths", data=episode_lengths)
        f.attrs["token_dim"] = 16
        
    return str(file_path)

def test_unified_dataset_loading(dummy_h5_data):
    dataset = UnifiedEpisodeDataset(dummy_h5_data)
    
    assert dataset.total_timesteps == 30
    assert len(dataset.episode_lengths) == 2
    assert dataset.episode_starts[0] == 0
    assert dataset.episode_starts[1] == 10

def test_cross_episode_slice(dummy_h5_data):
    dataset = UnifiedEpisodeDataset(dummy_h5_data)
    
    # Slice across boundary: 8 to 12 (Ep1[8:10] + Ep2[0:2])
    # Ep1 vals = 1.0, Ep2 vals = 2.0
    slice_data = dataset.get_cross_episode_slice("tokens", 8, 4)
    
    assert slice_data.shape == (4, 16)
    # Check values
    assert torch.all(slice_data[0] == 1.0)
    assert torch.all(slice_data[1] == 1.0)
    assert torch.all(slice_data[2] == 2.0) # Boundary crossed
    assert torch.all(slice_data[3] == 2.0)

def test_continuous_view_masks(dummy_h5_data):
    dataset = UnifiedEpisodeDataset(dummy_h5_data)
    
    # Create view with sequence length 10
    # Sample starting at index 5 (Mid Ep1 to Mid Ep2)
    # Indices: 5..14
    # Boundary at index 10 (which is 5th element in sequence 5,6,7,8,9, | 10,11,12)
    # Actually: 5,6,7,8,9 (Ep1), 10,11,12,13,14 (Ep2)
    # Ep IDs: 1,1,1,1,1, 2,2,2,2,2
    
    indices = [5]
    view = ContinuousView(dataset, indices, seq_len=10)
    
    batch = view[0]
    
    # Check Shapes
    assert batch["states"].shape == (10, 16)
    assert batch["seq_idx"].shape == (10,)
    assert batch["reset_mask"].shape == (10,)
    
    # Check Seq Ids
    expected_ids = torch.tensor([1,1,1,1,1, 2,2,2,2,2], dtype=torch.int32)
    assert torch.all(batch["seq_idx"] == expected_ids)
    
    # Check Reset Mask
    # Previous ID of index 5 is index 4 (ID=1).
    # So index 5 (start of seq) matches prev. No reset at t=0 of seq.
    # Boundary at 10. Prev(10)=9 (ID=1). Curr(10)=10 (ID=2).
    # Reset should be True at the index where ID changes.
    # In the SEQUENCE (0-9):
    # Seq[0]=Ind[5] (ID=1). Prev=ID=1. Diff=F.
    # ...
    # Seq[4]=Ind[9] (ID=1). Prev=ID=1. Diff=F.
    # Seq[5]=Ind[10] (ID=2). Prev=ID=1. Diff=T.
    
    expected_mask = torch.tensor([False, False, False, False, False, True, False, False, False, False])
    assert torch.all(batch["reset_mask"] == expected_mask)

def test_data_loader_pipeline(dummy_h5_data):
    # Test creation of loader
    loader, _ = create_continuous_data_loader(
        dummy_h5_data,
        batch_size=2,
        seq_len=5,
        validation_split=0.0,
        num_workers=0 
    )
    
    # Run one batch
    batch = next(iter(loader))
    assert "states" in batch
    assert "reset_mask" in batch
    assert batch["states"].shape == (2, 5, 16)
