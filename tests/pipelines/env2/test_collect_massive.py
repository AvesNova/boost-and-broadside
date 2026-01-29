import pytest
import shutil
from pathlib import Path
import h5py
import torch
import os

from env2.collect_massive import run_collection

class MockArgs:
    def __init__(self, output_dir, device="cpu"):
        self.num_envs = 2
        self.total_steps = 500
        self.output_dir = output_dir
        self.seed = 42
        self.device = device

def test_collect_massive_pipeline(tmp_path):
    """
    Verify that collect_massive runs and produces a valid HDF5 file.
    """
    output_dir = tmp_path / "data_collect"
    
    # Run Collection
    args = MockArgs(str(output_dir), device="cpu")
    run_collection(args)
    
    # Verify Output
    h5_path = output_dir / "aggregated_data.h5"
    assert h5_path.exists(), "HDF5 file was not created"
    
    with h5py.File(h5_path, "r") as f:
        # Check Dataset existence
        assert "tokens" in f
        assert "actions" in f
        assert "rewards" in f
        assert "returns" in f
        
        # Check shapes
        # Total transitions = num_envs * total_steps = 2 * 500 = 1000
        # Since we only save finished episodes, we might miss the last partial episode.
        # Expect at least 80% coverage.
        total_transitions = args.num_envs * args.total_steps
        
        assert f["tokens"].shape[0] >= total_transitions * 0.8
        assert f["actions"].shape[0] >= total_transitions * 0.8
        
        # Check attributes if any (AsyncCollector typically saves metadata?)
        pass

if __name__ == "__main__":
    pytest.main([__file__])
