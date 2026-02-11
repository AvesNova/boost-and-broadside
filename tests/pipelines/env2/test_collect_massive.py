
import pytest
import h5py
from pathlib import Path

from boost_and_broadside.env2.collect_massive import run_collection, CollectionArgs

def test_collect_massive_pipeline(tmp_path):
    """
    Verify that collect_massive runs and produces a valid HDF5 file.
    """
    output_dir = tmp_path / "data_collect"
    
    # Run Collection
    args = CollectionArgs(
        num_envs=2,
        total_steps=500,
        output_dir=str(output_dir),
        seed=42,
        device="cpu",
        min_skill=0.1,
        max_skill=1.0,
        expert_ratio=0.5,
        random_dist="beta"
    )
    
    run_collection(args)
    
    # Verify Output
    h5_path = output_dir / "aggregated_data.h5"
    assert h5_path.exists(), "HDF5 file was not created"
    
    with h5py.File(h5_path, "r") as f:
        # Check Dataset existence (Granular vs Tokens)
        assert "position" in f
        assert "velocity" in f
        assert "health" in f
        assert "power" in f
        assert "actions" in f
        assert "rewards" in f
        assert "returns" in f
        assert "expert_actions" in f
        assert "agent_skills" in f
        
        # Check shapes
        # Total transitions = num_envs * total_steps = 2 * 500 = 1000
        # Since we only save finished episodes, we might miss the last partial episode.
        # Expect at least coverage.
        
        assert f["position"].shape[0] > 0, f"Collected too few samples: {f['position'].shape[0]}"
        assert f["actions"].shape[0] > 0, f"Collected too few actions: {f['actions'].shape[0]}"
        assert f["agent_skills"].shape[0] > 0, "No skills recorded"

if __name__ == "__main__":
    pytest.main([__file__])
