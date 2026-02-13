
import pytest
import torch
from pathlib import Path
import h5py
import numpy as np

from boost_and_broadside.env2.collector import AsyncCollector
from boost_and_broadside.train.unified_dataset import UnifiedEpisodeDataset
from boost_and_broadside.core.constants import STATE_DIM, StateFeature
# Legacy normalization constants removed, use hardcoded 100.0/180.0
NORM_HEALTH = 100.0
NORM_VELOCITY = 180.0

@pytest.fixture
def temp_h5_path(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d / "test_data.h5")

def test_granular_storage_cycle(temp_h5_path):
    """
    Test that we can write granular data and read it back as tokens.
    """
    num_envs = 2
    max_ships = 4
    device = torch.device("cpu") # Test on CPU for simplicity
    
    # 1. Create Collector
    collector = AsyncCollector(temp_h5_path, num_envs, max_ships, device, save_interval=1)
    
    # 2. Simulate Data
    # Create fake obs dict
    
    # Generate random data
    # Fix shapes: Pos/Vel/Att are (N, M) complex
    obs = {
        "position": torch.randn(num_envs, max_ships, dtype=torch.complex64),
        "velocity": torch.randn(num_envs, max_ships, dtype=torch.complex64),
        "health": torch.rand(num_envs, max_ships),
        "power": torch.rand(num_envs, max_ships),
        "attitude": torch.randn(num_envs, max_ships, dtype=torch.complex64), 
        "ang_vel": torch.randn(num_envs, max_ships),
        "is_shooting": torch.randint(0, 2, (num_envs, max_ships), dtype=torch.bool),
        "team_id": torch.randint(0, 2, (num_envs, max_ships), dtype=torch.int32)
    }
    
    actions = torch.randint(0, 3, (num_envs, max_ships, 3))
    rewards = torch.randn(num_envs, max_ships)
    dones = torch.zeros(num_envs, dtype=torch.bool)
    
    # Step 1: Normal step
    collector.step(obs, actions, rewards, dones)
    
    # Step 2: Terminate one env to trigger write
    dones[0] = True
    # We update obs for next step (mock)
    collector.step(obs, actions, rewards, dones)
    
    # Flush
    collector.close()
    
    # 3. Verify File Structure
    with h5py.File(temp_h5_path, "r") as f:
        assert "tokens" not in f
        assert "position" in f
        assert "velocity" in f
        assert f["position"].dtype == np.float32
        assert f["velocity"].dtype == np.float16
        
        # Check lengths
        ep_lens = f["episode_lengths"][:]
        assert len(ep_lens) > 0
        total_steps = f["position"].shape[0]
        assert total_steps == ep_lens.sum()
        
    # 4. Verify Loading (UnifiedEpisodeDataset)
    ds = UnifiedEpisodeDataset(temp_h5_path)
    
    # Check total timesteps
    assert ds.total_timesteps == total_steps
    
    # Request "tokens"
    # UnifiedEpisodeDataset.get_slice("tokens", start, end)
    tokens = ds.get_slice("tokens", 0, total_steps)
    
    assert tokens.shape == (total_steps, max_ships, STATE_DIM)
    assert tokens.dtype == torch.bfloat16
    
    # Verify content reconstruction
    # Let's check Health
    # We wrote from 'obs' which was buffered. Use the data in file to compare exact values (modulo precision)
    
    with h5py.File(temp_h5_path, "r") as f:
        file_health = torch.from_numpy(f["health"][:]).float()
        token_health = tokens[..., StateFeature.HEALTH].float()
        
        # f16 vs f32 tolerance
        # f16 vs f32 tolerance
        # Tokens are now RAW health
        assert torch.allclose(file_health, token_health, atol=1e-2)
        
        # Check Velocity 
        # New Layout
        file_vel = torch.from_numpy(f["velocity"][:]).float()
        token_vel_x = tokens[..., StateFeature.VX].float()
        token_vel_y = tokens[..., StateFeature.VY].float()
        
        # Verify raw values
        assert torch.allclose(file_vel[..., 0], token_vel_x, atol=1e-2)
        assert torch.allclose(file_vel[..., 1], token_vel_y, atol=1e-2)

    print("Test Passed!")

if __name__ == "__main__":
    # If run directly
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        test_granular_storage_cycle(p)
