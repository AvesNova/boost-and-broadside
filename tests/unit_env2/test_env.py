
import pytest
import torch
import numpy as np
from src.env2.env import TensorEnv
from src.env2.state import ShipConfig

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_reset_nvm(device):
    """Test reset with NvM config."""
    num_envs = 2
    max_ships = 8
    config = ShipConfig()
    env = TensorEnv(num_envs=num_envs, config=config, device=device, max_ships=max_ships)
    
    # 2 vs 4
    obs = env.reset(options={"team_sizes": (2, 4)})
    state = env.state
    
    # Check shapes
    assert state.ship_pos.shape == (num_envs, max_ships)
    
    # Check alive counts
    # Team 0: 2 ships
    # Team 1: 4 ships
    # Total alive: 6
    # Indices [0, 1] -> Team 0
    # Indices [2, 3, 4, 5] -> Team 1
    # Indices [6, 7] -> Dead
    
    alive = state.ship_alive.cpu().numpy()
    teams = state.ship_team_id.cpu().numpy()
    
    for b in range(num_envs):
        assert np.sum(alive[b]) == 6
        assert np.sum(alive[b] & (teams[b] == 0)) == 2
        assert np.sum(alive[b] & (teams[b] == 1)) == 4

def test_step_mechanics(device):
    """Test that step advances state."""
    config = ShipConfig()
    env = TensorEnv(num_envs=1, config=config, device=device, max_ships=2)
    env.reset(options={"team_sizes": (1, 1), "random_pos": True})
    
    initial_step = env.state.step_count.clone()
    
    # Action: All Coast (0), Straight (0), No Shoot (0)
    actions = torch.zeros((1, 2, 3), dtype=torch.long, device=device)
    
    obs, rew, done, _, _ = env.step(actions)
    
    # Check step count advanced
    assert (env.state.step_count > initial_step).all()
    
    # Check return types
    assert isinstance(obs, dict)
    assert "position" in obs
    assert obs["position"].shape == (1, 2)
    assert rew.shape == (1, 2)
    
    # Check positions changed (velocity is initialized to 100)
    assert torch.abs(env.state.ship_vel).mean() > 0

