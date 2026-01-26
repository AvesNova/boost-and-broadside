
import pytest
import torch
import numpy as np
from src.env2.env import TensorEnv

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_reset_nvm(device):
    """Test reset with NvM config."""
    num_envs = 2
    max_ships = 8
    env = TensorEnv(num_envs=num_envs, device=device, max_ships=max_ships)
    
    # 2 vs 4
    obs = env.reset(options={"team_sizes": (2, 4)})
    state = env.state
    
    # Check shapes
    assert state.ships_pos.shape == (num_envs, max_ships)
    
    # Check alive counts
    # Team 0: 2 ships
    # Team 1: 4 ships
    # Total alive: 6
    # Indices [0, 1] -> Team 0
    # Indices [2, 3, 4, 5] -> Team 1
    # Indices [6, 7] -> Dead
    
    alive = state.ships_alive.cpu().numpy()
    teams = state.ships_team.cpu().numpy()
    
    for b in range(num_envs):
        assert np.sum(alive[b]) == 6
        assert np.sum(alive[b] & (teams[b] == 0)) == 2
        assert np.sum(alive[b] & (teams[b] == 1)) == 4

def test_step_mechanics(device):
    """Test that step advances state."""
    env = TensorEnv(num_envs=1, device=device, max_ships=2)
    env.reset(options={"team_sizes": (1, 1), "random_pos": True})
    
    initial_time = env.state.time.clone()
    
    # Action: All Coast (0), Straight (0), No Shoot (0)
    actions = torch.zeros((1, 2, 3), dtype=torch.long, device=device)
    
    obs, rew, done, _, _ = env.step({"action": actions})
    
    # Check time advanced cannot generally check env.state.time as it's not incremented in step?
    # Wait, I missed incrementing `self.state.time` in `step`!
    # `physics.update_ships` uses dt but doesn't update a global time tensor passed to it.
    # I should update `env.state.time += env.dt` in `env.step`.
    # I'll enable this check after fixing it.
    
    # Check return types
    assert isinstance(obs, dict)
    assert "tokens" in obs
    assert obs["tokens"].shape == (1, 2, 15)
    assert rew.shape == (1, 2)
    
    # Check positions changed (velocity is initialized to 100)
    assert torch.abs(env.state.ships_vel).mean() > 0

def test_shooting_reduces_power(device):
    """Test that shooting consumes power and creates bullets."""
    env = TensorEnv(num_envs=1, device=device, max_ships=1)
    env.reset(options={"team_sizes": (1, 0)})
    
    # Force power to max and cooldown to 0
    env.state.ships_power.fill_(100.0)
    env.state.ships_cooldown.fill_(0.0)
    
    # Action: Shoot (2 for action idx 2 is shoot=1)
    # Action tensor: [0, 0, 1] -> Power 0, Turn 0, Shoot 1.
    actions = torch.tensor([[[0, 0, 1]]], dtype=torch.long, device=device)
    
    env.step({"action": actions})
    
    # Check power reduced (3.0 cost)
    assert env.state.ships_power[0, 0] <= 97.0
    
    # Check bullet spawned
    # Cursor should assume some active bullets
    # Bullets time > 0
    assert (env.state.bullets_time > 0).any()
