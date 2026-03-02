"""Fix D: rewards buffer stores team scalar (B, 1) not per-ship (B, N)."""
import pytest
import torch
from boost_and_broadside.train.rl.buffer import GPUBuffer


@pytest.fixture
def small_buffer():
    return GPUBuffer(
        num_steps=8,
        num_envs=4,
        obs_shapes={"state": (6, 5)},  # (num_ships=6, features=5)
        action_shape=(6, 3),
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu")
    )


def test_rewards_shape_is_team_scalar(small_buffer):
    """Buffer.rewards should have shape (num_steps, num_envs, 1)."""
    buf = small_buffer
    assert buf.rewards.shape == (buf.num_steps, buf.num_envs, 1), \
        f"Expected rewards shape ({buf.num_steps}, {buf.num_envs}, 1), got {buf.rewards.shape}"


def test_add_aggregates_reward_to_scalar(small_buffer):
    """add() should average per-ship rewards to scalar before storing."""
    buf = small_buffer
    
    obs = {"state": torch.zeros(4, 6, 5)}
    action = torch.zeros(4, 6, 3, dtype=torch.float32)
    logprob = torch.zeros(4, 6)
    done = torch.zeros(4)
    value = torch.zeros(4, 6)
    
    # Per-ship reward: different values per ship
    per_ship_reward = torch.arange(24, dtype=torch.float32).reshape(4, 6)  # (B, N)
    expected_mean = per_ship_reward.mean(dim=-1, keepdim=True)  # (B, 1)

    buf.add(obs, action, logprob, per_ship_reward, done, value)

    stored = buf.rewards[0]  # (B, 1)
    assert stored.shape == (4, 1), f"Expected stored reward shape (4, 1), got {stored.shape}"
    assert torch.allclose(stored, expected_mean), \
        f"Stored reward {stored} != mean of per-ship rewards {expected_mean}"


def test_gae_works_with_scalar_reward(small_buffer):
    """compute_gae should work without error when rewards are (T, B, 1)."""
    buf = small_buffer
    
    obs = {"state": torch.zeros(4, 6, 5)}
    action = torch.zeros(4, 6, 3, dtype=torch.float32)
    logprob = torch.zeros(4, 6)
    done = torch.zeros(4)
    value = torch.ones(4, 6)
    reward = torch.ones(4, 6)  # (B, N) — gets aggregated to (B, 1)
    
    for _ in range(8):
        buf.add(obs, action, logprob, reward, done, value)
    
    next_value = torch.ones(4, 6)
    next_done = torch.zeros(4)
    
    # Should not raise
    buf.compute_gae(next_value, next_done)
    
    # advantages and returns should have valid shapes
    assert buf.advantages.shape == (8, 4, 6)
    assert buf.returns.shape == (8, 4, 6)
