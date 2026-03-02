"""Fix A: prev_action tensor in TensorState and env observation."""
import pytest
import torch
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv


@pytest.fixture
def simple_env():
    config = ShipConfig()
    env = TensorEnv(num_envs=4, config=config, device=torch.device("cpu"),
                    max_ships=4, max_episode_steps=100)
    return env


def test_prev_action_in_obs_after_reset(simple_env):
    """After reset, prev_action should be in obs, all zeros, correct shape."""
    env = simple_env
    obs = env.reset(seed=42)

    assert "prev_action" in obs, "prev_action missing from obs dict"
    pa = obs["prev_action"]
    assert pa.shape == (env.num_envs, env.max_ships, 3), \
        f"Expected shape ({env.num_envs}, {env.max_ships}, 3), got {pa.shape}"
    assert pa.dtype == torch.float32
    assert pa.sum() == 0.0, "prev_action should be all zeros after reset"


def test_prev_action_matches_taken_action(simple_env):
    """After step(action), prev_action in next obs matches that action."""
    env = simple_env
    env.reset(seed=0)

    actions = torch.tensor([[1, 3, 1], [0, 2, 0], [2, 6, 1], [1, 1, 0]],
                            dtype=torch.long).unsqueeze(0)  # (1, 4, 3)
    actions = actions.expand(env.num_envs, -1, -1)  # (4, 4, 3)

    obs, *_ = env.step(actions)

    pa = obs["prev_action"]
    expected = actions.float()
    assert torch.allclose(pa, expected), \
        "prev_action does not match the action that was passed to step()"


def test_prev_action_zeroed_on_reset(simple_env):
    """prev_action should be zeroed for envs that reset mid-rollout."""
    env = simple_env
    # Give initial actions
    env.reset(seed=1)
    actions = torch.ones((env.num_envs, env.max_ships, 3), dtype=torch.long)
    env.step(actions)

    # Now manually trigger a reset on env 0 and env 2
    mask = torch.tensor([True, False, True, False])
    env._reset_envs(mask)

    # Check prev_action is zero for reset envs, unchanged for others
    pa = env.state.prev_action
    assert pa[0].sum() == 0.0, "prev_action should be zeroed for reset envs"
    assert pa[2].sum() == 0.0, "prev_action should be zeroed for reset envs"
    # env 1 and 3 should still have the old action
    assert pa[1].sum() == env.max_ships * 3, "Non-reset envs should keep prev_action"
    assert pa[3].sum() == env.max_ships * 3, "Non-reset envs should keep prev_action"


def test_prev_action_in_state(simple_env):
    """TensorState.prev_action field exists and updates correctly."""
    env = simple_env
    env.reset(seed=2)
    actions = torch.full((env.num_envs, env.max_ships, 3), 2, dtype=torch.long)
    env.step(actions)

    assert hasattr(env.state, "prev_action"), "TensorState missing prev_action"
    assert env.state.prev_action.shape == (env.num_envs, env.max_ships, 3)
    assert env.state.prev_action.dtype == torch.float32
    assert (env.state.prev_action == 2.0).all()
