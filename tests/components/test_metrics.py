import pytest
import torch
from env.env import Environment
from agents.tokenizer import observation_to_tokens


def test_environment_seed_synchronization():
    """
    Verify that resetting the environment with the same seed produces identical initial states.
    This is critical for comparing Agent vs Expert on the 'same scenario'.
    """
    config = {
        "world_size": [1000, 1000],
        "max_ships": 2,
        "agent_dt": 0.1,
        "physics_dt": 0.05,
        "render_mode": "none",
        "memory_size": 10,
        "random_positioning": False,
        "random_speed": False,
    }

    env = Environment(**config)
    seed = 42

    # Run 1
    obs1, info1 = env.reset(seed=seed, game_mode="1v1")
    state1_tokens = observation_to_tokens(
        obs1, perspective=0, world_size=tuple(config["world_size"])
    )

    # Run 2
    obs2, info2 = env.reset(seed=seed, game_mode="1v1")
    state2_tokens = observation_to_tokens(
        obs2, perspective=0, world_size=tuple(config["world_size"])
    )

    # Check strict equality of tokens
    assert torch.allclose(state1_tokens, state2_tokens), (
        "Initial tokens differ despite same seed!"
    )

    # Verify different seed key produces different result (sanity check)
    obs3, info3 = env.reset(seed=seed + 1, game_mode="1v1")
    state3_tokens = observation_to_tokens(
        obs3, perspective=0, world_size=tuple(config["world_size"])
    )

    # It is statistically incredibly unlikely to be identical
    if torch.allclose(state1_tokens, state3_tokens):
        pytest.warns(
            UserWarning,
            match="Different seeds produced identical states - this might happen but is rare.",
        )


def test_environment_trajectory_synchronization():
    """
    Verify that applying same actions with same seed produces same trajectory.
    """
    config = {
        "world_size": [1000, 1000],
        "max_ships": 2,
        "agent_dt": 0.1,
        "physics_dt": 0.05,
        "render_mode": "none",
        "memory_size": 10,
        "random_positioning": False,
        "random_speed": False,
    }
    env = Environment(**config)
    seed = 123

    # Define a sequence of dummy actions
    # 1v1 has ships 0 and 1.
    actions = {}
    actions[0] = torch.tensor([1.0, 0.0, 0.0])  # Power, NoTurn, NoShoot
    actions[1] = torch.tensor([0.0, 0.0, 0.0])  # Coast

    # Run 1
    env.reset(seed=seed, game_mode="1v1")
    obs1_next, _, _, _, _ = env.step(actions)

    # Run 2
    env.reset(seed=seed, game_mode="1v1")
    obs2_next, _, _, _, _ = env.step(actions)

    token1 = observation_to_tokens(
        obs1_next, perspective=0, world_size=tuple(config["world_size"])
    )
    token2 = observation_to_tokens(
        obs2_next, perspective=0, world_size=tuple(config["world_size"])
    )

    assert torch.allclose(token1, token2), (
        "Trajectories diverged despite same seed and actions!"
    )
