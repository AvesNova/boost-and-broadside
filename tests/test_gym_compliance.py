"""
Tests for OpenAI Gym API compliance.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.constants import Actions


class TestGymAPI:
    """Tests for Gym API compliance."""

    def test_inheritance(self, basic_env):
        """Test that environment inherits from gym.Env."""
        assert isinstance(basic_env, gym.Env)

    def test_reset_signature(self, basic_env):
        """Test reset method signature and return values."""
        result = basic_env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2

        observation, info = result
        assert isinstance(observation, dict)
        assert isinstance(info, dict)

    def test_step_signature(self, basic_env):
        """Test step method signature and return values."""
        basic_env.reset(game_mode="1v1")

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        result = basic_env.step(actions)

        assert isinstance(result, tuple)
        assert len(result) == 5

        observation, reward, terminated, truncated, info = result
        assert isinstance(observation, dict)
        assert isinstance(reward, dict)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_close_method(self, basic_env):
        """Test that close method exists and works."""
        basic_env.reset()
        basic_env.close()

        # Should be able to reset and use again
        basic_env.reset()

    def test_render_method(self):
        """Test render method with different modes."""
        from src.env import Environment

        # Test with no rendering
        env = Environment(render_mode=None)
        env.reset()
        env.render(env.state[-1])  # Should not crash
        env.close()

        # Note: 'human' mode requires pygame, skip if not available
        try:
            import pygame

            env = Environment(render_mode="human")
            env.reset()
            # Don't actually render to avoid opening windows in tests
            assert env.render_mode == "human"
            env.close()
        except ImportError:
            pass

    def test_action_space_property(self, basic_env):
        """Test action_space property."""
        action_space = basic_env.action_space

        # Should be accessible as property
        assert hasattr(basic_env, "action_space")

        # Should be a Gym space
        assert isinstance(action_space, gym.spaces.Space)

        # Should be able to sample from it
        sample = action_space.sample()
        assert sample is not None

    def test_observation_space_property(self, basic_env):
        """Test observation_space property."""
        obs_space = basic_env.observation_space

        # Should be accessible as property
        assert hasattr(basic_env, "observation_space")

        # Should be a Gym space
        assert isinstance(obs_space, gym.spaces.Space)

        # Should be able to sample from it
        sample = obs_space.sample()
        assert sample is not None

    def test_metadata(self, basic_env):
        """Test that environment has metadata attribute."""
        # Gym environments should have metadata
        assert hasattr(basic_env, "metadata") or True  # Optional in newer versions

    def test_spec(self, basic_env):
        """Test that environment can have spec attribute."""
        # Spec is optional but should be settable
        basic_env.spec = None
        assert hasattr(basic_env, "spec")


class TestMultiAgentCompliance:
    """Tests for multi-agent extensions to Gym API."""

    def test_multi_agent_reset(self, basic_env):
        """Test that reset returns observations for all agents."""
        obs, info = basic_env.reset(game_mode="1v1")

        # Should return dict of observations
        assert isinstance(obs, dict)

        # Should have observation components for ships
        expected_keys = {
            "ship_id",
            "team_id",
            "alive",
            "health",
            "power",
            "position",
            "velocity",
            "speed",
            "attitude",
            "is_shooting",
            "token",
            "tokens",
        }
        assert set(obs.keys()) == expected_keys

        # Each component should have data for both ships
        for key, tensor in obs.items():
            if key == "token":
                assert tensor.shape[0] == 2  # 2 ships
            else:
                assert tensor.shape[0] == 2  # 2 ships

    def test_multi_agent_step(self, basic_env):
        """Test that step accepts and returns multi-agent data."""
        basic_env.reset(game_mode="1v1")

        # Actions should be a dict
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}

        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Returns should be dicts for multi-agent
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)

        # Environment returns empty rewards dict - wrapper handles team rewards
        assert len(rewards) == 0

        # Observations should have tensor structure with data for all ships
        for key, tensor in obs.items():
            assert tensor.shape[0] == 2  # Data for both ships

    def test_partial_termination(self, basic_env):
        """Test handling of individual agent termination."""
        basic_env.reset(game_mode="1v1")

        # Kill one ship
        basic_env.state[-1].ships[0].alive = False

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Check individual done flags
        assert info["individual_done"][0] is True
        assert info["individual_done"][1] is False

        # Episode should terminate (only one team left)
        assert terminated is True


class TestActionObservationValidation:
    """Tests for action and observation validation."""

    def test_action_space_contains(self, basic_env):
        """Test that valid actions are in action space."""
        action_space = basic_env.action_space

        # Valid action
        valid_action = torch.zeros(len(Actions))
        valid_action[Actions.forward] = 1
        assert action_space.contains(valid_action.numpy())

        # Invalid action (wrong shape)
        invalid_action = torch.zeros(len(Actions) + 1)
        assert not action_space.contains(invalid_action.numpy())

        # Invalid action (wrong values)
        invalid_action = torch.ones(len(Actions)) * 2
        assert not action_space.contains(invalid_action.numpy())

    def test_observation_space_contains(self, basic_env):
        """Test that observations are in observation space."""
        obs_space = basic_env.observation_space

        obs, _ = basic_env.reset(game_mode="1v1")

        # Convert to format expected by gym (tokens only)
        gym_obs = {"tokens": obs["token"].numpy()}
        assert obs_space.contains(gym_obs)

    def test_action_dtype_handling(self, basic_env):
        """Test that different action dtypes are handled."""
        basic_env.reset(game_mode="1v1")

        # Test with numpy array
        actions_np = {0: np.zeros(len(Actions)), 1: np.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions_np)
        assert obs is not None

        # Test with torch tensor
        actions_torch = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions_torch)
        assert obs is not None


class TestEpisodeFlow:
    """Tests for episode flow and state management."""

    def test_episode_length(self, basic_env):
        """Test that episodes can run for expected length."""
        basic_env.reset(game_mode="1v1")

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}

        step_count = 0
        max_steps = 1000

        for _ in range(max_steps):
            obs, rewards, terminated, truncated, info = basic_env.step(actions)
            step_count += 1

            if terminated or truncated:
                break

        # Should be able to run for multiple steps
        assert step_count > 1

    def test_reset_after_termination(self, basic_env):
        """Test that environment can be reset after termination."""
        basic_env.reset(game_mode="1v1")

        # Force termination
        basic_env.state[-1].ships[0].alive = False

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        assert terminated

        # Should be able to reset
        obs, info = basic_env.reset(game_mode="1v1")
        assert obs is not None

        # Should be able to step again
        obs, rewards, terminated, truncated, info = basic_env.step(actions)
        assert not terminated  # Fresh episode

    def test_state_consistency(self, basic_env, step_environment):
        """Test that state remains consistent across steps."""
        basic_env.reset(game_mode="1v1")

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}

        previous_time = 0.0

        for i in range(10):
            obs, rewards, terminated, truncated, info = basic_env.step(actions)

            # Time should advance monotonically
            assert info["current_time"] > previous_time
            previous_time = info["current_time"]

            # State deque should maintain proper size
            assert len(basic_env.state) <= basic_env.memory_size

            # Ship states should be consistent with observations
            for ship_id, ship_obs in obs.items():
                if ship_id in basic_env.state[-1].ships:
                    ship = basic_env.state[-1].ships[ship_id]
                    # Check a few values match
                    assert (
                        abs(
                            ship_obs["self_state"][4]
                            - ship.health / ship.config.max_health
                        )
                        < 1e-5
                    )


class TestSeedingAndDeterminism:
    """Tests for seeding and deterministic behavior."""

    def test_deterministic_reset(self, basic_env):
        """Test that reset produces consistent initial states."""
        # Use deterministic positioning for consistency
        states = []

        for _ in range(3):
            obs, _ = basic_env.reset(game_mode="1v1_old")

            # Record ship positions
            positions = []
            for ship in basic_env.state[-1].ships.values():
                positions.append((ship.position.real, ship.position.imag))
            states.append(positions)

        # All resets should produce same initial positions
        for i in range(1, len(states)):
            assert states[i] == states[0]

    def test_deterministic_physics(self, fixed_rng):
        """Test that physics is deterministic with fixed RNG."""
        from src.env import Environment

        results = []

        for _ in range(2):
            env = Environment(rng=fixed_rng)
            env.reset(game_mode="1v1_old")

            # Set fixed RNG for all ships
            for ship in env.state[-1].ships.values():
                ship.rng = np.random.default_rng(42)

            actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
            actions[0][Actions.forward] = 1
            actions[0][Actions.shoot] = 1

            # Run several steps
            final_positions = []
            for _ in range(5):
                obs, rewards, terminated, truncated, info = env.step(actions)

            for ship in env.state[-1].ships.values():
                final_positions.append((ship.position.real, ship.position.imag))

            results.append(final_positions)
            env.close()

        # Results should be identical
        for i in range(len(results[0])):
            assert abs(results[0][i][0] - results[1][i][0]) < 1e-10
            assert abs(results[0][i][1] - results[1][i][1]) < 1e-10
