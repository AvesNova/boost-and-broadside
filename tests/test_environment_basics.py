"""
Tests for basic environment operations and state management.
"""

import pytest
import numpy as np
import torch
from gymnasium import spaces

from src.constants import Actions


class TestEnvironmentInitialization:
    """Tests for environment initialization."""

    def test_basic_initialization(self, basic_env):
        """Test that environment initializes with correct parameters."""
        assert basic_env.world_size == (800, 600)
        assert basic_env.memory_size == 1
        assert basic_env.max_ships == 2
        assert basic_env.agent_dt == 0.02
        assert basic_env.physics_dt == 0.02
        assert basic_env.physics_substeps == 1

    def test_substep_calculation(self, env_with_substeps):
        """Test physics substep calculation."""
        assert env_with_substeps.agent_dt == 0.04
        assert env_with_substeps.physics_dt == 0.02
        assert env_with_substeps.physics_substeps == 2

    def test_invalid_timestep_ratio(self):
        """Test that non-integer timestep ratios fail."""
        from src.env import Environment

        with pytest.raises(
            AssertionError, match="agent_dt must be multiple of physics_dt"
        ):
            Environment(agent_dt=0.03, physics_dt=0.02)  # Not a multiple of 0.02

    def test_state_initialization(self, basic_env):
        """Test that state deque initializes correctly."""
        assert len(basic_env.state) == 0
        assert basic_env.state.maxlen == basic_env.memory_size
        assert basic_env.current_time == 0.0


class TestReset:
    """Tests for environment reset."""

    def test_reset_1v1(self, basic_env):
        """Test 1v1 reset creates correct initial state."""
        obs, info = basic_env.reset(game_mode="1v1_old")

        assert len(basic_env.state) == 1
        state = basic_env.state[0]

        # Check ships were created
        assert len(state.ships) == 2
        assert 0 in state.ships
        assert 1 in state.ships

        # Check initial positions (from one_vs_one_reset)
        ship0 = state.ships[0]
        ship1 = state.ships[1]

        # Ships should be positioned at expected ratios of world size
        assert ship0.position.real == 0.25 * basic_env.world_size[0]
        assert ship0.position.imag == 0.40 * basic_env.world_size[1]
        assert ship1.position.real == 0.75 * basic_env.world_size[0]
        assert ship1.position.imag == 0.60 * basic_env.world_size[1]

        # Check velocities (opposing)
        assert ship0.velocity.real > 0
        assert ship1.velocity.real < 0

    def test_reset_clears_state(self, basic_env):
        """Test that reset clears previous state."""
        # First reset
        basic_env.reset(game_mode="1v1")

        # Step to add more state
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Reset again
        basic_env.reset(game_mode="1v1")

        assert len(basic_env.state) == 1
        assert basic_env.current_time == 0.0

    def test_reset_returns_observations(self, basic_env):
        """Test that reset returns proper observations."""
        obs, info = basic_env.reset(game_mode="1v1")

        assert isinstance(obs, dict)

        # Check new observation structure with ship data as tensors
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

        # Check tensor shapes for 2 ships
        for key in obs:
            if key in ["token", "tokens"]:
                assert obs[key].shape == (2, 10)  # 2 ships, 10 token dimensions
            else:
                assert obs[key].shape == (2, 1)  # 2 ships, 1 dimension each

    def test_invalid_game_mode(self, basic_env):
        """Test that invalid game mode raises error."""
        with pytest.raises(ValueError, match="Unknown game mode"):
            basic_env.reset(game_mode="invalid")


class TestObservations:
    """Tests for observation generation."""

    def test_observation_structure(self, basic_env):
        """Test observation has correct structure."""
        obs, _ = basic_env.reset(game_mode="1v1")

        # Check that all observation components are tensors with correct shapes
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

        # Check tensor types and shapes for max_ships=2
        for key, tensor in obs.items():
            assert isinstance(tensor, torch.Tensor)
            if key in ["token", "tokens"]:
                assert tensor.shape == (basic_env.max_ships, 10)
                assert tensor.dtype == torch.float32
            else:
                assert tensor.shape == (basic_env.max_ships, 1)
                # Check dtype based on expected type
                if key in ["ship_id", "team_id", "alive", "health", "is_shooting"]:
                    assert tensor.dtype in [torch.int64, torch.int32]
                elif key in ["power", "speed"]:
                    assert tensor.dtype == torch.float32
                elif key in ["position", "velocity", "attitude"]:
                    assert tensor.dtype == torch.complex64

    def test_token_normalization(self, basic_env):
        """Test that ship tokens contain properly normalized values."""
        obs, _ = basic_env.reset(game_mode="1v1")

        token_tensor = obs["token"]

        # Test both ships
        for ship_idx in range(basic_env.max_ships):
            token = token_tensor[ship_idx, :]

            # Team ID should be 0 or 1 for 1v1
            assert token[0].item() in [0, 1]

            # Health ratio should be between 0 and 1
            assert 0 <= token[1].item() <= 1

            # Power ratio should be between 0 and 1
            assert 0 <= token[2].item() <= 1

            # Position should be normalized to [0, 1]
            assert 0 <= token[3].item() <= 1  # x position
            assert 0 <= token[4].item() <= 1  # y position

            # Velocity normalized by 180.0
            assert abs(token[5].item()) < 1.0  # vx
            assert abs(token[6].item()) < 1.0  # vy

            # Attitude is unit vector components
            attitude_mag = np.sqrt(token[7].item() ** 2 + token[8].item() ** 2)
            assert abs(attitude_mag - 1.0) < 1e-5

            # is_shooting should be 0 or 1
            assert token[9].item() in [0, 1]

    def test_team_differentiation(self, basic_env):
        """Test that ships have different team IDs in 1v1."""
        obs, _ = basic_env.reset(game_mode="1v1")

        team_ids = obs["team_id"]

        # In 1v1, ships should be on different teams
        assert team_ids[0, 0].item() != team_ids[1, 0].item()

        # Teams should be 0 and 1
        team_set = {team_ids[0, 0].item(), team_ids[1, 0].item()}
        assert team_set == {0, 1}

    def test_empty_observation_before_reset(self, basic_env):
        """Test that observations before reset return empty tensors."""
        # This should fail gracefully or return empty observations
        try:
            obs = basic_env.get_observation()
            # If it succeeds, should return empty tensors
            for key, tensor in obs.items():
                if key == "token":
                    assert torch.all(tensor == 0)
                else:
                    assert torch.all(tensor == 0)
        except IndexError:
            # Expected behavior - no state to observe yet
            pass

    def test_shooting_observations(self, basic_env, step_environment):
        """Test that shooting state is captured in observations."""
        basic_env.reset(game_mode="1v1")

        # Make ship 0 shoot
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1

        # Step to trigger shooting
        obs, _, _, _, _ = basic_env.step(actions)

        # Ship 0 should be marked as shooting
        is_shooting = obs["is_shooting"]
        assert is_shooting[0, 0].item() == 1  # Ship 0 is shooting
        assert is_shooting[1, 0].item() == 0  # Ship 1 is not shooting

    def test_position_observations(self, basic_env):
        """Test that ship positions are captured correctly in observations."""
        obs, _ = basic_env.reset(game_mode="1v1")
        state = basic_env.state[-1]

        positions = obs["position"]

        # Check that positions match ship states
        for ship_id, ship in state.ships.items():
            observed_position = positions[ship_id, 0]
            assert observed_position == ship.position

        # Check that positions are complex numbers
        assert positions.dtype == torch.complex64


class TestWorldWrapping:
    """Tests for toroidal world wrapping."""

    def test_ship_position_wrapping(self, basic_env):
        """Test that ship positions wrap at boundaries."""
        basic_env.reset(game_mode="1v1_old")

        # Move ship 0 past right boundary
        ship0 = basic_env.state[-1].ships[0]
        initial_velocity = ship0.velocity
        ship0.position = basic_env.world_size[0] + 50.0 + 300.0j

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        # Position should wrap, accounting for movement during physics step
        wrapped_ship = basic_env.state[-1].ships[0]
        # Expected position: (initial + velocity*dt) % world_size
        expected_real = (
            basic_env.world_size[0] + 50.0 + initial_velocity.real * basic_env.agent_dt
        ) % basic_env.world_size[0]
        assert abs(wrapped_ship.position.real - expected_real) < 1.0
        assert abs(wrapped_ship.position.imag - 300.0) < 1.0

    def test_ship_wrapping_all_boundaries(self, basic_env):
        """Test wrapping on all four boundaries."""
        basic_env.reset(game_mode="1v1_old")
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        ship = basic_env.state[-1].ships[0]
        velocity = ship.velocity
        dt = basic_env.agent_dt

        test_cases = [
            # (initial_position, description)
            (-50.0 + 300.0j, "Left"),
            (basic_env.world_size[0] + 50.0 + 300.0j, "Right"),
            (400.0 - 50.0j, "Top"),
            (400.0 + basic_env.world_size[1] + 50.0j, "Bottom"),
        ]

        for initial, description in test_cases:
            # Reset environment to get clean state for each test case
            basic_env.reset(game_mode="1v1")
            ship = basic_env.state[-1].ships[0]
            velocity = ship.velocity  # Get fresh velocity after reset

            ship.position = initial
            basic_env.step(actions)

            # Calculate expected position after movement and wrapping
            moved_pos = initial + velocity * dt
            expected_real = moved_pos.real % basic_env.world_size[0]
            expected_imag = moved_pos.imag % basic_env.world_size[1]

            wrapped = basic_env.state[-1].ships[0].position
            assert (
                abs(wrapped.real - expected_real) < 1.0
            ), f"{description} boundary wrapping failed"
            assert (
                abs(wrapped.imag - expected_imag) < 1.0
            ), f"{description} boundary wrapping failed"

    def test_bullet_position_wrapping(self, basic_env):
        """Test that bullet positions wrap at boundaries."""
        basic_env.reset(game_mode="1v1")

        # Add bullets at boundaries
        state = basic_env.state[-1]
        state.bullets.add_bullet(0, -10.0, 300.0, 100.0, 0.0, 1.0)
        state.bullets.add_bullet(0, 810.0, 300.0, 100.0, 0.0, 1.0)
        state.bullets.add_bullet(0, 400.0, -10.0, 0.0, 100.0, 1.0)
        state.bullets.add_bullet(0, 400.0, 610.0, 0.0, 100.0, 1.0)

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        basic_env.step(actions)

        bullets = basic_env.state[-1].bullets

        # Check all bullets wrapped correctly
        assert 0 <= bullets.x[0] < basic_env.world_size[0]
        assert 0 <= bullets.x[1] < basic_env.world_size[0]
        assert 0 <= bullets.y[2] < basic_env.world_size[1]
        assert 0 <= bullets.y[3] < basic_env.world_size[1]


class TestActionSpaceObservationSpace:
    """Tests for Gym space definitions."""

    def test_action_space(self, basic_env):
        """Test action space is correctly defined."""
        action_space = basic_env.action_space

        assert isinstance(action_space, spaces.MultiBinary)
        assert action_space.shape == (len(Actions),)

        # Test sample action
        sample = action_space.sample()
        assert sample.shape == (len(Actions),)
        assert np.all((sample == 0) | (sample == 1))

    def test_observation_space(self, basic_env):
        """Test observation space is correctly defined."""
        obs_space = basic_env.observation_space

        assert isinstance(obs_space, spaces.Dict)

        # Check new observation space structure with tokens
        assert "tokens" in obs_space.spaces

        # Check token space shape
        tokens_space = obs_space.spaces["tokens"]
        assert isinstance(tokens_space, spaces.Box)
        assert tokens_space.shape == (basic_env.max_ships, 10)
        assert tokens_space.dtype == np.float32

    def test_observation_matches_space(self, basic_env):
        """Test that actual observations match the defined space."""
        obs, _ = basic_env.reset(game_mode="1v1")
        obs_space = basic_env.observation_space

        # Create observation dict matching gym's expected format (tokens only)
        gym_obs = {"tokens": obs["token"].numpy()}  # Convert to numpy for gym

        # Use Gym's contains method to check
        assert obs_space.contains(gym_obs)
