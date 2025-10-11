"""
Tests for data collection formats and Monte Carlo return calculations.
"""

import pytest
import torch
from pathlib import Path
import tempfile

from src.collect_data import (
    compute_mc_returns,
    add_mc_returns,
    save_episodes,
    load_episodes,
)


class TestMonteCarloReturns:
    """Tests for Monte Carlo return calculation."""

    def test_simple_mc_returns(self):
        """Test MC returns for simple reward sequence."""
        rewards = [1.0, 0.0, 0.0, 1.0]
        gamma = 0.9

        returns = compute_mc_returns(rewards, gamma)

        # Manual calculation:
        # G3 = 1.0
        # G2 = 0.0 + 0.9 * 1.0 = 0.9
        # G1 = 0.0 + 0.9 * 0.9 = 0.81
        # G0 = 1.0 + 0.9 * 0.81 = 1.729
        expected = [1.729, 0.81, 0.9, 1.0]

        assert len(returns) == len(rewards)
        for actual, exp in zip(returns, expected):
            assert abs(actual - exp) < 1e-5

    def test_mc_returns_all_zeros(self):
        """Test MC returns with all zero rewards."""
        rewards = [0.0, 0.0, 0.0, 0.0]
        returns = compute_mc_returns(rewards, gamma=0.99)

        assert all(r == 0.0 for r in returns)
        assert len(returns) == len(rewards)

    def test_mc_returns_single_reward(self):
        """Test MC returns with single reward."""
        rewards = [5.0]
        returns = compute_mc_returns(rewards, gamma=0.95)

        assert len(returns) == 1
        assert returns[0] == 5.0

    def test_mc_returns_gamma_zero(self):
        """Test MC returns with gamma=0 (no discounting)."""
        rewards = [1.0, 2.0, 3.0]
        returns = compute_mc_returns(rewards, gamma=0.0)

        # With gamma=0, returns should equal immediate rewards
        assert returns == rewards

    def test_mc_returns_gamma_one(self):
        """Test MC returns with gamma=1 (full discounting)."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        returns = compute_mc_returns(rewards, gamma=1.0)

        # With gamma=1, each return is sum of all future rewards
        expected = [4.0, 3.0, 2.0, 1.0]
        assert returns == expected

    def test_mc_returns_negative_rewards(self):
        """Test MC returns with negative rewards."""
        rewards = [-1.0, 2.0, -0.5]
        gamma = 0.9
        returns = compute_mc_returns(rewards, gamma)

        # Manual calculation:
        # G2 = -0.5
        # G1 = 2.0 + 0.9 * (-0.5) = 1.55
        # G0 = -1.0 + 0.9 * 1.55 = 0.395
        expected = [0.395, 1.55, -0.5]

        for actual, exp in zip(returns, expected):
            assert abs(actual - exp) < 1e-5


class TestEpisodeDataFormat:
    """Tests for episode data structure validation."""

    @pytest.fixture
    def valid_episode_data(self):
        """Create valid episode data structure."""
        return {
            "game_mode": "2v2",
            "team_assignments": {0: [0, 1], 1: [2, 3]},
            "agent_types": {0: "scripted", 1: "scripted"},
            "observations": [
                {  # Initial observation
                    "tokens": torch.zeros((4, 10)),
                    "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
                    "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
                    "alive": torch.ones((4, 1)),
                },
                {  # After step 1
                    "tokens": torch.zeros((4, 10)),
                    "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
                    "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
                    "alive": torch.ones((4, 1)),
                },
                {  # After step 2 (episode_length=2 needs 3 observations)
                    "tokens": torch.zeros((4, 10)),
                    "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
                    "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
                    "alive": torch.ones((4, 1)),
                },
            ],
            "actions": {
                0: [
                    {0: torch.zeros(6), 1: torch.zeros(6)},
                    {0: torch.ones(6), 1: torch.zeros(6)},
                ],
                1: [
                    {2: torch.zeros(6), 3: torch.zeros(6)},
                    {2: torch.zeros(6), 3: torch.ones(6)},
                ],
            },
            "rewards": {
                0: [0.1, -0.2],
                1: [-0.1, 0.2],
            },
            "episode_length": 2,
            "terminated": True,
            "truncated": False,
            "outcome": {0: 1.0, 1: -1.0},
        }

    def test_episode_required_fields(self, valid_episode_data):
        """Test that episode data has all required fields."""
        required_fields = {
            "game_mode",
            "team_assignments",
            "agent_types",
            "observations",
            "actions",
            "rewards",
            "episode_length",
            "terminated",
            "truncated",
            "outcome",
        }

        assert set(valid_episode_data.keys()) >= required_fields

    def test_episode_sequence_lengths(self, valid_episode_data):
        """Test that observation/action/reward sequences have consistent lengths."""
        episode_length = valid_episode_data["episode_length"]

        # Observations should have episode_length + 1 (initial + per step)
        assert len(valid_episode_data["observations"]) == episode_length + 1

        # Actions and rewards should have episode_length
        for team_id in valid_episode_data["actions"]:
            assert len(valid_episode_data["actions"][team_id]) == episode_length
            assert len(valid_episode_data["rewards"][team_id]) == episode_length

    def test_team_consistency(self, valid_episode_data):
        """Test that team data is consistent across structures."""
        team_assignments = valid_episode_data["team_assignments"]
        actions = valid_episode_data["actions"]
        rewards = valid_episode_data["rewards"]
        agent_types = valid_episode_data["agent_types"]

        # All teams should appear in all structures
        assert set(team_assignments.keys()) == set(actions.keys())
        assert set(team_assignments.keys()) == set(rewards.keys())
        assert set(team_assignments.keys()) == set(agent_types.keys())

    def test_action_tensor_format(self, valid_episode_data):
        """Test that actions are properly formatted tensors."""
        for team_id, team_actions in valid_episode_data["actions"].items():
            for step_actions in team_actions:
                for ship_id, action in step_actions.items():
                    assert isinstance(action, torch.Tensor)
                    assert action.shape == (6,)  # 6 actions per ship
                    assert action.dtype == torch.float32

    def test_observation_tensor_format(self, valid_episode_data):
        """Test that observations are properly formatted."""
        for obs in valid_episode_data["observations"]:
            assert "tokens" in obs
            tokens = obs["tokens"]
            assert isinstance(tokens, torch.Tensor)
            assert tokens.dtype == torch.float32
            assert tokens.shape[1] == 10  # 10-dim tokens

    def test_add_mc_returns(self, valid_episode_data):
        """Test adding Monte Carlo returns to episode."""
        episode_with_returns = add_mc_returns(valid_episode_data, gamma=0.99)

        # Should have mc_returns field
        assert "mc_returns" in episode_with_returns

        # Should have returns for each team
        for team_id in valid_episode_data["rewards"]:
            assert team_id in episode_with_returns["mc_returns"]

            rewards = valid_episode_data["rewards"][team_id]
            returns = episode_with_returns["mc_returns"][team_id]

            assert len(returns) == len(rewards)

            # First return should include discounted future rewards
            # For rewards [0.1, -0.2] with gamma=0.99: G0 = 0.1 + 0.99*(-0.2) = -0.098
            expected_first_return = (
                rewards[0] + 0.99 * rewards[1]
            )  # gamma=0.99 from above
            assert abs(returns[0] - expected_first_return) < 1e-6

    def test_episode_outcome_validation(self, valid_episode_data):
        """Test that episode outcomes are valid."""
        if valid_episode_data["terminated"]:
            # Terminated episodes should have outcomes
            assert "outcome" in valid_episode_data
            outcomes = valid_episode_data["outcome"]

            # Outcomes should sum to approximately zero (zero-sum game)
            total_outcome = sum(outcomes.values())
            assert abs(total_outcome) < 1e-5

            # Each outcome should be in [-1, 1] range
            for outcome in outcomes.values():
                assert -1.0 <= outcome <= 1.0


class TestDataFileOperations:
    """Tests for data file saving and loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for file testing."""
        episodes = []
        for i in range(3):
            episode = {
                "game_mode": "1v1",
                "episode_length": 2,
                "rewards": {0: [0.1, -0.1], 1: [-0.1, 0.1]},
                "terminated": True,
                "episode_id": i,
            }
            episodes.append(episode)
        return episodes

    def test_save_load_uncompressed(self, temp_dir, sample_episodes):
        """Test saving and loading uncompressed episodes."""
        filepath = temp_dir / "episodes.pkl"

        save_episodes(sample_episodes, filepath, compress=False)
        assert filepath.exists()

        loaded_episodes = load_episodes(filepath)
        assert loaded_episodes == sample_episodes

    def test_save_load_compressed(self, temp_dir, sample_episodes):
        """Test saving and loading compressed episodes."""
        filepath = temp_dir / "episodes.pkl"

        save_episodes(sample_episodes, filepath, compress=True)
        assert (filepath.parent / "episodes.pkl.gz").exists()

        loaded_episodes = load_episodes(filepath.parent / "episodes.pkl.gz")
        assert loaded_episodes == sample_episodes

    def test_file_size_compression(self, temp_dir):
        """Test that compression reduces file size."""
        # Create larger dataset to see compression effect
        large_episodes = []
        for i in range(100):
            episode = {
                "game_mode": "2v2",
                "episode_length": 50,
                "observations": [torch.randn(4, 10) for _ in range(51)],
                "rewards": {0: [0.1] * 50, 1: [-0.1] * 50},
                "episode_id": i,
            }
            large_episodes.append(episode)

        # Save uncompressed
        uncompressed_path = temp_dir / "uncompressed.pkl"
        save_episodes(large_episodes, uncompressed_path, compress=False)

        # Save compressed
        compressed_path = temp_dir / "compressed.pkl"
        save_episodes(large_episodes, compressed_path, compress=True)

        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = (compressed_path.parent / "compressed.pkl.gz").stat().st_size

        # Compressed should be significantly smaller
        assert compressed_size < uncompressed_size * 0.8

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading from nonexistent file."""
        nonexistent_path = temp_dir / "missing.pkl"

        with pytest.raises(FileNotFoundError):
            load_episodes(nonexistent_path)

    def test_save_empty_episodes(self, temp_dir):
        """Test saving empty episode list."""
        filepath = temp_dir / "empty.pkl"

        save_episodes([], filepath, compress=False)
        loaded = load_episodes(filepath)

        assert loaded == []

    def test_episode_data_integrity(self, temp_dir):
        """Test that complex episode data maintains integrity through save/load."""
        complex_episode = {
            "game_mode": "3v3",
            "team_assignments": {0: [0, 1, 2], 1: [3, 4, 5]},
            "observations": [
                {
                    "tokens": torch.randn(6, 10),
                    "position": torch.randn(6, 1, dtype=torch.complex64),
                },
                {
                    "tokens": torch.randn(6, 10),
                    "position": torch.randn(6, 1, dtype=torch.complex64),
                },
            ],
            "actions": {
                0: [{0: torch.randint(0, 2, (6,)), 1: torch.randint(0, 2, (6,))}],
                1: [{3: torch.randint(0, 2, (6,)), 4: torch.randint(0, 2, (6,))}],
            },
            "rewards": {0: [0.123], 1: [-0.456]},
            "mc_returns": {0: [0.789], 1: [-0.987]},
        }

        filepath = temp_dir / "complex.pkl"
        save_episodes([complex_episode], filepath, compress=True)
        loaded = load_episodes(filepath.parent / "complex.pkl.gz")

        assert len(loaded) == 1
        loaded_episode = loaded[0]

        # Check basic structure
        assert loaded_episode["game_mode"] == complex_episode["game_mode"]
        assert loaded_episode["team_assignments"] == complex_episode["team_assignments"]

        # Check tensor data integrity
        for i, obs in enumerate(loaded_episode["observations"]):
            assert torch.allclose(
                obs["tokens"], complex_episode["observations"][i]["tokens"]
            )
            assert torch.allclose(
                obs["position"], complex_episode["observations"][i]["position"]
            )

        # Check action data
        for team_id in [0, 1]:
            for step_idx, step_actions in enumerate(loaded_episode["actions"][team_id]):
                for ship_id, action in step_actions.items():
                    original_action = complex_episode["actions"][team_id][step_idx][
                        ship_id
                    ]
                    assert torch.allclose(action, original_action)


class TestDataValidation:
    """Tests for data validation utilities."""

    def test_validate_episode_schema(self):
        """Test schema validation for episode data."""
        valid_episode = {
            "game_mode": "1v1",
            "team_assignments": {0: [0], 1: [1]},
            "observations": [
                {"tokens": torch.zeros(2, 10)},  # Initial observation
                {"tokens": torch.zeros(2, 10)},  # After step 1
            ],
            "actions": {0: [{}], 1: [{}]},
            "rewards": {0: [0.0], 1: [0.0]},
            "episode_length": 1,
            "terminated": True,
            "truncated": False,
        }

        # Should not raise any exceptions
        self._validate_episode_structure(valid_episode)

    def test_invalid_episode_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_episode = {
            "game_mode": "1v1",
            # Missing other required fields
        }

        with pytest.raises(KeyError):
            self._validate_episode_structure(invalid_episode)

    def test_invalid_episode_length_mismatch(self):
        """Test validation with mismatched sequence lengths."""
        invalid_episode = {
            "game_mode": "1v1",
            "team_assignments": {0: [0], 1: [1]},
            "observations": [{"tokens": torch.zeros(2, 10)}],
            "actions": {0: [{}], 1: [{}]},
            "rewards": {0: [0.0, 0.0], 1: [0.0]},  # Mismatched lengths
            "episode_length": 1,
            "terminated": True,
            "truncated": False,
        }

        with pytest.raises(ValueError):
            self._validate_episode_structure(invalid_episode)

    def _validate_episode_structure(self, episode):
        """Helper method to validate episode structure."""
        required_fields = {
            "game_mode",
            "team_assignments",
            "observations",
            "actions",
            "rewards",
            "episode_length",
            "terminated",
            "truncated",
        }

        # Check required fields
        for field in required_fields:
            if field not in episode:
                raise KeyError(f"Missing required field: {field}")

        # Check sequence length consistency
        episode_length = episode["episode_length"]

        # Observations should be episode_length + 1
        if len(episode["observations"]) != episode_length + 1:
            raise ValueError("Observation sequence length mismatch")

        # Actions and rewards should be episode_length for each team
        for team_id in episode["actions"]:
            if len(episode["actions"][team_id]) != episode_length:
                raise ValueError(f"Action sequence length mismatch for team {team_id}")
            if len(episode["rewards"][team_id]) != episode_length:
                raise ValueError(f"Reward sequence length mismatch for team {team_id}")
