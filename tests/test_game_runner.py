"""
Tests for the UnifiedGameRunner system.
"""

import pytest
import torch


from src.game_runner import (
    UnifiedGameRunner,
    create_standard_runner,
    create_human_runner,
)
from src.agents import Agent


class MockAgent(Agent):
    """Mock agent for testing that returns predictable actions."""

    def __init__(self, agent_type: str = "mock", action_pattern: dict = None):
        self.agent_type = agent_type
        self.action_pattern = action_pattern or {}
        self.call_count = 0
        self.last_obs = None
        self.last_ship_ids = None

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        self.call_count += 1
        self.last_obs = obs_dict
        self.last_ship_ids = ship_ids

        actions = {}
        for ship_id in ship_ids:
            if ship_id in self.action_pattern:
                actions[ship_id] = torch.tensor(
                    self.action_pattern[ship_id], dtype=torch.float32
                )
            else:
                # Default to alternating pattern based on call count
                pattern = (
                    [1, 0, 1, 0, 0, 0]
                    if self.call_count % 2 == 1
                    else [0, 1, 0, 1, 0, 1]
                )
                actions[ship_id] = torch.tensor(pattern, dtype=torch.float32)

        return actions

    def get_agent_type(self) -> str:
        return self.agent_type


class TestUnifiedGameRunnerInitialization:
    """Tests for game runner initialization and setup."""

    @pytest.fixture
    def basic_env_config(self):
        """Basic environment configuration for testing."""
        return {
            "world_size": (800, 600),
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }

    @pytest.fixture
    def team_assignments(self):
        """Standard team assignments."""
        return {0: [0, 1], 1: [2, 3]}

    def test_runner_initialization(self, basic_env_config, team_assignments):
        """Test basic runner initialization."""
        runner = UnifiedGameRunner(basic_env_config, team_assignments)

        assert runner.env_config == basic_env_config
        assert runner.team_assignments == team_assignments
        assert runner.env is None
        assert len(runner.agents) == 0

    def test_setup_environment(self, basic_env_config, team_assignments):
        """Test environment setup."""
        runner = UnifiedGameRunner(basic_env_config, team_assignments)
        env = runner.setup_environment()

        assert runner.env is not None
        assert runner.env.world_size == (800, 600)
        assert runner.env.max_ships == 4
        assert env is runner.env

    def test_setup_environment_with_render_mode(
        self, basic_env_config, team_assignments
    ):
        """Test environment setup with rendering."""
        runner = UnifiedGameRunner(basic_env_config, team_assignments)
        env = runner.setup_environment(render_mode="human")

        assert runner.env.render_mode == "human"

    def test_agent_assignment(self, basic_env_config, team_assignments):
        """Test agent assignment to teams."""
        runner = UnifiedGameRunner(basic_env_config, team_assignments)

        mock_agent_0 = MockAgent("test_agent_0")
        mock_agent_1 = MockAgent("test_agent_1")

        runner.assign_agent(0, mock_agent_0)
        runner.assign_agent(1, mock_agent_1)

        assert len(runner.agents) == 2
        assert runner.agents[0] is mock_agent_0
        assert runner.agents[1] is mock_agent_1


class TestEpisodeExecution:
    """Tests for episode execution and data collection."""

    @pytest.fixture
    def setup_runner_with_agents(self):
        """Create runner with environment and mock agents."""
        env_config = {
            "world_size": (1200, 800),
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }
        team_assignments = {0: [0, 1], 1: [2, 3]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        runner.setup_environment()

        # Create mock agents with predictable behavior
        agent_0 = MockAgent("scripted", {0: [1, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0]})
        agent_1 = MockAgent("scripted", {2: [0, 0, 1, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0]})

        runner.assign_agent(0, agent_0)
        runner.assign_agent(1, agent_1)

        return runner

    def test_single_episode_basic(self, setup_runner_with_agents):
        """Test basic single episode execution."""
        runner = setup_runner_with_agents

        episode_data = runner.run_episode(
            game_mode="2v2", collect_data=False, max_steps=10
        )

        # Check basic episode structure
        assert "game_mode" in episode_data
        assert "episode_length" in episode_data
        assert "terminated" in episode_data
        assert "truncated" in episode_data
        assert "team_assignments" in episode_data
        assert "agent_types" in episode_data

        # Episode should complete within max_steps
        assert episode_data["episode_length"] <= 10
        assert isinstance(episode_data["terminated"], bool)
        assert isinstance(episode_data["truncated"], bool)

    def test_episode_data_collection_enabled(self, setup_runner_with_agents):
        """Test episode with data collection enabled."""
        runner = setup_runner_with_agents

        episode_data = runner.run_episode(
            game_mode="1v1", collect_data=True, max_steps=5
        )

        # Should have data collection fields
        assert "observations" in episode_data
        assert "actions" in episode_data
        assert "rewards" in episode_data
        assert "outcome" in episode_data

        # Check data structure
        episode_length = episode_data["episode_length"]

        # Observations: initial + one per step
        assert len(episode_data["observations"]) == episode_length + 1

        # Actions and rewards: one per step, per team
        for team_id in episode_data["team_assignments"]:
            if team_id in episode_data["actions"]:
                assert len(episode_data["actions"][team_id]) == episode_length
                assert len(episode_data["rewards"][team_id]) == episode_length

    def test_episode_data_collection_disabled(self, setup_runner_with_agents):
        """Test episode with data collection disabled."""
        runner = setup_runner_with_agents

        episode_data = runner.run_episode(
            game_mode="1v1", collect_data=False, max_steps=5
        )

        # Should not have data collection fields or they should be empty
        assert episode_data.get("observations", []) == []
        assert episode_data.get("actions", {}) == {}
        assert episode_data.get("rewards", {}) == {}

    def test_episode_max_steps_truncation(self, setup_runner_with_agents):
        """Test episode truncation at max_steps."""
        runner = setup_runner_with_agents

        # Use very small max_steps to force truncation
        episode_data = runner.run_episode(
            game_mode="2v2", collect_data=False, max_steps=2
        )

        # Should be truncated at max_steps
        assert episode_data["episode_length"] <= 2

        # If not terminated naturally, should be truncated
        if not episode_data["terminated"]:
            assert episode_data["episode_length"] == 2

    def test_agent_action_calls(self, setup_runner_with_agents):
        """Test that agents are called correctly during episode."""
        runner = setup_runner_with_agents

        episode_data = runner.run_episode(
            game_mode="1v1", collect_data=False, max_steps=3
        )

        # Both agents should have been called
        agent_0 = runner.agents[0]
        agent_1 = runner.agents[1]

        assert agent_0.call_count > 0
        assert agent_1.call_count > 0

        # Agents should have received ship IDs
        assert agent_0.last_ship_ids is not None
        assert agent_1.last_ship_ids is not None

        # Should have received observations
        assert agent_0.last_obs is not None
        assert agent_1.last_obs is not None


class TestTeamAssignmentUpdates:
    """Tests for dynamic team assignment handling."""

    @pytest.fixture
    def nvn_runner(self):
        """Create runner for nvn mode testing."""
        env_config = {
            "world_size": (1200, 800),
            "max_ships": 8,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }
        # Initial team assignments will be overridden by nvn mode
        team_assignments = {0: [0, 1], 1: [2, 3]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        runner.setup_environment()

        return runner

    def test_team_assignment_updates_nvn(self, nvn_runner):
        """Test that team assignments update correctly in nvn mode."""
        runner = nvn_runner

        # Add agents that can handle variable team sizes
        flexible_agent_0 = MockAgent("flexible_0")
        flexible_agent_1 = MockAgent("flexible_1")

        runner.assign_agent(0, flexible_agent_0)
        runner.assign_agent(1, flexible_agent_1)

        episode_data = runner.run_episode(
            game_mode="nvn", collect_data=False, max_steps=5
        )

        # Team assignments should be updated from environment
        actual_teams = episode_data["team_assignments"]

        # Should have two teams
        assert len(actual_teams) == 2
        assert 0 in actual_teams
        assert 1 in actual_teams

        # Each team should have at least one ship
        assert len(actual_teams[0]) >= 1
        assert len(actual_teams[1]) >= 1

        # All ships should be assigned
        all_ships = set()
        for ships in actual_teams.values():
            all_ships.update(ships)

        # Should have some ships assigned
        assert len(all_ships) > 0

    def test_get_actual_team_assignments(self, nvn_runner):
        """Test _get_actual_team_assignments method."""
        runner = nvn_runner

        # Before reset, should return default assignments
        default_assignments = runner._get_actual_team_assignments()
        assert default_assignments == {0: [0, 1], 1: [2, 3]}

        # After reset, should get assignments from environment
        runner.env.reset(game_mode="2v2")
        actual_assignments = runner._get_actual_team_assignments()

        # Should have updated assignments
        assert len(actual_assignments) == 2
        assert 0 in actual_assignments
        assert 1 in actual_assignments


class TestMultipleEpisodes:
    """Tests for running multiple episodes."""

    @pytest.fixture
    def batch_runner(self):
        """Create runner for batch episode testing."""
        env_config = {
            "world_size": (1200, 800),
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }
        team_assignments = {0: [0, 1], 1: [2, 3]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        runner.setup_environment()

        # Fast agents for batch testing
        agent_0 = MockAgent("batch_0")
        agent_1 = MockAgent("batch_1")

        runner.assign_agent(0, agent_0)
        runner.assign_agent(1, agent_1)

        return runner

    def test_run_multiple_episodes(self, batch_runner):
        """Test running multiple episodes."""
        runner = batch_runner

        episodes = runner.run_multiple_episodes(
            n_episodes=3, game_mode="1v1", collect_data=False, max_steps=5
        )

        assert len(episodes) == 3

        for episode in episodes:
            assert "episode_length" in episode
            assert "game_mode" in episode
            assert episode["game_mode"] == "1v1"
            assert episode["episode_length"] <= 5

    def test_progress_callback(self, batch_runner):
        """Test progress callback functionality."""
        runner = batch_runner
        callback_calls = []

        def progress_callback(episode_num, total, episode_data):
            callback_calls.append((episode_num, total, episode_data["episode_length"]))

        episodes = runner.run_multiple_episodes(
            n_episodes=3,
            game_mode="2v2",
            collect_data=False,
            progress_callback=progress_callback,
            max_steps=3,
        )

        assert len(callback_calls) == 3
        assert callback_calls[0][0] == 1  # First episode
        assert callback_calls[0][1] == 3  # Total episodes
        assert callback_calls[2][0] == 3  # Last episode

        # All callbacks should have episode data
        for call in callback_calls:
            assert isinstance(call[2], int)  # episode_length

    def test_batch_data_collection(self, batch_runner):
        """Test data collection across multiple episodes."""
        runner = batch_runner

        episodes = runner.run_multiple_episodes(
            n_episodes=2, game_mode="1v1", collect_data=True, max_steps=3
        )

        assert len(episodes) == 2

        for episode in episodes:
            # Each episode should have data
            assert "observations" in episode
            assert "actions" in episode
            assert "rewards" in episode

            # Data should be consistent
            episode_length = episode["episode_length"]
            assert len(episode["observations"]) == episode_length + 1


class TestWinStatistics:
    """Tests for win rate calculation and statistics."""

    @pytest.fixture
    def stats_runner(self):
        """Create runner for statistics testing."""
        env_config = {
            "world_size": (800, 600),
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }
        team_assignments = {0: [0, 1], 1: [2, 3]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        return runner

    def test_win_stats_all_wins(self, stats_runner):
        """Test win statistics with all wins."""
        episodes = [
            {
                "terminated": True,
                "outcome": {0: 1.0, 1: -1.0},  # Team 0 wins
                "episode_length": 10,
            },
            {
                "terminated": True,
                "outcome": {0: 1.0, 1: -1.0},  # Team 0 wins
                "episode_length": 15,
            },
            {
                "terminated": True,
                "outcome": {0: 1.0, 1: -1.0},  # Team 0 wins
                "episode_length": 8,
            },
        ]

        stats = stats_runner.get_win_stats(episodes, team_id=0)

        assert stats["wins"] == 3
        assert stats["losses"] == 0
        assert stats["draws"] == 0
        assert stats["win_rate"] == 1.0
        assert stats["avg_length"] == (10 + 15 + 8) / 3

    def test_win_stats_mixed_outcomes(self, stats_runner):
        """Test win statistics with mixed outcomes."""
        episodes = [
            {
                "terminated": True,
                "outcome": {0: 1.0, 1: -1.0},  # Team 0 wins
                "episode_length": 10,
            },
            {
                "terminated": True,
                "outcome": {0: -1.0, 1: 1.0},  # Team 0 loses
                "episode_length": 12,
            },
            {
                "terminated": True,
                "outcome": {0: 0.0, 1: 0.0},  # Draw
                "episode_length": 20,
            },
            {
                "terminated": False,  # Not terminated, shouldn't count
                "outcome": {},
                "episode_length": 5,
            },
        ]

        stats = stats_runner.get_win_stats(episodes, team_id=0)

        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["draws"] == 1
        assert stats["win_rate"] == 0.5  # 1 win out of 2 decisive games
        assert stats["avg_length"] == (10 + 12 + 20) / 3  # Only terminated episodes

    def test_win_stats_no_terminated_episodes(self, stats_runner):
        """Test win statistics with no terminated episodes."""
        episodes = [
            {
                "terminated": False,
                "outcome": {},
                "episode_length": 5,
            },
            {
                "terminated": False,
                "outcome": {},
                "episode_length": 3,
            },
        ]

        stats = stats_runner.get_win_stats(episodes, team_id=0)

        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_length"] == 0.0

    def test_win_stats_empty_episodes(self, stats_runner):
        """Test win statistics with empty episode list."""
        episodes = []

        stats = stats_runner.get_win_stats(episodes, team_id=0)

        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_length"] == 0.0


class TestFactoryFunctions:
    """Tests for runner factory functions."""

    def test_create_standard_runner(self):
        """Test standard runner factory."""
        runner = create_standard_runner(world_size=(1000, 600), max_ships=6)

        assert runner.env_config["world_size"] == (1000, 600)
        assert runner.env_config["max_ships"] == 6
        assert runner.env_config["agent_dt"] == 0.04
        assert runner.env_config["physics_dt"] == 0.02

        # Should have default team assignments
        assert runner.team_assignments == {0: [0, 1], 1: [2, 3]}

    def test_create_human_runner(self):
        """Test human runner factory."""
        runner = create_human_runner(world_size=(1400, 900))

        assert runner.env_config["world_size"] == (1400, 900)
        assert runner.env_config["render_mode"] == "human"
        assert runner.env_config["max_ships"] == 8

        # Should have human vs multiple opponents setup
        assert runner.team_assignments == {0: [0], 1: [1, 2, 3]}


class TestResourceManagement:
    """Tests for proper resource cleanup."""

    def test_runner_close(self):
        """Test runner resource cleanup."""
        env_config = {
            "world_size": (800, 600),
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
        }
        team_assignments = {0: [0, 1], 1: [2, 3]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        runner.setup_environment()

        # Environment should be initialized
        assert runner.env is not None

        # Close should clean up
        runner.close()

        # Environment should be cleaned up
        assert runner.env is None

    def test_multiple_close_calls(self):
        """Test that multiple close calls don't cause errors."""
        runner = create_standard_runner()
        runner.setup_environment()

        # Should handle multiple close calls gracefully
        runner.close()
        runner.close()  # Should not raise exception

        assert runner.env is None


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_run_episode_without_environment(self):
        """Test running episode without setting up environment."""
        env_config = {"world_size": (800, 600), "max_ships": 4}
        team_assignments = {0: [0], 1: [1]}

        runner = UnifiedGameRunner(env_config, team_assignments)
        # Don't call setup_environment()

        with pytest.raises(ValueError, match="Environment not setup"):
            runner.run_episode()

    # def test_run_episode_without_agents(self):
    #     """Test running episode without assigning agents."""
    #     runner = create_standard_runner()
    #     runner.setup_environment()
    #     # Don't assign any agents

    #     # Should handle gracefully (though might not produce meaningful results)
    #     episode_data = runner.run_episode(game_mode="1v1", max_steps=2)

    #     assert "episode_length" in episode_data
    #     # Episode should terminate quickly due to no actions

    def test_invalid_game_mode(self):
        """Test running episode with invalid game mode."""
        runner = create_standard_runner()
        runner.setup_environment()

        agent = MockAgent("test")
        runner.assign_agent(0, agent)
        runner.assign_agent(1, agent)

        with pytest.raises(ValueError, match="Unknown game mode"):
            runner.run_episode(game_mode="invalid_mode")
