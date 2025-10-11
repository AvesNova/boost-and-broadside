"""
Tests for the unified agent system integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.agents import (
    ScriptedAgentProvider,
    RLAgentProvider,
    HumanAgentProvider,
    RandomAgentProvider,
    SelfPlayAgentProvider,
    create_scripted_agent,
    create_rl_agent,
    create_human_agent,
    create_selfplay_agent,
)
from src.constants import Actions


class TestScriptedAgentProvider:
    """Tests for scripted agent functionality."""

    @pytest.fixture
    def scripted_config(self):
        """Standard scripted agent configuration."""
        return {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
        }

    @pytest.fixture
    def world_size(self):
        """Standard world size for testing."""
        return (1200, 800)

    @pytest.fixture
    def sample_observations(self):
        """Create sample observation dict for testing."""
        return {
            "tokens": torch.randn(4, 10),
            "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
            "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
            "alive": torch.tensor([1, 1, 1, 0]).reshape(-1, 1),  # Ship 3 is dead
            "health": torch.tensor([80.0, 90.0, 70.0, 0.0]).reshape(-1, 1),
            "power": torch.tensor([50.0, 60.0, 45.0, 0.0]).reshape(-1, 1),
            "position": torch.tensor(
                [100 + 200j, 150 + 250j, 500 + 300j, 600 + 400j]
            ).reshape(-1, 1),
            "velocity": torch.tensor([50 + 0j, 60 + 10j, -30 + 20j, 0 + 0j]).reshape(
                -1, 1
            ),
            "speed": torch.tensor([50.0, 61.0, 36.0, 0.0]).reshape(-1, 1),
            "attitude": torch.tensor(
                [1 + 0j, 0.9 + 0.44j, -0.8 + 0.6j, 1 + 0j]
            ).reshape(-1, 1),
            "is_shooting": torch.tensor([0, 1, 0, 0]).reshape(-1, 1),
        }

    def test_scripted_agent_initialization(self, scripted_config, world_size):
        """Test scripted agent provider initialization."""
        agent = ScriptedAgentProvider(scripted_config, world_size)

        assert agent.scripted_config == scripted_config
        assert agent.world_size == world_size
        assert len(agent.agents) == 0
        assert agent.get_agent_type() == "scripted"

    def test_scripted_agent_creation_on_demand(
        self, scripted_config, world_size, sample_observations
    ):
        """Test that scripted agents are created on demand."""
        agent_provider = ScriptedAgentProvider(scripted_config, world_size)

        ship_ids = [0, 1]
        actions = agent_provider.get_actions(sample_observations, ship_ids)

        # Should have created agents for requested ships
        assert 0 in agent_provider.agents
        assert 1 in agent_provider.agents
        assert 2 not in agent_provider.agents  # Not requested

        # Should return actions for requested ships
        assert set(actions.keys()) == {0, 1}
        for ship_id in ship_ids:
            assert isinstance(actions[ship_id], torch.Tensor)
            assert actions[ship_id].shape == (6,)
            assert actions[ship_id].dtype == torch.float32

    def test_scripted_agent_alive_ship_handling(
        self, scripted_config, world_size, sample_observations
    ):
        """Test that scripted agents only generate actions for alive ships."""
        agent_provider = ScriptedAgentProvider(scripted_config, world_size)

        # Request actions for alive and dead ships
        ship_ids = [0, 3]  # Ship 0 alive, ship 3 dead
        actions = agent_provider.get_actions(sample_observations, ship_ids)

        # Should get actions for both, but dead ship should get zero actions
        assert len(actions) == 2

        # Alive ship should have non-zero actions (at least some possibility)
        alive_actions = actions[0]
        assert alive_actions.shape == (6,)

        # Dead ship should have zero actions
        dead_actions = actions[3]
        assert torch.all(dead_actions == 0.0)

    def test_scripted_agent_action_consistency(
        self, scripted_config, world_size, sample_observations
    ):
        """Test that scripted agents produce consistent actions for same state."""
        agent_provider = ScriptedAgentProvider(scripted_config, world_size)

        ship_ids = [0, 1]

        # Get actions twice for same state
        actions1 = agent_provider.get_actions(sample_observations, ship_ids)
        actions2 = agent_provider.get_actions(sample_observations, ship_ids)

        # Should be identical (scripted agents are deterministic)
        for ship_id in ship_ids:
            assert torch.allclose(actions1[ship_id], actions2[ship_id])

    def test_scripted_agent_ship_out_of_bounds(
        self, scripted_config, world_size, sample_observations
    ):
        """Test handling of ship IDs beyond observation bounds."""
        agent_provider = ScriptedAgentProvider(scripted_config, world_size)

        # Request action for ship ID beyond observation array
        ship_ids = [10]  # Only 4 ships in sample_observations
        actions = agent_provider.get_actions(sample_observations, ship_ids)

        # Should handle gracefully with zero actions
        assert 10 in actions
        assert torch.all(actions[10] == 0.0)


class TestRLAgentProvider:
    """Tests for RL agent functionality."""

    @pytest.fixture
    def mock_transformer_model(self):
        """Mock transformer model for testing."""
        model = Mock()
        model.get_actions.return_value = {
            "actions": torch.tensor(
                [
                    [
                        [1, 0, 1, 0, 0, 0],  # Ship 0 actions
                        [0, 1, 0, 1, 0, 1],  # Ship 1 actions
                        [1, 1, 0, 0, 1, 0],  # Ship 2 actions
                        [0, 0, 0, 0, 0, 0],
                    ]  # Ship 3 actions
                ],
                dtype=torch.float32,
            )
        }
        return model

    @pytest.fixture
    def mock_ppo_model(self):
        """Mock PPO model for testing."""
        model = Mock()
        # PPO returns flattened actions and values
        model.predict.return_value = (
            np.array(
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32
            ),  # 2 ships * 6 actions
            None,  # values (not used in test)
        )
        return model

    @pytest.fixture
    def mock_team_controller(self):
        """Mock team controller for testing."""
        controller = Mock()
        return controller

    @pytest.fixture
    def sample_observations(self):
        """Sample observations for RL agent testing."""
        return {
            "tokens": torch.randn(4, 10),
            "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
            "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
            "alive": torch.tensor([1, 1, 1, 1]).reshape(-1, 1),
            "health": torch.tensor([100.0, 80.0, 90.0, 70.0]).reshape(-1, 1),
            "power": torch.tensor([75.0, 50.0, 60.0, 45.0]).reshape(-1, 1),
            "position": torch.tensor(
                [100 + 200j, 150 + 250j, 500 + 300j, 600 + 400j]
            ).reshape(-1, 1),
            "velocity": torch.tensor([50 + 0j, 60 + 10j, -30 + 20j, 40 + 30j]).reshape(
                -1, 1
            ),
            "speed": torch.tensor([50.0, 61.0, 36.0, 50.0]).reshape(-1, 1),
            "attitude": torch.tensor(
                [1 + 0j, 0.9 + 0.44j, -0.8 + 0.6j, 0.8 - 0.6j]
            ).reshape(-1, 1),
            "is_shooting": torch.tensor([0, 1, 0, 1]).reshape(-1, 1),
        }

    def test_transformer_rl_agent(
        self, mock_transformer_model, mock_team_controller, sample_observations
    ):
        """Test transformer-based RL agent."""
        agent = RLAgentProvider(
            mock_transformer_model, mock_team_controller, "transformer"
        )

        ship_ids = [0, 1, 2]
        actions = agent.get_actions(sample_observations, ship_ids)

        # Should call model's get_actions method
        assert mock_transformer_model.get_actions.called

        # Should return actions for requested ships
        assert set(actions.keys()) == {0, 1, 2}

        for ship_id in ship_ids:
            assert isinstance(actions[ship_id], torch.Tensor)
            assert actions[ship_id].shape == (6,)

    def test_ppo_rl_agent(
        self, mock_ppo_model, mock_team_controller, sample_observations
    ):
        """Test PPO-based RL agent."""
        agent = RLAgentProvider(mock_ppo_model, mock_team_controller, "ppo")

        ship_ids = [0, 1]
        actions = agent.get_actions(sample_observations, ship_ids)

        # Should call model's predict method
        assert mock_ppo_model.predict.called

        # Should return actions for requested ships
        assert set(actions.keys()) == {0, 1}

        for ship_id in ship_ids:
            assert isinstance(actions[ship_id], torch.Tensor)
            assert actions[ship_id].shape == (6,)

    def test_rl_agent_error_handling(self, mock_team_controller, sample_observations):
        """Test RL agent error handling when model fails."""
        # Create model that raises exception
        broken_model = Mock()
        broken_model.get_actions.side_effect = RuntimeError("Model error")

        agent = RLAgentProvider(broken_model, mock_team_controller, "transformer")

        ship_ids = [0, 1]
        actions = agent.get_actions(sample_observations, ship_ids)

        # Should handle error gracefully with random actions
        assert set(actions.keys()) == {0, 1}

        for ship_id in ship_ids:
            assert isinstance(actions[ship_id], torch.Tensor)
            assert actions[ship_id].shape == (6,)
            # Should be binary values (random actions)
            assert torch.all((actions[ship_id] == 0) | (actions[ship_id] == 1))

    def test_rl_agent_ship_mask_creation(
        self, mock_transformer_model, mock_team_controller, sample_observations
    ):
        """Test ship mask creation for transformer models."""
        # Modify observations to have one dead ship
        sample_observations["alive"] = torch.tensor([1, 1, 0, 1]).reshape(-1, 1)

        agent = RLAgentProvider(
            mock_transformer_model, mock_team_controller, "transformer"
        )
        ship_ids = [0, 1, 2, 3]

        agent.get_actions(sample_observations, ship_ids)

        # Should have created mask and called model
        assert mock_transformer_model.get_actions.called

        # Check the call arguments to verify mask creation
        call_args = mock_transformer_model.get_actions.call_args
        obs_batch = call_args[0][0]
        ship_mask = call_args[0][1]

        assert isinstance(ship_mask, torch.Tensor)
        assert ship_mask.shape == (1, 4)  # batch_size=1, max_ships=4

    def test_rl_agent_type_identification(
        self, mock_transformer_model, mock_team_controller
    ):
        """Test agent type identification."""
        transformer_agent = RLAgentProvider(
            mock_transformer_model, mock_team_controller, "transformer"
        )
        assert transformer_agent.get_agent_type() == "rl_transformer"

        ppo_agent = RLAgentProvider(mock_transformer_model, mock_team_controller, "ppo")
        assert ppo_agent.get_agent_type() == "rl_ppo"

        bc_agent = RLAgentProvider(mock_transformer_model, mock_team_controller, "bc")
        assert bc_agent.get_agent_type() == "rl_bc"


class TestHumanAgentProvider:
    """Tests for human agent functionality."""

    @pytest.fixture
    def mock_renderer(self):
        """Mock renderer for human agent testing."""
        renderer = Mock()
        renderer.update_human_actions.return_value = None
        renderer.get_human_actions.return_value = {
            0: torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32),
            1: torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.float32),
        }
        return renderer

    def test_human_agent_initialization(self, mock_renderer):
        """Test human agent provider initialization."""
        agent = HumanAgentProvider(mock_renderer)

        assert agent.renderer is mock_renderer
        assert agent.get_agent_type() == "human"

    def test_human_agent_actions(self, mock_renderer):
        """Test human agent action retrieval."""
        agent = HumanAgentProvider(mock_renderer)
        sample_obs = {"tokens": torch.randn(2, 10)}

        ship_ids = [0, 1, 2]
        actions = agent.get_actions(sample_obs, ship_ids)

        # Should update human input
        assert mock_renderer.update_human_actions.called
        assert mock_renderer.get_human_actions.called

        # Should return actions for all requested ships
        assert set(actions.keys()) == {0, 1, 2}

        # Ships with human input should get those actions
        assert torch.equal(
            actions[0], torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32)
        )
        assert torch.equal(
            actions[1], torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.float32)
        )

        # Ship without human input should get zero actions
        assert torch.equal(actions[2], torch.zeros(6, dtype=torch.float32))


class TestRandomAgentProvider:
    """Tests for random agent functionality."""

    def test_random_agent_initialization(self):
        """Test random agent initialization."""
        agent = RandomAgentProvider(rng=np.random.default_rng(seed=42))
        assert agent.get_agent_type() == "random"

        # Test without seed
        agent_no_seed = RandomAgentProvider()
        assert agent_no_seed.get_agent_type() == "random"

    def test_random_agent_actions(self):
        """Test random agent action generation."""
        agent = RandomAgentProvider(rng=np.random.default_rng(seed=42))
        sample_obs = {"tokens": torch.randn(3, 10)}

        ship_ids = [0, 1, 2]
        actions = agent.get_actions(sample_obs, ship_ids)

        # Should return actions for all requested ships
        assert set(actions.keys()) == {0, 1, 2}

        for ship_id in ship_ids:
            action = actions[ship_id]
            assert isinstance(action, torch.Tensor)
            assert action.shape == (6,)
            assert action.dtype == torch.float32
            # Should be binary values
            assert torch.all((action == 0.0) | (action == 1.0))

    def test_random_agent_determinism(self):
        """Test that same seed produces same actions."""
        agent1 = RandomAgentProvider(rng=np.random.default_rng(seed=123))
        agent2 = RandomAgentProvider(rng=np.random.default_rng(seed=123))

        sample_obs = {"tokens": torch.randn(2, 10)}
        ship_ids = [0, 1]

        actions1 = agent1.get_actions(sample_obs, ship_ids)
        actions2 = agent2.get_actions(sample_obs, ship_ids)

        # Same seed should produce same actions
        for ship_id in ship_ids:
            assert torch.equal(actions1[ship_id], actions2[ship_id])


class TestSelfPlayAgentProvider:
    """Tests for self-play agent functionality."""

    @pytest.fixture
    def mock_team_controller(self):
        """Mock team controller."""
        return Mock()

    @pytest.fixture
    def mock_model_class(self):
        """Mock model class for self-play testing."""
        model_class = Mock()
        model_instance = Mock()
        model_instance.state_dict.return_value = {"param1": torch.tensor([1.0, 2.0])}
        model_instance.load_state_dict = Mock()
        model_instance.eval = Mock()
        model_class.return_value = model_instance
        return model_class, model_instance

    def test_selfplay_agent_initialization(self, mock_team_controller):
        """Test self-play agent initialization."""
        agent = SelfPlayAgentProvider(mock_team_controller, memory_size=10)

        assert agent.team_controller is mock_team_controller
        assert agent.memory_size == 10
        assert len(agent.model_memory) == 0
        assert agent.current_opponent is None
        assert agent.get_agent_type() == "self_play_random"

    def test_selfplay_add_model_to_memory(self, mock_team_controller):
        """Test adding models to self-play memory."""
        agent = SelfPlayAgentProvider(mock_team_controller, memory_size=3)

        # Create mock models
        model1 = Mock()
        model1.state_dict.return_value = {"param": torch.tensor([1.0])}

        model2 = Mock()
        model2.state_dict.return_value = {"param": torch.tensor([2.0])}

        # Add models
        agent.add_model_to_memory(model1)
        assert len(agent.model_memory) == 1

        agent.add_model_to_memory(model2)
        assert len(agent.model_memory) == 2

        # Add more models to test memory limit
        for i in range(3, 10):
            model = Mock()
            model.state_dict.return_value = {"param": torch.tensor([float(i)])}
            agent.add_model_to_memory(model)

        # Should be limited by memory_size
        assert len(agent.model_memory) == 3

    def test_selfplay_update_opponent(self, mock_team_controller, mock_model_class):
        """Test updating self-play opponent."""
        model_class, model_instance = mock_model_class
        agent = SelfPlayAgentProvider(mock_team_controller, memory_size=5)

        # Add some models to memory first
        for i in range(3):
            model = Mock()
            model.state_dict.return_value = {"param": torch.tensor([float(i)])}
            agent.add_model_to_memory(model)

        # Update opponent
        model_config = {"embedding_dim": 64}
        agent.update_opponent(model_class, model_config)

        # Should have created new model and loaded from memory
        assert model_class.called
        assert model_instance.load_state_dict.called
        assert model_instance.eval.called

        # Should have current opponent
        assert agent.current_opponent is not None
        assert agent.get_agent_type().startswith("self_play_")

    def test_selfplay_empty_memory_fallback(
        self, mock_team_controller, mock_model_class
    ):
        """Test self-play behavior with empty memory."""
        model_class, model_instance = mock_model_class
        agent = SelfPlayAgentProvider(mock_team_controller, memory_size=5)

        # Try to update opponent with no models in memory
        agent.update_opponent(model_class, {})

        # Should fallback to random opponent
        assert agent.current_opponent is not None
        assert agent.current_opponent.get_agent_type() == "random"

    def test_selfplay_get_actions(self, mock_team_controller):
        """Test self-play agent action generation."""
        agent = SelfPlayAgentProvider(mock_team_controller, memory_size=5)

        sample_obs = {"tokens": torch.randn(2, 10)}
        ship_ids = [0, 1]

        # With no opponent set, should use fallback
        actions = agent.get_actions(sample_obs, ship_ids)

        assert set(actions.keys()) == {0, 1}
        for ship_id in ship_ids:
            assert isinstance(actions[ship_id], torch.Tensor)
            assert actions[ship_id].shape == (6,)


class TestFactoryFunctions:
    """Tests for agent factory functions."""

    def test_create_scripted_agent(self):
        """Test scripted agent factory function."""
        world_size = (1200, 800)
        config = {"max_shooting_range": 600.0}

        agent = create_scripted_agent(world_size, config)

        assert isinstance(agent, ScriptedAgentProvider)
        assert agent.world_size == world_size
        assert agent.scripted_config["max_shooting_range"] == 600.0

    def test_create_scripted_agent_defaults(self):
        """Test scripted agent factory with defaults."""
        world_size = (800, 600)

        agent = create_scripted_agent(world_size)

        assert isinstance(agent, ScriptedAgentProvider)
        assert "max_shooting_range" in agent.scripted_config
        assert "angle_threshold" in agent.scripted_config

    def test_create_rl_agent(self):
        """Test RL agent factory function."""
        mock_model = Mock()
        mock_controller = Mock()

        agent = create_rl_agent(mock_model, mock_controller, "transformer")

        assert isinstance(agent, RLAgentProvider)
        assert agent.model is mock_model
        assert agent.team_controller is mock_controller
        assert agent.model_type == "transformer"

    def test_create_human_agent(self):
        """Test human agent factory function."""
        mock_renderer = Mock()

        agent = create_human_agent(mock_renderer)

        assert isinstance(agent, HumanAgentProvider)
        assert agent.renderer is mock_renderer

    def test_create_selfplay_agent(self):
        """Test self-play agent factory function."""
        mock_controller = Mock()

        agent = create_selfplay_agent(mock_controller, memory_size=25)

        assert isinstance(agent, SelfPlayAgentProvider)
        assert agent.team_controller is mock_controller
        assert agent.memory_size == 25


class TestAgentActionConsistency:
    """Tests for action format consistency across agent types."""

    @pytest.fixture
    def sample_observations(self):
        """Standard observations for consistency testing."""
        return {
            "tokens": torch.randn(4, 10),
            "ship_id": torch.tensor([0, 1, 2, 3]).reshape(-1, 1),
            "team_id": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
            "alive": torch.ones((4, 1)),
            "health": torch.tensor([100.0, 80.0, 90.0, 70.0]).reshape(-1, 1),
            "power": torch.tensor([75.0, 50.0, 60.0, 45.0]).reshape(-1, 1),
            "position": torch.tensor(
                [100 + 200j, 150 + 250j, 500 + 300j, 600 + 400j]
            ).reshape(-1, 1),
            "velocity": torch.tensor([50 + 0j, 60 + 10j, -30 + 20j, 40 + 30j]).reshape(
                -1, 1
            ),
            "speed": torch.tensor([50.0, 61.0, 36.0, 50.0]).reshape(-1, 1),
            "attitude": torch.tensor(
                [1 + 0j, 0.9 + 0.44j, -0.8 + 0.6j, 0.8 - 0.6j]
            ).reshape(-1, 1),
            "is_shooting": torch.tensor([0, 1, 0, 1]).reshape(-1, 1),
        }

    def test_all_agents_action_format(self, sample_observations):
        """Test that all agent types return consistent action formats."""
        agents = [
            create_scripted_agent((1200, 800)),
            RandomAgentProvider(rng=np.random.default_rng(seed=42)),
        ]

        # Add RL agent with mock model
        mock_model = Mock()
        mock_model.get_actions.return_value = {
            "actions": torch.zeros((1, 4, 6), dtype=torch.float32)
        }
        agents.append(create_rl_agent(mock_model, Mock(), "transformer"))

        # Add human agent with mock renderer
        mock_renderer = Mock()
        mock_renderer.update_human_actions.return_value = None
        mock_renderer.get_human_actions.return_value = {}
        agents.append(create_human_agent(mock_renderer))

        ship_ids = [0, 1]

        for agent in agents:
            actions = agent.get_actions(sample_observations, ship_ids)

            # All agents should return same format
            assert isinstance(actions, dict)
            assert set(actions.keys()) == {0, 1}

            for ship_id in ship_ids:
                action = actions[ship_id]
                assert isinstance(action, torch.Tensor)
                assert action.shape == (6,)
                assert action.dtype == torch.float32
                # Values should be valid (finite)
                assert torch.all(torch.isfinite(action))

    def test_action_value_ranges(self, sample_observations):
        """Test that action values are in expected ranges."""
        agents = [
            create_scripted_agent((1200, 800)),
            RandomAgentProvider(rng=np.random.default_rng(seed=42)),
        ]

        ship_ids = [0, 1]

        for agent in agents:
            actions = agent.get_actions(sample_observations, ship_ids)

            for ship_id in ship_ids:
                action = actions[ship_id]

                # Actions should be binary (0 or 1) or continuous [0, 1]
                assert torch.all(action >= 0.0)
                assert torch.all(action <= 1.0)

    def test_dead_ship_zero_actions(self):
        """Test that all agent types return zero actions for dead ships."""
        # Observations with one dead ship
        dead_ship_obs = {
            "tokens": torch.randn(2, 10),
            "ship_id": torch.tensor([0, 1]).reshape(-1, 1),
            "team_id": torch.tensor([0, 1]).reshape(-1, 1),
            "alive": torch.tensor([1, 0]).reshape(-1, 1),  # Ship 1 is dead
            "health": torch.tensor([80.0, 0.0]).reshape(-1, 1),
            "power": torch.tensor([60.0, 0.0]).reshape(-1, 1),
            "position": torch.tensor([100 + 200j, 150 + 250j]).reshape(-1, 1),
            "velocity": torch.tensor([50 + 0j, 0 + 0j]).reshape(-1, 1),
            "speed": torch.tensor([50.0, 0.0]).reshape(-1, 1),
            "attitude": torch.tensor([1 + 0j, 1 + 0j]).reshape(-1, 1),
            "is_shooting": torch.tensor([0, 0]).reshape(-1, 1),
        }

        agents = [
            create_scripted_agent((1200, 800)),
            RandomAgentProvider(rng=np.random.default_rng(seed=42)),
        ]

        ship_ids = [0, 1]  # Request actions for both alive and dead ship

        for agent in agents:
            actions = agent.get_actions(dead_ship_obs, ship_ids)

            # Dead ship should have zero actions
            dead_ship_actions = actions[1]
            assert torch.all(dead_ship_actions == 0.0)

            # Alive ship may have non-zero actions
            alive_ship_actions = actions[0]
            assert isinstance(alive_ship_actions, torch.Tensor)
            assert alive_ship_actions.shape == (6,)


class TestAgentErrorRecovery:
    """Tests for agent error handling and recovery."""

    def test_agent_with_corrupted_observations(self):
        """Test agent behavior with malformed observations."""
        corrupted_obs = {
            "tokens": torch.tensor([[float("nan"), 1.0, 2.0]]),  # NaN values
            "alive": torch.tensor([[1]]),
        }

        agent = RandomAgentProvider(rng=np.random.default_rng(seed=42))
        ship_ids = [0]

        # Should handle gracefully without crashing
        actions = agent.get_actions(corrupted_obs, ship_ids)

        assert 0 in actions
        assert isinstance(actions[0], torch.Tensor)
        assert actions[0].shape == (6,)

    def test_agent_with_empty_ship_ids(self):
        """Test agent behavior with empty ship ID list."""
        sample_obs = {"tokens": torch.randn(2, 10)}

        agents = [
            create_scripted_agent((1200, 800)),
            RandomAgentProvider(rng=np.random.default_rng(seed=42)),
        ]

        for agent in agents:
            actions = agent.get_actions(sample_obs, [])

            # Should return empty dict for empty ship list
            assert actions == {}

    def test_agent_with_large_ship_ids(self):
        """Test agent behavior with ship IDs beyond observation bounds."""
        sample_obs = {
            "tokens": torch.randn(2, 10),
            "alive": torch.ones((2, 1)),
        }

        agents = [
            create_scripted_agent((1200, 800)),
            RandomAgentProvider(rng=np.random.default_rng(seed=42)),
        ]

        large_ship_ids = [10, 20, 100]  # Way beyond observation size

        for agent in agents:
            actions = agent.get_actions(sample_obs, large_ship_ids)

            # Should handle gracefully
            assert set(actions.keys()) == set(large_ship_ids)

            for ship_id in large_ship_ids:
                action = actions[ship_id]
                assert isinstance(action, torch.Tensor)
                assert action.shape == (6,)
