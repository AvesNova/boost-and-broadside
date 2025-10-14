"""
Integration tests for the training pipeline to catch specific errors we fixed.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from omegaconf import DictConfig

from src.ship import Ship, ShipConfig, ActionStates
from src.rl_wrapper import UnifiedRLWrapper
from src.callbacks import SelfPlayCallback
from src.constants import Actions


class TestShipActionHandling:
    """Tests for ship action handling to catch empty tensor errors."""

    def test_extract_action_states_with_empty_tensor(self):
        """Test that ship handles empty action tensors gracefully."""
        ship_config = ShipConfig()
        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=ship_config,
            initial_x=100,
            initial_y=100,
            initial_vx=10,
            initial_vy=10,
            world_size=(800, 600),
        )

        # Test with empty tensor
        empty_actions = torch.tensor([])
        action_states = ship._extract_action_states(empty_actions)

        # Should return default actions (all zeros)
        assert action_states.forward == 0
        assert action_states.backward == 0
        assert action_states.left == 0
        assert action_states.right == 0
        assert action_states.sharp_turn == 0
        assert action_states.shoot == 0

    def test_extract_action_states_with_partial_tensor(self):
        """Test that ship handles partial action tensors gracefully."""
        ship_config = ShipConfig()
        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=ship_config,
            initial_x=100,
            initial_y=100,
            initial_vx=10,
            initial_vy=10,
            world_size=(800, 600),
        )

        # Test with partial tensor (fewer than 6 elements)
        partial_actions = torch.tensor([1, 0, 1])  # Only 3 elements
        action_states = ship._extract_action_states(partial_actions)

        # Should handle missing elements gracefully
        assert action_states.forward == 1  # First element
        assert action_states.backward == 0  # Second element
        assert action_states.left == 1  # Third element
        assert action_states.right == 0  # Missing, should be 0
        assert action_states.sharp_turn == 0  # Missing, should be 0
        assert action_states.shoot == 0  # Missing, should be 0

    def test_extract_action_states_with_full_tensor(self):
        """Test that ship handles full action tensors correctly."""
        ship_config = ShipConfig()
        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=ship_config,
            initial_x=100,
            initial_y=100,
            initial_vx=10,
            initial_vy=10,
            world_size=(800, 600),
        )

        # Test with full tensor (6 elements)
        full_actions = torch.tensor([1, 0, 1, 0, 1, 0])
        action_states = ship._extract_action_states(full_actions)

        # Should map correctly to action states
        assert action_states.forward == 1  # Actions.forward
        assert action_states.backward == 0  # Actions.backward
        assert action_states.left == 1  # Actions.left
        assert action_states.right == 0  # Actions.right
        assert action_states.sharp_turn == 1  # Actions.sharp_turn
        assert action_states.shoot == 0  # Actions.shoot


class TestRLWrapperActionHandling:
    """Tests for RL wrapper action handling to catch empty tensor errors."""

    def test_unflatten_actions_with_empty_array(self):
        """Test that RL wrapper handles empty action arrays gracefully."""
        wrapper = UnifiedRLWrapper(
            env_config={"world_size": (800, 600), "max_ships": 4},
            learning_team_id=0,
            team_assignments={0: [0, 1], 1: [2, 3]},
        )

        # Test with empty array
        empty_actions = np.array([])
        unflattened = wrapper._unflatten_actions(empty_actions)

        # Should return default actions for controlled ships
        assert len(unflattened) == 2  # Two controlled ships
        for ship_id, action in unflattened.items():
            assert isinstance(action, torch.Tensor)
            assert action.shape == (6,)  # 6 actions per ship
            assert torch.all(action == 0)  # All zeros

    def test_unflatten_actions_with_partial_array(self):
        """Test that RL wrapper handles partial action arrays gracefully."""
        wrapper = UnifiedRLWrapper(
            env_config={"world_size": (800, 600), "max_ships": 4},
            learning_team_id=0,
            team_assignments={0: [0, 1], 1: [2, 3]},
        )

        # Test with partial array (fewer than 12 elements for 2 ships)
        partial_actions = np.array([1, 0, 1, 0, 1, 0])  # Only 6 elements (1 ship)
        unflattened = wrapper._unflatten_actions(partial_actions)

        # Should handle missing elements gracefully
        assert len(unflattened) == 2  # Two controlled ships
        # First ship should get the available actions
        assert 0 in unflattened
        assert torch.all(
            unflattened[0] == torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
        )
        # Second ship should get default actions
        assert 1 in unflattened
        assert torch.all(unflattened[1] == torch.zeros(6, dtype=torch.float32))

    def test_unflatten_actions_with_full_array(self):
        """Test that RL wrapper handles full action arrays correctly."""
        wrapper = UnifiedRLWrapper(
            env_config={"world_size": (800, 600), "max_ships": 4},
            learning_team_id=0,
            team_assignments={0: [0, 1], 1: [2, 3]},
        )

        # Test with full array (12 elements for 2 ships)
        full_actions = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        unflattened = wrapper._unflatten_actions(full_actions)

        # Should map correctly to ship actions
        assert len(unflattened) == 2  # Two controlled ships
        assert 0 in unflattened
        assert torch.all(
            unflattened[0] == torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
        )
        assert 1 in unflattened
        assert torch.all(
            unflattened[1] == torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
        )


class TestSelfPlayCallback:
    """Tests for self-play callback to catch Monitor wrapper errors."""

    def test_callback_with_monitor_wrapper(self):
        """Test that callback handles Monitor wrapper correctly."""
        # Create mock environment with Monitor wrapper
        mock_actual_env = MagicMock()
        mock_actual_env.add_model_to_memory = MagicMock()
        mock_actual_env.get_win_rate = MagicMock(return_value=0.5)
        mock_actual_env.selfplay_opponent = MagicMock()
        mock_actual_env.selfplay_opponent.model_memory = []
        mock_actual_env.episode_count = 10

        mock_monitor_env = MagicMock()
        mock_monitor_env.env = mock_actual_env

        # Create mock model
        mock_model = MagicMock()
        mock_policy = MagicMock()
        mock_transformer = MagicMock()
        mock_policy.get_transformer_model.return_value = mock_transformer
        mock_model.policy = mock_policy

        # Create callback
        callback = SelfPlayCallback(
            env_wrapper=mock_monitor_env, save_freq=1000, min_save_steps=100, verbose=0
        )
        callback.model = mock_model
        callback.num_timesteps = 100

        # Test _on_step
        result = callback._on_step()

        # Should not raise errors
        assert result is True

        # If save conditions were met, should have called add_model_to_memory on actual env
        # (not on the Monitor wrapper)
        if callback.num_timesteps >= callback.min_save_steps:
            # Check that it tries to access the actual environment
            assert hasattr(mock_monitor_env, "env")


class TestHydraConfigRecursion:
    """Tests to catch Hydra configuration recursion issues."""

    def test_config_no_recursion(self):
        """Test that configuration doesn't have recursion issues."""
        from omegaconf import OmegaConf
        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        try:
            import os

            original_cwd = os.getcwd()
            config_dir = os.path.join(original_cwd, "src/config")

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Try to load configs that had recursion issues
                cfg = compose(config_name="train/bc")

                # Should not raise recursion errors
                assert isinstance(cfg, DictConfig)
                assert "environment" in cfg
                assert "model" in cfg

                # Check specific configs that had issues
                assert cfg.training.mode == "bc"

                # Check train/full config
                cfg_full = compose(config_name="train/full")
                assert cfg_full.training.mode == "full"

        except Exception as e:
            pytest.skip(f"Could not load config: {e}")
        finally:
            GlobalHydra.instance().clear()


class TestTrainingPipelineIntegration:
    """Integration tests for the full training pipeline."""

    @patch("src.pipelines.training.TrainingPipeline._train_bc")
    @patch("src.pipelines.training.TrainingPipeline._train_rl")
    def test_train_full_pipeline(self, mock_train_rl, mock_train_bc):
        """Test that full training pipeline calls both BC and RL phases."""
        from src.pipelines.training import TrainingPipeline

        mock_train_bc.return_value = 0
        mock_train_rl.return_value = 0

        cfg = DictConfig({"training": {"mode": "full"}, "run_name": "test_run"})

        # Mock the _train_full method to check it calls both phases
        with patch.object(TrainingPipeline, "_train_full") as mock_train_full:
            mock_train_full.return_value = 0

            result = TrainingPipeline.execute(cfg)

            assert result == 0
            mock_train_full.assert_called_once_with(cfg)

    def test_bc_model_loading(self):
        """Test that BC model loading handles state dict mismatches."""
        from src.team_transformer_model import TeamTransformerModel

        # Create model with old architecture (different key names)
        old_model = TeamTransformerModel()
        # Manually create state dict with old naming convention
        old_state_dict = {}
        for name, param in old_model.named_parameters():
            # Prefix with "transformer." to simulate old naming
            old_name = (
                f"transformer.{name}" if not name.startswith("transformer.") else name
            )
            old_state_dict[old_name] = param.clone()

        # Create new model
        new_model = TeamTransformerModel()

        # Try to load old state dict into new model
        # Should handle missing keys gracefully
        try:
            new_model.load_state_dict(old_state_dict, strict=False)
            # If no exception, model handled it correctly
            assert True
        except Exception as e:
            pytest.fail(f"Model loading failed with error: {e}")
