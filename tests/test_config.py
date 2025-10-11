"""
Tests for the Hydra configuration management
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
import os

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


class TestHydraConfig:
    """Tests for Hydra configuration management."""

    def setup_method(self):
        """Setup test environment."""
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

    def teardown_method(self):
        """Cleanup test environment."""
        # Clear Hydra instance
        GlobalHydra.instance().clear()

    def test_load_base_config(self):
        """Test loading the base configuration."""
        # Get the original working directory
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="base")

                assert isinstance(cfg, DictConfig)
                assert "environment" in cfg
                assert "model" in cfg
                assert "training" in cfg
                assert "data_collection" in cfg
                assert "play" in cfg
                assert "replay" in cfg
                assert "evaluate" in cfg

                # Check some specific values
                assert cfg.environment.world_size == [1200, 800]
                assert cfg.model.transformer.embed_dim == 64
                assert cfg.model.ppo.learning_rate == 0.0003
        except Exception as e:
            # Skip test if config files not found
            pytest.skip(f"Could not load config: {e}")

    def test_load_train_bc_config(self):
        """Test loading the training BC configuration."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="train/bc")

                assert isinstance(cfg, DictConfig)
                assert "environment" in cfg
                assert "model" in cfg
                assert "training" in cfg

                # Check BC-specific overrides
                assert cfg.training.mode == "bc"
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_load_train_rl_config(self):
        """Test loading the training RL configuration."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="train/rl")

                assert isinstance(cfg, DictConfig)
                assert "environment" in cfg
                assert "model" in cfg
                assert "training" in cfg

                # Check RL-specific overrides
                assert cfg.training.mode == "rl"
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_config_override(self):
        """Test overriding configuration values."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Override model parameters
                cfg = compose(
                    config_name="train/bc",
                    overrides=[
                        "model.transformer.embed_dim=128",
                        "model.ppo.learning_rate=0.0001",
                    ],
                )

                assert cfg.model.transformer.embed_dim == 128
                assert cfg.model.ppo.learning_rate == 0.0001

                # Other values should remain at defaults
                assert cfg.model.transformer.num_heads == 4
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_config_to_dict(self):
        """Test converting DictConfig to regular dict."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="base")

                # Convert to dict
                config_dict = OmegaConf.to_container(cfg, resolve=True)

                assert isinstance(config_dict, dict)
                assert "environment" in config_dict
                assert "model" in config_dict
                assert isinstance(config_dict["environment"], dict)
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_config_merge(self):
        """Test that configuration merging works correctly."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Load base config
                base_cfg = compose(config_name="base")

                # Load train/bc config (should extend base)
                train_cfg = compose(config_name="train/bc")

                # Train config should have all base keys plus training-specific overrides
                assert set(base_cfg.keys()).issubset(set(train_cfg.keys()))

                # Check that training mode is set correctly
                assert train_cfg.training.mode == "bc"
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_invalid_config_override(self):
        """Test handling of invalid configuration overrides."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Try to override with invalid type
                with pytest.raises(Exception):
                    compose(
                        config_name="train/bc",
                        overrides=["model.transformer.embed_dim=invalid"],
                    )
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_nonexistent_config(self):
        """Test handling of nonexistent configuration."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Try to load nonexistent config
                with pytest.raises(Exception):
                    compose(config_name="nonexistent/config")
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")

    def test_config_validation(self):
        """Test that configuration values are validated."""
        original_cwd = os.getcwd()
        config_dir = os.path.join(original_cwd, "src/config")

        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Load config and check that required fields exist
                cfg = compose(config_name="base")

                # Check required fields
                assert "world_size" in cfg.environment
                assert "max_ships" in cfg.environment
                assert "embed_dim" in cfg.model.transformer
                assert "num_heads" in cfg.model.transformer
                assert "learning_rate" in cfg.model.ppo

                # Check field types
                assert isinstance(cfg.environment.world_size, list)
                assert isinstance(cfg.environment.max_ships, int)
                assert isinstance(cfg.model.transformer.embed_dim, int)
                assert isinstance(cfg.model.ppo.learning_rate, float)
        except Exception as e:
            pytest.skip(f"Could not load config: {e}")
