"""
Tests for the configuration management utilities
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config import (
    load_config,
    get_default_config,
    merge_configs,
    validate_config,
    save_config,
)


class TestConfigManagement:
    """Tests for configuration management utilities."""

    def test_get_default_config(self):
        """Test getting default configurations."""
        # Test training default
        training_config = get_default_config("training")
        assert isinstance(training_config, dict)
        assert "model" in training_config
        assert "training" in training_config

        # Test data_collection default
        data_config = get_default_config("data_collection")
        assert isinstance(data_config, dict)
        assert "data_collection" in data_config

        # Test evaluation default
        eval_config = get_default_config("evaluation")
        assert isinstance(eval_config, dict)
        # Evaluation config may not have an "evaluation" key, just check it's a dict

        # Test full default
        full_config = get_default_config("full")
        assert isinstance(full_config, dict)
        assert "model" in full_config
        assert "training" in full_config

    def test_load_config_with_file(self):
        """Test loading configuration from file."""
        config_data = {
            "model": {"transformer": {"d_model": 256, "n_heads": 8}},
            "training": {"batch_size": 64},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Load with default
            loaded_config = load_config(config_path, get_default_config("training"))

            assert loaded_config["model"]["transformer"]["d_model"] == 256
            assert loaded_config["model"]["transformer"]["n_heads"] == 8
            assert loaded_config["training"]["batch_size"] == 64

            # Should have defaults merged in
            assert "bc" in loaded_config["model"]
            assert "ppo" in loaded_config["model"]

        finally:
            Path(config_path).unlink()

    def test_load_config_without_file(self):
        """Test loading configuration without file (uses defaults)."""
        default_config = get_default_config("training")
        loaded_config = load_config("", default_config)

        assert loaded_config == default_config

    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        # This should not raise FileNotFoundError, but should return default config
        loaded_config = load_config(
            "nonexistent_file.yaml", get_default_config("training")
        )
        assert loaded_config == get_default_config("training")

    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        base_config = {
            "model": {
                "transformer": {"d_model": 128, "n_heads": 4, "n_layers": 4},
                "bc": {"learning_rate": 1e-4},
            },
            "training": {"batch_size": 32},
        }

        override_config = {
            "model": {
                "transformer": {"d_model": 256, "dropout": 0.1}  # Override  # New field
            },
            "training": {"batch_size": 64},  # Override
        }

        merged = merge_configs(base_config, override_config)

        # Check overrides
        assert merged["model"]["transformer"]["d_model"] == 256
        assert merged["training"]["batch_size"] == 64

        # Check preserved values
        assert merged["model"]["transformer"]["n_heads"] == 4
        assert merged["model"]["transformer"]["n_layers"] == 4
        assert merged["model"]["bc"]["learning_rate"] == 1e-4

        # Check new values
        assert merged["model"]["transformer"]["dropout"] == 0.1

    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            "model": {
                "transformer": {
                    "token_dim": 32,
                    "embed_dim": 256,
                    "num_heads": 8,
                    "num_layers": 6,
                    "max_ships": 4,
                    "num_actions": 4,
                }
            },
            "training": {"batch_size": 64, "total_timesteps": 1000000},
        }

        # Valid config should pass
        # validate_config returns None on success, raises ValueError on failure
        validate_config(valid_config)

        # Test invalid config
        invalid_config = {
            "environment": {"world_size": [500, 500, 500]},  # Invalid world_size
        }

        with pytest.raises(
            ValueError, match="world_size must be a list of two integers"
        ):
            validate_config(invalid_config)

        # Test invalid max_ships
        invalid_config = {
            "environment": {"max_ships": 0},  # Invalid max_ships
        }

        with pytest.raises(ValueError, match="max_ships must be a positive integer"):
            validate_config(invalid_config)

        # Test missing required transformer field
        invalid_config = {
            "model": {
                "transformer": {
                    "embed_dim": 256,
                    "num_heads": 8,
                    "num_layers": 6,
                    "max_ships": 4,
                    "num_actions": 4,
                    # Missing token_dim
                }
            }
        }

        with pytest.raises(
            ValueError, match="Missing required transformer field: token_dim"
        ):
            validate_config(invalid_config)

    def test_save_config(self):
        """Test saving configuration to file."""
        config_data = {"model": {"transformer": {"d_model": 256, "n_heads": 8}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name

        try:
            save_config(config_data, config_path)

            # Verify file was created and contains correct data
            assert Path(config_path).exists()

            with open(config_path, "r") as f:
                loaded_data = yaml.safe_load(f)

            assert loaded_data == config_data

        finally:
            Path(config_path).unlink()

    def test_load_config_with_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name

        try:
            # This should not raise YAMLError, but should return default config
            loaded_config = load_config(config_path, get_default_config("training"))
            assert loaded_config == get_default_config("training")
        finally:
            Path(config_path).unlink()

    def test_get_default_config_invalid_context(self):
        """Test getting default config for invalid context."""
        # This should raise ValueError for invalid context
        with pytest.raises(ValueError, match="Invalid context 'invalid_context'"):
            get_default_config("invalid_context")
