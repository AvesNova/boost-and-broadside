"""
Tests for the CLI argument parsing utilities
"""

import pytest
import argparse
from unittest.mock import patch

from src.cli_args import (
    add_config_argument,
    add_run_name_argument,
    add_output_argument,
    add_model_argument,
    add_bc_model_argument,
    add_episode_file_argument,
    add_data_dir_argument,
    add_skip_bc_argument,
    create_subparser,
    get_training_arguments,
    get_rl_training_arguments,
    get_full_training_arguments,
    get_data_collection_arguments,
    get_playback_arguments,
    get_evaluation_arguments,
    validate_args,
)


class TestCLIArgumentHelpers:
    """Tests for CLI argument helper functions."""

    def test_add_config_argument(self):
        """Test adding config argument."""
        parser = argparse.ArgumentParser()
        add_config_argument(parser)

        # Test parsing
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

        # Test default
        args = parser.parse_args([])
        assert args.config == "src/unified_training.yaml"

    def test_add_config_argument_custom_default(self):
        """Test adding config argument with custom default."""
        parser = argparse.ArgumentParser()
        add_config_argument(parser, default="custom.yaml")

        args = parser.parse_args([])
        assert args.config == "custom.yaml"

    def test_add_run_name_argument(self):
        """Test adding run-name argument."""
        parser = argparse.ArgumentParser()
        add_run_name_argument(parser)

        args = parser.parse_args(["--run-name", "test_run"])
        assert args.run_name == "test_run"

        # Test default (should be None)
        args = parser.parse_args([])
        assert args.run_name is None

    def test_add_output_argument(self):
        """Test adding output argument."""
        parser = argparse.ArgumentParser()
        add_output_argument(parser)

        args = parser.parse_args(["--output", "/tmp/output"])
        assert args.output == "/tmp/output"

    def test_add_model_argument(self):
        """Test adding model argument."""
        parser = argparse.ArgumentParser()
        add_model_argument(parser, required=True)

        args = parser.parse_args(["--model", "model.pt"])
        assert args.model == "model.pt"

    def test_add_model_argument_optional(self):
        """Test adding optional model argument."""
        parser = argparse.ArgumentParser()
        add_model_argument(parser, required=False)

        args = parser.parse_args(["--model", "model.pt"])
        assert args.model == "model.pt"

    def test_add_bc_model_argument(self):
        """Test adding BC model argument."""
        parser = argparse.ArgumentParser()
        add_bc_model_argument(parser)

        args = parser.parse_args(["--bc-model", "bc_model.pt"])
        assert args.bc_model == "bc_model.pt"

        # Test default
        args = parser.parse_args([])
        assert args.bc_model is None

    def test_add_episode_file_argument(self):
        """Test adding episode file argument."""
        parser = argparse.ArgumentParser()
        add_episode_file_argument(parser, required=True)

        args = parser.parse_args(["--episode-file", "episode.pkl"])
        assert args.episode_file == "episode.pkl"

    def test_add_data_dir_argument(self):
        """Test adding data directory argument."""
        parser = argparse.ArgumentParser()
        add_data_dir_argument(parser)

        args = parser.parse_args(["--data-dir", "/data"])
        assert args.data_dir == "/data"

        # Test default
        args = parser.parse_args([])
        assert args.data_dir == "data"

    def test_add_skip_bc_argument(self):
        """Test adding skip BC argument."""
        parser = argparse.ArgumentParser()
        add_skip_bc_argument(parser)

        # Test not set
        args = parser.parse_args([])
        assert args.skip_bc is False

        # Test set
        args = parser.parse_args(["--skip-bc"])
        assert args.skip_bc is True

    def test_create_subparser(self):
        """Test creating subparser with arguments."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        arguments = [
            {"args": ["--config"], "kwargs": {"type": str, "help": "Config file"}},
            {"args": ["--run-name"], "kwargs": {"type": str, "help": "Run name"}},
        ]

        subparser = create_subparser(subparsers, "test", "Test command", arguments)

        args = subparser.parse_args(["--config", "test.yaml", "--run-name", "test"])
        assert args.config == "test.yaml"
        assert args.run_name == "test"

    def test_get_training_arguments(self):
        """Test getting training arguments."""
        args = get_training_arguments()

        assert len(args) == 2
        assert args[0]["args"] == ["--config"]
        assert args[1]["args"] == ["--run-name"]
        assert "default" in args[0]["kwargs"]
        assert "help" in args[0]["kwargs"]

    def test_get_rl_training_arguments(self):
        """Test getting RL training arguments."""
        args = get_rl_training_arguments()

        assert len(args) == 3
        assert args[0]["args"] == ["--config"]
        assert args[1]["args"] == ["--run-name"]
        assert args[2]["args"] == ["--bc-model"]

    def test_get_full_training_arguments(self):
        """Test getting full training arguments."""
        args = get_full_training_arguments()

        assert len(args) == 3
        assert args[0]["args"] == ["--config"]
        assert args[1]["args"] == ["--run-name"]
        assert args[2]["args"] == ["--skip-bc"]
        assert args[2]["kwargs"]["action"] == "store_true"

    def test_get_data_collection_arguments(self):
        """Test getting data collection arguments."""
        args = get_data_collection_arguments()

        assert len(args) == 2
        assert args[0]["args"] == ["--config"]
        assert args[1]["args"] == ["--output"]

    def test_get_playback_arguments(self):
        """Test getting playback arguments."""
        args = get_playback_arguments()

        assert len(args) == 1
        assert args[0]["args"] == ["--config"]

    def test_get_evaluation_arguments(self):
        """Test getting evaluation arguments."""
        args = get_evaluation_arguments()

        assert len(args) == 2
        assert args[0]["args"] == ["--model"]
        assert args[0]["kwargs"]["required"] is True
        assert args[1]["args"] == ["--config"]

    @patch("pathlib.Path.exists")
    def test_validate_args_valid(self, mock_exists):
        """Test argument validation with valid arguments."""
        mock_exists.return_value = True

        parser = argparse.ArgumentParser()
        add_config_argument(parser)
        add_model_argument(parser)

        args = parser.parse_args(["--config", "test.yaml", "--model", "model.pt"])

        # Should not raise exception
        validate_args(args, "test")

    @patch("pathlib.Path.exists")
    def test_validate_args_missing_config(self, mock_exists):
        """Test argument validation with missing config file."""
        mock_exists.return_value = False

        parser = argparse.ArgumentParser()
        add_config_argument(parser)

        args = parser.parse_args(["--config", "missing.yaml"])

        with pytest.raises(ValueError, match="Config file not found"):
            validate_args(args, "test")

    @patch("pathlib.Path.exists")
    def test_validate_args_missing_model(self, mock_exists):
        """Test argument validation with missing model file."""
        mock_exists.return_value = False

        parser = argparse.ArgumentParser()
        add_model_argument(parser)

        args = parser.parse_args(["--model", "missing.pt"])

        with pytest.raises(ValueError, match="Model file not found"):
            validate_args(args, "test")

    @patch("pathlib.Path.exists")
    def test_validate_args_missing_bc_model(self, mock_exists):
        """Test argument validation with missing BC model file."""
        mock_exists.return_value = False

        parser = argparse.ArgumentParser()
        add_bc_model_argument(parser)

        args = parser.parse_args(["--bc-model", "missing.pt"])

        with pytest.raises(ValueError, match="BC model file not found"):
            validate_args(args, "test")

    @patch("pathlib.Path.exists")
    def test_validate_args_missing_episode_file(self, mock_exists):
        """Test argument validation with missing episode file."""
        mock_exists.return_value = False

        parser = argparse.ArgumentParser()
        add_episode_file_argument(parser)

        args = parser.parse_args(["--episode-file", "missing.pkl"])

        with pytest.raises(ValueError, match="Episode file not found"):
            validate_args(args, "test")

    @patch("pathlib.Path.exists")
    def test_validate_args_missing_data_dir(self, mock_exists):
        """Test argument validation with missing data directory."""
        mock_exists.return_value = False

        parser = argparse.ArgumentParser()
        add_data_dir_argument(parser)

        args = parser.parse_args(["--data-dir", "missing_dir"])

        with pytest.raises(ValueError, match="Data directory not found"):
            validate_args(args, "test")

    def test_validate_args_no_attributes(self):
        """Test argument validation with no relevant attributes."""
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])

        # Should not raise exception
        validate_args(args, "test")
