"""
Tests for the unified command-line interface in main.py
"""

import pytest
import sys
from unittest.mock import patch, MagicMock, Mock, ANY
import argparse


class TestMainInterface:
    """Tests for the main.py unified interface."""

    @patch("argparse.ArgumentParser.print_help")
    def test_main_help(self, mock_print_help):
        """Test that main.py help works."""
        with patch("sys.argv", ["main.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from main import main

                main()

            # Should exit with code 0 for help
            assert exc_info.value.code == 0
            mock_print_help.assert_called_once()

    @patch("argparse.ArgumentParser.print_help")
    def test_train_help(self, mock_print_help):
        """Test that train command help works."""
        with patch("sys.argv", ["main.py", "train", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from main import main

                main()

            # Should exit with code 0 for help
            assert exc_info.value.code == 0

    @patch("src.pipelines.training.TrainingPipeline.add_subparsers")
    @patch("src.pipelines.data_collection.DataCollectionPipeline.add_subparsers")
    @patch("src.pipelines.playback.PlaybackPipeline.add_subparsers")
    @patch("src.pipelines.evaluation.EvaluationPipeline.add_subparsers")
    def test_parser_creation(self, mock_eval, mock_replay, mock_collect, mock_train):
        """Test that parser is created correctly with all subparsers."""
        # Mock the add_subparsers methods to return mock parsers
        mock_train.return_value = MagicMock()
        mock_collect.return_value = MagicMock()
        mock_replay.return_value = MagicMock()
        mock_eval.return_value = MagicMock()

        from main import create_parser

        parser = create_parser()

        # Verify all subparsers were added
        mock_train.assert_called_once()
        mock_collect.assert_called_once()
        mock_replay.assert_called_once()
        mock_eval.assert_called_once()

    @patch("src.pipelines.training.TrainingPipeline.execute")
    def test_train_bc_command_routing(self, mock_execute):
        """Test that train bc command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "train", "bc", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.data_collection.DataCollectionPipeline.execute")
    def test_collect_bc_command_routing(self, mock_execute):
        """Test that collect bc command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "collect", "bc", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.training.TrainingPipeline.execute")
    def test_train_rl_command_routing(self, mock_execute):
        """Test that train rl command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "train", "rl", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.training.TrainingPipeline.execute")
    def test_train_full_command_routing(self, mock_execute):
        """Test that train full command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "train", "full", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.data_collection.DataCollectionPipeline.execute")
    def test_collect_selfplay_command_routing(self, mock_execute):
        """Test that collect selfplay command is routed correctly."""
        mock_execute.return_value = 0

        with patch(
            "sys.argv", ["main.py", "collect", "selfplay", "--config", "test.yaml"]
        ):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.data_collection.PlayPipeline.execute")
    def test_play_human_command_routing(self, mock_execute):
        """Test that play human command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "play", "human", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.playback.PlaybackPipeline.execute")
    def test_replay_episode_command_routing(self, mock_execute):
        """Test that replay episode command is routed correctly."""
        mock_execute.return_value = 0

        with patch(
            "sys.argv",
            ["main.py", "replay", "episode", "--episode-file", "test.pkl.gz"],
        ):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.playback.PlaybackPipeline.execute")
    def test_replay_browse_command_routing(self, mock_execute):
        """Test that replay browse command is routed correctly."""
        mock_execute.return_value = 0

        with patch("sys.argv", ["main.py", "replay", "browse", "--data-dir", "data/"]):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("src.pipelines.evaluation.EvaluationPipeline.execute")
    def test_evaluate_model_command_routing(self, mock_execute):
        """Test that evaluate model command is routed correctly."""
        mock_execute.return_value = 0

        with patch(
            "sys.argv",
            [
                "main.py",
                "evaluate",
                "model",
                "--model",
                "test.pt",
                "--config",
                "test.yaml",
            ],
        ):
            from main import main

            result = main()

            # Should call the execute method
            mock_execute.assert_called_once()

    @patch("sys.argv", ["main.py", "invalid_command"])
    @patch("argparse.ArgumentParser.print_help")
    @patch("argparse.ArgumentParser.exit")
    def test_invalid_command(self, mock_exit, mock_print_help):
        """Test that invalid commands return error."""
        # Make exit not actually exit
        mock_exit.return_value = None

        from main import main

        result = main()

        # Should return error code
        assert result == 1
        # Check that exit was called with error code 2 (may be called multiple times)
        assert mock_exit.call_count >= 1
        assert all(call[0][0] == 2 for call in mock_exit.call_args_list)

    @patch("sys.argv", ["main.py"])
    @patch("argparse.ArgumentParser.print_help")
    def test_no_command(self, mock_print_help):
        """Test that no command shows help."""
        from main import main

        result = main()

        # Should return error code and show help
        assert result == 1
        mock_print_help.assert_called_once()

    @patch("src.pipelines.training.TrainingPipeline.execute")
    def test_train_command_error_handling(self, mock_execute):
        """Test that train command handles errors correctly."""
        mock_execute.side_effect = Exception("Test error")

        with patch("sys.argv", ["main.py", "train", "bc", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should return error code when exception occurs
            assert result == 1

    @patch("src.pipelines.training.TrainingPipeline.execute")
    def test_train_command_keyboard_interrupt(self, mock_execute):
        """Test that train command handles keyboard interrupt correctly."""
        mock_execute.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["main.py", "train", "bc", "--config", "test.yaml"]):
            from main import main

            result = main()

            # Should return error code when interrupted
            assert result == 1

    def test_argument_parsing(self):
        """Test that arguments are parsed correctly."""
        from main import create_parser

        parser = create_parser()

        # Test train bc command
        args = parser.parse_args(
            ["train", "bc", "--config", "test.yaml", "--run-name", "test_run"]
        )
        assert args.command == "train"
        assert args.train_command == "bc"  # Note: train_command, not subcommand
        assert args.config == "test.yaml"
        assert args.run_name == "test_run"

        # Test collect bc command
        args = parser.parse_args(["collect", "bc", "--config", "test.yaml"])
        assert args.command == "collect"
        assert args.collect_command == "bc"  # Note: collect_command, not subcommand
        assert args.config == "test.yaml"

        # Test replay episode command
        args = parser.parse_args(["replay", "episode", "--episode-file", "test.pkl.gz"])
        assert args.command == "replay"
        assert args.replay_command == "episode"  # Note: replay_command, not subcommand
        assert args.episode_file == "test.pkl.gz"
