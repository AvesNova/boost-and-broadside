"""
Tests for the pipeline modules
"""

import pytest
import argparse
from unittest.mock import patch, MagicMock

from src.pipelines.training import TrainingPipeline
from src.pipelines.data_collection import DataCollectionPipeline, PlayPipeline
from src.pipelines.playback import PlaybackPipeline
from src.pipelines.evaluation import EvaluationPipeline


class TestTrainingPipeline:
    """Tests for the training pipeline."""

    def test_add_subparsers(self):
        """Test adding training subparsers."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        train_parser = TrainingPipeline.add_subparsers(subparsers)

        # Check that subparser was created
        assert train_parser is not None
        # The prog name will be 'pytest train' when running tests
        assert train_parser.prog.endswith(" train")

    @patch("src.pipelines.training.TrainingPipeline._train_bc")
    def test_execute_bc(self, mock_train_bc):
        """Test executing BC training."""
        mock_train_bc.return_value = 0

        args = argparse.Namespace()
        args.train_command = "bc"

        result = TrainingPipeline.execute(args)

        assert result == 0
        mock_train_bc.assert_called_once_with(args)

    @patch("src.pipelines.training.TrainingPipeline._train_rl")
    def test_execute_rl(self, mock_train_rl):
        """Test executing RL training."""
        mock_train_rl.return_value = 0

        args = argparse.Namespace()
        args.train_command = "rl"

        result = TrainingPipeline.execute(args)

        assert result == 0
        mock_train_rl.assert_called_once_with(args)

    @patch("src.pipelines.training.TrainingPipeline._train_full")
    def test_execute_full(self, mock_train_full):
        """Test executing full training."""
        mock_train_full.return_value = 0

        args = argparse.Namespace()
        args.train_command = "full"

        result = TrainingPipeline.execute(args)

        assert result == 0
        mock_train_full.assert_called_once_with(args)

    def test_execute_invalid_command(self):
        """Test executing invalid training command."""
        args = argparse.Namespace()
        args.train_command = "invalid"

        result = TrainingPipeline.execute(args)

        assert result == 1

    @patch("src.pipelines.training.load_config")
    @patch("src.pipelines.training.setup_logging")
    @patch("src.pipelines.training.setup_directories")
    @patch("src.pipelines.training.train_bc_model")
    def test_train_bc(
        self, mock_train_bc, mock_setup_dirs, mock_setup_logging, mock_load_config
    ):
        """Test BC training implementation."""
        # Mock dependencies
        mock_config = {
            "data_collection": {"bc_data": {"output_dir": "test_data"}},
            "model": {"transformer": {"d_model": 256}, "bc": {"learning_rate": 1e-4}},
        }
        mock_load_config.return_value = mock_config

        mock_setup_dirs.return_value = (MagicMock(), MagicMock())
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        mock_model = MagicMock()

        # Mock data files
        with patch("src.pipelines.training.Path") as mock_path:
            mock_path.return_value.glob.return_value = ["file1.pkl", "file2.pkl"]

            with patch(
                "src.pipelines.training.create_bc_model", return_value=mock_model
            ):
                with patch("src.pipelines.training.save_model"):
                    args = argparse.Namespace(config="test.yaml", run_name="test_run")

                    result = TrainingPipeline._train_bc(args)

                    assert result == 0

    @patch("src.pipelines.training.load_config")
    def test_train_bc_no_data(self, mock_load_config):
        """Test BC training with no data."""
        mock_config = {"data_collection": {"bc_data": {"output_dir": "test_data"}}}
        mock_load_config.return_value = mock_config

        # Mock no data files
        with patch("src.pipelines.training.Path") as mock_path:
            mock_path.return_value.glob.return_value = []

            args = argparse.Namespace(config="test.yaml", run_name="test_run")

            result = TrainingPipeline._train_bc(args)

            assert result == 1


class TestDataCollectionPipeline:
    """Tests for the data collection pipeline."""

    def test_add_subparsers(self):
        """Test adding data collection subparsers."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        collect_parser = DataCollectionPipeline.add_subparsers(subparsers)

        # Check that subparser was created
        assert collect_parser is not None
        # The prog name will be 'pytest collect' when running tests
        assert collect_parser.prog.endswith(" collect")

    @patch("src.pipelines.data_collection.DataCollectionPipeline._collect_bc")
    def test_execute_bc(self, mock_collect_bc):
        """Test executing BC data collection."""
        mock_collect_bc.return_value = 0

        args = argparse.Namespace()
        args.collect_command = "bc"

        result = DataCollectionPipeline.execute(args)

        assert result == 0
        mock_collect_bc.assert_called_once_with(args)

    @patch("src.pipelines.data_collection.DataCollectionPipeline._collect_selfplay")
    def test_execute_selfplay(self, mock_collect_selfplay):
        """Test executing self-play data collection."""
        mock_collect_selfplay.return_value = 0

        args = argparse.Namespace()
        args.collect_command = "selfplay"

        result = DataCollectionPipeline.execute(args)

        assert result == 0
        mock_collect_selfplay.assert_called_once_with(args)

    def test_execute_invalid_command(self):
        """Test executing invalid collection command."""
        args = argparse.Namespace()
        args.collect_command = "invalid"

        result = DataCollectionPipeline.execute(args)

        assert result == 1

    @patch("src.pipelines.data_collection.load_config")
    @patch("src.pipelines.data_collection.setup_logging")
    @patch("src.pipelines.data_collection.collect_bc_data")
    def test_collect_bc(
        self, mock_collect_bc_data, mock_setup_logging, mock_load_config
    ):
        """Test BC data collection implementation."""
        mock_config = {"data_collection": {"bc_data": {"output_dir": "test_data"}}}
        mock_load_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        args = argparse.Namespace(config="test.yaml", output="custom_output")

        result = DataCollectionPipeline._collect_bc(args)

        assert result == 0
        mock_collect_bc_data.assert_called_once()


class TestPlayPipeline:
    """Tests for the play pipeline."""

    def test_add_subparsers(self):
        """Test adding play subparsers."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        play_parser = PlayPipeline.add_subparsers(subparsers)

        # Check that subparser was created
        assert play_parser is not None
        # The prog name will be 'pytest play' when running tests
        assert play_parser.prog.endswith(" play")

    @patch("src.pipelines.data_collection.run_human_play")
    def test_execute_human(self, mock_run_human_play):
        """Test executing human play."""
        mock_run_human_play.return_value = None

        args = argparse.Namespace()
        args.play_command = "human"
        args.config = None  # Add the missing config attribute

        result = PlayPipeline.execute(args)

        assert result == 0
        mock_run_human_play.assert_called_once()

    def test_execute_invalid_command(self):
        """Test executing invalid play command."""
        args = argparse.Namespace()
        args.play_command = "invalid"

        result = PlayPipeline.execute(args)

        assert result == 1

    @patch("src.pipelines.data_collection.load_config")
    @patch("src.pipelines.data_collection.setup_logging")
    @patch("src.pipelines.data_collection.run_human_play")
    def test_play_human(
        self, mock_run_human_play, mock_setup_logging, mock_load_config
    ):
        """Test human play implementation."""
        mock_config = {"environment": {"world_size": (800, 600)}}
        mock_load_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        args = argparse.Namespace(config="test.yaml")

        result = PlayPipeline._play_human(args)

        assert result == 0
        mock_run_human_play.assert_called_once()


class TestPlaybackPipeline:
    """Tests for the playback pipeline."""

    def test_add_subparsers(self):
        """Test adding replay subparsers."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        replay_parser = PlaybackPipeline.add_subparsers(subparsers)

        # Check that subparser was created
        assert replay_parser is not None
        # The prog name will be 'pytest replay' when running tests
        assert replay_parser.prog.endswith(" replay")

    @patch("src.pipelines.playback.PlaybackPipeline._replay_episode")
    def test_execute_episode(self, mock_replay_episode):
        """Test executing episode replay."""
        mock_replay_episode.return_value = 0

        args = argparse.Namespace()
        args.replay_command = "episode"

        result = PlaybackPipeline.execute(args)

        assert result == 0
        mock_replay_episode.assert_called_once_with(args)

    @patch("src.pipelines.playback.PlaybackPipeline._browse_episodes")
    def test_execute_browse(self, mock_browse_episodes):
        """Test executing episode browsing."""
        mock_browse_episodes.return_value = 0

        args = argparse.Namespace()
        args.replay_command = "browse"

        result = PlaybackPipeline.execute(args)

        assert result == 0
        mock_browse_episodes.assert_called_once_with(args)

    def test_execute_invalid_command(self):
        """Test executing invalid replay command."""
        args = argparse.Namespace()
        args.replay_command = "invalid"

        result = PlaybackPipeline.execute(args)

        assert result == 1

    @patch("src.pipelines.playback.validate_file_exists")
    @patch("src.pipelines.playback.load_config")
    @patch("src.pipelines.playback.setup_logging")
    @patch("src.pipelines.playback.run_playback")
    def test_replay_episode(
        self, mock_run_playback, mock_setup_logging, mock_load_config, mock_validate
    ):
        """Test episode replay implementation."""
        mock_validate.return_value = MagicMock()
        mock_config = {"environment": {"world_size": (800, 600)}}
        mock_load_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        args = argparse.Namespace(episode_file="test_episode.pkl", config="test.yaml")

        result = PlaybackPipeline._replay_episode(args)

        assert result == 0
        mock_run_playback.assert_called_once()


class TestEvaluationPipeline:
    """Tests for the evaluation pipeline."""

    def test_add_subparsers(self):
        """Test adding evaluation subparsers."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        eval_parser = EvaluationPipeline.add_subparsers(subparsers)

        # Check that subparser was created
        assert eval_parser is not None
        # The prog name will be 'pytest evaluate' when running tests
        assert eval_parser.prog.endswith(" evaluate")

    @patch("src.pipelines.evaluation.EvaluationPipeline._evaluate_model")
    def test_execute_model(self, mock_evaluate_model):
        """Test executing model evaluation."""
        mock_evaluate_model.return_value = 0

        args = argparse.Namespace()
        args.evaluate_command = "model"

        result = EvaluationPipeline.execute(args)

        assert result == 0
        mock_evaluate_model.assert_called_once_with(args)

    def test_execute_invalid_command(self):
        """Test executing invalid evaluation command."""
        args = argparse.Namespace()
        args.evaluate_command = "invalid"

        result = EvaluationPipeline.execute(args)

        assert result == 1

    @patch("src.pipelines.evaluation.validate_file_exists")
    @patch("src.pipelines.evaluation.load_config")
    @patch("src.pipelines.evaluation.setup_logging")
    @patch("src.pipelines.evaluation.evaluate_model")
    def test_evaluate_model(
        self, mock_evaluate_model, mock_setup_logging, mock_load_config, mock_validate
    ):
        """Test model evaluation implementation."""
        mock_validate.return_value = MagicMock()
        mock_config = {"environment": {"world_size": (800, 600)}}
        mock_load_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        mock_stats = {"win_rate": 0.75, "avg_score": 10.5}
        mock_evaluate_model.return_value = mock_stats

        args = argparse.Namespace(model="test_model.pt", config="test.yaml")

        result = EvaluationPipeline._evaluate_model(args)

        assert result == 0
        mock_evaluate_model.assert_called_once()
