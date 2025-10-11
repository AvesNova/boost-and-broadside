"""
Tests for the common utilities
"""

import pytest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.utils import (
    setup_directories,
    setup_logging,
    generate_run_name,
    save_config_copy,
    InterruptHandler,
    validate_file_exists,
    format_duration,
    ensure_dir,
)


class TestDirectorySetup:
    """Tests for directory setup utilities."""

    def test_setup_directories(self):
        """Test setting up directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = "test_run"
            checkpoint_dir, log_dir = setup_directories(base_name, base_dir=temp_dir)

            # Check directories were created
            assert checkpoint_dir.exists()
            assert log_dir.exists()

            # Check directory names
            assert checkpoint_dir.name == f"{base_name}_checkpoints"
            assert log_dir.name == f"{base_name}_logs"

            # Check they're in the right place
            assert checkpoint_dir.parent == Path(temp_dir)
            assert log_dir.parent == Path(temp_dir)

    def test_setup_directories_default_base(self):
        """Test setting up directories with default base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = "test_run"

            with patch("src.utils.Path.cwd", return_value=Path(temp_dir)):
                checkpoint_dir, log_dir = setup_directories(base_name)

            # Check directories were created in centralized directories
            assert checkpoint_dir.exists()
            assert log_dir.exists()
            assert checkpoint_dir.parent == Path(temp_dir) / "checkpoints"
            assert log_dir.parent == Path(temp_dir) / "logs"

    def test_ensure_dir_new(self):
        """Test ensuring directory exists for new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"

            # Directory shouldn't exist initially
            assert not new_dir.exists()

            # Ensure it exists
            ensure_dir(new_dir)

            # Directory should now exist
            assert new_dir.exists()

    def test_ensure_dir_existing(self):
        """Test ensuring directory exists for existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)

            # Directory should exist
            assert existing_dir.exists()

            # Ensure it exists (should not raise error)
            ensure_dir(existing_dir)

            # Directory should still exist
            assert existing_dir.exists()


class TestLoggingSetup:
    """Tests for logging setup utilities."""

    def test_setup_logging(self):
        """Test setting up logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "test_run"
            logger = setup_logging(run_name, log_dir=Path(temp_dir))

            try:
                # Check logger was created
                assert logger is not None
                assert isinstance(logger, logging.Logger)
                assert logger.name == run_name

                # Check log file was created
                log_file = Path(temp_dir) / f"{run_name}.log"
                assert log_file.exists()

                # Test logging
                logger.info("Test message")

                # Check message was written to file
                with open(log_file, "r") as f:
                    content = f.read()
                    assert "Test message" in content
            finally:
                # Properly close handlers to avoid Windows file lock issues
                if hasattr(logger, "close_handlers"):
                    logger.close_handlers()

    def test_setup_logging_default_dir(self):
        """Test setting up logging with default directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "test_run"

            with patch("src.utils.Path.cwd", return_value=Path(temp_dir)):
                logger = setup_logging(run_name)

            try:
                # Check log file was created in centralized logs directory
                log_file = Path(temp_dir) / "logs" / f"{run_name}.log"
                assert log_file.exists()
            finally:
                # Properly close handlers to avoid Windows file lock issues
                if hasattr(logger, "close_handlers"):
                    logger.close_handlers()


class TestRunNameGeneration:
    """Tests for run name generation."""

    @patch("src.utils.datetime")
    def test_generate_run_name(self, mock_datetime):
        """Test generating run names."""
        # Mock datetime to get predictable results
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20231201_120000"
        mock_datetime.now.return_value = mock_now

        # Test with prefix
        run_name = generate_run_name("test")
        assert run_name == "test_20231201_120000"

        # Test with different prefix
        run_name = generate_run_name("training")
        assert run_name == "training_20231201_120000"


class TestConfigCopy:
    """Tests for config copy utilities."""

    def test_save_config_copy(self):
        """Test saving config copy."""
        config_data = {
            "model": {"transformer": {"d_model": 256, "n_heads": 8}},
            "training": {"batch_size": 64},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_config_copy(config_data, Path(temp_dir))

            # Check config file was created
            config_file = Path(temp_dir) / "config.yaml"
            assert config_file.exists()

            # Check content
            import yaml

            with open(config_file, "r") as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config == config_data


class TestInterruptHandler:
    """Tests for interrupt handler."""

    def test_interrupt_handler_context(self):
        """Test interrupt handler as context manager."""
        with patch("signal.signal") as mock_signal:
            with InterruptHandler("Test message"):
                pass

            # Check signal handlers were set and restored
            assert mock_signal.call_count == 2  # Set and restore

    def test_interrupt_handler_with_interrupt(self):
        """Test interrupt handler with actual interrupt."""
        with patch("signal.signal") as mock_signal:
            # Mock signal to raise KeyboardInterrupt
            def mock_handler(signum, frame):
                raise KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                with InterruptHandler("Test message"):
                    # Simulate interrupt
                    mock_signal.side_effect = mock_handler


class TestFileValidation:
    """Tests for file validation utilities."""

    def test_validate_file_exists_valid(self):
        """Test validating existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = validate_file_exists(temp_path, "Test file")
            assert result == Path(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_file_exists_missing(self):
        """Test validating missing file."""
        with pytest.raises(FileNotFoundError, match="Test file not found"):
            validate_file_exists("nonexistent_file.txt", "Test file")

    def test_validate_file_exists_path_object(self):
        """Test validating file with Path object."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = validate_file_exists(temp_path, "Test file")
            assert result == temp_path
        finally:
            temp_path.unlink()


class TestProgressBar:
    """Tests for progress bar utilities."""

    def test_create_progress_bar(self):
        """Test creating progress bar."""
        from src.utils import create_progress_bar

        with patch("tqdm.tqdm") as mock_tqdm:
            iterable = range(10)
            create_progress_bar(iterable, desc="Test")

            mock_tqdm.assert_called_once_with(iterable, desc="Test")

    def test_create_progress_bar_disabled(self):
        """Test creating disabled progress bar."""
        from src.utils import create_progress_bar

        with patch("tqdm.tqdm") as mock_tqdm:
            iterable = range(10)
            create_progress_bar(iterable, desc="Test", disable=True)

            mock_tqdm.assert_called_once_with(iterable, desc="Test", disable=True)


class TestTimeFormatting:
    """Tests for time formatting utilities."""

    def test_format_time_seconds(self):
        """Test formatting time in seconds."""
        from src.utils import format_duration

        result = format_duration(45)
        assert "45.00s" in result

    def test_format_time_minutes(self):
        """Test formatting time in minutes."""
        from src.utils import format_duration

        result = format_duration(125)  # 2 minutes 5 seconds
        assert "2m" in result
        assert "5.00s" in result

    def test_format_time_hours(self):
        """Test formatting time in hours."""
        from src.utils import format_duration

        result = format_duration(3665)  # 1 hour 1 minute 5 seconds
        assert "1h" in result
        assert "1m" in result
        assert "5.00s" in result

    def test_format_time_zero(self):
        """Test formatting zero time."""
        from src.utils import format_duration

        result = format_duration(0)
        assert "0.00s" in result
