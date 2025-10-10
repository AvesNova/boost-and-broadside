"""
Tests for recovery manager utilities
"""

import pytest
import tempfile
import shutil
import pickle
import gzip
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.recovery_manager import (
    RecoveryManager,
    auto_resume_collection,
    auto_resume_training,
)


class TestRecoveryManager:
    """Tests for the RecoveryManager class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def recovery_manager(self, temp_dir):
        """Create a RecoveryManager instance"""
        return RecoveryManager(temp_dir)

    def test_recovery_manager_initialization(self, temp_dir):
        """Test RecoveryManager initialization"""
        manager = RecoveryManager(temp_dir)
        assert manager.output_dir == temp_dir
        assert manager.logger is not None

    def test_save_and_load_collection_state(self, recovery_manager, temp_dir):
        """Test saving and loading collection state"""
        state = {
            "collected_episodes": 50,
            "episodes_to_collect": 100,
            "current_mode": "2v2",
        }

        # Save state
        recovery_manager.save_collection_state(state)

        # Check file exists
        state_path = temp_dir / "collection_state.pkl"
        assert state_path.exists()

        # Load state
        loaded_state = recovery_manager.load_collection_state()

        assert loaded_state["collected_episodes"] == 50
        assert loaded_state["episodes_to_collect"] == 100
        assert loaded_state["current_mode"] == "2v2"
        assert "last_saved" in loaded_state

    def test_load_nonexistent_collection_state(self, recovery_manager):
        """Test loading non-existent collection state"""
        state = recovery_manager.load_collection_state()
        assert state is None

    def test_save_and_load_training_state(self, recovery_manager, temp_dir):
        """Test saving and loading training state"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save training state
        recovery_manager.save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=10,
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.9],
            best_val_loss=0.9,
            config={"learning_rate": 0.001},
        )

        # Check file exists
        state_path = temp_dir / "training_state.pkl"
        assert state_path.exists()

        # Create new model and optimizer for loading
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Load training state
        loaded_state = recovery_manager.load_training_state(new_model, new_optimizer)

        assert loaded_state is not None
        assert loaded_state["epoch"] == 10
        assert loaded_state["train_losses"] == [1.0, 0.9, 0.8]
        assert loaded_state["val_losses"] == [1.1, 1.0, 0.9]
        assert loaded_state["best_val_loss"] == 0.9
        assert loaded_state["config"]["learning_rate"] == 0.001

        # Check model parameters are loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_load_nonexistent_training_state(self, recovery_manager, temp_dir):
        """Test loading non-existent training state"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        state = recovery_manager.load_training_state(model, optimizer)
        assert state is None

    def test_find_latest_checkpoint(self, recovery_manager, temp_dir):
        """Test finding the latest checkpoint"""
        # Create some checkpoint files
        checkpoint1 = temp_dir / "checkpoint_100.pt"
        checkpoint2 = temp_dir / "checkpoint_200.pt"
        checkpoint3 = temp_dir / "best_model.pt"

        # Write dummy data
        checkpoint1.write_text("dummy1")
        checkpoint2.write_text("dummy2")
        checkpoint3.write_text("dummy3")

        # Make checkpoint2 the newest
        import time

        time.sleep(0.1)
        checkpoint2.touch()

        # Find latest checkpoint
        latest = recovery_manager.find_latest_checkpoint()

        assert latest == checkpoint2

    def test_find_latest_checkpoint_none(self, recovery_manager, temp_dir):
        """Test finding latest checkpoint when none exist"""
        latest = recovery_manager.find_latest_checkpoint()
        assert latest is None

    def test_find_latest_checkpoint_pattern(self, recovery_manager, temp_dir):
        """Test finding latest checkpoint with specific pattern"""
        # Create some checkpoint files
        checkpoint1 = temp_dir / "checkpoint_100.pt"
        checkpoint2 = temp_dir / "model_200.pt"
        checkpoint3 = temp_dir / "best_checkpoint_300.pt"

        # Write dummy data
        checkpoint1.write_text("dummy1")
        checkpoint2.write_text("dummy2")
        checkpoint3.write_text("dummy3")

        # Make checkpoint3 the newest
        import time

        time.sleep(0.1)
        checkpoint3.touch()

        # Find latest checkpoint with specific pattern
        latest = recovery_manager.find_latest_checkpoint(pattern="*checkpoint*.pt")

        assert latest == checkpoint3

    def test_resume_bc_collection(self, recovery_manager, temp_dir):
        """Test resuming BC data collection"""
        # Save a valid collection state
        state = {
            "collected_episodes": 50,
            "episodes_to_collect": 100,
            "current_mode": "2v2",
        }
        recovery_manager.save_collection_state(state)

        # Try to resume
        config = {"test": "config"}
        resumed_state = recovery_manager.resume_bc_collection(config)

        assert resumed_state is not None
        assert resumed_state["collected_episodes"] == 50
        assert resumed_state["episodes_to_collect"] == 100

    def test_resume_bc_collection_completed(self, recovery_manager, temp_dir):
        """Test resuming BC collection when already completed"""
        # Save a completed collection state
        state = {
            "collected_episodes": 100,
            "episodes_to_collect": 100,
            "current_mode": "2v2",
        }
        recovery_manager.save_collection_state(state)

        # Try to resume
        config = {"test": "config"}
        resumed_state = recovery_manager.resume_bc_collection(config)

        assert resumed_state is None  # Already completed

    def test_resume_bc_collection_invalid_state(self, recovery_manager, temp_dir):
        """Test resuming BC collection with invalid state"""
        # Save an invalid collection state
        state = {"invalid": "state"}
        recovery_manager.save_collection_state(state)

        # Try to resume
        config = {"test": "config"}
        resumed_state = recovery_manager.resume_bc_collection(config)

        assert resumed_state is None  # Invalid state

    def test_resume_bc_collection_no_state(self, recovery_manager, temp_dir):
        """Test resuming BC collection with no state"""
        # Try to resume without saving state
        config = {"test": "config"}
        resumed_state = recovery_manager.resume_bc_collection(config)

        assert resumed_state is None  # No state to resume from

    def test_resume_rl_training(self, recovery_manager, temp_dir):
        """Test resuming RL training"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save training state
        recovery_manager.save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=10,
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.9],
            best_val_loss=0.9,
            config={"learning_rate": 0.001},
        )

        # Create new model and optimizer for loading
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Try to resume
        config = {"epochs": 50}
        resumed_state = recovery_manager.resume_rl_training(
            new_model, new_optimizer, config
        )

        assert resumed_state is not None
        assert resumed_state["epoch"] == 10
        assert resumed_state["train_losses"] == [1.0, 0.9, 0.8]

    def test_resume_rl_training_completed(self, recovery_manager, temp_dir):
        """Test resuming RL training when already completed"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save completed training state
        recovery_manager.save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=50,
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.9],
            best_val_loss=0.9,
            config={"learning_rate": 0.001},
        )

        # Create new model and optimizer for loading
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Try to resume
        config = {"epochs": 50}
        resumed_state = recovery_manager.resume_rl_training(
            new_model, new_optimizer, config
        )

        assert resumed_state is None  # Already completed

    def test_resume_rl_training_no_state(self, recovery_manager, temp_dir):
        """Test resuming RL training with no state"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Try to resume without saving state
        config = {"epochs": 50}
        resumed_state = recovery_manager.resume_rl_training(model, optimizer, config)

        assert resumed_state is None  # No state to resume from

    def test_create_recovery_report(self, recovery_manager, temp_dir):
        """Test creating a recovery report"""
        # Create some mock files
        collection_state = {
            "collected_episodes": 50,
            "episodes_to_collect": 100,
            "last_saved": "2023-01-01T00:00:00",
        }
        recovery_manager.save_collection_state(collection_state)

        # Create a checkpoint
        checkpoint = temp_dir / "checkpoint_100.pt"
        checkpoint.write_text("dummy")

        # Create worker directories
        worker0 = temp_dir / "worker_0"
        worker1 = temp_dir / "worker_1"
        worker0.mkdir()
        worker1.mkdir()

        # Create report
        report = recovery_manager.create_recovery_report()

        # Check report content
        assert report["output_dir"] == str(temp_dir)
        assert report["collection_state"]["found"] is True
        assert report["training_state"]["found"] is False
        assert len(report["checkpoints"]) == 1
        assert len(report["worker_data"]) == 2
        assert "generated_at" in report

        # Check report file was saved
        report_path = temp_dir / "recovery_report.yaml"
        assert report_path.exists()

        with open(report_path, "r") as f:
            saved_report = yaml.safe_load(f)

        assert saved_report["output_dir"] == str(temp_dir)


class TestAutoResumeFunctions:
    """Tests for the auto-resume convenience functions"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_auto_resume_collection(self, temp_dir):
        """Test auto_resume_collection function"""
        # Create a recovery manager and save state
        manager = RecoveryManager(temp_dir)
        state = {
            "collected_episodes": 50,
            "episodes_to_collect": 100,
            "current_mode": "2v2",
        }
        manager.save_collection_state(state)

        # Try auto-resume
        config = {"test": "config"}
        resumed_state = auto_resume_collection(config, temp_dir)

        assert resumed_state is not None
        assert resumed_state["collected_episodes"] == 50

    def test_auto_resume_collection_no_state(self, temp_dir):
        """Test auto_resume_collection with no state"""
        config = {"test": "config"}
        resumed_state = auto_resume_collection(config, temp_dir)

        assert resumed_state is None

    def test_auto_resume_collection_with_workers(self, temp_dir):
        """Test auto_resume_collection with worker data"""
        # Create worker directories
        worker0 = temp_dir / "worker_0"
        worker1 = temp_dir / "worker_1"
        worker0.mkdir()
        worker1.mkdir()

        # Try auto-resume with patch to suppress print
        with patch("builtins.print") as mock_print:
            resumed_state = auto_resume_collection({"test": "config"}, temp_dir)

        assert resumed_state is None
        mock_print.assert_called_with(
            "Use data_aggregator.aggregate_worker_data() to combine the data"
        )

    def test_auto_resume_training(self, temp_dir):
        """Test auto_resume_training function"""
        # Create a recovery manager and save state
        manager = RecoveryManager(temp_dir)

        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        manager.save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=10,
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.9],
            best_val_loss=0.9,
            config={"learning_rate": 0.001},
        )

        # Create new model and optimizer for loading
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Try auto-resume
        config = {"epochs": 50}
        with patch("builtins.print") as mock_print:
            resumed_state = auto_resume_training(
                new_model, new_optimizer, config, temp_dir
            )

        assert resumed_state is not None
        mock_print.assert_called_with("Resuming training from epoch 10")

    def test_auto_resume_training_no_state(self, temp_dir):
        """Test auto_resume_training with no state"""
        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Try auto-resume with patch to suppress print
        config = {"epochs": 50}
        with patch("builtins.print") as mock_print:
            resumed_state = auto_resume_training(model, optimizer, config, temp_dir)

        assert resumed_state is None

    def test_auto_resume_training_with_checkpoint(self, temp_dir):
        """Test auto_resume_training with checkpoint but no state"""
        # Create a checkpoint
        checkpoint = temp_dir / "checkpoint_100.pt"
        checkpoint.write_text("dummy")

        # Create a simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Try auto-resume with patch to suppress print
        config = {"epochs": 50}
        with patch("builtins.print") as mock_print:
            resumed_state = auto_resume_training(model, optimizer, config, temp_dir)

        assert resumed_state is None
        # The actual implementation prints a different message
        mock_print.assert_called_with(
            "Manually load this checkpoint to resume training"
        )
        mock_print.assert_called_with(
            "Manually load this checkpoint to resume training"
        )
