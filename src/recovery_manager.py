"""
Recovery manager for resuming interrupted data collection and training
"""

import pickle
import gzip
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from datetime import datetime
import logging

from .utils import setup_logging


class RecoveryManager:
    """
    Manages recovery of interrupted data collection and training processes
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.logger = setup_logging("recovery_manager")

    def save_collection_state(
        self,
        state: Dict[str, Any],
        filename: str = "collection_state.pkl",
    ):
        """
        Save the current state of data collection for recovery

        Args:
            state: Current collection state
            filename: Filename for the state file
        """
        state_path = self.output_dir / filename
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp
        state["last_saved"] = datetime.now().isoformat()

        try:
            with gzip.open(state_path, "wb") as f:
                pickle.dump(state, f)
            self.logger.info(f"Collection state saved to {state_path}")
        except Exception as e:
            self.logger.error(f"Error saving collection state: {e}")
            raise

    def load_collection_state(
        self,
        filename: str = "collection_state.pkl",
    ) -> Optional[Dict[str, Any]]:
        """
        Load the collection state for recovery

        Args:
            filename: Filename of the state file

        Returns:
            Collection state if found, None otherwise
        """
        state_path = self.output_dir / filename

        if not state_path.exists():
            self.logger.info(f"No collection state found at {state_path}")
            return None

        try:
            with gzip.open(state_path, "rb") as f:
                state = pickle.load(f)

            self.logger.info(f"Collection state loaded from {state_path}")
            self.logger.info(f"State saved at: {state.get('last_saved', 'unknown')}")

            return state
        except Exception as e:
            self.logger.error(f"Error loading collection state: {e}")
            return None

    def save_training_state(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_losses: List[float],
        val_losses: List[float],
        best_val_loss: float,
        config: Dict[str, Any],
        filename: str = "training_state.pkl",
    ):
        """
        Save the current state of training for recovery

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            train_losses: Training losses
            val_losses: Validation losses
            best_val_loss: Best validation loss
            config: Training configuration
            filename: Filename for the state file
        """
        state_path = self.output_dir / filename
        state_path.parent.mkdir(parents=True, exist_ok=True)

        training_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "config": config,
            "last_saved": datetime.now().isoformat(),
        }

        try:
            with gzip.open(state_path, "wb") as f:
                pickle.dump(training_state, f)
            self.logger.info(f"Training state saved to {state_path}")
        except Exception as e:
            self.logger.error(f"Error saving training state: {e}")
            raise

    def load_training_state(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str = "training_state.pkl",
    ) -> Optional[Dict[str, Any]]:
        """
        Load the training state for recovery

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            filename: Filename of the state file

        Returns:
            Training state if found, None otherwise
        """
        state_path = self.output_dir / filename

        if not state_path.exists():
            self.logger.info(f"No training state found at {state_path}")
            return None

        try:
            with gzip.open(state_path, "rb") as f:
                training_state = pickle.load(f)

            # Load model and optimizer states
            model.load_state_dict(training_state["model_state_dict"])
            optimizer.load_state_dict(training_state["optimizer_state_dict"])

            self.logger.info(f"Training state loaded from {state_path}")
            self.logger.info(f"Resuming from epoch {training_state['epoch']}")
            self.logger.info(
                f"State saved at: {training_state.get('last_saved', 'unknown')}"
            )

            return training_state
        except Exception as e:
            self.logger.error(f"Error loading training state: {e}")
            return None

    def find_latest_checkpoint(
        self, pattern: str = "*checkpoint*.pt"
    ) -> Optional[Path]:
        """
        Find the latest checkpoint file matching a pattern

        Args:
            pattern: Glob pattern for checkpoint files

        Returns:
            Path to the latest checkpoint, or None if not found
        """
        checkpoints = list(self.output_dir.glob(pattern))

        if not checkpoints:
            return None

        # Sort by modification time
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

        self.logger.info(f"Latest checkpoint found: {latest_checkpoint}")
        return latest_checkpoint

    def resume_bc_collection(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Resume BC data collection from a saved state

        Args:
            config: Collection configuration

        Returns:
            Collection state if found and valid, None otherwise
        """
        # Load collection state
        state = self.load_collection_state()

        if state is None:
            return None

        # Check if state is valid for resuming
        if "collected_episodes" not in state or "episodes_to_collect" not in state:
            self.logger.warning("Invalid collection state, cannot resume")
            return None

        collected_episodes = state["collected_episodes"]
        episodes_to_collect = state["episodes_to_collect"]

        if collected_episodes >= episodes_to_collect:
            self.logger.info("Collection already completed")
            return None

        self.logger.info(
            f"Resuming collection: {collected_episodes}/{episodes_to_collect} episodes"
        )

        return state

    def resume_rl_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Resume RL training from a saved state

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            config: Training configuration

        Returns:
            Training state if found and valid, None otherwise
        """
        # Load training state
        state = self.load_training_state(model, optimizer)

        if state is None:
            return None

        # Check if training is already complete
        epochs = config.get("epochs", 50)
        if state["epoch"] >= epochs:
            self.logger.info("Training already completed")
            return None

        self.logger.info(f"Resuming training from epoch {state['epoch']}/{epochs}")

        return state

    def create_recovery_report(self) -> Dict[str, Any]:
        """
        Create a report of recoverable states

        Returns:
            Dictionary with recovery information
        """
        report = {
            "output_dir": str(self.output_dir),
            "collection_state": None,
            "training_state": None,
            "checkpoints": [],
            "worker_data": [],
            "generated_at": datetime.now().isoformat(),
        }

        # Check for collection state
        collection_state = self.load_collection_state()
        if collection_state:
            report["collection_state"] = {
                "found": True,
                "last_saved": collection_state.get("last_saved"),
                "collected_episodes": collection_state.get("collected_episodes"),
                "episodes_to_collect": collection_state.get("episodes_to_collect"),
            }
        else:
            report["collection_state"] = {"found": False}

        # Check for training state
        training_state_path = self.output_dir / "training_state.pkl"
        if training_state_path.exists():
            report["training_state"] = {
                "found": True,
                "last_modified": datetime.fromtimestamp(
                    training_state_path.stat().st_mtime
                ).isoformat(),
            }
        else:
            report["training_state"] = {"found": False}

        # Find checkpoints
        checkpoints = list(self.output_dir.glob("*checkpoint*.pt"))
        checkpoints.extend(self.output_dir.glob("*checkpoint*.pkl"))

        for checkpoint in checkpoints:
            report["checkpoints"].append(
                {
                    "path": str(checkpoint),
                    "size": checkpoint.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        checkpoint.stat().st_mtime
                    ).isoformat(),
                }
            )

        # Find worker data directories
        worker_dirs = list(self.output_dir.glob("worker_*"))
        for worker_dir in worker_dirs:
            if worker_dir.is_dir():
                report["worker_data"].append(
                    {
                        "path": str(worker_dir),
                        "modified": datetime.fromtimestamp(
                            worker_dir.stat().st_mtime
                        ).isoformat(),
                    }
                )

        # Save report
        report_path = self.output_dir / "recovery_report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report, f)

        self.logger.info(f"Recovery report saved to {report_path}")

        return report


def auto_resume_collection(
    config: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """
    Automatically attempt to resume data collection

    Args:
        config: Collection configuration
        output_dir: Output directory

    Returns:
        Collection state if resumable, None otherwise
    """
    recovery_manager = RecoveryManager(output_dir)

    # Try to resume collection
    state = recovery_manager.resume_bc_collection(config)

    if state:
        print(f"Resuming data collection from {state['collected_episodes']} episodes")
        return state

    # Check if there's worker data that can be aggregated
    from .data_aggregator import find_worker_directories

    worker_dirs = find_worker_directories(output_dir)
    if worker_dirs:
        print(f"Found worker data from {len(worker_dirs)} workers")
        print("Use data_aggregator.aggregate_worker_data() to combine the data")

    return None


def auto_resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """
    Automatically attempt to resume training

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        config: Training configuration
        output_dir: Output directory

    Returns:
        Training state if resumable, None otherwise
    """
    recovery_manager = RecoveryManager(output_dir)

    # Try to resume training
    state = recovery_manager.resume_rl_training(model, optimizer, config)

    if state:
        print(f"Resuming training from epoch {state['epoch']}")
        return state

    # Try to find latest checkpoint
    latest_checkpoint = recovery_manager.find_latest_checkpoint()
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Manually load this checkpoint to resume training")

    return None
