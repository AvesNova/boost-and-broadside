"""
Data aggregation utilities for combining data from multiple workers
"""

import pickle
import gzip
import yaml
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
from datetime import datetime
import logging

from .utils import setup_logging


class DataAggregator:
    """
    Utility class for aggregating data from multiple worker processes
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.logger = setup_logging("data_aggregator")

    def aggregate_bc_data(
        self,
        worker_dirs: List[Union[str, Path]],
        output_filename: str = "aggregated_bc_data.pkl",
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Aggregate BC data from multiple worker directories

        Args:
            worker_dirs: List of worker output directories
            output_filename: Name of the output file
            compress: Whether to compress the output

        Returns:
            Dictionary with aggregation statistics
        """
        self.logger.info(f"Aggregating BC data from {len(worker_dirs)} workers")

        all_episodes = []
        worker_metadata = []
        episodes_per_mode = {}

        # Load data from each worker
        for worker_dir in worker_dirs:
            worker_path = Path(worker_dir)

            # Load metadata
            metadata_path = worker_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    worker_metadata.append(metadata)

                    # Track episodes per mode
                    game_mode = metadata.get("game_mode", "unknown")
                    episodes_count = metadata.get("episodes_collected", 0)
                    episodes_per_mode[game_mode] = (
                        episodes_per_mode.get(game_mode, 0) + episodes_count
                    )

            # Load final data
            final_data_path = worker_path / "final_data.pkl"
            if final_data_path.exists():
                episodes = self._load_episodes(final_data_path)
                all_episodes.extend(episodes)
                self.logger.info(
                    f"Loaded {len(episodes)} episodes from {worker_path.name}"
                )
            else:
                self.logger.warning(f"No final data found in {worker_path}")

        # Save aggregated data
        output_path = self.output_dir / output_filename
        self._save_episodes(all_episodes, output_path, compress)

        # Create and save aggregation statistics
        stats = {
            "total_episodes": len(all_episodes),
            "total_samples": sum(ep.get("episode_length", 0) for ep in all_episodes),
            "workers": len(worker_dirs),
            "episodes_per_mode": episodes_per_mode,
            "worker_metadata": worker_metadata,
            "aggregation_time": datetime.now().isoformat(),
            "output_file": str(output_path),
        }

        stats_path = output_path.with_suffix(".yaml")
        with open(stats_path, "w") as f:
            yaml.dump(stats, f)

        self.logger.info(f"Aggregated {len(all_episodes)} episodes to {output_path}")

        return stats

    def aggregate_rl_experiences(
        self,
        worker_dirs: List[Union[str, Path]],
        output_filename: str = "aggregated_rl_experiences.pkl",
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Aggregate RL experiences from multiple worker directories

        Args:
            worker_dirs: List of worker output directories
            output_filename: Name of the output file
            compress: Whether to compress the output

        Returns:
            Dictionary with aggregation statistics
        """
        self.logger.info(f"Aggregating RL experiences from {len(worker_dirs)} workers")

        all_experiences = []
        worker_metadata = []
        total_timesteps = 0

        # Load data from each worker
        for worker_dir in worker_dirs:
            worker_path = Path(worker_dir)

            # Load metadata
            metadata_path = worker_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    worker_metadata.append(metadata)
                    total_timesteps += metadata.get("timesteps_collected", 0)

            # Load final experiences
            final_data_path = worker_path / "final_experiences.pkl"
            if final_data_path.exists():
                with open(final_data_path, "rb") as f:
                    data = pickle.load(f)
                    experiences = data.get("experiences", [])
                    all_experiences.extend(experiences)
                    self.logger.info(
                        f"Loaded {len(experiences)} experiences from {worker_path.name}"
                    )
            else:
                self.logger.warning(f"No final experiences found in {worker_path}")

        # Save aggregated experiences
        output_path = self.output_dir / output_filename
        self._save_experiences(all_experiences, output_path, compress)

        # Create and save aggregation statistics
        stats = {
            "total_experiences": len(all_experiences),
            "total_timesteps": total_timesteps,
            "workers": len(worker_dirs),
            "worker_metadata": worker_metadata,
            "aggregation_time": datetime.now().isoformat(),
            "output_file": str(output_path),
        }

        stats_path = output_path.with_suffix(".yaml")
        with open(stats_path, "w") as f:
            yaml.dump(stats, f)

        self.logger.info(
            f"Aggregated {len(all_experiences)} experiences to {output_path}"
        )

        return stats

    def merge_checkpoints(
        self,
        checkpoint_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        merge_strategy: str = "latest",
    ) -> Dict[str, Any]:
        """
        Merge multiple checkpoints into one

        Args:
            checkpoint_paths: List of checkpoint file paths
            output_path: Path to save the merged checkpoint
            merge_strategy: Strategy for merging ("latest", "best", "average")

        Returns:
            Dictionary with merge statistics
        """
        self.logger.info(
            f"Merging {len(checkpoint_paths)} checkpoints using strategy: {merge_strategy}"
        )

        checkpoints = []
        for path in checkpoint_paths:
            checkpoint = self._load_checkpoint(path)
            if checkpoint:
                checkpoints.append((path, checkpoint))

        if not checkpoints:
            raise ValueError("No valid checkpoints found")

        # Apply merge strategy
        if merge_strategy == "latest":
            # Use the most recent checkpoint
            latest_path, latest_checkpoint = max(
                checkpoints, key=lambda x: Path(x[0]).stat().st_mtime
            )
            merged_checkpoint = latest_checkpoint
            strategy_info = f"Used latest checkpoint: {latest_path}"

        elif merge_strategy == "best":
            # Use the checkpoint with best performance (if available)
            best_checkpoint = None
            best_metric = float("-inf")
            best_path = None

            for path, checkpoint in checkpoints:
                # Look for validation loss or other metrics
                if "val_loss" in checkpoint:
                    if checkpoint["val_loss"] < best_metric:
                        best_metric = checkpoint["val_loss"]
                        best_checkpoint = checkpoint
                        best_path = path
                elif "win_rate" in checkpoint:
                    if checkpoint["win_rate"] > best_metric:
                        best_metric = checkpoint["win_rate"]
                        best_checkpoint = checkpoint
                        best_path = path

            if best_checkpoint:
                merged_checkpoint = best_checkpoint
                strategy_info = (
                    f"Used best checkpoint: {best_path} (metric: {best_metric})"
                )
            else:
                # Fallback to latest
                latest_path, latest_checkpoint = max(
                    checkpoints, key=lambda x: Path(x[0]).stat().st_mtime
                )
                merged_checkpoint = latest_checkpoint
                strategy_info = (
                    f"No performance metrics found, used latest: {latest_path}"
                )

        elif merge_strategy == "average":
            # Average model weights (for neural network checkpoints)
            if len(checkpoints) < 2:
                raise ValueError("Need at least 2 checkpoints for averaging")

            # Extract model state dicts
            state_dicts = [ckpt[1].get("model_state_dict") for ckpt in checkpoints]
            if not all(state_dicts):
                raise ValueError("Not all checkpoints contain model_state_dict")

            # Average the weights
            averaged_state_dict = {}
            for key in state_dicts[0].keys():
                if all(key in sd for sd in state_dicts[1:]):
                    # Average the tensors
                    averaged_tensor = sum(sd[key] for sd in state_dicts) / len(
                        state_dicts
                    )
                    averaged_state_dict[key] = averaged_tensor

            # Create merged checkpoint
            merged_checkpoint = checkpoints[0][1].copy()
            merged_checkpoint["model_state_dict"] = averaged_state_dict
            merged_checkpoint["merge_info"] = {
                "strategy": "average",
                "num_checkpoints": len(checkpoints),
                "checkpoint_paths": [str(p[0]) for p in checkpoints],
            }

            strategy_info = f"Averaged {len(checkpoints)} checkpoints"

        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Save merged checkpoint
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._save_checkpoint(merged_checkpoint, output_path)

        # Create merge statistics
        stats = {
            "input_checkpoints": [str(p[0]) for p in checkpoints],
            "output_checkpoint": str(output_path),
            "merge_strategy": merge_strategy,
            "strategy_info": strategy_info,
            "merge_time": datetime.now().isoformat(),
        }

        self.logger.info(f"Merged checkpoint saved to {output_path}")
        self.logger.info(strategy_info)

        return stats

    def _load_episodes(self, filepath: Path) -> List[Dict]:
        """Load episodes from file with compression support"""
        if not filepath.exists():
            return []

        try:
            if filepath.suffix == ".gz":
                with gzip.open(filepath, "rb") as f:
                    return pickle.load(f)
            else:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading episodes from {filepath}: {e}")
            return []

    def _save_episodes(self, episodes: List[Dict], filepath: Path, compress: bool):
        """Save episodes to file with optional compression"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if compress and not filepath.name.endswith(".gz"):
                filepath = filepath.with_suffix(filepath.suffix + ".gz")

            if filepath.suffix == ".gz":
                with gzip.open(filepath, "wb") as f:
                    pickle.dump(episodes, f)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(episodes, f)
        except Exception as e:
            self.logger.error(f"Error saving episodes to {filepath}: {e}")
            raise

    def _save_experiences(
        self, experiences: List[Dict], filepath: Path, compress: bool
    ):
        """Save experiences to file with optional compression"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if compress and not filepath.name.endswith(".gz"):
                filepath = filepath.with_suffix(filepath.suffix + ".gz")

            if filepath.suffix == ".gz":
                with gzip.open(filepath, "wb") as f:
                    pickle.dump(experiences, f)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(experiences, f)
        except Exception as e:
            self.logger.error(f"Error saving experiences to {filepath}: {e}")
            raise

    def _load_checkpoint(self, filepath: Union[str, Path]) -> Optional[Dict]:
        """Load checkpoint from file"""
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        try:
            if filepath.suffix == ".gz":
                with gzip.open(filepath, "rb") as f:
                    return pickle.load(f)
            else:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {filepath}: {e}")
            return None

    def _save_checkpoint(self, checkpoint: Dict, filepath: Union[str, Path]):
        """Save checkpoint to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint to {filepath}: {e}")
            raise


def find_worker_directories(base_dir: Union[str, Path]) -> List[Path]:
    """
    Find all worker directories in a base directory

    Args:
        base_dir: Base directory to search

    Returns:
        List of worker directory paths
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Look for directories matching "worker_N" pattern
    worker_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("worker_"):
            try:
                # Extract worker ID
                worker_id = int(item.name.split("_")[1])
                worker_dirs.append(item)
            except (IndexError, ValueError):
                continue

    return sorted(worker_dirs, key=lambda x: int(x.name.split("_")[1]))


def aggregate_worker_data(
    base_dir: Union[str, Path],
    data_type: str = "bc",
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to aggregate data from worker directories

    Args:
        base_dir: Base directory containing worker directories
        data_type: Type of data ("bc" or "rl")
        output_filename: Optional custom output filename

    Returns:
        Dictionary with aggregation statistics
    """
    # Find worker directories
    worker_dirs = find_worker_directories(base_dir)

    if not worker_dirs:
        raise ValueError(f"No worker directories found in {base_dir}")

    # Create aggregator
    aggregator = DataAggregator(base_dir)

    # Determine output filename
    if output_filename is None:
        if data_type == "bc":
            output_filename = "aggregated_bc_data.pkl"
        elif data_type == "rl":
            output_filename = "aggregated_rl_experiences.pkl"
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    # Aggregate data
    if data_type == "bc":
        return aggregator.aggregate_bc_data(worker_dirs, output_filename)
    elif data_type == "rl":
        return aggregator.aggregate_rl_experiences(worker_dirs, output_filename)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
