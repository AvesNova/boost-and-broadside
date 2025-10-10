"""
Parallel processing utilities for data collection and training
"""

import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pickle
import gzip
import numpy as np
from datetime import datetime
import logging

from .utils import setup_logging, ensure_dir


class WorkerProcess(mp.Process):
    """
    Base class for worker processes with proper signal handling
    """

    def __init__(self, worker_id: int, config: Dict[str, Any], **kwargs):
        super().__init__()
        self.worker_id = worker_id
        self.config = config
        self.kwargs = kwargs
        self._should_stop = False

    def run(self):
        """Main worker process with signal handling"""
        # Ignore interrupt signals in worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            self.execute()
        except Exception as e:
            self.log_error(f"Worker {self.worker_id} failed: {e}")
            traceback.print_exc()
            return 1
        return 0

    def execute(self):
        """Override this method in subclasses"""
        raise NotImplementedError

    def log_info(self, message: str):
        """Log info message with worker ID"""
        print(f"[Worker {self.worker_id}] {message}")

    def log_error(self, message: str):
        """Log error message with worker ID"""
        print(f"[Worker {self.worker_id}] ERROR: {message}")

    def stop(self):
        """Signal the worker to stop"""
        self._should_stop = True


class DataCollectionWorker(WorkerProcess):
    """
    Worker process for collecting data in parallel
    """

    def __init__(self, worker_id: int, config: Dict[str, Any], **kwargs):
        super().__init__(worker_id, config, **kwargs)
        self.output_dir = kwargs.get("output_dir")
        self.episodes_to_collect = kwargs.get("episodes_to_collect", 100)
        self.checkpoint_freq = kwargs.get("checkpoint_freq", 50)
        self.game_mode = kwargs.get("game_mode", "2v2")

    def execute(self):
        """Execute data collection"""
        self.log_info(
            f"Starting data collection for {self.episodes_to_collect} episodes"
        )

        # Setup worker-specific output directory
        worker_dir = Path(self.output_dir) / f"worker_{self.worker_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging for this worker
        logger = setup_logging(f"worker_{self.worker_id}", log_dir=worker_dir)

        # Import here to avoid issues with multiprocessing
        from game_runner import create_standard_runner
        from agents import create_scripted_agent
        from collect_data import add_mc_returns, save_episodes

        # Setup environment and agents
        env_config = self.config.get("environment", {})
        runner = create_standard_runner(
            world_size=tuple(env_config.get("world_size", [1200, 800])),
            max_ships=env_config.get("max_ships", 8),
        )
        runner.setup_environment()

        scripted_agent = create_scripted_agent(
            world_size=tuple(env_config.get("world_size", [1200, 800])),
            config=self.config.get("scripted_agent", {}),
        )

        runner.assign_agent(0, scripted_agent)
        runner.assign_agent(1, scripted_agent)

        # Collect episodes with periodic checkpointing
        all_episodes = []
        collected_episodes = 0

        try:
            while (
                collected_episodes < self.episodes_to_collect and not self._should_stop
            ):
                # Determine batch size (don't exceed checkpoint frequency)
                batch_size = min(
                    self.checkpoint_freq, self.episodes_to_collect - collected_episodes
                )

                self.log_info(f"Collecting batch of {batch_size} episodes")

                # Collect a batch of episodes
                episodes = runner.run_multiple_episodes(
                    n_episodes=batch_size,
                    game_mode=self.game_mode,
                    collect_data=True,
                    max_steps=env_config.get("max_episode_steps", 10000),
                )

                # Add MC returns to episodes
                episodes_with_returns = []
                for episode in episodes:
                    episode_with_returns = add_mc_returns(
                        episode,
                        self.config.get("data_collection", {})
                        .get("bc_data", {})
                        .get("gamma", 0.99),
                    )
                    episodes_with_returns.append(episode_with_returns)
                    all_episodes.append(episode_with_returns)

                # Save checkpoint
                checkpoint_path = (
                    worker_dir / f"checkpoint_{collected_episodes + batch_size}.pkl"
                )
                save_episodes(episodes_with_returns, checkpoint_path, compress=True)
                self.log_info(
                    f"Saved checkpoint with {len(episodes_with_returns)} episodes"
                )

                collected_episodes += batch_size

                # Check if we should stop
                if self._should_stop:
                    self.log_info("Received stop signal, finishing current batch")
                    break

        finally:
            runner.close()

        # Save final data
        final_path = worker_dir / "final_data.pkl"
        save_episodes(all_episodes, final_path, compress=True)

        # Save metadata
        metadata = {
            "worker_id": self.worker_id,
            "episodes_collected": len(all_episodes),
            "game_mode": self.game_mode,
            "collection_time": datetime.now().isoformat(),
        }

        metadata_path = worker_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self.log_info(f"Completed collection of {len(all_episodes)} episodes")

        return {
            "worker_id": self.worker_id,
            "episodes_collected": len(all_episodes),
            "output_dir": str(worker_dir),
        }


class WorkerPool:
    """
    Manages a pool of worker processes for parallel execution
    """

    def __init__(self, worker_class: type, num_workers: Optional[int] = None):
        self.worker_class = worker_class
        self.num_workers = num_workers or os.cpu_count()
        self.workers: List[WorkerProcess] = []
        self.results: List[Any] = []

    def start_workers(
        self, config: Dict[str, Any], **worker_kwargs
    ) -> List[WorkerProcess]:
        """Start all worker processes"""
        self.workers = []

        for worker_id in range(self.num_workers):
            worker = self.worker_class(worker_id, config, **worker_kwargs)
            worker.start()
            self.workers.append(worker)

        return self.workers

    def wait_for_completion(self, timeout: Optional[float] = None) -> List[Any]:
        """Wait for all workers to complete and return results"""
        results = []

        for worker in self.workers:
            worker.join(timeout=timeout)

            if worker.exitcode == 0:
                # Worker completed successfully
                results.append({"worker_id": worker.worker_id, "status": "success"})
            else:
                # Worker failed
                results.append(
                    {
                        "worker_id": worker.worker_id,
                        "status": "failed",
                        "exitcode": worker.exitcode,
                    }
                )

        self.results = results
        return results

    def stop_workers(self):
        """Signal all workers to stop"""
        for worker in self.workers:
            if worker.is_alive():
                worker.stop()

        # Give workers a moment to stop gracefully
        time.sleep(0.1)

        # Force terminate if needed
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()

    def is_alive(self) -> bool:
        """Check if any workers are still running"""
        return any(worker.is_alive() for worker in self.workers)


def aggregate_worker_data(output_dirs: List[str], output_path: str) -> Dict[str, Any]:
    """
    Aggregate data from multiple worker directories

    Args:
        output_dirs: List of worker output directories
        output_path: Path to save the aggregated data

    Returns:
        Dictionary with aggregation statistics
    """
    all_episodes = []
    metadata_list = []

    # Load data from each worker
    for worker_dir in output_dirs:
        worker_path = Path(worker_dir)

        # Load metadata
        metadata_path = worker_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                metadata_list.append(metadata)

        # Load final data
        final_data_path = worker_path / "final_data.pkl"
        if final_data_path.exists():
            episodes = load_episodes(final_data_path)
            all_episodes.extend(episodes)

    # Save aggregated data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_episodes:
        # Import save_episodes from collect_data to avoid circular import
        from collect_data import save_episodes

        save_episodes(all_episodes, output_path, compress=True)

    # Save aggregation stats
    stats = {
        "total_episodes": len(all_episodes),
        "workers": len(output_dirs),
        "aggregation_time": datetime.now().isoformat(),
        "worker_metadata": metadata_list,
    }

    stats_path = output_path.with_suffix(".yaml")
    import yaml

    with open(stats_path, "w") as f:
        yaml.dump(stats, f)

    return stats


def load_episodes(filepath: Union[str, Path]) -> List[Dict]:
    """Load episodes from file with compression support"""
    filepath = Path(filepath)

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
        print(f"Error loading episodes from {filepath}: {e}")
        return []


def save_checkpoint(data: Any, filepath: Union[str, Path], compress: bool = True):
    """Save data checkpoint with optional compression"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if compress and not filepath.name.endswith(".gz"):
            filepath = filepath.with_suffix(filepath.suffix + ".gz")

        if filepath.suffix == ".gz":
            with gzip.open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving checkpoint to {filepath}: {e}")
        raise


def load_checkpoint(filepath: Union[str, Path]) -> Any:
    """Load data checkpoint with compression support"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    try:
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        raise
