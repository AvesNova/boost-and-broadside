"""
Tests for parallel processing utilities
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import multiprocessing as mp
import pickle
import gzip

from src.parallel_utils import (
    WorkerProcess,
    DataCollectionWorker,
    WorkerPool,
    aggregate_worker_data,
    save_checkpoint,
    load_checkpoint,
)


class TestWorkerProcess:
    """Tests for the base WorkerProcess class"""

    def test_worker_process_initialization(self):
        """Test worker process initialization"""
        config = {"test": "config"}
        worker = WorkerProcess(worker_id=0, config=config)

        assert worker.worker_id == 0
        assert worker.config == config
        assert not worker._should_stop

    def test_worker_process_log_methods(self):
        """Test worker process logging methods"""
        config = {"test": "config"}
        worker = WorkerProcess(worker_id=1, config=config)

        # Test log_info
        with patch("builtins.print") as mock_print:
            worker.log_info("Test message")
            mock_print.assert_called_once_with("[Worker 1] Test message")

        # Test log_error
        with patch("builtins.print") as mock_print:
            worker.log_error("Error message")
            mock_print.assert_called_once_with("[Worker 1] ERROR: Error message")

    def test_worker_process_stop(self):
        """Test stopping a worker process"""
        config = {"test": "config"}
        worker = WorkerProcess(worker_id=0, config=config)

        assert not worker._should_stop
        worker.stop()
        assert worker._should_stop


class TestDataCollectionWorker:
    """Tests for the DataCollectionWorker class"""

    def test_data_collection_worker_initialization(self):
        """Test data collection worker initialization"""
        config = {"environment": {"world_size": [800, 600]}}
        kwargs = {
            "output_dir": "/tmp/test",
            "episodes_to_collect": 100,
            "checkpoint_freq": 50,
            "game_mode": "2v2",
        }

        worker = DataCollectionWorker(worker_id=2, config=config, **kwargs)

        assert worker.worker_id == 2
        assert worker.output_dir == "/tmp/test"
        assert worker.episodes_to_collect == 100
        assert worker.checkpoint_freq == 50
        assert worker.game_mode == "2v2"


class TestWorkerPool:
    """Tests for the WorkerPool class"""

    def test_worker_pool_initialization(self):
        """Test worker pool initialization"""
        pool = WorkerPool(DataCollectionWorker, num_workers=2)

        assert pool.worker_class == DataCollectionWorker
        assert pool.num_workers == 2
        assert len(pool.workers) == 0
        assert len(pool.results) == 0

    def test_worker_pool_default_num_workers(self):
        """Test worker pool with default number of workers"""
        with patch("os.cpu_count", return_value=8):
            pool = WorkerPool(DataCollectionWorker)
            assert pool.num_workers == 8

    @patch("multiprocessing.Process.start")
    def test_worker_pool_start_workers(self, mock_start):
        """Test starting workers in a pool"""
        pool = WorkerPool(DataCollectionWorker, num_workers=2)
        config = {"test": "config"}
        kwargs = {"output_dir": "/tmp/test"}

        workers = pool.start_workers(config, **kwargs)

        assert len(workers) == 2
        assert len(pool.workers) == 2
        assert mock_start.call_count == 2

    def test_worker_pool_wait_for_completion(self):
        """Test waiting for worker completion"""
        pool = WorkerPool(DataCollectionWorker, num_workers=2)

        # Create mock workers
        mock_worker1 = MagicMock()
        mock_worker1.exitcode = 0
        mock_worker1.worker_id = 0

        mock_worker2 = MagicMock()
        mock_worker2.exitcode = 1
        mock_worker2.worker_id = 1

        pool.workers = [mock_worker1, mock_worker2]

        results = pool.wait_for_completion()

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[0]["worker_id"] == 0
        assert results[1]["status"] == "failed"
        assert results[1]["worker_id"] == 1
        assert results[1]["exitcode"] == 1

    def test_worker_pool_is_alive(self):
        """Test checking if any workers are alive"""
        pool = WorkerPool(DataCollectionWorker, num_workers=2)

        # Create mock workers
        mock_worker1 = MagicMock()
        mock_worker1.is_alive.return_value = False

        mock_worker2 = MagicMock()
        mock_worker2.is_alive.return_value = True

        pool.workers = [mock_worker1, mock_worker2]

        assert pool.is_alive() == True

        # Both dead
        mock_worker2.is_alive.return_value = False
        assert pool.is_alive() == False


class TestCheckpointFunctions:
    """Tests for checkpoint saving and loading functions"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoint tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_save_checkpoint_uncompressed(self, temp_dir):
        """Test saving uncompressed checkpoint"""
        data = {"test": "data", "numbers": [1, 2, 3]}
        filepath = temp_dir / "test_checkpoint.pkl"

        save_checkpoint(data, filepath, compress=False)

        assert filepath.exists()
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == data

    def test_save_checkpoint_compressed(self, temp_dir):
        """Test saving compressed checkpoint"""
        data = {"test": "data", "numbers": [1, 2, 3]}
        filepath = temp_dir / "test_checkpoint.pkl"

        save_checkpoint(data, filepath, compress=True)

        compressed_path = temp_dir / "test_checkpoint.pkl.gz"
        assert compressed_path.exists()

        with gzip.open(compressed_path, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == data

    def test_load_checkpoint_uncompressed(self, temp_dir):
        """Test loading uncompressed checkpoint"""
        data = {"test": "data", "numbers": [1, 2, 3]}
        filepath = temp_dir / "test_checkpoint.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        loaded_data = load_checkpoint(filepath)
        assert loaded_data == data

    def test_load_checkpoint_compressed(self, temp_dir):
        """Test loading compressed checkpoint"""
        data = {"test": "data", "numbers": [1, 2, 3]}
        filepath = temp_dir / "test_checkpoint.pkl.gz"

        with gzip.open(filepath, "wb") as f:
            pickle.dump(data, f)

        loaded_data = load_checkpoint(filepath)
        assert loaded_data == data

    def test_load_checkpoint_nonexistent(self, temp_dir):
        """Test loading non-existent checkpoint"""
        filepath = temp_dir / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_checkpoint(filepath)


class TestAggregateWorkerData:
    """Tests for worker data aggregation"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for aggregation tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_aggregate_worker_data(self, temp_dir):
        """Test aggregating data from multiple workers"""
        # Create mock worker directories
        worker1_dir = temp_dir / "worker_0"
        worker2_dir = temp_dir / "worker_1"

        worker1_dir.mkdir()
        worker2_dir.mkdir()

        # Create mock episodes
        episodes1 = [
            {"episode_id": 0, "episode_length": 10},
            {"episode_id": 1, "episode_length": 15},
        ]

        episodes2 = [
            {"episode_id": 2, "episode_length": 12},
            {"episode_id": 3, "episode_length": 8},
        ]

        # Create mock metadata
        metadata1 = {
            "worker_id": 0,
            "episodes_collected": 2,
            "game_mode": "2v2",
        }

        metadata2 = {
            "worker_id": 1,
            "episodes_collected": 2,
            "game_mode": "2v2",
        }

        # Save worker data
        final_data_path1 = worker1_dir / "final_data.pkl"
        with open(final_data_path1, "wb") as f:
            pickle.dump(episodes1, f)

        metadata_path1 = worker1_dir / "metadata.pkl"
        with open(metadata_path1, "wb") as f:
            pickle.dump(metadata1, f)

        final_data_path2 = worker2_dir / "final_data.pkl"
        with open(final_data_path2, "wb") as f:
            pickle.dump(episodes2, f)

        metadata_path2 = worker2_dir / "metadata.pkl"
        with open(metadata_path2, "wb") as f:
            pickle.dump(metadata2, f)

        # Aggregate data
        output_path = temp_dir / "aggregated_data.pkl"
        stats = aggregate_worker_data(
            [str(worker1_dir), str(worker2_dir)], str(output_path)
        )

        # Check stats
        assert stats["total_episodes"] == 4
        assert stats["workers"] == 2
        assert len(stats["worker_metadata"]) == 2
        # Check output file (might be compressed)
        output_path = temp_dir / "aggregated_data.pkl"
        output_path_gz = temp_dir / "aggregated_data.pkl.gz"
        assert output_path.exists() or output_path_gz.exists()

        # Check aggregated data
        if output_path_gz.exists():
            with gzip.open(output_path_gz, "rb") as f:
                aggregated_episodes = pickle.load(f)
        else:
            with open(output_path, "rb") as f:
                aggregated_episodes = pickle.load(f)

        assert len(aggregated_episodes) == 4
        assert all(ep in aggregated_episodes for ep in episodes1)
        assert all(ep in aggregated_episodes for ep in episodes2)

    def test_aggregate_worker_data_empty(self, temp_dir):
        """Test aggregating data from empty worker list"""
        output_path = temp_dir / "aggregated_data.pkl"

        stats = aggregate_worker_data([], str(output_path))

        assert stats["total_episodes"] == 0
        assert stats["workers"] == 0
        assert not output_path.exists()
