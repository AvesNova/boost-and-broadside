"""
Data collection pipeline module - handles data collection and human play
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

# Import shared utilities
# Removed old config import - now using Hydra DictConfig
from ..utils import setup_logging, generate_run_name, InterruptHandler
from ..cli_args import get_data_collection_arguments

# Import existing data collection functions
from ..collect_data import collect_bc_data, run_human_play, collect_selfplay_data

# Import parallel processing utilities
from ..parallel_utils import WorkerPool, DataCollectionWorker, aggregate_worker_data


class DataCollectionPipeline:
    """Handles all data collection operations"""

    @staticmethod
    def add_subparsers(subparsers):
        """Add data collection subcommands to the argument parser"""
        collect_parser = subparsers.add_parser(
            "collect", help="Data collection operations"
        )
        collect_subparsers = collect_parser.add_subparsers(
            dest="collect_command", help="Collection command"
        )

        # BC data collection
        bc_parser = collect_subparsers.add_parser(
            "bc", help="Collect behavior cloning data"
        )
        for arg in get_data_collection_arguments():
            bc_parser.add_argument(*arg["args"], **arg["kwargs"])

        # Self-play data collection
        selfplay_parser = collect_subparsers.add_parser(
            "selfplay", help="Collect self-play data"
        )
        for arg in get_data_collection_arguments():
            selfplay_parser.add_argument(*arg["args"], **arg["kwargs"])

        return collect_parser

    @staticmethod
    def execute(cfg: DictConfig) -> int:
        """Execute the appropriate data collection command with DictConfig"""
        try:
            with InterruptHandler("Data collection interrupted by user"):
                # If config is nested under 'collect', extract it
                if "collect" in cfg and isinstance(cfg.collect, DictConfig):
                    cfg = cfg.collect

                # Get command from config or from command structure
                collect_command = None

                # Try to get from collect_command first (directly from config)
                if "collect_command" in cfg:
                    collect_command = cfg.collect_command
                # Try to get from command (from CLI args)
                elif "command" in cfg:
                    collect_command = cfg.command
                # Fallback to data_collection.mode
                elif cfg.get("data_collection", {}).get("mode"):
                    collect_command = cfg.data_collection.mode

                if collect_command == "bc":
                    return DataCollectionPipeline._collect_bc(cfg)
                elif collect_command == "selfplay":
                    return DataCollectionPipeline._collect_selfplay(cfg)
                else:
                    print(f"Unknown collection command: {collect_command}")
                    return 1
        except Exception as e:
            print(f"Error during data collection: {e}")
            return 1

    @staticmethod
    def _collect_bc(cfg: DictConfig) -> int:
        """Execute behavior cloning data collection with DictConfig"""
        # Add timestamp to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a mutable copy of the config
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Update output directory with timestamp
        if (
            "data_collection" in config_dict
            and "bc_data" in config_dict["data_collection"]
        ):
            bc_config = config_dict["data_collection"]["bc_data"]
            bc_config["output_dir"] = (
                f"{bc_config.get('output_dir', 'data/bc_pretraining')}_{timestamp}"
            )

        print("=" * 60)
        print("COLLECTING BEHAVIOR CLONING DATA")
        print("=" * 60)
        print(
            f"Output directory: {config_dict['data_collection']['bc_data']['output_dir']}"
        )

        # Setup logging
        run_name = generate_run_name("collect_bc")
        logger = setup_logging(run_name)

        # Check if parallel processing is enabled
        parallel_config = config_dict.get("parallel_processing", {})
        if parallel_config.get("enabled", False):
            return DataCollectionPipeline._collect_bc_parallel(config_dict, run_name)
        else:
            # Use original sequential collection
            try:
                collect_bc_data(config_dict)
                print("BC data collection completed successfully!")
                return 0
            except Exception as e:
                print(f"Error during BC data collection: {e}")
                return 1

    @staticmethod
    def _collect_bc_parallel(config: dict, run_name: str) -> int:
        """Execute parallel behavior cloning data collection"""
        print("Starting parallel BC data collection...")

        # Extract configuration
        bc_config = config["data_collection"]["bc_data"]
        parallel_config = config["parallel_processing"]

        # Calculate episodes per worker
        total_episodes = sum(bc_config["episodes_per_mode"].values())
        num_workers = parallel_config.get("num_workers", 4)
        episodes_per_worker = total_episodes // num_workers
        remaining_episodes = total_episodes % num_workers

        print(f"Total episodes to collect: {total_episodes}")
        print(f"Number of workers: {num_workers}")
        print(
            f"Episodes per worker: {episodes_per_worker} (with {remaining_episodes} extra)"
        )

        # Setup worker pool
        worker_pool = WorkerPool(DataCollectionWorker, num_workers)

        # Prepare worker arguments
        output_dir = bc_config["output_dir"]
        checkpoint_freq = parallel_config["data_collection"]["checkpoint_frequency"]

        # Start workers for each game mode
        game_modes = bc_config["game_modes"]
        episodes_per_mode = bc_config["episodes_per_mode"]

        # We'll distribute episodes across workers by game mode
        worker_args = []
        worker_id = 0

        for game_mode in game_modes:
            mode_episodes = episodes_per_mode[game_mode]
            episodes_per_worker_for_mode = mode_episodes // num_workers
            remaining_mode_episodes = mode_episodes % num_workers

            for i in range(num_workers):
                episodes_to_collect = episodes_per_worker_for_mode
                if i < remaining_mode_episodes:
                    episodes_to_collect += 1

                if episodes_to_collect > 0:
                    worker_args.append(
                        {
                            "output_dir": output_dir,
                            "episodes_to_collect": episodes_to_collect,
                            "checkpoint_freq": checkpoint_freq,
                            "game_mode": game_mode,
                        }
                    )
                    worker_id += 1

        # Start workers
        print(f"Starting {len(worker_args)} workers...")
        workers = worker_pool.start_workers(config, **worker_args[0])

        # Start remaining workers with their specific args
        for i, args in enumerate(worker_args[1:], 1):
            # Create additional workers as needed
            if i >= len(workers):
                worker = DataCollectionWorker(i, config, **args)
                worker.start()
                workers.append(worker)

        # Wait for completion or timeout
        timeout = parallel_config["data_collection"].get("timeout", 3600)

        try:
            results = worker_pool.wait_for_completion(timeout=timeout)

            # Check results
            successful_workers = sum(1 for r in results if r["status"] == "success")
            print(f"Completed: {successful_workers}/{len(workers)} workers successful")

            if successful_workers < len(workers):
                print("Warning: Some workers failed")
                for result in results:
                    if result["status"] == "failed":
                        print(
                            f"  Worker {result['worker_id']} failed with code {result['exitcode']}"
                        )

            # Aggregate data from all workers
            print("Aggregating data from all workers...")
            worker_dirs = [
                str(Path(output_dir) / f"worker_{w.worker_id}") for w in workers
            ]

            aggregated_path = Path(output_dir) / "aggregated_bc_data.pkl"
            stats = aggregate_worker_data(worker_dirs, str(aggregated_path))

            print(f"Aggregated {stats['total_episodes']} episodes")
            print(f"Final dataset saved to: {aggregated_path}")

            # Save collection stats
            import yaml

            stats_path = Path(output_dir) / "parallel_collection_stats.yaml"
            with open(stats_path, "w") as f:
                yaml.dump(stats, f)

            print("Parallel BC data collection completed successfully!")
            return 0

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
            worker_pool.stop_workers()
            return 1

        except Exception as e:
            print(f"Error during parallel BC data collection: {e}")
            worker_pool.stop_workers()
            return 1

        finally:
            worker_pool.stop_workers()

    @staticmethod
    def _collect_selfplay(cfg: DictConfig) -> int:
        """Execute self-play data collection with DictConfig"""
        # Create a mutable copy of the config
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Add timestamp to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if (
            "data_collection" in config_dict
            and "selfplay_data" in config_dict["data_collection"]
        ):
            config_dict["data_collection"]["selfplay_data"][
                "output_dir"
            ] = f"{config_dict['data_collection']['selfplay_data'].get('output_dir', 'data/selfplay')}_{timestamp}"

        print("=" * 60)
        print("COLLECTING SELF-PLAY DATA")
        print("=" * 60)
        print(
            f"Output directory: {config_dict['data_collection']['selfplay_data']['output_dir']}"
        )

        # Setup logging
        run_name = generate_run_name("collect_selfplay")
        logger = setup_logging(run_name)

        # Execute data collection
        try:
            collect_selfplay_data(config_dict)
            print("Self-play data collection completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during self-play data collection: {e}")
            return 1


class PlayPipeline:
    """Handles play operations"""

    @staticmethod
    def add_subparsers(subparsers):
        """Add play subcommands to the argument parser"""
        play_parser = subparsers.add_parser("play", help="Play operations")
        play_subparsers = play_parser.add_subparsers(
            dest="play_command", help="Play command"
        )

        # Human play
        human_parser = play_subparsers.add_parser("human", help="Human vs AI play")
        human_parser.add_argument("--config", type=str, help="Config file path")

        return play_parser

    @staticmethod
    def execute(cfg: DictConfig) -> int:
        """Execute the appropriate play command with DictConfig"""
        try:
            with InterruptHandler("Game interrupted by user"):
                # If config is nested under 'play', extract it
                if "play" in cfg and isinstance(cfg.play, DictConfig):
                    cfg = cfg.play

                # Get command from config or from command structure
                play_command = None

                # Try to get from play_command first (directly from config)
                if "play_command" in cfg:
                    play_command = cfg.play_command
                # Try to get from command (from CLI args)
                elif "command" in cfg:
                    play_command = cfg.command
                # Fallback to play.mode
                elif cfg.get("play", {}).get("mode"):
                    play_command = cfg.play.mode

                if play_command == "human":
                    return PlayPipeline._play_human(cfg)
                else:
                    print(f"Unknown play command: {play_command}")
                    return 1
        except Exception as e:
            print(f"Error during play: {e}")
            return 1

    @staticmethod
    def _play_human(cfg: DictConfig) -> int:
        """Execute human play with DictConfig"""
        # Convert config to dict for compatibility
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        print("=" * 60)
        print("HUMAN VS AI PLAY")
        print("=" * 60)
        print("Controls: WASD/Arrow Keys (move), Space (shoot), Shift (sharp turn)")
        print("Close window or Ctrl+C to quit")

        # Setup logging
        run_name = generate_run_name("play_human")
        logger = setup_logging(run_name)

        try:
            run_human_play(config_dict)
            return 0
        except Exception as e:
            print(f"Error during human play: {e}")
            return 1
