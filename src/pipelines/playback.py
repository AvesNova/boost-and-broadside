"""
Playback pipeline module - handles human play and episode replay
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf

# Import shared utilities
# Removed old config import - now using Hydra DictConfig
from ..utils import (
    setup_logging,
    generate_run_name,
    InterruptHandler,
    validate_file_exists,
)
from ..cli_args import get_playback_arguments, get_evaluation_arguments

# Import existing playback functions
from ..collect_data import run_playback, browse_episodes, evaluate_model


class PlaybackPipeline:
    """Handles all playback operations"""

    @staticmethod
    def add_subparsers(subparsers):
        """Add playback subcommands to the argument parser"""
        replay_parser = subparsers.add_parser("replay", help="Replay operations")
        replay_subparsers = replay_parser.add_subparsers(
            dest="replay_command", help="Replay command"
        )

        # Episode replay
        episode_parser = replay_subparsers.add_parser(
            "episode", help="Replay saved episode"
        )
        episode_parser.add_argument(
            "--episode-file",
            type=str,
            required=True,
            help="Episode file path for replay",
        )
        episode_parser.add_argument("--config", type=str, help="Config file path")

        # Browse episodes
        browse_parser = replay_subparsers.add_parser(
            "browse", help="Browse episode data"
        )
        browse_parser.add_argument(
            "--data-dir", type=str, default="data", help="Data directory path"
        )
        browse_parser.add_argument("--config", type=str, help="Config file path")

        return replay_parser

    @staticmethod
    def execute(cfg: DictConfig) -> int:
        """Execute the appropriate playback command with DictConfig"""
        try:
            with InterruptHandler("Playback interrupted by user"):
                # Get command from config or from command structure
                replay_command = None

                # Try to get from config first
                if cfg.get("replay", {}).get("mode"):
                    replay_command = cfg.replay.mode
                # Fallback to detecting from command structure
                elif "replay_command" in cfg:
                    replay_command = cfg.replay_command

                if replay_command == "episode":
                    return PlaybackPipeline._replay_episode(cfg)
                elif replay_command == "browse":
                    return PlaybackPipeline._browse_episodes(cfg)
                else:
                    print(f"Unknown replay command: {replay_command}")
                    return 1
        except Exception as e:
            print(f"Error during playback: {e}")
            return 1

    @staticmethod
    def _replay_episode(cfg: DictConfig) -> int:
        """Execute episode replay with DictConfig"""
        # Validate episode file exists
        episode_file = validate_file_exists(cfg.get("episode_file"), "Episode file")

        # Convert config to dict for compatibility
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        print("=" * 60)
        print("REPLAYING EPISODE")
        print("=" * 60)
        print(f"Episode file: {episode_file}")

        # Setup logging
        run_name = generate_run_name("replay_episode")
        logger = setup_logging(run_name)

        try:
            run_playback(config_dict, str(episode_file))
            print("Episode replay completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during episode replay: {e}")
            return 1

    @staticmethod
    def _browse_episodes(cfg: DictConfig) -> int:
        """Execute episode browsing with DictConfig"""
        # Validate data directory exists
        data_dir = Path(cfg.get("data_dir", "data"))
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            return 1

        # Convert config to dict for compatibility
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        print("=" * 60)
        print("BROWSING EPISODE DATA")
        print("=" * 60)
        print(f"Data directory: {data_dir}")

        # Setup logging
        run_name = generate_run_name("browse_episodes")
        logger = setup_logging(run_name)

        try:
            browse_episodes(config_dict, str(data_dir))
            print("Episode browsing completed!")
            return 0
        except Exception as e:
            print(f"Error during episode browsing: {e}")
            return 1


class EvaluationPipeline:
    """Handles all evaluation operations"""

    @staticmethod
    def add_subparsers(subparsers):
        """Add evaluation subcommands to the argument parser"""
        eval_parser = subparsers.add_parser("evaluate", help="Evaluation operations")
        eval_subparsers = eval_parser.add_subparsers(
            dest="evaluate_command", help="Evaluation command"
        )

        # Model evaluation
        model_parser = eval_subparsers.add_parser(
            "model", help="Evaluate trained model"
        )
        for arg in get_evaluation_arguments():
            model_parser.add_argument(*arg["args"], **arg["kwargs"])

        return eval_parser

    @staticmethod
    def execute(cfg: DictConfig) -> int:
        """Execute the appropriate evaluation command with DictConfig"""
        try:
            with InterruptHandler("Evaluation interrupted by user"):
                # Get command from config or from command structure
                evaluate_command = None

                # Try to get from config first
                if cfg.get("evaluate", {}).get("mode"):
                    evaluate_command = cfg.evaluate.mode
                # Fallback to detecting from command structure
                elif "evaluate_command" in cfg:
                    evaluate_command = cfg.evaluate_command

                if evaluate_command == "model":
                    return EvaluationPipeline._evaluate_model(cfg)
                else:
                    print(f"Unknown evaluation command: {evaluate_command}")
                    return 1
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1

    @staticmethod
    def _evaluate_model(cfg: DictConfig) -> int:
        """Execute model evaluation with DictConfig"""
        # Validate model file exists
        model_file = validate_file_exists(cfg.get("model"), "Model file")

        # Convert config to dict for compatibility
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        print("=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        print(f"Model: {model_file}")

        # Setup logging
        run_name = generate_run_name("evaluate_model")
        logger = setup_logging(run_name)

        try:
            stats = evaluate_model(str(model_file), config_dict)
            print("Model evaluation completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return 1
