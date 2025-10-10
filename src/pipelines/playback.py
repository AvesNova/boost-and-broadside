"""
Playback pipeline module - handles human play and episode replay
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Import shared utilities
from ..config import load_config, get_default_config
from ..utils import (
    setup_logging,
    generate_run_name,
    InterruptHandler,
    validate_file_exists,
)
from ..cli_args import get_playback_arguments, get_evaluation_arguments

# Import existing playback functions
from collect_data import run_playback, browse_episodes, evaluate_model


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
    def execute(args) -> int:
        """Execute the appropriate playback command"""
        try:
            with InterruptHandler("Playback interrupted by user"):
                if args.replay_command == "episode":
                    return PlaybackPipeline._replay_episode(args)
                elif args.replay_command == "browse":
                    return PlaybackPipeline._browse_episodes(args)
                else:
                    print(f"Unknown replay command: {args.replay_command}")
                    return 1
        except Exception as e:
            print(f"Error during playback: {e}")
            return 1

    @staticmethod
    def _replay_episode(args) -> int:
        """Execute episode replay"""
        # Validate episode file exists
        episode_file = validate_file_exists(args.episode_file, "Episode file")

        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config())
        else:
            config = get_default_config()

        print("=" * 60)
        print("REPLAYING EPISODE")
        print("=" * 60)
        print(f"Episode file: {episode_file}")
        print(f"Config: {args.config or 'default'}")

        # Setup logging
        run_name = generate_run_name("replay_episode")
        logger = setup_logging(run_name)

        try:
            run_playback(config, str(episode_file))
            print("Episode replay completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during episode replay: {e}")
            return 1

    @staticmethod
    def _browse_episodes(args) -> int:
        """Execute episode browsing"""
        # Validate data directory exists
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            return 1

        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config())
        else:
            config = get_default_config()

        print("=" * 60)
        print("BROWSING EPISODE DATA")
        print("=" * 60)
        print(f"Data directory: {data_dir}")
        print(f"Config: {args.config or 'default'}")

        # Setup logging
        run_name = generate_run_name("browse_episodes")
        logger = setup_logging(run_name)

        try:
            browse_episodes(config, str(data_dir))
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
    def execute(args) -> int:
        """Execute the appropriate evaluation command"""
        try:
            with InterruptHandler("Evaluation interrupted by user"):
                if args.evaluate_command == "model":
                    return EvaluationPipeline._evaluate_model(args)
                else:
                    print(f"Unknown evaluation command: {args.evaluate_command}")
                    return 1
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1

    @staticmethod
    def _evaluate_model(args) -> int:
        """Execute model evaluation"""
        # Validate model file exists
        model_file = validate_file_exists(args.model, "Model file")

        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config())
        else:
            config = get_default_config()

        print("=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        print(f"Model: {model_file}")
        print(f"Config: {args.config or 'default'}")

        # Setup logging
        run_name = generate_run_name("evaluate_model")
        logger = setup_logging(run_name)

        try:
            stats = evaluate_model(str(model_file), config)
            print("Model evaluation completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return 1
