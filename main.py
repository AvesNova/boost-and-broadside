#!/usr/bin/env python3
"""
Boost and Broadside - Unified Command Line Interface

This is the main entry point for all Boost and Broadside functionality.
It provides a unified command-line interface for training, data collection,
evaluation, and playback operations using Hydra for configuration management.
"""

import argparse
import sys
from typing import Dict, Any

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Import pipeline modules
from src.pipelines import (
    TrainingPipeline,
    DataCollectionPipeline,
    PlayPipeline,
    PlaybackPipeline,
    EvaluationPipeline,
)
from src.pipelines.data_collection import PlayPipeline


def create_parser():
    """Create the main argument parser with Hydra support"""
    parser = argparse.ArgumentParser(
        description="Boost and Broadside - Physics-based ship combat with transformer-based AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Add conflict_handler to allow Hydra overrides
        conflict_handler="resolve",
        epilog="""
Examples:
  # Training
  python main.py train bc
  python main.py train rl model.transformer.embed_dim=128
  python main.py train full training.rl.total_timesteps=5000000

  # Data Collection
  python main.py collect bc
  python main.py collect selfplay data_collection.selfplay_data.output_dir=custom_data

  # Playing
  python main.py play human

  # Evaluation
  python main.py evaluate model --model checkpoints/best_model.pt

  # Replay
  python main.py replay episode --episode-file data/bc_pretraining/1v1_episodes.pkl.gz

  # Browse
  python main.py replay browse --data-dir data/

  # Hydra overrides
  python main.py train bc model.transformer.embed_dim=128 model.ppo.learning_rate=0.0001
        """,
    )

    # Add subparsers for each pipeline
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add pipeline subparsers
    TrainingPipeline.add_subparsers(subparsers)
    DataCollectionPipeline.add_subparsers(subparsers)
    PlayPipeline.add_subparsers(subparsers)
    PlaybackPipeline.add_subparsers(subparsers)
    EvaluationPipeline.add_subparsers(subparsers)

    return parser


def get_command_specific_config(command_parts: list) -> str:
    """Get command-specific config based on command parts"""
    if not command_parts:
        return "base"

    # Map commands to config files
    config_map = {
        ("train", "bc"): "train/bc",
        ("train", "rl"): "train/rl",
        (
            "train",
            "full",
        ): "train/full_simple",  # Use the simple config to avoid recursion
        ("collect", "bc"): "collect/bc",
        ("collect", "selfplay"): "collect/selfplay",
        ("play", "human"): "play/human",
        ("replay", "episode"): "replay/episode",
        ("replay", "browse"): "replay/browse",
        ("evaluate", "model"): "evaluate/model",
    }

    return config_map.get(tuple(command_parts), "base")


def main() -> int:
    """Main entry point with full Hydra configuration"""
    # Parse command line arguments to determine the command
    parser = create_parser()
    cli_args, unknown_args = parser.parse_known_args()

    # If no command is specified, print help
    if not hasattr(cli_args, "command") or cli_args.command is None:
        parser.print_help()
        return 1

    # Determine command parts for config selection
    command_parts = []
    if hasattr(cli_args, "command") and cli_args.command:
        command_parts.append(cli_args.command)

    # Add subcommand if exists
    subcommand_attr = f"{cli_args.command}_command"
    if hasattr(cli_args, subcommand_attr) and getattr(cli_args, subcommand_attr):
        command_parts.append(getattr(cli_args, subcommand_attr))

    # Get command-specific configuration
    command_config = get_command_specific_config(command_parts)

    # Get the original working directory (where Hydra was launched from)
    import os

    original_cwd = os.getcwd()
    config_dir = os.path.join(original_cwd, "src/config")

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Disable struct mode to allow dynamic keys
            from hydra.core.config_store import ConfigStore

            cs = ConfigStore.instance()
            cs.store(name="base_config", node={})
            # Compose the configuration with command-specific overrides
            # Include any unknown args as Hydra overrides
            overrides = unknown_args

            # Add CLI arguments as overrides
            cli_dict = vars(cli_args)
            for key, value in cli_dict.items():
                if key != "command" and value is not None and key != "config":
                    # Convert args with underscores to config with dots
                    config_key = key.replace("_", ".")
                    # Use + to append to config instead of overriding
                    overrides.append(f"+{config_key}={value}")

            # Create the configuration
            cfg = compose(
                config_name=command_config,
                overrides=overrides,
                return_hydra_config=False,
            )

            # Execute the appropriate pipeline with the Hydra config
            try:
                if cli_args.command == "train":
                    return TrainingPipeline.execute(cfg)
                elif cli_args.command == "collect":
                    return DataCollectionPipeline.execute(cfg)
                elif cli_args.command == "play":
                    return PlayPipeline.execute(cfg)
                elif cli_args.command == "replay":
                    return PlaybackPipeline.execute(cfg)
                elif cli_args.command == "evaluate":
                    return EvaluationPipeline.execute(cfg)
                else:
                    print(f"Unknown command: {cli_args.command}")
                    parser.print_help()
                    return 1
            except KeyboardInterrupt:
                print("\nOperation interrupted by user")
                return 1
            except Exception as e:
                print(f"Error: {e}")
                return 1

    except Exception as e:
        print(f"Error initializing Hydra: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
