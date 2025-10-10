#!/usr/bin/env python3
"""
Boost and Broadside - Unified Command Line Interface

This is the main entry point for all Boost and Broadside functionality.
It provides a unified command-line interface for training, data collection,
evaluation, and playback operations.
"""

import argparse
import sys
from typing import Dict, Any

# Import pipeline modules
from src.pipelines import (
    TrainingPipeline,
    DataCollectionPipeline,
    PlaybackPipeline,
    EvaluationPipeline,
)
from src.pipelines.data_collection import PlayPipeline


def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="Boost and Broadside - Physics-based ship combat with transformer-based AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train bc --config src/unified_training.yaml
  python main.py train rl --config src/unified_training.yaml --bc-model model.pt
  python main.py train full --config src/unified_training.yaml --skip-bc

  # Data Collection
  python main.py collect bc --config src/unified_training.yaml
  python main.py collect selfplay --config src/unified_training.yaml

  # Playing
  python main.py play human --config src/unified_training.yaml

  # Evaluation
  python main.py evaluate model --model checkpoints/best_model.pt --config src/unified_training.yaml

  # Replay
  python main.py replay episode --episode-file data/bc_pretraining/1v1_episodes.pkl.gz

  # Browse
  python main.py replay browse --data-dir data/
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


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command is specified, print help
    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
        return 1

    # Execute the appropriate pipeline
    try:
        if args.command == "train":
            return TrainingPipeline.execute(args)
        elif args.command == "collect":
            return DataCollectionPipeline.execute(args)
        elif args.command == "play":
            return PlayPipeline.execute(args)
        elif args.command == "replay":
            return PlaybackPipeline.execute(args)
        elif args.command == "evaluate":
            return EvaluationPipeline.execute(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
