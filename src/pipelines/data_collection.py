"""
Data collection pipeline module - handles data collection and human play
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import shared utilities
from ..config import load_config, get_default_config
from ..utils import setup_logging, generate_run_name, InterruptHandler
from ..cli_args import get_data_collection_arguments

# Import existing data collection functions
from collect_data import collect_bc_data, run_human_play, collect_selfplay_data


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
    def execute(args) -> int:
        """Execute the appropriate data collection command"""
        try:
            with InterruptHandler("Data collection interrupted by user"):
                if args.collect_command == "bc":
                    return DataCollectionPipeline._collect_bc(args)
                elif args.collect_command == "selfplay":
                    return DataCollectionPipeline._collect_selfplay(args)
                else:
                    print(f"Unknown collection command: {args.collect_command}")
                    return 1
        except Exception as e:
            print(f"Error during data collection: {e}")
            return 1

    @staticmethod
    def _collect_bc(args) -> int:
        """Execute behavior cloning data collection"""
        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config("data_collection"))
        else:
            config = get_default_config("data_collection")

        # Override output directory if specified
        if args.output:
            if "data_collection" not in config:
                config["data_collection"] = {}
            if "bc_data" not in config["data_collection"]:
                config["data_collection"]["bc_data"] = {}
            config["data_collection"]["bc_data"]["output_dir"] = args.output

        # Add timestamp to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "data_collection" in config and "bc_data" in config["data_collection"]:
            bc_config = config["data_collection"]["bc_data"]
            bc_config["output_dir"] = (
                f"{bc_config.get('output_dir', 'data/bc_pretraining')}_{timestamp}"
            )

        print("=" * 60)
        print("COLLECTING BEHAVIOR CLONING DATA")
        print("=" * 60)
        print(f"Config: {args.config or 'default'}")
        print(f"Output directory: {config['data_collection']['bc_data']['output_dir']}")

        # Setup logging
        run_name = generate_run_name("collect_bc")
        logger = setup_logging(run_name)

        # Execute data collection
        try:
            collect_bc_data(config)
            print("BC data collection completed successfully!")
            return 0
        except Exception as e:
            print(f"Error during BC data collection: {e}")
            return 1

    @staticmethod
    def _collect_selfplay(args) -> int:
        """Execute self-play data collection"""
        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config("data_collection"))
        else:
            config = get_default_config("data_collection")

        # Override output directory if specified
        if args.output:
            config["output_dir"] = args.output
        else:
            # Add timestamp to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config["output_dir"] = f"data/selfplay_{timestamp}"

        print("=" * 60)
        print("COLLECTING SELF-PLAY DATA")
        print("=" * 60)
        print(f"Config: {args.config or 'default'}")
        print(f"Output directory: {config['output_dir']}")

        # Setup logging
        run_name = generate_run_name("collect_selfplay")
        logger = setup_logging(run_name)

        # Execute data collection
        try:
            collect_selfplay_data(config)
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
    def execute(args) -> int:
        """Execute the appropriate play command"""
        try:
            with InterruptHandler("Game interrupted by user"):
                if args.play_command == "human":
                    return PlayPipeline._play_human(args)
                else:
                    print(f"Unknown play command: {args.play_command}")
                    return 1
        except Exception as e:
            print(f"Error during play: {e}")
            return 1

    @staticmethod
    def _play_human(args) -> int:
        """Execute human play"""
        # Load configuration
        if args.config:
            config = load_config(args.config, get_default_config())
        else:
            config = get_default_config()

        print("=" * 60)
        print("HUMAN VS AI PLAY")
        print("=" * 60)
        print("Controls: WASD/Arrow Keys (move), Space (shoot), Shift (sharp turn)")
        print("Close window or Ctrl+C to quit")

        # Setup logging
        run_name = generate_run_name("play_human")
        logger = setup_logging(run_name)

        try:
            run_human_play(config)
            return 0
        except Exception as e:
            print(f"Error during human play: {e}")
            return 1
