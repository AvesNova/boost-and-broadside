"""
Shared argument parsing utilities for Boost and Broadside
"""

import argparse
from typing import Dict, Any, List, Optional


def add_config_argument(
    parser: argparse.ArgumentParser, default: str = "src/unified_training.yaml"
) -> None:
    """
    Add standard config argument to a parser

    Args:
        parser: Argument parser to add the argument to
        default: Default value for the config argument
    """
    parser.add_argument(
        "--config",
        type=str,
        default=default,
        help="Config file path",
    )


def add_run_name_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add standard run-name argument to a parser

    Args:
        parser: Argument parser to add the argument to
    """
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for checkpoints and logs",
    )


def add_output_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add standard output directory argument to a parser

    Args:
        parser: Argument parser to add the argument to
    """
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory override",
    )


def add_model_argument(parser: argparse.ArgumentParser, required: bool = False) -> None:
    """
    Add model path argument to a parser

    Args:
        parser: Argument parser to add the argument to
        required: Whether the argument is required
    """
    parser.add_argument(
        "--model",
        type=str,
        required=required,
        help="Model path",
    )


def add_bc_model_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add BC model path argument to a parser

    Args:
        parser: Argument parser to add the argument to
    """
    parser.add_argument(
        "--bc-model",
        type=str,
        default=None,
        help="Path to BC model for RL initialization",
    )


def add_episode_file_argument(
    parser: argparse.ArgumentParser, required: bool = False
) -> None:
    """
    Add episode file argument to a parser

    Args:
        parser: Argument parser to add the argument to
        required: Whether the argument is required
    """
    parser.add_argument(
        "--episode-file",
        type=str,
        required=required,
        help="Episode file path",
    )


def add_data_dir_argument(
    parser: argparse.ArgumentParser, default: str = "data"
) -> None:
    """
    Add data directory argument to a parser

    Args:
        parser: Argument parser to add the argument to
        default: Default value for the data directory
    """
    parser.add_argument(
        "--data-dir",
        type=str,
        default=default,
        help="Data directory path",
    )


def add_skip_bc_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add skip BC argument to a parser

    Args:
        parser: Argument parser to add the argument to
    """
    parser.add_argument(
        "--skip-bc",
        action="store_true",
        help="Skip BC pretraining in full mode",
    )


def create_subparser(
    subparsers, name: str, help: str, arguments: Optional[List[Dict[str, Any]]] = None
) -> argparse.ArgumentParser:
    """
    Create a subparser with standard arguments

    Args:
        subparsers: Subparsers object to add to
        name: Name of the subcommand
        help: Help text for the subcommand
        arguments: List of argument dictionaries to add

    Returns:
        Created subparser
    """
    parser = subparsers.add_parser(name, help=help)

    if arguments:
        for arg in arguments:
            parser.add_argument(*arg["args"], **arg["kwargs"])

    return parser


def get_training_arguments() -> List[Dict[str, Any]]:
    """
    Get standard arguments for training commands

    Returns:
        List of argument dictionaries
    """
    return [
        {
            "args": ["--config"],
            "kwargs": {
                "type": str,
                "default": "src/unified_training.yaml",
                "help": "Config file path",
            },
        },
        {
            "args": ["--run-name"],
            "kwargs": {
                "type": str,
                "default": None,
                "help": "Run name for checkpoints and logs",
            },
        },
    ]


def get_rl_training_arguments() -> List[Dict[str, Any]]:
    """
    Get arguments specific to RL training

    Returns:
        List of argument dictionaries
    """
    args = get_training_arguments()
    args.append(
        {
            "args": ["--bc-model"],
            "kwargs": {
                "type": str,
                "default": None,
                "help": "Path to BC model for RL initialization",
            },
        }
    )
    return args


def get_full_training_arguments() -> List[Dict[str, Any]]:
    """
    Get arguments for full training pipeline

    Returns:
        List of argument dictionaries
    """
    args = get_training_arguments()
    args.append(
        {
            "args": ["--skip-bc"],
            "kwargs": {
                "action": "store_true",
                "help": "Skip BC pretraining in full mode",
            },
        }
    )
    return args


def get_data_collection_arguments() -> List[Dict[str, Any]]:
    """
    Get arguments for data collection commands

    Returns:
        List of argument dictionaries
    """
    return [
        {
            "args": ["--config"],
            "kwargs": {
                "type": str,
                "help": "Config file path",
            },
        },
        {
            "args": ["--output"],
            "kwargs": {
                "type": str,
                "help": "Output directory override",
            },
        },
    ]


def get_playback_arguments() -> List[Dict[str, Any]]:
    """
    Get arguments for playback commands

    Returns:
        List of argument dictionaries
    """
    return [
        {
            "args": ["--config"],
            "kwargs": {
                "type": str,
                "help": "Config file path",
            },
        },
    ]


def get_evaluation_arguments() -> List[Dict[str, Any]]:
    """
    Get arguments for evaluation commands

    Returns:
        List of argument dictionaries
    """
    return [
        {
            "args": ["--model"],
            "kwargs": {
                "type": str,
                "required": True,
                "help": "Model path for evaluation",
            },
        },
        {
            "args": ["--config"],
            "kwargs": {
                "type": str,
                "help": "Config file path",
            },
        },
    ]


def validate_args(args: argparse.Namespace, command: str) -> None:
    """
    Validate common arguments

    Args:
        args: Parsed arguments
        command: Command context for validation

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate config file exists
    if hasattr(args, "config") and args.config:
        from pathlib import Path

        if not Path(args.config).exists():
            raise ValueError(f"Config file not found: {args.config}")

    # Validate model file exists
    if hasattr(args, "model") and args.model:
        from pathlib import Path

        if not Path(args.model).exists():
            raise ValueError(f"Model file not found: {args.model}")

    # Validate BC model file exists
    if hasattr(args, "bc_model") and args.bc_model:
        from pathlib import Path

        if not Path(args.bc_model).exists():
            raise ValueError(f"BC model file not found: {args.bc_model}")

    # Validate episode file exists
    if hasattr(args, "episode_file") and args.episode_file:
        from pathlib import Path

        if not Path(args.episode_file).exists():
            raise ValueError(f"Episode file not found: {args.episode_file}")

    # Validate data directory exists
    if hasattr(args, "data_dir") and args.data_dir:
        from pathlib import Path

        if not Path(args.data_dir).exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")
