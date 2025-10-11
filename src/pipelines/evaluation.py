"""
Evaluation pipeline module - handles model evaluation
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
from ..cli_args import get_evaluation_arguments

# Import existing evaluation functions
from ..collect_data import evaluate_model


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
