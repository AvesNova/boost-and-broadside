"""
Evaluation pipeline module - handles model evaluation
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
from ..cli_args import get_evaluation_arguments

# Import existing evaluation functions
from collect_data import evaluate_model


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
