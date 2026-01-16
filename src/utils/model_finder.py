"""
Utility functions for finding trained models in the models directory.

Provides functions to locate the most recent or best-performing models
for BC and RL training.
"""

import csv
from pathlib import Path
from typing import Literal

MODELS_DIR = Path("models")


def find_most_recent_model(
    model_type: Literal["bc", "rl", "world_model"],
) -> str | None:
    """
    Find the most recent model file in the models/{model_type} directory.

    Args:
        model_type: Type of model to search for ("bc" or "rl").

    Returns:
        Absolute path to the model file, or None if not found.
    """
    search_dir = MODELS_DIR / model_type
    if not search_dir.exists():
        return None

    # Get all run directories
    run_dirs = [
        d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        return None

    # Sort by name (timestamp) descending
    run_dirs.sort(key=lambda x: x.name, reverse=True)

    for run_dir in run_dirs:
        # Check for best model first
        if model_type == "bc":
            model_path = run_dir / "best_bc_model.pth"
        elif model_type == "world_model":
            model_path = run_dir / "best_world_model.pth"
        else:
            # RL: Check for final zip, then transformer pth, then WorldModel pth
            paths_to_check = [
                run_dir / "final_rl_model.zip",
                run_dir / "final_rl_transformer.pth",
                run_dir / "final_world_model.pth",
            ]
            for p in paths_to_check:
                if p.exists():
                    return str(p.absolute())
            model_path = None  # Logic continues below

        if model_path and model_path.exists():
            return str(model_path.absolute())

        # Fallback to final model if best not found
        if model_type == "bc":
            model_path = run_dir / "final_bc_model.pth"
        elif model_type == "world_model":
            model_path = run_dir / "final_world_model.pth"

        if model_type in ["bc", "world_model"] and model_path and model_path.exists():
            return str(model_path.absolute())

        # RL Fallback: Check checkpoints directory
        if model_type == "rl":
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                # Find latest zip
                zips = list(ckpt_dir.glob("*.zip"))
                if zips:
                    # Sort by modification time or name (steps)
                    # Name format: rl_model_STEP_steps.zip. Extract STEP.
                    def get_step(p):
                        try:
                            return int(p.stem.split("_")[2])
                        except:
                            return 0

                    latest_zip = max(zips, key=get_step)
                    return str(latest_zip.absolute())

    return None

    return None


def find_best_model(model_type: Literal["bc", "rl", "world_model"]) -> str | None:
    """
    Find the best model based on validation loss from training logs.

    For BC models, parses training_log.csv to find the run with the lowest
    validation loss. For RL models, currently returns None as SB3 logging
    is not yet parsed.

    Args:
        model_type: Type of model to search for ("bc" or "rl").

    Returns:
        Absolute path to the best model file, or None if not found.
    """
    search_dir = MODELS_DIR / model_type
    if not search_dir.exists():
        return None

    run_dirs = [
        d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        return None

    best_val_loss = float("inf")
    best_model_path = None

    for run_dir in run_dirs:
        # RL best finding not yet implemented (would need to parse SB3 logs)
        if model_type == "rl":
            continue

        log_path = run_dir / "training_log.csv"
        if not log_path.exists():
            continue

        try:
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                min_run_loss = float("inf")
                for row in reader:
                    try:
                        val_loss = float(row["val_loss"])
                        if val_loss < min_run_loss:
                            min_run_loss = val_loss
                    except (ValueError, KeyError):
                        continue

                if min_run_loss < best_val_loss:
                    # Check if model file exists
                    if model_type == "bc":
                        best_path = run_dir / "best_bc_model.pth"
                        final_path = run_dir / "final_bc_model.pth"
                    elif model_type == "world_model":
                        best_path = run_dir / "best_world_model.pth"
                        final_path = run_dir / "final_world_model.pth"
                    else:
                        continue

                    if best_path.exists():
                        best_val_loss = min_run_loss
                        best_model_path = str(best_path.absolute())
                    elif final_path.exists():
                        best_val_loss = min_run_loss
                        best_model_path = str(final_path.absolute())

        except Exception:
            continue

    return best_model_path
