import os
import csv
from pathlib import Path
from typing import Optional, Literal

MODELS_DIR = Path("models")

def find_most_recent_model(model_type: Literal["bc", "rl"]) -> Optional[str]:
    """
    Find the most recent model file in the models/{model_type} directory.
    Returns the absolute path to the model file or None if not found.
    """
    search_dir = MODELS_DIR / model_type
    if not search_dir.exists():
        return None
    
    # Get all run directories
    run_dirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) descending
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    
    for run_dir in run_dirs:
        # Check for best model first
        if model_type == "bc":
            model_path = run_dir / "best_bc_model.pth"
        else:
            model_path = run_dir / "final_rl_model.zip" # RL saves zip
            # Actually RL saves final_rl_transformer.pth for the agent
            model_path = run_dir / "final_rl_transformer.pth"
            
        if model_path.exists():
            return str(model_path.absolute())
        
        # Fallback to final model if best not found (for BC)
        if model_type == "bc":
            model_path = run_dir / "final_bc_model.pth"
            if model_path.exists():
                return str(model_path.absolute())
            
    return None

def find_best_model(model_type: Literal["bc", "rl"]) -> Optional[str]:
    """
    Find the best model based on validation loss from training logs.
    Returns the absolute path to the model file or None if not found.
    """
    search_dir = MODELS_DIR / model_type
    if not search_dir.exists():
        return None
        
    run_dirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
        
    best_val_loss = float("inf")
    best_model_path = None
    
    for run_dir in run_dirs:
        # RL logs might be different (monitor.csv), but we used custom CSV for BC.
        # For RL, SB3 logs to monitor.csv if configured, or we can check tensorboard.
        # But our train_rl doesn't write a custom CSV like BC does.
        # So find_best_model might only work for BC for now unless we parse SB3 logs.
        if model_type == "rl":
            # For now, just return most recent for RL as we don't have easy CSV parsing yet
            # Or we could skip RL best finding.
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
                    model_path = run_dir / "best_bc_model.pth"
                    if model_path.exists():
                        best_val_loss = min_run_loss
                        best_model_path = str(model_path.absolute())
                    else:
                        # Try final model
                        model_path = run_dir / "final_bc_model.pth"
                        if model_path.exists():
                            best_val_loss = min_run_loss
                            best_model_path = str(model_path.absolute())
                            
        except Exception:
            continue
            
    return best_model_path
