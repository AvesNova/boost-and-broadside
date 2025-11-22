import os
import csv
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")

def find_most_recent_model() -> Optional[str]:
    """
    Find the most recent model file (best_bc_model.pth) in the models directory.
    Returns the absolute path to the model file or None if not found.
    """
    if not MODELS_DIR.exists():
        return None
    
    # Get all run directories
    run_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) descending
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    
    for run_dir in run_dirs:
        model_path = run_dir / "best_bc_model.pth"
        if model_path.exists():
            return str(model_path.absolute())
        
        # Fallback to final model if best not found
        model_path = run_dir / "final_bc_model.pth"
        if model_path.exists():
            return str(model_path.absolute())
            
    return None

def find_best_model() -> Optional[str]:
    """
    Find the best model based on validation loss from training logs.
    Returns the absolute path to the model file or None if not found.
    """
    if not MODELS_DIR.exists():
        return None
        
    run_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
        
    best_val_loss = float("inf")
    best_model_path = None
    
    for run_dir in run_dirs:
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
