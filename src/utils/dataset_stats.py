import h5py
import numpy as np
import torch
import logging
from pathlib import Path
from core.constants import TurnActions, NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS

log = logging.getLogger(__name__)

def calculate_action_counts(data_path: str) -> dict[str, np.ndarray]:
    """
    Calculate or retrieve action counts from HDF5 dataset.
    Returns dictionary with keys: 'power', 'turn', 'shoot'.
    """
    try:
        # Open in r+ to ensure we can write attributes if needed
        # Use 'r' if we fail to open 'r+' (e.g. read-only file)
        f = h5py.File(data_path, "r+")
        mode = "r+"
    except OSError:
        f = h5py.File(data_path, "r")
        mode = "r"
        log.warning(f"Could not open {data_path} in write mode. Caching to HDF5 attributes will be disabled.")

    with f:
        # Check attributes first
        if "action_counts_power" in f.attrs and \
           "action_counts_turn" in f.attrs and \
           "action_counts_shoot" in f.attrs:
            
            log.info("Loading cached action counts from HDF5 attributes.")
            return {
                "power": f.attrs["action_counts_power"],
                "turn": f.attrs["action_counts_turn"],
                "shoot": f.attrs["action_counts_shoot"]
            }
        
        log.info("Calculating action counts from scratch (this may take a moment)...")
        
        if "actions" not in f:
            log.warning("No 'actions' dataset found. Returning ones.")
            return {
                "power": np.ones(NUM_POWER_ACTIONS),
                "turn": np.ones(NUM_TURN_ACTIONS),
                "shoot": np.ones(NUM_SHOOT_ACTIONS)
            }

        # Load all actions. 
        # For very large datasets, this might need chunked processing.
        # Assuming fits in memory for now.
        actions = f["actions"][:] 
        
        # Flatten: (N, MaxShips, 3) -> (N*MaxShips, 3)
        flat_actions = actions.reshape(-1, 3)
        
        # Power
        counts_p = np.bincount(flat_actions[:, 0], minlength=NUM_POWER_ACTIONS)
        # Turn
        counts_t = np.bincount(flat_actions[:, 1], minlength=NUM_TURN_ACTIONS)
        # Shoot
        counts_s = np.bincount(flat_actions[:, 2], minlength=NUM_SHOOT_ACTIONS)
        
        # Cache if possible
        if mode == "r+":
            try:
                f.attrs["action_counts_power"] = counts_p
                f.attrs["action_counts_turn"] = counts_t
                f.attrs["action_counts_shoot"] = counts_s
                log.info("Cached action counts to HDF5 attributes.")
            except Exception as e:
                log.warning(f"Failed to cache attributes: {e}")
        
        return {
            "power": counts_p,
            "turn": counts_t,
            "shoot": counts_s
        }

def compute_class_weights(counts: np.ndarray, cap: float = 10.0, power: float = 0.5) -> torch.Tensor:
    """
    Compute weights: w = min(cap, (1 / freq)^power)
    freq = (count + 1) / (total + num_classes) [Laplace smoothing]
    """
    total = counts.sum()
    num_classes = len(counts)
    if total == 0:
        return torch.ones(num_classes)
        
    # Standard frequency with Laplace smoothing to handle zero counts
    freq = (counts + 1.0) / (total + num_classes)
    
    # Use torch for calculation
    x = torch.tensor(freq, dtype=torch.float32)
    
    # Compute inverse frequency raised to the power (usually 0.5 for sqrt)
    # w = (1/x)^power = x^(-power)
    weights = torch.pow(x, -power)
    
    # Clip to maximum weight to avoid instability
    clipped = torch.clamp(weights, max=cap)
    
    return clipped

def apply_turn_exceptions(weights: torch.Tensor) -> torch.Tensor:
    """Apply exceptions for AIR_BRAKE and SHARP_AIR_BRAKE."""
    # indices: AIR_BRAKE=5, SHARP_AIR_BRAKE=6
    if len(weights) != NUM_TURN_ACTIONS:
        log.warning(f"Turn weights length {len(weights)} != {NUM_TURN_ACTIONS}. Skipping exceptions.")
        return weights
        
    weights[TurnActions.AIR_BRAKE] = 1e-7
    weights[TurnActions.SHARP_AIR_BRAKE] = 1e-7
    return weights

def normalize_weights(weights: torch.Tensor, counts: np.ndarray) -> torch.Tensor:
    """
    Normalize weights such that sum(weight * freq) = 1.
    """
    total = counts.sum()
    if total == 0:
        return weights
        
    # Ensure freqs is on same device/dtype as weights
    freqs = torch.tensor(counts / total, dtype=weights.dtype, device=weights.device)
    
    expected_val = torch.sum(weights * freqs)
    
    if expected_val < 1e-9:
        return weights
        
    return weights / expected_val
