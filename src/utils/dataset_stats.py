import h5py
import torch
import numpy as np
import logging
import json
from tqdm import tqdm

log = logging.getLogger(__name__)

def calculate_action_stats(data_path: str, batch_size: int = 10000, max_weight: float = 10.0, epsilon: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Calculate or retrieve class weights for action heads from HDF5 metadata.
    
    Args:
        data_path: Path to the HDF5 file.
        batch_size: Chunk size for reading data during calculation.
        max_weight: Maximum allowed weight value (weights are clamped to this).
        epsilon: Small value added to counts to avoid division by zero.
        
    Returns:
        dict containing 'power', 'turn', 'shoot' tensors with class weights.
    """
    with h5py.File(data_path, "r+") as f:
        # ... (Loading logic remains same) ...
        # 1. Check if metadata exists
        if "action_weights_sqrt" in f.attrs:
            log.info("Loading pre-computed action weights from metadata.")
            weights_json = f.attrs["action_weights_sqrt"]
            weights_dict = json.loads(weights_json)
            
            w_p = torch.tensor(weights_dict["power"])
            w_t = torch.tensor(weights_dict["turn"])
            w_s = torch.tensor(weights_dict["shoot"])
            
            # Clamp loaded weights
            return {
                "power": torch.clamp(w_p, max=max_weight),
                "turn": torch.clamp(w_t, max=max_weight),
                "shoot": torch.clamp(w_s, max=max_weight)
            }
            
        # ... (Calculation logic) ...

        # Calculate Weights: N_total / (N_classes * (Count_class + epsilon))
        def get_weights(counts):
            total = counts.sum()
            n_classes = len(counts)
            if total == 0: return np.ones(n_classes)
            # Add epsilon to counts (Laplace-like smoothing if >= 1, or stability if small)
            # Use sqrt for less extreme weighting
            weights = np.sqrt(total / (n_classes * (counts + epsilon)))
            return weights
        log.info("Calculating action statistics (this measures full dataset)...")
        
        # Dimensions are (N, T, Ships, ActionDim) or similar.
        # Check shapes
        if "actions" not in f or "action_masks" not in f:
            log.warning("actions or action_masks not found in dataset. Using uniform weights.")
            return {
                "power": torch.ones(3),
                "turn": torch.ones(7),
                "shoot": torch.ones(2)
            }
            
        total_samples = f["actions"].shape[0]
        
        # Counters
        counts_p = np.zeros(3, dtype=np.int64)
        counts_t = np.zeros(7, dtype=np.int64)
        counts_s = np.zeros(2, dtype=np.int64)
        
        total_valid = 0
        
        for i in tqdm(range(0, total_samples, batch_size), desc="Scanning dataset"):
            end = min(i + batch_size, total_samples)
            
            actions = f["actions"][i:end] 
            
            masks = f["action_masks"][i:end] 
            
            # Flatten to (Samples, 3)
            # Actions: (B, Ships, 3) -> (B*Ships, 3)
            # Masks: (B, Ships) -> (B*Ships,)
            
            act_flat = actions.reshape(-1, 3)
            mask_flat = masks.reshape(-1)
            
            if len(act_flat) != len(mask_flat):
                log.warning(f"Shape mismatch: actions {act_flat.shape}, masks {mask_flat.shape}. Skipping chunk.")
                continue

            # Filtering
            # mask_flat is boolean-like (1 for valid, 0 for invalid/padding)
            valid_idx = mask_flat > 0
            
            valid_actions = act_flat[valid_idx]
            
            if len(valid_actions) == 0:
                continue
                
            # Count Power (idx 0)
            p_ids, p_counts = np.unique(valid_actions[:, 0].astype(int), return_counts=True)
            for pid, pcount in zip(p_ids, p_counts):
                if 0 <= pid < 3: counts_p[pid] += pcount
                
            # Count Turn (idx 1)
            t_ids, t_counts = np.unique(valid_actions[:, 1].astype(int), return_counts=True)
            for tid, tcount in zip(t_ids, t_counts):
                if 0 <= tid < 7: counts_t[tid] += tcount
                
            # Count Shoot (idx 2)
            s_ids, s_counts = np.unique(valid_actions[:, 2].astype(int), return_counts=True)
            for sid, scount in zip(s_ids, s_counts):
                if 0 <= sid < 2: counts_s[sid] += scount
                
            total_valid += len(valid_actions)
            
        log.info(f"Total valid samples processed: {total_valid}")
        log.info(f"Counts Power: {counts_p}")
        log.info(f"Counts Turn: {counts_t}")
        log.info(f"Counts Shoot: {counts_s}")
        
        # Calculate Weights: N_total / (N_classes * Count_class)
        # Add epsilon to counts to avoid div zero
        def get_weights(counts):
            total = counts.sum()
            n_classes = len(counts)
            if total == 0: return np.ones(n_classes)
            # Use sqrt for less extreme weighting
            weights = np.sqrt(total / (n_classes * (counts + 1e-6)))
            return weights
            
        w_p = get_weights(counts_p)
        w_t = get_weights(counts_t)
        w_s = get_weights(counts_s)
        
        # Save to metadata (Save original unclamped weights)
        weights_dict = {
            "power": w_p.tolist(),
            "turn": w_t.tolist(),
            "shoot": w_s.tolist()
        }
        
        log.info(f"Saving weights to metadata: {weights_dict}")
        f.attrs["action_weights_sqrt"] = json.dumps(weights_dict)
        
        # Return CLAMPED weights
        return {
            "power": torch.clamp(torch.tensor(w_p, dtype=torch.float32), max=max_weight),
            "turn": torch.clamp(torch.tensor(w_t, dtype=torch.float32), max=max_weight),
            "shoot": torch.clamp(torch.tensor(w_s, dtype=torch.float32), max=max_weight)
        }
