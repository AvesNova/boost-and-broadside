#!/usr/bin/env python3
"""
Optimized utility script to compute dataset statistics for the Yemong Dynamics model.
Includes Min/Max and separate Raw vs Normalized tables.
"""

import torch
import numpy as np
from pathlib import Path
import logging
import time

from boost_and_broadside.train.unified_dataset import UnifiedEpisodeDataset
from boost_and_broadside.models.components.encoders import RelationalEncoder
from boost_and_broadside.core.constants import (
    StateFeature, TargetFeature, TARGET_DIM, STATE_DIM,
    NORM_VELOCITY, NORM_ANGULAR_VELOCITY, NORM_HEALTH, NORM_POWER
)
from boost_and_broadside.train.data_loader import load_bc_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def compute_stats_vectorized(data: torch.Tensor, max_samples: int = 1000000) -> dict:
    """Computes statistics for a tensor (N, D) or (N,) on the current device."""
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    
    # Cap samples for expensive operations (quantile/median)
    if data.shape[0] > max_samples:
        indices = torch.randperm(data.shape[0], device=data.device)[:max_samples]
        data_subset = data[indices]
    else:
        data_subset = data

    with torch.no_grad():
        # 1. Basic Stats (Full data)
        means = data.mean(dim=0)
        stds = data.std(dim=0)
        mins = data.min(dim=0).values
        maxs = data.max(dim=0).values
        sem = stds / np.sqrt(data.shape[0])
        
        # 2. Quantiles (Expensive - Use subset)
        q = torch.quantile(data_subset, torch.tensor([0.25, 0.5, 0.75], device=data.device), dim=0)
        
    return {
        "mean": means.cpu(),
        "std": stds.cpu(),
        "min": mins.cpu(),
        "max": maxs.cpu(),
        "median": q[1].cpu(),
        "q1": q[0].cpu(),
        "q3": q[2].cpu(),
        "sem": sem.cpu()
    }

def main():
    t_start = time.time()
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = load_bc_data() 
    log.info(f"Loading data from {data_path}")
    
    dataset = UnifiedEpisodeDataset(data_path)
    total_steps = dataset.total_timesteps
    
    # Target: 500,000 steps
    num_steps = min(500000, total_steps - 1) 
    log.info(f"Processing {num_steps:,} contiguous steps on {device}.")
    
    rel_encoder = RelationalEncoder(d_model=256, n_layers=6).to(device)
    
    # 2. Load Raw and Normalized
    abs_start = 0 
    abs_end = abs_start + num_steps + 1
    
    log.info("Loading HDF5 data...")
    # Raw components
    health = dataset.get_slice("health", abs_start, abs_end).to(device).float()
    power = dataset.get_slice("power", abs_start, abs_end).to(device).float()
    vel = dataset.get_slice("velocity", abs_start, abs_end).to(device).float()
    ang = dataset.get_slice("ang_vel", abs_start, abs_end).to(device).float()
    
    # Auxiliary for relational/deltas
    pos = dataset.get_slice("position", abs_start, abs_end).to(device)
    att = dataset.get_slice("attitude", abs_start, abs_end).to(device)
    T_total, N, _ = vel.shape

    # Normalized tokens (post-norm)
    tokens = dataset.get_slice("tokens", abs_start, abs_end).to(device).float()

    # Identify episode boundaries
    is_boundary = torch.zeros(num_steps, dtype=torch.bool, device=device)
    for start in dataset.episode_starts:
        if abs_start < start < abs_end:
            is_boundary[int(start - abs_start - 1)] = True
    valid_mask = ~is_boundary
    
    # 3. State Stats (Pre and Post)
    log.info("Computing state statistics...")
    # Raw State (T, N, 5)
    raw_states = torch.stack([
        health, power, vel[..., 0], vel[..., 1], ang
    ], dim=-1)
    all_raw_states = raw_states[:-1].reshape(-1, STATE_DIM)
    all_norm_states = tokens[:-1].reshape(-1, STATE_DIM)
    
    # 4. Target Stats (Pre and Post)
    log.info("Computing target correlations...")
    # Pre-norm Target (Raw Deltas)
    d_pos = pos[1:] - pos[:-1]
    W, H = 1024.0, 1024.0
    d_pos[..., 0] = d_pos[..., 0] - torch.round(d_pos[..., 0] / W) * W
    d_pos[..., 1] = d_pos[..., 1] - torch.round(d_pos[..., 1] / H) * H
    
    raw_d_state = raw_states[1:] - raw_states[:-1]
    raw_targets = torch.zeros((num_steps, N, TARGET_DIM), device=device)
    raw_targets[..., 0:2] = d_pos
    raw_targets[..., 2:] = raw_d_state
    all_raw_targets = raw_targets[valid_mask].reshape(-1, TARGET_DIM)
    
    # Post-norm Target (Residual Deltas)
    norm_d_state = tokens[1:] - tokens[:-1]
    norm_targets = torch.zeros((num_steps, N, TARGET_DIM), device=device)
    norm_targets[..., TargetFeature.DX:TargetFeature.DY+1] = d_pos # dx, dy are always raw
    norm_targets[..., TargetFeature.DVX] = norm_d_state[..., StateFeature.VX]
    norm_targets[..., TargetFeature.DVY] = norm_d_state[..., StateFeature.VY]
    norm_targets[..., TargetFeature.DHEALTH] = norm_d_state[..., StateFeature.HEALTH]
    norm_targets[..., TargetFeature.DPOWER] = norm_d_state[..., StateFeature.POWER]
    norm_targets[..., TargetFeature.DANG_VEL] = norm_d_state[..., StateFeature.ANG_VEL]
    all_norm_targets = norm_targets[valid_mask].reshape(-1, TARGET_DIM)
    
    # 5. Relational Features (Chunked)
    log.info("Computing relational features (chunked)...")
    batch_size = 50000 
    rel_feat_list = []
    for i in range(0, num_steps, batch_size):
        end_idx = min(i + batch_size, num_steps)
        chunk_pos = pos[i:end_idx].unsqueeze(0).float()
        chunk_vel = vel[i:end_idx].unsqueeze(0)
        chunk_att = att[i:end_idx].unsqueeze(0).float()
        with torch.no_grad():
            chunk_rel = rel_encoder.compute_analytic_features(
                chunk_pos, chunk_vel, att=chunk_att, world_size=(1024.0, 1024.0)
            )
            rel_feat_list.append(chunk_rel[0, ..., :50].reshape(-1, 50).cpu())
            
    # Free raw inputs
    del health, power, vel, ang, tokens, pos, att, raw_targets, norm_targets, d_pos, raw_d_state, norm_d_state
    torch.cuda.empty_cache()

    # 6. Final Stats Calculation
    log.info("Processing statistics...")
    raw_state_stats = compute_stats_vectorized(all_raw_states)
    norm_state_stats = compute_stats_vectorized(all_norm_states)
    raw_target_stats = compute_stats_vectorized(all_raw_targets)
    norm_target_stats = compute_stats_vectorized(all_norm_targets)
    
    all_rel_cpu = torch.cat(rel_feat_list, dim=0)
    del rel_feat_list
    
    rel_res = {}
    for key in ["mean", "std", "min", "max", "median", "q1", "q3", "sem"]:
        rel_res[key] = torch.zeros(50)
        
    for i in range(50):
        col = all_rel_cpu[:, i].to(device)
        res = compute_stats_vectorized(col)
        for key in rel_res: rel_res[key][i] = res[key][0]
    
    # 7. Report Generation
    data_dir = Path(data_path).parent
    report_path = data_dir / "dataset_stats_report.md"
    
    summary_header = f"# Dataset Statistics Report\n\n- **Source**: `{data_path}`\n- **Steps**: {num_steps:,}\n- **Total Samples**: {all_raw_states.shape[0]:,}\n- **Execution Time**: {time.time() - t_start:.2f}s\n\n"
    
    def format_table(name, stats_dict, enum_class):
        s = stats_dict
        table = f"### {name}\n"
        table += "| Feature | Mean | Std | SEM | Min | Q1 | Median | Q3 | Max |\n"
        table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for i, feat in enumerate(enum_class):
            table += f"| {feat.name} | {s['mean'][i]:.4f} | {s['std'][i]:.4f} | {s['sem'][i]:.6f} | {s['min'][i]:.4f} | {s['q1'][i]:.4f} | {s['median'][i]:.4f} | {s['q3'][i]:.4f} | {s['max'][i]:.4f} |\n"
        return table + "\n"

    def write_csv(path, datasets):
        """datasets: list of (prefix, stats_dict, enum_or_names)"""
        with open(path, "w") as f:
            f.write("Feature|Mean|Std|SEM|Min|Q1|Median|Q3|Max\n")
            for prefix, s, names in datasets:
                for i, name in enumerate(names):
                    feat_name = f"{prefix}_{name}" if prefix else name
                    f.write(f"{feat_name}|{s['mean'][i]:.6f}|{s['std'][i]:.6f}|{s['sem'][i]:.8f}|{s['min'][i]:.6f}|{s['q1'][i]:.6f}|{s['median'][i]:.6f}|{s['q3'][i]:.6f}|{s['max'][i]:.6f}\n")

    with open(report_path, "w") as f:
        f.write(summary_header)
        
        f.write("## 1. State Features\n")
        f.write(format_table("Raw State Features (Pre-Normalization)", raw_state_stats, StateFeature))
        f.write(format_table("Normalized State Tokens (Post-Normalization)", norm_state_stats, StateFeature))
        
        f.write("## 2. Target Features\n")
        f.write(format_table("Raw Target Deltas (Pre-Normalization)", raw_target_stats, TargetFeature))
        f.write(format_table("Normalized Target Deltas (Post-Normalization)", norm_target_stats, TargetFeature))
        
        f.write("## 3. Relational Features\n")
        f.write("| Index | Feature Description | Mean | Std | Min | Q1 | Median | Q3 | Max |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        rel_names = ["dx", "dy", "dvx", "dvy", "dist", "inv_dist", "rel_speed", "closing", "dir_x", "dir_y", "log_dist", "tti", "cos_ata", "sin_ata", "cos_aa", "sin_aa", "cos_hca", "sin_hca"]
        for i in range(50):
            name = rel_names[i] if i < len(rel_names) else f"Fourier_{i-18}"
            f.write(f"| {i} | {name} | {rel_res['mean'][i]:.4f} | {rel_res['std'][i]:.4f} | {rel_res['min'][i]:.4f} | {rel_res['q1'][i]:.4f} | {rel_res['median'][i]:.4f} | {rel_res['q3'][i]:.4f} | {rel_res['max'][i]:.4f} |\n")

    # Write CSVs
    raw_csv_path = data_dir / "raw_stats.csv"
    norm_csv_path = data_dir / "norm_stats.csv"
    
    rel_names = rel_names + [f"Fourier_{i}" for i in range(32)]
    
    write_csv(raw_csv_path, [
        ("State", raw_state_stats, [f.name for f in StateFeature]),
        ("Target", raw_target_stats, [f.name for f in TargetFeature]),
        ("Relational", rel_res, rel_names)
    ])
    
    write_csv(norm_csv_path, [
        ("State", norm_state_stats, [f.name for f in StateFeature]),
        ("Target", norm_target_stats, [f.name for f in TargetFeature])
    ])

    log.info(f"Report saved to {report_path}")
    log.info(f"CSVs saved to {raw_csv_path} and {norm_csv_path}")

if __name__ == "__main__":
    main()
