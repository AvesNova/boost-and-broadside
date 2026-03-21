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
    StateFeature, TargetFeature, TARGET_DIM, STATE_DIM
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
    
    data_f64 = data.to(torch.float64)

    with torch.no_grad():
        # 1. Basic Stats (Full data)
        means = data_f64.mean(dim=0)
        stds = data_f64.std(dim=0)
        mins = data_f64.min(dim=0).values
        maxs = data_f64.max(dim=0).values
        rms = torch.sqrt(torch.mean(data_f64**2, dim=0))
        sem = stds / np.sqrt(data.shape[0])
        
        # 2. Quantiles (Expensive - Use subset)
        q = torch.quantile(data_subset.to(torch.float64), torch.tensor([0.25, 0.5, 0.75], device=data.device, dtype=torch.float64), dim=0)
        
    return {
        "mean": means.cpu(),
        "std": stds.cpu(),
        "rms": rms.cpu(),
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

    # Identify episode boundaries
    is_boundary = torch.zeros(num_steps, dtype=torch.bool, device=device)
    for start in dataset.episode_starts:
        if abs_start < start < abs_end:
            is_boundary[int(start - abs_start - 1)] = True
    valid_mask = ~is_boundary
    
    # 3. State Stats
    log.info("Computing state statistics...")
    # Raw State (T, N, 5)
    raw_states = torch.stack([
        health, power, vel[..., 0], vel[..., 1], ang
    ], dim=-1)
    all_raw_states = raw_states[:-1].reshape(-1, STATE_DIM)
    
    # 4. Target Stats
    log.info("Computing target correlations...")
    # Pre-norm Target (Raw Deltas)
    d_pos = pos[1:] - pos[:-1]
    W, H = 1024.0, 1024.0 # TODO: Pulled from dataset attrs?
    d_pos[..., 0] = d_pos[..., 0] - torch.round(d_pos[..., 0] / W) * W
    d_pos[..., 1] = d_pos[..., 1] - torch.round(d_pos[..., 1] / H) * H
    
    raw_d_state = raw_states[1:] - raw_states[:-1]
    raw_targets = torch.zeros((num_steps, N, TARGET_DIM), device=device)
    raw_targets[..., TargetFeature.DX:TargetFeature.DY+1] = d_pos
    raw_targets[..., TargetFeature.DVX:TargetFeature.DANG_VEL+1] = raw_d_state
    
    # Part D: Relational Targets in dataset_stats
    # Each ship i predicts how its "average neighbor" changed relative to it.
    sum_d_pos = d_pos.sum(dim=1, keepdim=True) # (T, 1, 2)
    sum_d_vel = raw_d_state[..., StateFeature.VX:StateFeature.VY+1].sum(dim=1, keepdim=True) # (T, 1, 2)
    mean_others_d_pos = (sum_d_pos - d_pos) / (N - 1 + 1e-6)
    mean_others_d_vel = (sum_d_vel - raw_d_state[..., StateFeature.VX:StateFeature.VY+1]) / (N - 1 + 1e-6)
    
    raw_targets[..., TargetFeature.DREL_X:TargetFeature.DREL_Y+1] = mean_others_d_pos - d_pos
    raw_targets[..., TargetFeature.DREL_VX] = (mean_others_d_vel - raw_d_state[..., StateFeature.VX:StateFeature.VY+1])[..., 0]
    raw_targets[..., TargetFeature.DREL_VY] = (mean_others_d_vel - raw_d_state[..., StateFeature.VX:StateFeature.VY+1])[..., 1]
    
    all_raw_targets = raw_targets[valid_mask].reshape(-1, TARGET_DIM)
    
    # 5. Relational Features (Chunked)
    log.info("Computing relational features (chunked)...")
    batch_size = 50000 
    rel_feat_list = []
    
    num_analytic = 54 # Matching RelationalEncoder
    for i in range(0, num_steps, batch_size):
        end_idx = min(i + batch_size, num_steps)
        chunk_abs_end = min(i + batch_size + 1, num_steps + 1)
        
        chunk_pos = pos[i:chunk_abs_end].unsqueeze(0).float()
        chunk_vel = vel[i:chunk_abs_end].unsqueeze(0)
        chunk_att = att[i:chunk_abs_end].unsqueeze(0).float()
        
        with torch.no_grad():
            chunk_rel = rel_encoder.compute_analytic_features(
                chunk_pos, chunk_vel, att=chunk_att, world_size=(1024.0, 1024.0)
            )
            # Analytic features at t are (1, T, N, N, 64) -> squeeze and take first 54
            # We want to average over neighbors or just take all edges?
            # Standard stats should be over all active edges.
            # (1, T, N, N, 64) -> (T*N*N, 64)
            rel_flat = chunk_rel[0, :end_idx-i, ..., :num_analytic].reshape(-1, num_analytic)
            rel_feat_list.append(rel_flat.cpu())

    # 6. Final Stats Calculation
    log.info("Processing statistics...")
    raw_state_stats = compute_stats_vectorized(all_raw_states)
    raw_target_stats = compute_stats_vectorized(all_raw_targets)
    
    all_rel_cpu = torch.cat(rel_feat_list, dim=0)
    del rel_feat_list
    
    rel_res = {}
    for key in ["mean", "std", "rms", "min", "max", "median", "q1", "q3", "sem"]:
        rel_res[key] = torch.zeros(num_analytic)
        
    for i in range(num_analytic):
        col = all_rel_cpu[:, i].to(device)
        res = compute_stats_vectorized(col)
        for key in rel_res: rel_res[key][i] = res[key][0]
        
    # 7. Report Generation
    data_dir = Path(data_path).parent
    report_path = data_dir / "dataset_stats_report.md"
    
    summary_header = f"# Dataset Statistics Report\n\n- **Source**: `{data_path}`\n- **Steps**: {num_steps:,}\n- **Total Samples**: {all_raw_states.shape[0]:,}\n- **Execution Time**: {time.time() - t_start:.2f}s\n\n"
    
    def format_table(name, stats_dict, enum_class=None, names=None):
        s = stats_dict
        table = f"### {name}\n"
        table += "| Feature | Mean | Std | RMS | Min | Q1 | Median | Q3 | Max |\n"
        table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        if enum_class:
            for i, feat in enumerate(enum_class):
                table += f"| {feat.name} | {s['mean'][i]:.4f} | {s['std'][i]:.4f} | {s['rms'][i]:.4f} | {s['min'][i]:.4f} | {s['q1'][i]:.4f} | {s['median'][i]:.4f} | {s['q3'][i]:.4f} | {s['max'][i]:.4f} |\n"
        elif names:
             for i, name in enumerate(names):
                table += f"| {name} | {s['mean'][i]:.4f} | {s['std'][i]:.4f} | {s['rms'][i]:.4f} | {s['min'][i]:.4f} | {s['q1'][i]:.4f} | {s['median'][i]:.4f} | {s['q3'][i]:.4f} | {s['max'][i]:.4f} |\n"
        return table + "\n"

    def write_csv(path, datasets):
        """datasets: list of (prefix, stats_dict, enum_or_names)"""
        with open(path, "w") as f:
            f.write("Feature|Mean|Std|RMS|SEM|Min|Q1|Median|Q3|Max\n")
            for prefix, s, names in datasets:
                for i, name in enumerate(names):
                    feat_name = f"{prefix}_{name}" if prefix else name
                    f.write(f"{feat_name}|{s['mean'][i]:.6f}|{s['std'][i]:.6f}|{s['rms'][i]:.6f}|{s['sem'][i]:.8f}|{s['min'][i]:.6f}|{s['q1'][i]:.6f}|{s['median'][i]:.6f}|{s['q3'][i]:.6f}|{s['max'][i]:.6f}\n")

    with open(report_path, "w") as f:
        f.write(summary_header)
        
        f.write("## 1. State Features\n")
        f.write(format_table("Raw State Features", raw_state_stats, StateFeature))
        
        f.write("## 2. Target Features\n")
        f.write(format_table("Raw Target Deltas", raw_target_stats, TargetFeature))
        
        f.write("## 3. Relational Features\n")
        rel_names = [
            "dvx", "dvy", "rel_speed", "closing", "log_dist", "tti",
            "heading_x", "heading_y", 
            "cos_ata", "sin_ata", "cos_aa", "sin_aa", "cos_hca", "sin_hca"
        ]
        rel_names = rel_names + [f"Fourier_{i}" for i in range(40)]
        f_table = format_table("Analytic Relational Features", rel_res, names=rel_names)
        f.write(f_table)
        

    # Write CSVs
    raw_csv_path = data_dir / "raw_stats.csv"
    
    write_csv(raw_csv_path, [
        ("State", raw_state_stats, [f.name for f in StateFeature]),
        ("Target", raw_target_stats, [f.name for f in TargetFeature]),
        ("Relational", rel_res, rel_names),
    ])
    
    log.info(f"Report saved to {report_path}")
    log.info(f"CSV saved to {raw_csv_path}")

if __name__ == "__main__":
    main()
