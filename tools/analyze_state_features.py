import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# We emulate indices from `StateFeature` locally here rather than importing
# just so the script is highly portable, but importing from core constants is also fine.
HEALTH_IDX = 0
POWER_IDX = 1
VX_IDX = 2
VY_IDX = 3
# Angel Vel is 4
# Position X and Y are often embedded or treated separately. 
# In `aggregated_data.h5`, position often is `dataset['pos']` while `dataset['state']` has health, power, vels.

def load_and_aggregate_features(h5_path: str, max_samples: int = 500_000, world_size: float = 1024.0):
    """
    Reads positions and states from aggregated_data.h5.
    Returns dictionaries of raw features, deltas, and euler integration errors.
    """
    
    def wrap_torus(delta, size):
        return (delta + size / 2) % size - size / 2
    print(f"Loading data from {h5_path}...")
    try:
        f = h5py.File(h5_path, 'r')
    except Exception as e:
        print(f"Error opening {h5_path}: {e}")
        return None, None, None

    # Determine dimensions based on first chunk or metadata if available
    # aggregated_data.h5 shapes typically:
    # positions: (Transitions, MaxShips, 2)
    # states:    (Transitions, MaxShips, StateDim (5))
    # alive:     (Transitions, MaxShips)
    
    pos_ds = f['position']
    vel_ds = f['velocity']
    health_ds = f['health']
    power_ds = f['power']
    ep_ids_ds = f.get('episode_ids')
    
    total_transitions = min(pos_ds.shape[0], max_samples)
    print(f"Aggregating {total_transitions} transitions (shape: {pos_ds.shape})...")

    pos = pos_ds[:total_transitions].astype(np.float32)
    vel = vel_ds[:total_transitions].astype(np.float32)
    health_raw = health_ds[:total_transitions].astype(np.float32)
    power_raw = power_ds[:total_transitions].astype(np.float32)
    
    if ep_ids_ds is not None:
        ep_ids = ep_ids_ds[:total_transitions]
    else:
        ep_ids = None
    
    # Infer alive map from health > 0
    alive = (health_raw > 0)

    # Filter purely valid (alive) ship ticks
    # pos is [N, 8, 2] -> x, y
    x = pos[alive, 0]
    y = pos[alive, 1]
    
    health = health_raw[alive]
    power = power_raw[alive]
    vx = vel[alive, 0]
    vy = vel[alive, 1]
    
    # Strip infinitesimal float noise that causes arctan2 to flip to +/- pi
    vx = np.where(np.abs(vx) < 1e-2, 0.0, vx)
    vy = np.where(np.abs(vy) < 1e-2, 0.0, vy)

    speed = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)

    raw_features = {
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'speed': speed,
        'angle': angle,
        'power': power,
        'health': health
    }

    # Compute Deltas (t+1 - t)
    # To avoid boundary anomalies, we'll only compute deltas where ship is alive at T AND T+1
    alive_t = alive[:-1]
    alive_t1 = alive[1:]
    
    # Valid transition mask
    valid_trans = alive_t & alive_t1
    
    # Strictly enforce episode boundaries if data exists
    if ep_ids is not None:
        valid_ep = (ep_ids[:-1] == ep_ids[1:])
        valid_ep = np.expand_dims(valid_ep, axis=-1)  # Broadcast (N,) to (N, 1) to match (N, 8) ships
        valid_trans = valid_trans & valid_ep
        
    x_t = pos[:-1, :, 0][valid_trans]
    x_t1 = pos[1:, :, 0][valid_trans]
    dx = wrap_torus(x_t1 - x_t, world_size)
    
    y_t = pos[:-1, :, 1][valid_trans]
    y_t1 = pos[1:, :, 1][valid_trans]
    dy = wrap_torus(y_t1 - y_t, world_size)
    
    dx = np.where(np.abs(dx) < 1e-2, 0.0, dx)
    dy = np.where(np.abs(dy) < 1e-2, 0.0, dy)
    
    dpos_mag = np.sqrt(dx**2 + dy**2)
    dpos_angle = np.arctan2(dy, dx)
    
    vx_t = vel[:-1, :, 0][valid_trans]
    vx_t1 = vel[1:, :, 0][valid_trans]
    dvx = vx_t1 - vx_t
    
    vy_t = vel[:-1, :, 1][valid_trans]
    vy_t1 = vel[1:, :, 1][valid_trans]
    dvy = vy_t1 - vy_t
    
    dvx = np.where(np.abs(dvx) < 1e-2, 0.0, dvx)
    dvy = np.where(np.abs(dvy) < 1e-2, 0.0, dvy)
    
    dvel_mag = np.sqrt(dvx**2 + dvy**2)
    dvel_angle = np.arctan2(dvy, dvx)
    
    speed_t = np.sqrt(vx_t**2 + vy_t**2)
    speed_t1 = np.sqrt(vx_t1**2 + vy_t1**2)
    dspeed = speed_t1 - speed_t
    
    # Calculate Ship Reference Frame (Ship velocity vector = +X axis)
    # Cos = vx / speed, Sin = vy / speed
    safe_speed = np.where(speed_t < 1e-6, 1.0, speed_t)
    cos_t = vx_t / safe_speed
    sin_t = vy_t / safe_speed
    cos_t = np.where(speed_t < 1e-6, 1.0, cos_t)
    sin_t = np.where(speed_t < 1e-6, 0.0, sin_t)

    # Rotate dx, dy by -theta to align with the ship's heading
    dx_local = dx * cos_t + dy * sin_t
    dy_local = -dx * sin_t + dy * cos_t
    
    dx_local = np.where(np.abs(dx_local) < 1e-2, 0.0, dx_local)
    dy_local = np.where(np.abs(dy_local) < 1e-2, 0.0, dy_local)
    dpos_local_angle = np.arctan2(dy_local, dx_local)
    
    dvx_local = dvx * cos_t + dvy * sin_t
    dvy_local = -dvx * sin_t + dvy * cos_t
    
    dvx_local = np.where(np.abs(dvx_local) < 1e-2, 0.0, dvx_local)
    dvy_local = np.where(np.abs(dvy_local) < 1e-2, 0.0, dvy_local)
    dvel_local_angle = np.arctan2(dvy_local, dvx_local)
    
    power_t = power_raw[:-1, :][valid_trans]
    power_t1 = power_raw[1:, :][valid_trans]
    dpower = power_t1 - power_t
    
    health_t = health_raw[:-1, :][valid_trans]
    health_t1 = health_raw[1:, :][valid_trans]
    dhealth = health_t1 - health_t

    deltas = {
        'delta_x': dx,
        'delta_y': dy,
        'delta_pos_mag': dpos_mag,
        'delta_pos_angle': dpos_angle,
        'delta_vx': dvx,
        'delta_vy': dvy,
        'delta_vel_mag': dvel_mag,
        'delta_vel_angle': dvel_angle,
        'delta_x_local': dx_local,
        'delta_y_local': dy_local,
        'delta_pos_local_angle': dpos_local_angle,
        'delta_vx_local': dvx_local,
        'delta_vy_local': dvy_local,
        'delta_vel_local_angle': dvel_local_angle,
        'delta_speed': dspeed,
        'delta_power': dpower,
        'delta_health': dhealth
    }
    
    # Compute Semi-Implicit Euler Integration Errors
    # Assuming dt=1 for internal logic, or adjusting if velocities are natively scaled
    # Euler: x' = x + v * dt
    # Error: x_actual' - (x + v * dt)
    
    # The actual integration in game engine might be x' = x + vx' * dt
    # So error based on vx_t vs vx_t1 might vary. We'll test standard Euler: x + vx_t.
    dt = 1.0 # Or whatever your physics timestep scaling is
    
    x_euler = x_t + vx_t * dt
    euler_error_x = wrap_torus(x_t1 - x_euler, world_size)
    
    y_euler = y_t + vy_t * dt
    euler_error_y = wrap_torus(y_t1 - y_euler, world_size)
    
    errors = {
        'euler_error_x': euler_error_x,
        'euler_error_y': euler_error_y
    }
    
    f.close()
    return raw_features, deltas, errors

def plot_distributions(data_dict, output_dir: Path, prefix: str, log_y: bool = False):
    """
    Renders histogram/density plots for a dictionary of 1D numpy arrays.
    """
    sns.set_theme(style="whitegrid")
    
    for key, values in tqdm(data_dict.items(), desc=f"Plotting {prefix}"):
        plt.figure(figsize=(10, 6))
        
        # Subsample if extremely large to speed up KDE calculation
        if len(values) > 100_000:
            plot_vals = np.random.choice(values, size=100_000, replace=False)
        else:
            plot_vals = values
            
        # Turn off KDE if we are logging Y, as seaborn KDE can fail on log scales with sharp zeros
        sns.histplot(plot_vals, kde=not log_y, bins=100, color='steelblue', stat='density')
        
        if log_y:
            plt.yscale('log')
            # Prevent pure 0-count bins from crashing the log scale bounds
            bottom_limit = 1e-6
            plt.ylim(bottom=bottom_limit)
        
        # Plot mean and std lines
        mean = np.mean(plot_vals)
        std = np.std(plot_vals)
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(mean - std, color='orange', linestyle=':', label=f'1 Std Dev')
        plt.axvline(mean + std, color='orange', linestyle=':')
        
        # Title and Labels
        plt.title(f"{prefix.title()} Distribution: {key}", fontsize=14, pad=15)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        save_path = output_dir / f"{prefix}_{key}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

def plot_2d_distributions(x_vals, y_vals, output_dir: Path, prefix: str, name_x: str, name_y: str, log_colors: bool = False):
    """
    Renders 2D Histogram/Heatmap plots for paired features like (x, y) or (vx, vy).
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    
    # Subsample if extremely large
    if len(x_vals) > 500_000:
        idx = np.random.choice(len(x_vals), size=500_000, replace=False)
        x_plot, y_plot = x_vals[idx], y_vals[idx]
    else:
        x_plot, y_plot = x_vals, y_vals

    norm = mcolors.LogNorm() if log_colors else None
    
    h = plt.hist2d(x_plot, y_plot, bins=100, cmap='inferno', norm=norm, density=True)
    plt.colorbar(h[3], label='Density' + (' (Log Scale)' if log_colors else ''))
    
    # Title and Labels
    plt.title(f"{prefix.title()} 2D Heatmap: {name_x} vs {name_y}", fontsize=14, pad=15)
    plt.xlabel(name_x.upper(), fontsize=12)
    plt.ylabel(name_y.upper(), fontsize=12)
    plt.tight_layout()
    
    # Save mechanism
    save_path = output_dir / f"{prefix}_2D_{name_x}_{name_y}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="State Feature Distribution Analyzer")
    parser.add_argument("--data_path", type=str, default="data/bc_pretraining/aggregated_data.h5", help="Path to aggregated setup.")
    parser.add_argument("--out_dir", type=str, default="outputs/feature_analysis", help="Output directory for charts.")
    parser.add_argument("--samples", type=int, default=1_000_000, help="Max transitions to process.")
    parser.add_argument("--world_size", type=float, default=1024.0, help="World width/height for toroidal space.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("🚀 Yemong State Feature Analyzer")
    print("=" * 50)
    
    raw, deltas, errors = load_and_aggregate_features(args.data_path, max_samples=args.samples, world_size=args.world_size)
    
    if raw is None:
        return
        
    print(f"\nExtracted {len(raw['x'])} valid state records, and {len(deltas['delta_x'])} valid transitions.")
    
    plot_distributions(raw, out_dir, prefix="raw", log_y=False)
    plot_distributions(deltas, out_dir, prefix="delta", log_y=True)
    plot_distributions(errors, out_dir, prefix="integration_error", log_y=True)
    
    # 2D Heatmaps
    print("Generating 2D Spatial & Velocity Heatmaps...")
    plot_2d_distributions(raw['x'], raw['y'], out_dir, prefix="raw", name_x="x", name_y="y", log_colors=False)
    plot_2d_distributions(raw['vx'], raw['vy'], out_dir, prefix="raw", name_x="vx", name_y="vy", log_colors=True)
    plot_2d_distributions(deltas['delta_x'], deltas['delta_y'], out_dir, prefix="delta", name_x="delta_x", name_y="delta_y", log_colors=True)
    plot_2d_distributions(deltas['delta_vx'], deltas['delta_vy'], out_dir, prefix="delta", name_x="delta_vx", name_y="delta_vy", log_colors=True)
    
    print("Generating 2D Spatial & Velocity Heatmaps (Ship Reference Frame)...")
    plot_2d_distributions(deltas['delta_x_local'], deltas['delta_y_local'], out_dir, prefix="delta_local", name_x="delta_x_local", name_y="delta_y_local", log_colors=True)
    plot_2d_distributions(deltas['delta_vx_local'], deltas['delta_vy_local'], out_dir, prefix="delta_local", name_x="delta_vx_local", name_y="delta_vy_local", log_colors=True)
    
    # Angle vs Radius (Magnitude) Polar Heatmaps
    print("Generating 2D Polar (Angle vs Radius) Heatmaps...")
    plot_2d_distributions(raw['angle'], raw['speed'], out_dir, prefix="polar", name_x="vel_angle", name_y="vel_magnitude", log_colors=True)
    plot_2d_distributions(deltas['delta_pos_angle'], deltas['delta_pos_mag'], out_dir, prefix="polar", name_x="dpos_angle", name_y="dpos_magnitude", log_colors=True)
    plot_2d_distributions(deltas['delta_vel_angle'], deltas['delta_vel_mag'], out_dir, prefix="polar", name_x="dvel_angle", name_y="dvel_magnitude", log_colors=True)
    
    print("Generating 2D Polar Heatmaps (Ship Reference Frame)...")
    plot_2d_distributions(deltas['delta_pos_local_angle'], deltas['delta_pos_mag'], out_dir, prefix="polar_local", name_x="dpos_local_angle", name_y="dpos_magnitude", log_colors=True)
    plot_2d_distributions(deltas['delta_vel_local_angle'], deltas['delta_vel_mag'], out_dir, prefix="polar_local", name_x="dvel_local_angle", name_y="dvel_magnitude", log_colors=True)
    
    print(f"\n✅ All distribution plots saved to {out_dir.absolute()}")

if __name__ == "__main__":
    main()
