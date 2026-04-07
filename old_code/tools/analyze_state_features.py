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


def load_and_aggregate_features(
    h5_path: str, max_samples: int = 500_000, world_size: float = 1024.0
):
    """
    Reads positions and states from aggregated_data.h5.
    Returns dictionaries of raw features, deltas, and euler integration errors.
    """

    def wrap_torus(delta, size):
        return (delta + size / 2) % size - size / 2

    print(f"Loading data from {h5_path}...")
    try:
        f = h5py.File(h5_path, "r")
    except Exception as e:
        print(f"Error opening {h5_path}: {e}")
        return None, None, None

    # Determine dimensions based on first chunk or metadata if available
    # aggregated_data.h5 shapes typically:
    # positions: (Transitions, MaxShips, 2)
    # states:    (Transitions, MaxShips, StateDim (5))
    # alive:     (Transitions, MaxShips)

    pos_ds = f["position"]
    vel_ds = f["velocity"]
    health_ds = f["health"]
    power_ds = f["power"]
    ep_ids_ds = f.get("episode_ids")

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
    alive = health_raw > 0

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
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "angle": angle,
        "power": power,
        "health": health,
    }

    # Compute Deltas (t+1 - t)
    # To avoid boundary anomalies, we'll only compute deltas where ship is alive at T AND T+1
    alive_t = alive[:-1]
    alive_t1 = alive[1:]

    # Valid transition mask
    valid_trans = alive_t & alive_t1

    # Strictly enforce episode boundaries if data exists
    if ep_ids is not None:
        valid_ep = ep_ids[:-1] == ep_ids[1:]
        valid_ep = np.expand_dims(
            valid_ep, axis=-1
        )  # Broadcast (N,) to (N, 1) to match (N, 8) ships
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
        "delta_x": dx,
        "delta_y": dy,
        "delta_pos_mag": dpos_mag,
        "delta_pos_angle": dpos_angle,
        "delta_vx": dvx,
        "delta_vy": dvy,
        "delta_vel_mag": dvel_mag,
        "delta_vel_angle": dvel_angle,
        "delta_x_local": dx_local,
        "delta_y_local": dy_local,
        "delta_pos_local_angle": dpos_local_angle,
        "delta_vx_local": dvx_local,
        "delta_vy_local": dvy_local,
        "delta_vel_local_angle": dvel_local_angle,
        "delta_speed": dspeed,
        "delta_power": dpower,
        "delta_health": dhealth,
    }

    # Calculate empirical physics clock dynamically (to map error safely into spatial components)
    valid_speed = speed_t > 1e-2
    if np.any(valid_speed):
        dt = np.median(dpos_mag[valid_speed] / speed_t[valid_speed])
    else:
        dt = 1.0

    # Compute Semi-Implicit Euler Integration Errors (spatial deviations)
    x_euler = x_t + vx_t * dt
    y_euler = y_t + vy_t * dt

    euler_error_x = wrap_torus(x_t1 - x_euler, world_size)
    euler_error_y = wrap_torus(y_t1 - y_euler, world_size)

    euler_error_x = np.where(np.abs(euler_error_x) < 1e-2, 0.0, euler_error_x)
    euler_error_y = np.where(np.abs(euler_error_y) < 1e-2, 0.0, euler_error_y)

    error_mag = np.sqrt(euler_error_x**2 + euler_error_y**2)
    error_angle = np.arctan2(euler_error_y, euler_error_x)

    error_x_local = euler_error_x * cos_t + euler_error_y * sin_t
    error_y_local = -euler_error_x * sin_t + euler_error_y * cos_t

    error_x_local = np.where(np.abs(error_x_local) < 1e-2, 0.0, error_x_local)
    error_y_local = np.where(np.abs(error_y_local) < 1e-2, 0.0, error_y_local)
    error_local_angle = np.arctan2(error_y_local, error_x_local)

    errors = {
        "x": euler_error_x,
        "y": euler_error_y,
        "mag": error_mag,
        "angle": error_angle,
    }
    errors_local = {
        "x_local": error_x_local,
        "y_local": error_y_local,
        "local_angle": error_local_angle,
    }

    f.close()
    return raw_features, deltas, errors, errors_local


def apply_symlog(arr, linthresh=1):
    """
    Applies a smooth symmetric logarithm based on log10.
    Useful for mapping speeds, deltas, and multi-scale variables safely across zero.
    """
    return np.sign(arr) * np.log10(1.0 + np.abs(arr) / linthresh)


def generate_symlog_data(data_dict):
    """
    Creates a mapped dictionary with symlog applied to appropriate features.
    Leaves absolute locations, angles, and health points unmodified.
    """
    out = {}
    for k, v in data_dict.items():
        if "angle" in k or k in ["x", "y", "health"]:
            out[k] = v
        else:
            out[k] = apply_symlog(v)
    return out


def plot_distributions(
    data_dict,
    output_dir: Path,
    prefix: str,
    log_y: bool = False,
    max_plot_samples: int = 100_000,
):
    """
    Renders histogram/density plots for a dictionary of 1D numpy arrays.
    """
    sns.set_theme(style="whitegrid")

    for key, values in tqdm(data_dict.items(), desc=f"Plotting {prefix}"):
        plt.figure(figsize=(10, 6))

        # Subsample if extremely large to speed up KDE calculation
        if len(values) > max_plot_samples:
            plot_vals = np.random.choice(values, size=max_plot_samples, replace=False)
        else:
            plot_vals = values

        # Turn off KDE if we are logging Y, as seaborn KDE can fail on log scales with sharp zeros
        sns.histplot(
            plot_vals, kde=not log_y, bins=100, color="steelblue", stat="density"
        )

        if log_y:
            plt.yscale("log")
            # Prevent pure 0-count bins from crashing the log scale bounds
            bottom_limit = 1e-6
            plt.ylim(bottom=bottom_limit)

        # Plot mean and std lines
        mean = np.mean(plot_vals)
        std = np.std(plot_vals)
        plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        plt.axvline(mean - std, color="orange", linestyle=":", label=f"1 Std Dev")
        plt.axvline(mean + std, color="orange", linestyle=":")

        # Title and Labels
        plt.title(f"{prefix.title()} Distribution: {key}", fontsize=14, pad=15)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()

        save_path = output_dir / f"{prefix}_{key}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_2d_distributions(
    x_vals,
    y_vals,
    output_dir: Path,
    prefix: str,
    name_x: str,
    name_y: str,
    log_colors: bool = False,
    max_plot_samples: int = 500_000,
):
    """
    Renders 2D Histogram/Heatmap plots for paired features like (x, y) or (vx, vy).
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))

    # Subsample if extremely large
    if len(x_vals) > max_plot_samples:
        idx = np.random.choice(len(x_vals), size=max_plot_samples, replace=False)
        x_plot, y_plot = x_vals[idx], y_vals[idx]
    else:
        x_plot, y_plot = x_vals, y_vals

    norm = mcolors.LogNorm() if log_colors else None

    h = plt.hist2d(x_plot, y_plot, bins=100, cmap="inferno", norm=norm, density=True)
    plt.colorbar(h[3], label="Density" + (" (Log Scale)" if log_colors else ""))

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
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/bc_pretraining/aggregated_data.h5",
        help="Path to aggregated setup.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/feature_analysis",
        help="Output directory for charts.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000_000,
        help="Max transitions to load from dataset.",
    )
    parser.add_argument(
        "--plot_1d_samples",
        type=int,
        default=100_000,
        help="Max samples to render in 1D density plots.",
    )
    parser.add_argument(
        "--plot_2d_samples",
        type=int,
        default=500_000,
        help="Max samples to render in 2D Heatmaps.",
    )
    parser.add_argument(
        "--world_size",
        type=float,
        default=1024.0,
        help="World width/height for toroidal space.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("🚀 Yemong State Feature Analyzer")
    print("=" * 50)

    raw, deltas, errors, errors_local = load_and_aggregate_features(
        args.data_path, max_samples=args.samples, world_size=args.world_size
    )

    if raw is None:
        return

    print(
        f"\nExtracted {len(raw['x'])} valid state records, and {len(deltas['delta_x'])} valid transitions."
    )

    plot_distributions(
        raw, out_dir, prefix="raw", log_y=False, max_plot_samples=args.plot_1d_samples
    )
    plot_distributions(
        deltas,
        out_dir,
        prefix="delta",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )
    plot_distributions(
        errors,
        out_dir,
        prefix="error",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )
    plot_distributions(
        errors_local,
        out_dir,
        prefix="error_local",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )

    # 2D Heatmaps
    print("Generating 2D Spatial & Velocity Heatmaps...")
    plot_2d_distributions(
        raw["x"],
        raw["y"],
        out_dir,
        prefix="raw",
        name_x="x",
        name_y="y",
        log_colors=False,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        raw["vx"],
        raw["vy"],
        out_dir,
        prefix="raw",
        name_x="vx",
        name_y="vy",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_x"],
        deltas["delta_y"],
        out_dir,
        prefix="delta",
        name_x="delta_x",
        name_y="delta_y",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_vx"],
        deltas["delta_vy"],
        out_dir,
        prefix="delta",
        name_x="delta_vx",
        name_y="delta_vy",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    print("Generating 2D Spatial & Velocity Heatmaps (Ship Reference Frame)...")
    plot_2d_distributions(
        deltas["delta_x_local"],
        deltas["delta_y_local"],
        out_dir,
        prefix="delta_local",
        name_x="delta_x_local",
        name_y="delta_y_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_vx_local"],
        deltas["delta_vy_local"],
        out_dir,
        prefix="delta_local",
        name_x="delta_vx_local",
        name_y="delta_vy_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    # Error Heatmaps
    print("Generating 2D Error Heatmaps...")
    plot_2d_distributions(
        errors["x"],
        errors["y"],
        out_dir,
        prefix="error",
        name_x="x",
        name_y="y",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        errors_local["x_local"],
        errors_local["y_local"],
        out_dir,
        prefix="error_local",
        name_x="x_local",
        name_y="y_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    # Angle vs Radius (Magnitude) Polar Heatmaps
    print("Generating 2D Polar (Angle vs Radius) Heatmaps...")
    plot_2d_distributions(
        raw["angle"],
        raw["speed"],
        out_dir,
        prefix="polar",
        name_x="vel_angle",
        name_y="vel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_pos_angle"],
        deltas["delta_pos_mag"],
        out_dir,
        prefix="polar",
        name_x="dpos_angle",
        name_y="dpos_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_vel_angle"],
        deltas["delta_vel_mag"],
        out_dir,
        prefix="polar",
        name_x="dvel_angle",
        name_y="dvel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        errors["angle"],
        errors["mag"],
        out_dir,
        prefix="error_polar",
        name_x="angle",
        name_y="mag",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    print("Generating 2D Polar Heatmaps (Ship Reference Frame)...")
    plot_2d_distributions(
        deltas["delta_pos_local_angle"],
        deltas["delta_pos_mag"],
        out_dir,
        prefix="polar_local",
        name_x="dpos_local_angle",
        name_y="dpos_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        deltas["delta_vel_local_angle"],
        deltas["delta_vel_mag"],
        out_dir,
        prefix="polar_local",
        name_x="dvel_local_angle",
        name_y="dvel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        errors_local["local_angle"],
        errors["mag"],
        out_dir,
        prefix="error_polar_local",
        name_x="local_angle",
        name_y="mag",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    # Generate Original Markdown Report
    print("Generating Markdown Plot Report...")
    generate_markdown_report(
        out_dir,
        report_filename="report.md",
        prefix="",
        title="Yemong State Feature Analysis Report",
    )

    # Generate Symlog Plots
    print("\n" + "=" * 50)
    print("Generating Symlog Distributions (this may take a moment)...")
    sym_raw = generate_symlog_data(raw)
    sym_deltas = generate_symlog_data(deltas)
    sym_errors = generate_symlog_data(errors)
    sym_errors_local = generate_symlog_data(errors_local)

    plot_distributions(
        sym_raw,
        out_dir,
        prefix="symlog_raw",
        log_y=False,
        max_plot_samples=args.plot_1d_samples,
    )
    plot_distributions(
        sym_deltas,
        out_dir,
        prefix="symlog_delta",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )
    plot_distributions(
        sym_errors,
        out_dir,
        prefix="symlog_error",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )
    plot_distributions(
        sym_errors_local,
        out_dir,
        prefix="symlog_error_local",
        log_y=True,
        max_plot_samples=args.plot_1d_samples,
    )

    print("Generating Symlog 2D Heatmaps...")
    plot_2d_distributions(
        sym_raw["x"],
        sym_raw["y"],
        out_dir,
        prefix="symlog_raw",
        name_x="x",
        name_y="y",
        log_colors=False,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_raw["vx"],
        sym_raw["vy"],
        out_dir,
        prefix="symlog_raw",
        name_x="vx",
        name_y="vy",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_x"],
        sym_deltas["delta_y"],
        out_dir,
        prefix="symlog_delta",
        name_x="delta_x",
        name_y="delta_y",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_vx"],
        sym_deltas["delta_vy"],
        out_dir,
        prefix="symlog_delta",
        name_x="delta_vx",
        name_y="delta_vy",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    plot_2d_distributions(
        sym_deltas["delta_x_local"],
        sym_deltas["delta_y_local"],
        out_dir,
        prefix="symlog_delta_local",
        name_x="delta_x_local",
        name_y="delta_y_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_vx_local"],
        sym_deltas["delta_vy_local"],
        out_dir,
        prefix="symlog_delta_local",
        name_x="delta_vx_local",
        name_y="delta_vy_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    plot_2d_distributions(
        sym_errors["x"],
        sym_errors["y"],
        out_dir,
        prefix="symlog_error",
        name_x="x",
        name_y="y",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_errors_local["x_local"],
        sym_errors_local["y_local"],
        out_dir,
        prefix="symlog_error_local",
        name_x="x_local",
        name_y="y_local",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    print("Generating Symlog Polar Heatmaps...")
    plot_2d_distributions(
        sym_raw["angle"],
        sym_raw["speed"],
        out_dir,
        prefix="symlog_polar",
        name_x="vel_angle",
        name_y="vel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_pos_angle"],
        sym_deltas["delta_pos_mag"],
        out_dir,
        prefix="symlog_polar",
        name_x="dpos_angle",
        name_y="dpos_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_vel_angle"],
        sym_deltas["delta_vel_mag"],
        out_dir,
        prefix="symlog_polar",
        name_x="dvel_angle",
        name_y="dvel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_errors["angle"],
        sym_errors["mag"],
        out_dir,
        prefix="symlog_error_polar",
        name_x="angle",
        name_y="mag",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    plot_2d_distributions(
        sym_deltas["delta_pos_local_angle"],
        sym_deltas["delta_pos_mag"],
        out_dir,
        prefix="symlog_polar_local",
        name_x="dpos_local_angle",
        name_y="dpos_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_deltas["delta_vel_local_angle"],
        sym_deltas["delta_vel_mag"],
        out_dir,
        prefix="symlog_polar_local",
        name_x="dvel_local_angle",
        name_y="dvel_magnitude",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )
    plot_2d_distributions(
        sym_errors_local["local_angle"],
        sym_errors["mag"],
        out_dir,
        prefix="symlog_error_polar_local",
        name_x="local_angle",
        name_y="mag",
        log_colors=True,
        max_plot_samples=args.plot_2d_samples,
    )

    print("Generating Symlog Markdown Plot Report...")
    generate_markdown_report(
        out_dir,
        report_filename="report_symlog.md",
        prefix="symlog_",
        title="Yemong State Feature Analysis Report (Symlog Normalized)",
    )

    print(f"\n✅ All distribution plots saved to {out_dir.absolute()}")


def generate_markdown_report(
    out_dir: Path,
    report_filename="report.md",
    prefix="",
    title="Yemong State Feature Analysis Report",
):
    import glob

    def get_images(pattern: str, exclude_patterns: list = None):
        files = glob.glob(str(out_dir / f"{prefix}{pattern}"))
        if exclude_patterns:
            for exc in exclude_patterns:
                exc_files = glob.glob(str(out_dir / f"{prefix}{exc}"))
                files = [f for f in files if f not in exc_files]
        return sorted([Path(f).name for f in files])

    sections = [
        (
            "Raw 1D Cartesian Plots",
            get_images("raw_*.png", exclude_patterns=["raw_2D_*.png"]),
        ),
        ("Raw 2D Spatial & Velocity Heatmaps", get_images("raw_2D_*.png")),
        (
            "Global Delta 1D Cartesian Plots",
            get_images(
                "delta_*.png", exclude_patterns=["delta_2D_*.png", "delta_local_*.png"]
            ),
        ),
        ("Global Delta 2D Spatial & Velocity Heatmaps", get_images("delta_2D_*.png")),
        (
            "Local Delta (Ship Frame) 1D Cartesian Plots",
            get_images("delta_local_*.png", exclude_patterns=["delta_local_2D_*.png"]),
        ),
        (
            "Local Delta (Ship Frame) 2D Spatial & Velocity Heatmaps",
            get_images("delta_local_2D_*.png"),
        ),
        ("Global Polar Heatmaps (Angle vs Radius)", get_images("polar_2D_*.png")),
        ("Local Polar Heatmaps (Ship Frame)", get_images("polar_local_2D_*.png")),
        (
            "Global Integration Error 1D Plots",
            get_images(
                "error_*.png",
                exclude_patterns=[
                    "error_2D_*.png",
                    "error_local_*.png",
                    "error_polar_*.png",
                ],
            ),
        ),
        ("Global Integration Error 2D Heatmaps", get_images("error_2D_*.png")),
        (
            "Local Integration Error (Ship Frame) 1D",
            get_images("error_local_*.png", exclude_patterns=["error_local_2D_*.png"]),
        ),
        ("Local Integration Error (Ship Frame) 2D", get_images("error_local_2D_*.png")),
        ("Global Polar Error Heatmaps", get_images("error_polar_2D_*.png")),
        (
            "Local Polar Error Heatmaps (Ship Frame)",
            get_images("error_polar_local_2D_*.png"),
        ),
    ]

    css_path = out_dir / "report.css"
    with open(css_path, "w") as f:
        f.write("""
@page { size: A4 portrait; margin: 1cm; }
body { font-family: sans-serif; }
table { width: 100%; table-layout: fixed; border-collapse: collapse; page-break-inside: avoid; }
td { width: 50%; padding: 5px; text-align: center; vertical-align: middle; }
img { max-width: 100%; max-height: 290px; object-fit: contain; }
.page-break { page-break-before: always; }
h2 { text-align: center; font-size: 20px; margin-bottom: 20px; }
code { font-size: 11px; word-wrap: break-word; }
        """)

    report_path = out_dir / report_filename
    with open(report_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(
            "This document was automatically generated to group the feature distribution charts.\n\n"
        )

        first_page = True
        for title, images in sections:
            if not images:
                continue

            chunk_size = 6
            for chunk_idx in range(0, len(images), chunk_size):
                page_images = images[chunk_idx : chunk_idx + chunk_size]

                if not first_page:
                    f.write("<div class='page-break'></div>\n\n")
                first_page = False

                if chunk_idx == 0:
                    f.write(f"## {title}\n\n")
                else:
                    f.write(f"## {title} (Continued)\n\n")

                f.write("<table>\n")
                for i in range(0, len(page_images), 2):
                    f.write("<tr>\n")
                    img1 = page_images[i]
                    f.write(f"<td><img src='{img1}'><br><code>{img1}</code></td>\n")

                    if i + 1 < len(page_images):
                        img2 = page_images[i + 1]
                        f.write(f"<td><img src='{img2}'><br><code>{img2}</code></td>\n")
                    else:
                        f.write("<td></td>\n")
                    f.write("</tr>\n")
                f.write("</table>\n\n")

    # Generate PDF Report
    try:
        from md2pdf.core import md2pdf

        pdf_path = out_dir / report_filename.replace(".md", ".pdf")
        print(f"Converting Markdown Report '{report_filename}' to PDF...")
        md2pdf(pdf_path, md=report_path, css=css_path, base_url=out_dir)
        print(f"✅ PDF Report saved to {pdf_path.absolute()}")
    except ImportError:
        print(
            "⚠️ 'md2pdf' not found. Skipping PDF generation. Install with 'uv add md2pdf'."
        )
    except Exception as e:
        print(f"⚠️ Failed to generate PDF: {e}")


if __name__ == "__main__":
    main()
