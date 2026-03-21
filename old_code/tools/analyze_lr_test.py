
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def smooth_data(data, window_length=51, polyorder=3):
    """
    Smooth data using Savitzky-Golay filter if available, otherwise use rolling mean.
    """
    try:
        from scipy.signal import savgol_filter
        if window_length > len(data):
            window_length = len(data) // 2 * 2 + 1 
        return savgol_filter(data, window_length, polyorder)
    except ImportError:
        print("Scipy not found, using simple rolling mean.")
        return data.rolling(window=window_length, center=True, min_periods=1).mean().values

def analyze_lr_test(csv_path, args=None):
    print(f"Analyzing {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'learning_rate' not in df.columns or 'loss' not in df.columns:
        print("CSV must contain 'learning_rate' and 'loss' columns.")
        return

    # Sort by LR just in case
    df = df.sort_values('learning_rate')
    
    # Remove duplicates to avoid divide by zero in gradient
    df = df.drop_duplicates(subset=['learning_rate'])
    
    lrs = df['learning_rate'].values
    losses = df['loss'].values
    
    # 1. Smooth the loss
    # Adjust window size based on data length
    # Adjust window size based on data length
    # Need at least polyorder + 2 (usually 5)
    min_window = 7
    window_size = max(min_window, min(len(losses) // 10, 101))
    
    if window_size > len(losses):
        window_size = len(losses) // 2 * 2 + 1
        
    if window_size % 2 == 0: window_size += 1
    
    # If data is too small for default polyorder (3), adjust polyorder
    polyorder = 3
    if window_size <= polyorder:
         polyorder = max(1, window_size - 1)
    
    smoothed_losses = smooth_data(losses, window_length=window_size, polyorder=polyorder)

    # 2. Compute gradients
    # We want d(Loss) / d(log(LR))
    log_lrs = np.log10(lrs)
    grads = np.gradient(smoothed_losses, log_lrs)
    
    
    # 3. Find steepest descent (minimum gradient)
    # Filter for max LR if specified
    if args.max_lr:
        mask = lrs <= args.max_lr
        if not np.any(mask):
            print(f"No data points found with LR <= {args.max_lr}")
            return
        
        # We need to find indices in the ORIGINAL arrays that survived the mask
        # But we computed gradients on the full smoothed array. 
        # So we just mask the gradients/LRS for the SEARCH.
        valid_grads = grads[mask]
        valid_lrs = lrs[mask]
        
        min_grad_idx_valid = np.argmin(valid_grads)
        # Map back to original if needed, or just use valid_lrs
        steepest_lr = valid_lrs[min_grad_idx_valid]
        suggested_lr = steepest_lr
        
        # Update plot to show cutoff
        min_grad_full_idx = np.where(lrs == steepest_lr)[0][0] # Find index in full array for plotting point
        # Actually min_grad_idx is needed for plotting the red dot.
    
    else:
        min_grad_idx = np.argmin(grads)
        steepest_lr = lrs[min_grad_idx]
        suggested_lr = steepest_lr
        min_grad_full_idx = min_grad_idx

    print(f"Steepest descent found at LR: {steepest_lr:.2e}")
    print(f"Suggested Max LR: {suggested_lr:.2e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses, label='Raw Loss', alpha=0.3, color='gray')
    plt.semilogx(lrs, smoothed_losses, label='Smoothed Loss', linewidth=2, color='blue')
    
    # Mark the suggestion
    plt.scatter(steepest_lr, smoothed_losses[min_grad_full_idx], color='red', s=100, label=f'Suggested LR: {suggested_lr:.1e}', zorder=5)
    
    if args.max_lr:
        plt.axvline(x=args.max_lr, color='r', linestyle='--', label=f'Cutoff ({args.max_lr:.1e})')

    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Dynamic Y Limit based on max loss before cutoff (to avoid explosion obscuring the curve)
    if args.max_lr:
        mask = lrs <= args.max_lr
        max_y = np.max(losses[mask]) if np.any(mask) else 5.0
    else:
        max_y = np.max(losses)
        
    plt.ylim(top=max_y * 1.1, bottom=0.0)
    
    output_path = Path(csv_path).parent / 'lr_range_test.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    # plt.show() # Don't show in headless env

    return suggested_lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Learning Rate Range Test results.")
    parser.add_argument("csv_path", nargs='?', help="Path to training_log.csv")
    parser.add_argument("--latest", action="store_true", help="Find the latest run automatically")
    parser.add_argument("--max_lr", type=float, help="Limit analysis to LRs less than or equal to this value")
    
    args = parser.parse_args()
    
    path_to_analyze = args.csv_path
    
    if args.latest:
        # Assuming typical structure: models/world_model/run_YYYYMMDD_HHMMSS/training_log.csv
        base_dir = Path("models/world_model")
        if base_dir.exists():
            runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda x: x.name, reverse=True)
            if runs:
                latest_run = runs[0]
                possible_path = latest_run / "training_step_log.csv"
                if possible_path.exists():
                    path_to_analyze = str(possible_path)
                    print(f"Found latest run: {path_to_analyze}")
                else:
                    print(f"No training_step_log.csv found in {latest_run}, checking for training_log.csv")
                    # Fallback
                    path_to_analyze = str(latest_run / "training_log.csv")
            else:
                print("No runs found in models/world_model")
    
    if not path_to_analyze:
        print("Please provide a CSV path or use --latest.")
        sys.exit(1)
        
    analyze_lr_test(path_to_analyze, args)
