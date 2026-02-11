import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

def inspect_h5_file(file_path: Path) -> None:
    """
    Load and print information about a collected HDF5 data file.

    Args:
        file_path: Path to the HDF5 file
    """
    print(f"\n{'=' * 80}")
    print(f"Inspecting: {file_path}")
    print(f"{'=' * 80}\n")

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    try:
        with h5py.File(file_path, "r") as f:
            # 1. Metadata (Attributes)
            print("Metadata (Attributes):")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print()

            # 2. Datasets
            print("Datasets:")
            
            # Helper to print stats
            def print_ds_info(name, ds):
                print(f"  {name}: shape={ds.shape}, dtype={ds.dtype}")
                # Optional: Print simple stats for numeric types if small
                # Or just print range
                if np.issubdtype(ds.dtype, np.number) and ds.shape[0] > 0:
                    try:
                        # Quick stats on a sample to avoid full read
                         # For massive files, don't read all, maybe first 10k
                        sample_len = min(10000, ds.shape[0])
                        sample = ds[:sample_len]
                        if np.issubdtype(ds.dtype, np.floating):
                            print(f"    Range: [{np.nanmin(sample):.4f}, {np.nanmax(sample):.4f}]")
                        else:
                            print(f"    Range: [{np.min(sample)}, {np.max(sample)}]")
                    except Exception:
                        pass # Ignore errors on special types

            keys = sorted(list(f.keys()))
            for key in keys:
                print_ds_info(key, f[key])

            # 3. Validation Logic
            if "episode_lengths" in f:
                lengths = f["episode_lengths"][:]
                if len(lengths) > 0:
                    print("\nEpisode Length Stats:")
                    print(f"  Count: {len(lengths)}")
                    print(f"  Min/Max: {lengths.min()} / {lengths.max()}")
                    print(f"  Mean: {lengths.mean():.2f}")
                    print(f"  Total Steps: {lengths.sum()}")
            
            # 4. Check for Format
            if "tokens" in f:
                print("\n[INFO] File uses Monolithic Token Format.")
            elif "position" in f and "velocity" in f:
                print("\n[INFO] File uses Granular Feature Format.")
            else:
                print("\n[WARN] Unknown file format.")

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 data files.")
    parser.add_argument("file_path", type=str, nargs="?", help="Path to HDF5 file")
    parser.add_argument("--latest", action="store_true", help="Inspect latest file in data/massive_collection")
    args = parser.parse_args()

    if args.latest:
        data_dir = Path("data/massive_collection") # Or bc_pretraining?
        if not data_dir.exists():
             # Fallback
             data_dir = Path("data/bc_pretraining")
        
        if data_dir.exists():
            # Find all h5 files
            files = sorted(data_dir.rglob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                file_path = files[0]
                print(f"Selected latest file: {file_path}")
                inspect_h5_file(file_path)
            else:
                print(f"No .h5 files found in {data_dir}")
        else:
            print("No data directory found.")
    
    elif args.file_path:
        inspect_h5_file(Path(args.file_path))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
