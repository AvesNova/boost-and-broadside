from pathlib import Path
import sys

# Add src to sys.path



import h5py
import numpy as np

def inspect_data_file(file_path: Path) -> None:
    """
    Load and print information about a collected data file (HDF5)
    """
    print(f"\n{'=' * 80}")
    print(f"Inspecting: {file_path}")
    print(f"{'=' * 80}\n")

    try:
        with h5py.File(file_path, "r") as f:
            print("Attributes (Metadata):")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print("\nDatasets:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                    # Sample stats if numeric
                    if obj.shape[0] > 0 and np.issubdtype(obj.dtype, np.number):
                        try:
                            # Sample first chunk to be fast
                            sample = obj[:1000]
                            print(f"    mean={np.mean(sample):.4f}, min={np.min(sample)}, max={np.max(sample)}")
                            if np.sum(sample) == 0 and np.max(sample) == 0 and np.min(sample) == 0:
                                print("    [WARNING] First 1000 items are all zeros!")
                        except: pass
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")

            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Failed to inspect HDF5: {e}")


def main() -> None:
    """Inspect all collected data files"""

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            inspect_data_file(file_path)
        else:
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    else:
        data_dir = Path("data/bc_pretraining")
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)

        pkl_files = list(data_dir.rglob("*.pkl"))
        if not pkl_files:
            print(f"No .pkl files found in {data_dir}")
            sys.exit(1)

        print(f"Found {len(pkl_files)} data files\n")

        for pkl_file in sorted(pkl_files):
            inspect_data_file(pkl_file)


if __name__ == "__main__":
    main()
