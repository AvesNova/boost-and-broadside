"""Utility to inspect collected training data"""

import pickle
from pathlib import Path

import torch


def inspect_data_file(file_path: Path) -> None:
    """
    Load and print information about a collected data file

    Args:
        file_path: Path to the pickle file
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*80}\n")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if "metadata" in data:
        print("Metadata:")
        for key, value in data["metadata"].items():
            print(f"  {key}: {value}")
        print()

    print("Data tensors:")
    for key, value in data.items():
        if key == "metadata":
            continue
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    print(
                        f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}"
                    )
                else:
                    print(f"    {subkey}: type={type(subvalue)}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    if "episode_lengths" in data:
        lengths = data["episode_lengths"]
        print(f"\nEpisode lengths:")
        print(f"  Min: {lengths.min().item()}")
        print(f"  Max: {lengths.max().item()}")
        print(f"  Mean: {lengths.float().mean().item():.2f}")
        print(f"  Median: {lengths.float().median().item():.0f}")

    if "team_0" in data and "team_1" in data:
        if "rewards" in data["team_0"] and "rewards" in data["team_1"]:
            r0 = data["team_0"]["rewards"]
            r1 = data["team_1"]["rewards"]
            print(f"\nRewards:")
            print(
                f"  Team 0 - Mean: {r0.mean().item():.4f}, Sum: {r0.sum().item():.2f}"
            )
            print(
                f"  Team 1 - Mean: {r1.mean().item():.4f}, Sum: {r1.sum().item():.2f}"
            )


def main() -> None:
    """Inspect all collected data files"""
    import sys

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
