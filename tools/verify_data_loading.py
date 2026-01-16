from pathlib import Path
import sys

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train.data_loader import load_bc_data, create_unified_data_loaders


def verify():
    data_path = "data/bc_pretraining/debug/aggregated_data.pkl"
    print(f"Loading data from {data_path}...")
    print(f"Absolute path: {Path(data_path).resolve()}")

    data = load_bc_data(data_path)

    print("Data loaded. Keys:", data.keys())
    print("Team 0 tokens shape:", data["team_0"]["tokens"].shape)
    print("Team 0 actions shape:", data["team_0"]["actions"].shape)
    print("Episode lengths shape:", data["episode_lengths"].shape)
    print("Sum of episode lengths:", data["episode_lengths"].sum())

    print("Creating unified data loaders...")
    # Adjust parameters as needed for the debug dataset size
    ts_loader, tl_loader, vs_loader, vl_loader = create_unified_data_loaders(
        data,
        short_batch_size=32,
        long_batch_size=16,  # reduced for debug
        short_batch_len=32,
        long_batch_len=128,
        batch_ratio=4,
        validation_split=0.2,
        num_workers=0,
    )

    print("Data loaders created.")
    print("Train Short loader length:", len(ts_loader))
    print("Train Long loader length:", len(tl_loader))
    print("Val Short loader length:", len(vs_loader))
    print("Val Long loader length:", len(vl_loader))

    print("Iterating train short loader...")
    for i, (tokens, actions, returns, loss_mask, action_masks) in enumerate(ts_loader):
        print(
            f"Batch {i}: tokens={tokens.shape}, actions={actions.shape}, returns={returns.shape}"
        )
        if i >= 2:
            break


if __name__ == "__main__":
    verify()
