import pickle
from pathlib import Path
import torch
from src.train.data_loader import load_bc_data

def create_debug_data():
    print("Loading full dataset...")
    # Load the full dataset using existing loader
    # It finds the latest automatically
    data = load_bc_data()
    
    print("Creating debug dataset (10% size)...")
    debug_data = {}
    
    # Process team data
    for team_key in ["team_0", "team_1"]:
        team_data = data[team_key]
        debug_team = {}
        
        # Calculate split index (10%)
        total_steps = team_data["tokens"].shape[0]
        split_idx = int(total_steps * 0.1)
        
        print(f"  {team_key}: Keeping {split_idx} of {total_steps} steps")
        
        debug_team["tokens"] = team_data["tokens"][:split_idx]
        debug_team["actions"] = team_data["actions"][:split_idx]
        debug_team["rewards"] = team_data["rewards"][:split_idx]
        
        debug_data[team_key] = debug_team

    # Process metadata
    total_episodes = data["episode_lengths"].shape[0]
    ep_split_idx = int(total_episodes * 0.1)
    
    print(f"  Episodes: Keeping {ep_split_idx} of {total_episodes}")
    
    debug_data["episode_lengths"] = data["episode_lengths"][:ep_split_idx]
    debug_data["episode_ids"] = data["episode_ids"][:ep_split_idx] # Assuming this exists and matches length
    if "metadata" in data:
        debug_data["metadata"] = data["metadata"]

    # Save to a new directory
    output_dir = Path("data/bc_pretraining/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "aggregated_data.pkl"
    print(f"Saving to {output_path}...")
    
    with open(output_path, "wb") as f:
        pickle.dump(debug_data, f)
        
    print("Done!")

if __name__ == "__main__":
    create_debug_data()
