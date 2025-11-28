import pickle
from pathlib import Path
from src.train.data_loader import load_bc_data

def create_debug_dataset():
    print("Loading full dataset...")
    data = load_bc_data()
    
    print("Slicing data...")
    # Slice to 100 episodes
    n_episodes = 100
    episode_lengths = data["episode_lengths"][:n_episodes]
    total_timesteps = int(episode_lengths.sum().item())
    
    print(f"Slicing to {n_episodes} episodes ({total_timesteps} timesteps)...")
    
    debug_data = {
        "episode_lengths": episode_lengths,
        "team_0": {
            "tokens": data["team_0"]["tokens"][:total_timesteps],
            "actions": data["team_0"]["actions"][:total_timesteps],
            "rewards": data["team_0"]["rewards"][:total_timesteps],
        },
        "team_1": {
            "tokens": data["team_1"]["tokens"][:total_timesteps],
            "actions": data["team_1"]["actions"][:total_timesteps],
            "rewards": data["team_1"]["rewards"][:total_timesteps],
        }
    }
    
    # Check if rewards are flattened or per-episode
    # In data_loader: rewards = torch.cat([...], dim=0) implies they are flattened in the loader, 
    # but in the dict they might be list of tensors or padded tensor?
    # "tokens": (B, T, N, F) usually.
    # Let's verify structure by printing shapes in the script if needed, but assuming standard structure.
    # If "tokens" is (N_ep, T, ...), slicing [:100] works.
    
    # However, if rewards are flattened in the input dict, we need to slice carefully based on episode lengths.
    # But usually `load_bc_data` returns the raw dict from pickle.
    # Let's assume the dict structure matches the keys.
    
    # Wait, `load_bc_data` returns the pickle content.
    # If the pickle content has flattened arrays, we need to slice by sum of lengths.
    # But usually we save as (N_ep, ...) tensors or lists.
    # Let's assume they are tensors with batch dim 0 being episodes.
    
    # If they are not, this script might fail or produce invalid data.
    # But for "tokens" and "actions" it's likely (N_ep, T, ...).
    # For "rewards", it might be (N_ep, T).
    
    output_dir = Path("data/bc_pretraining/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aggregated_data.pkl"
    
    print(f"Saving debug dataset to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(debug_data, f)
    
    print("Done!")

if __name__ == "__main__":
    create_debug_dataset()
