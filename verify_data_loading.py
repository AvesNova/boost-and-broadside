from pathlib import Path
from src.train.data_loader import load_bc_data, create_world_model_data_loader

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
    
    print("Creating data loader...")
    train_loader, val_loader = create_world_model_data_loader(
        data, 
        batch_size=32,
        context_len=128,
        validation_split=0.2,
        num_workers=0
    )
    
    print("Data loader created.")
    print("Train loader length:", len(train_loader))
    
    print("Iterating train loader...")
    for i, (states, actions) in enumerate(train_loader):
        print(f"Batch {i}: states={states.shape}, actions={actions.shape}")
        if i >= 2:
            break
            
if __name__ == "__main__":
    verify()
