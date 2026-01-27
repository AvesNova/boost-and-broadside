
import h5py
import numpy as np

path = "data/bc_pretraining/test_collection/aggregated_data.h5"

try:
    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))
        if "episode_lengths" in f:
            print("Episodes:", len(f["episode_lengths"]))
            print("Total Steps:", f["tokens"].shape[0])
            print("Tokens Shape:", f["tokens"].shape)
            print("Actions Shape:", f["actions"].shape)
            print("Rewards Shape:", f["rewards"].shape)
            print("Returns Shape:", f["returns"].shape)
            
            # Check content
            print("First Reward:", f["rewards"][0, 0])
            print("First Action:", f["actions"][0, 0])
            
        else:
            print("No episode_lengths found.")
            
except Exception as e:
    print(f"Error: {e}")
