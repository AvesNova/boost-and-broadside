
import h5py
import numpy as np

path = "data/bc_pretraining/debug_collection/aggregated_data.h5"

try:
    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))
        if "episode_lengths" in f:
            print("Episodes:", len(f["episode_lengths"]))
            print("Total Steps:", f["tokens"].shape[0])
            print("Tokens Shape:", f["tokens"].shape)
            print("First Reward:", f["rewards"][0, 0])
            print("Last Reward:", f["rewards"][-1, 0])
            
        else:
            print("No episode_lengths found.")
            
except Exception as e:
    print(f"Error: {e}")
