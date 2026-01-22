import pytest
import os
import torch
import h5py
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Import the training function
from src.train.train_world_model import train_world_model

def test_world_model_training_pipeline(tmp_path):
    """Test the World Model training pipeline using config_test.yaml."""
    
    # 1. Load the test configuration
    print(f"DEBUG: Current CWD: {os.getcwd()}")
    config_path = "configs/config_test.yaml"
    if not os.path.exists(config_path):
        print(f"DEBUG: Config not found at {os.path.abspath(config_path)}")
        pytest.skip(f"Test config not found at {config_path}")
        
    cfg = OmegaConf.load(config_path)
    
    # 2. Override paths to use tmp_path
    data_dir = tmp_path / "data" / "bc_pretraining"
    data_dir.mkdir(parents=True)
    h5_path = data_dir / "aggregated_data.h5"
    
    output_dir = tmp_path / "models" / "world_model"
    output_dir.mkdir(parents=True)
    
    cfg.collect.output_dir = str(data_dir)
    # Critical: Override the hardcoded path in config_test.yaml
    cfg.train.bc_data_path = str(h5_path)
    
    # Force CPU for testing to avoid CUDA errors with dummy data or environment
    cfg.device = "cpu"
    # Also disable AMP just in case
    cfg.train.amp = False
    # We will patch get_latest_data_path or set the argument if possible.
    # train_world_model doesn't take data path in config usually, it searches.
    # So we'll need to mock `train.data_loader.get_latest_data_path`.
    
    # Also set mode to train_wm (implicit via calling the function, but for config correctness)
    cfg.mode = "train_wm"
    # Ensure offline wandb
    cfg.wandb.enabled = False
    
    # 3. Create Mini-Dataset from Real Data
    # Path from config.yaml
    real_data_path = Path(r"data\bc_pretraining\20260117_160241\aggregated_data.h5")
    
    if not real_data_path.exists():
        pytest.skip(f"Real data not found at {real_data_path}")

    # Copy a small slice (e.g. 2 episodes)
    num_episodes_to_keep = 2
    
    with h5py.File(real_data_path, "r") as src:
        with h5py.File(h5_path, "w") as dst:
            # Copy Attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            
            # Update num_episodes in attributes to match our slice
            dst.attrs["num_episodes"] = num_episodes_to_keep
            
            # Copy Datasets
            # We need to calculate how many timesteps correspond to these episodes
            episode_lengths = src["episode_lengths"][:num_episodes_to_keep]
            total_timesteps = episode_lengths.sum()
            dst.attrs["total_timesteps"] = total_timesteps
            
            dst.create_dataset("episode_lengths", data=episode_lengths)
            
            # For other datasets, we slice [0 : total_timesteps]
            for key in src.keys():
                if key == "episode_lengths": continue
                
                # Check shape to ensure we slice the time dimension correctly
                # All time-series data usually has shape (TotalTime, ...)
                # episode_ids is (TotalTime,)
                if src[key].shape[0] >= total_timesteps:
                    data_slice = src[key][:total_timesteps]
                    dst.create_dataset(key, data=data_slice)
                else:
                    # Metadata or small arrays? Just copy full if small?
                    # Or warn?
                    # Assuming unified dataset structure where dim0 is time.
                    dst.create_dataset(key, data=src[key][:])
        
    # 4. Run Training
    # We need to mock get_latest_data_path to return our h5_path
    
    # We also want to ensure it runs quickly but triggers SWA.
    # Set warmup steps to very low so SWA starts early.
    # config_test.yaml doesn't have scheduler, so we must create it.
    cfg.world_model.scheduler = OmegaConf.create({
        "type": "warmup_constant",
        "warmup": {
            "steps": 5,
            "start_lr": 1e-7
        }
    })
    
    # Force minimal accum steps so we update frequent
    cfg.world_model.gradient_accumulation_steps = 1
    
    # config_test has epochs=3. Let's force 2 epochs to verify epoch transition logic.
    cfg.world_model.epochs = 2
    # reduce batch size if needed, config_test is already small
    cfg.world_model.num_workers = 0
    
    with patch("src.train.data_loader.get_latest_data_path", return_value=str(h5_path)), \
         patch("torch.cuda.is_available", return_value=False):
        # We also need to patch os.getcwd if the script relies on it for output_dir relative paths
        # But we can override hydra output dir if we ran via main... 
        # Since we call `train_world_model` directly, it doesn't use Hydra's run dir logic 
        # unless `train_world_model` sets it up.
        # `train_world_model` accesses `os.getcwd()` for creating `output_dir`.
        
        # Let's just run it. It will write to `models/world_model/run_...` in current CWD.
        # We can chdir to tmp_path to keep it clean.
        
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            train_world_model(cfg)
        finally:
            os.chdir(orig_cwd)
            
    # 5. Assertions
    # Check if any model checkpoint was created
    # train_world_model saves to `dataset_path/models/world_model/run_...` usually?
    # No, it says `Output directory: models\world_model\run_...` in logs.
    # Since we chdir'd to tmp_path, it should be in `tmp_path/models/world_model/run_...`
    
    run_dirs = list((tmp_path / "models" / "world_model").glob("run_*"))
    assert len(run_dirs) > 0, "No run directory created"
    
    # Check for checkpoint
    checkpoints = list(run_dirs[0].glob("*.pt"))
    assert len(checkpoints) > 0, "No checkpoint (.pt) file created"

