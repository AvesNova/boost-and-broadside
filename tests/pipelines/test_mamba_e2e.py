
import pytest
import os
import torch
import numpy as np
import h5py
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Import the training function
from src.train.train_world_model import train_world_model

@pytest.fixture
def synthetic_mamba_data(tmp_path):
    """Create a synthetic HDF5 file valid for MambaBB (continuous view)."""
    h5_path = tmp_path / "mamba_data.h5"
    
    # 2 episodes, total 40 steps
    # Token dim 16
    # Shape: (TotalSteps, N_ships, D)
    N_ships = 4
    tokens = np.random.randn(40, N_ships, 16).astype(np.float32)
    actions = np.random.randint(0, 2, (40, N_ships, 3)).astype(np.int32)
    
    # Ep IDs: 0-19 -> ID 0, 20-39 -> ID 1
    episode_ids = np.zeros(40, dtype=np.int32)
    episode_ids[20:] = 1
    
    episode_lengths = np.array([20, 20], dtype=np.int32)
    
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("actions", data=actions)
        f.create_dataset("episode_ids", data=episode_ids)
        f.create_dataset("episode_lengths", data=episode_lengths)
        f.attrs["token_dim"] = 16
        f.attrs["num_actions"] = 3
        f.attrs["max_ships"] = N_ships 
        f.attrs["total_timesteps"] = 40
        f.attrs["num_episodes"] = 2
        
    return h5_path

def test_mamba_training_pipeline(tmp_path, synthetic_mamba_data):
    """
    Test the MambaBB training pipeline end-to-end.
    - Loads config/model/mamba_bb
    - Uses synthetic data
    - Runs 2 epochs on CPU
    """
    
    # 1. Load Base Config + Mamba Config
    # We construct a minimal cfg that mimics the hydra composition
    # Base keys needed: collect, train, world_model, environment, wandb
    
    # We can load the actual files if we know paths, or construct manually.
    # Construction is safer for unit tests to be self-contained, but we want to test
    # the actual mamba_bb.yaml config too if possible.
    
    mamba_cfg_path = Path("configs/model/mamba_bb.yaml")
    if not mamba_cfg_path.exists():
        pytest.skip("Mamba config not found")
        
    mamba_cfg = OmegaConf.load(mamba_cfg_path)
    
    # Base skeleton
    cfg = OmegaConf.create({
        "world_model": mamba_cfg,
        "environment": {
            "world_size": [100.0, 100.0],
            "n_ships": 4
        },
        "train": {
            "amp": False,
            "seed": 42,
            "bc_data_path": str(synthetic_mamba_data), # Explicit path
            "compile": False
        },
        "collect": {
            "output_dir": str(tmp_path / "data")
        },
        "wandb": {
            "enabled": False,
            "project": "test",
            "entity": "test",
            "group": "test"
        }
    })
    
    # 2. Overrides for Test Speed/Size
    cfg.world_model.embed_dim = 64 # Size must satisfy Mamba2 constraints
    cfg.world_model.n_layers = 2
    cfg.world_model.n_heads = 2
    cfg.world_model.short_batch_size = 2
    cfg.world_model.batch_size = 2
    cfg.world_model.seq_len = 5
    cfg.world_model.epochs = 2
    cfg.world_model.num_workers = 0
    cfg.world_model.batch_ratio = 1 # Required by trainer, though ignored by Mamba loader
    cfg.world_model.scheduler = OmegaConf.create({
        "type": "warmup_constant",
         "warmup": {"steps": 1, "start_lr": 1e-5}
    })
    
    
    # Override device
    device = "cpu"
    
    # 3. Paths
    h5_path = str(synthetic_mamba_data)
    
    # Mock Mamba2 (CUDA kernel) for CPU testing of the Pipeline
    class MockMamba2(torch.nn.Module):
        def __init__(self, d_model, d_state=128, expand=2):
            super().__init__()
            self.linear = torch.nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)

    # 4. Run
    
    with patch("src.train.data_loader.get_latest_data_path", return_value=h5_path), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("agents.mamba_bb.Mamba2", MockMamba2):
         
         # We need to ensure models are saved to output_dir. 
         output_dir = tmp_path / "models" / "mamba_result"
         output_dir.mkdir(parents=True)
         
         orig_cwd = os.getcwd()
         os.chdir(output_dir)
         try:
             train_world_model(cfg)
         except Exception as e:
             import traceback
             traceback.print_exc()
             pytest.fail(f"Training failed with error: {e}")
         finally:
             os.chdir(orig_cwd)
             
    # 5. Verify Artifacts
    # Expect 'world_model_final.pt' or best/last checkpoints in output_dir/models/world_model/runs/...
    # train_world_model creates models/world_model/run_DATE relative to CWD (which is output_dir)
    
    actual_run_root = output_dir / "models" / "world_model"
    run_dirs = list(actual_run_root.glob("run_*"))
    assert len(run_dirs) > 0, f"No run directory created in {actual_run_root}"
    
    ckpts = list(run_dirs[0].glob("*.pt"))
    assert len(ckpts) > 0, "No Valid/Final checkpoints found"
