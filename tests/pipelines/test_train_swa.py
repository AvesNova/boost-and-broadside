import pytest
import os
import torch
import h5py
import numpy as np
from boost_and_broadside.modes.train import train
from omegaconf import OmegaConf
from boost_and_broadside.core.constants import NORM_HEALTH, STATE_DIM

def test_train_swa_pipeline(default_config, tmp_path):
    """Test the World Model training pipeline with SWA."""
    # Update config for testing
    cfg = default_config.copy()
    cfg.mode = "train_wm" # Or just train with flags
    cfg.train.run_collect = False
    cfg.train.run_bc = False
    cfg.train.run_rl = False
    cfg.train.run_world_model = True
    
    # World Model Config
    cfg.world_model = OmegaConf.create({
        "embed_dim": 64,
        "n_layers": 2,
        "n_heads": 2,
        "n_ships": 4,
        "context_len": 16,
        "seq_len": 16, # Required for setup.py
        "learning_rate": 1e-3,
        "epochs": 3, # Need at least 3 epochs to trigger SWA (0, 1, 2)
        "batch_ratio": 1,
        "batch_size": 2,
        "short_batch_size": 2,
        "long_batch_size": 2,
        "short_batch_len": 8,
        "long_batch_len": 16,
        "num_workers": 0, # Main process for testing
        "prefetch_factor": None, # Disable prefetch for num_workers=0
        "noise_scale": 0.0,
        "rollout": {
            "enabled": False,
            "start_epoch": 0,
            "ramp_epochs": 1,
            "max_len_start": 1,
            "max_len_end": 1
        },
        "loss": {
            "type": "uncertainty",
            "use_focal_loss": False,
            "weighted_loss_cap": 5.0,
            "weighted_loss_power": 0.5
        },
        "scheduler": {
            "type": "warmup_constant",
            "warmup": {"steps": 1, "start_lr": 0.0}
        }
    })
    
    cfg.wandb = OmegaConf.create({
        "enabled": False,
        "project": "test",
        "mode": "offline",
        "log_frequency": 1
    })

    # Create Dummy HDF5 Data
    data_dir = tmp_path / "data" / "bc_pretraining" / "dummy_run"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "aggregated_data.h5"
    
    cfg.train.bc_data_path = str(data_path)
    
    # Create valid dummy data
    N = 200 # increased to 200
    MaxShips = 4
    TokenDim = STATE_DIM # 15
    NumActions = 3 
    
    with h5py.File(data_path, "w") as f:
        # Granular features replacing tokens
        f.create_dataset("position", data=np.random.randn(N, MaxShips, 2).astype(np.float32))
        f.create_dataset("velocity", data=np.random.randn(N, MaxShips, 2).astype(np.float32))
        f.create_dataset("health", data=np.random.rand(N, MaxShips).astype(np.float32) * NORM_HEALTH)
        f.create_dataset("power", data=np.random.rand(N, MaxShips).astype(np.float32) * 100.0)
        f.create_dataset("attitude", data=np.random.randn(N, MaxShips, 2).astype(np.float32))
        f.create_dataset("ang_vel", data=np.random.randn(N, MaxShips).astype(np.float32))
        f.create_dataset("is_shooting", data=np.random.randint(0, 2, (N, MaxShips)).astype(np.float32))
        f.create_dataset("team_ids", data=np.zeros((N, MaxShips), dtype=np.float32))
        
        # Actions: Last dim is 3 (Power, Turn, Shoot)
        actions = np.zeros((N, MaxShips, 3), dtype=np.int32)
        actions[..., 0] = np.random.randint(0, 3, size=(N, MaxShips))
        actions[..., 1] = np.random.randint(0, 7, size=(N, MaxShips))
        actions[..., 2] = np.random.randint(0, 2, size=(N, MaxShips))
        f.create_dataset("actions", data=actions)
        
        f.create_dataset("action_masks", data=np.ones((N, MaxShips, 12), dtype=np.bool_)) # 3+7+2=12
        f.create_dataset("rewards", data=np.zeros((N, MaxShips), dtype=np.float32))
        f.create_dataset("returns", data=np.zeros((N, MaxShips), dtype=np.float32))
        
        # Episode IDs: 10 episodes of 20 steps
        ep_ids = np.repeat(np.arange(10), 20)
        f.create_dataset("episode_ids", data=ep_ids.astype(np.int32))
        f.create_dataset("episode_lengths", data=np.full(10, 20, dtype=np.int32))
        
        f.attrs["token_dim"] = TokenDim

    # Run Training
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        (tmp_path / "models" / "world_model").mkdir(parents=True, exist_ok=True)
        
        cfg.train.compile = False # Disable compile for test
        train(cfg)
        
        # Verify SWA model is saved
        run_dirs = list((tmp_path / "models" / "world_model").glob("run_*"))
        assert len(run_dirs) > 0, "No run directory found"
        last_run_dir = run_dirs[-1]
        
        assert (last_run_dir / "final_world_model.pt").exists()
        
        # Load and verify keys
        ckpt = torch.load(last_run_dir / "final_world_model.pt")
        assert "model_state_dict" in ckpt
        assert "swa_model_state_dict" in ckpt, "SWA model state dict missing from checkpoint!"
        
    except Exception as e:
        import traceback
        with open(os.path.join(original_cwd, "error.log"), "w") as f:
            f.write(traceback.format_exc())
        pytest.fail(f"SWA Training pipeline failed: {e}")
        
    finally:
        os.chdir(original_cwd)
