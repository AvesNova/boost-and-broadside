import pytest
import os
from omegaconf import OmegaConf
from unittest.mock import patch

# Import the training function
from boost_and_broadside.train.train_world_model import train_world_model

def test_world_model_training_pipeline(tmp_path, synthetic_h5_data):
    """Test the World Model training pipeline using config_test.yaml."""
    
    # 1. Load the test configuration
    config_path = "configs/config_test.yaml"
    if not os.path.exists(config_path):
        pytest.skip(f"Test config not found at {config_path}")
        
    cfg = OmegaConf.load(config_path)
    
    # 2. Override paths to use tmp_path
    # Use the synthetic data path provided by the fixture
    h5_path = synthetic_h5_data
    
    output_dir = tmp_path / "models" / "world_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.collect.output_dir = str(tmp_path / "data" / "bc_pretraining")
    cfg.train.bc_data_path = str(h5_path)
    
    # Force CPU to avoid CUDA errors
    cfg.device = "cpu"
    cfg.train.amp = False
    
    # Ensure offline wandb
    cfg.wandb.enabled = False
    
    # 3. Scheduler & Optimization Config
    # Create scheduler config if missing (config_test might be minimal)
    cfg.world_model.scheduler = OmegaConf.create({
        "type": "warmup_constant",
        "warmup": {
            "steps": 2,
            "start_lr": 1e-7
        }
    })
    
    # Force minimal accum steps so we update frequently
    cfg.world_model.gradient_accumulation_steps = 1
    
    # Run minimal epochs
    cfg.world_model.epochs = 2
    cfg.world_model.num_workers = 0
    cfg.world_model.curriculum = {"enabled": False} # Disable filtering
    cfg.world_model.seq_len = 16 # Small seq len for small dataset
    cfg.train.compile = False
    
    # 4. Run Training
    # Mock get_latest_data_path to return our synthetic file
    # Mock torch.cuda.is_available to ensure CPU path is taken
    
    with patch("boost_and_broadside.train.data_loader.get_latest_data_path", return_value=str(h5_path)), \
         patch("torch.cuda.is_available", return_value=False):
        
        # Isolate execution directory
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            train_world_model(cfg)
        finally:
            os.chdir(orig_cwd)
            
    # 5. Assertions
    # Check if any model checkpoint was created in the tmp_path
    run_dirs = list((tmp_path / "models" / "world_model").glob("run_*"))
    assert len(run_dirs) > 0, "No run directory created"
    
    # Check for checkpoint
    checkpoints = list(run_dirs[0].glob("*.pt"))
    assert len(checkpoints) > 0, "No checkpoint (.pt) file created"


