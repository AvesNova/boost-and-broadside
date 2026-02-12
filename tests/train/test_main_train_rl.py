import pytest
import sys
from unittest.mock import patch
from hydra import compose, initialize
from omegaconf import OmegaConf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from main import my_app

import torch

def test_main_train_rl_execution():
    """
    Integration test that runs main.py with mode=train_rl.
    It mocks the training loop to avoid long execution but verifies 
    configuration loading and entry point logic.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Arguments simulating command line
    # Override total_timesteps to something very small for quick test
    overrides = [
        "mode=train_rl",
        "train.ppo.total_timesteps=32", # 1 iteration hopefully
        "train.ppo.num_envs=4",
        "train.ppo.num_steps=8",
        "train.ppo.num_minibatches=2",
        "train.ppo.update_epochs=1",
        "wandb.enabled=false" # Disable wandb for test
    ]
    
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config", overrides=overrides)
        
        # Mock PPOTrainer.train to just return (avoid running full loop if we trust unit test)
        # OR let it run a tiny bit.
        # Let's let it run to verify end-to-end integration of config + env + agent
        
        try:
             my_app(cfg)
        except Exception as e:
             pytest.fail(f"main.py failed with: {e}")
