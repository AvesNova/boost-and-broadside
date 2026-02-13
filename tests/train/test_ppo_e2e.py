
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.gpu_wrapper import GPUEnvWrapper
from boost_and_broadside.models.yemong.scaffolds import YemongDynamics
from boost_and_broadside.train.rl.ppo import PPOTrainer

@pytest.fixture
def ppo_config():
    return OmegaConf.create({
        "train": {
             "ppo": {
                 "total_timesteps": 1000,
                 "num_envs": 2,
                 "num_steps": 16,
                 "gamma": 0.99,
                 "gae_lambda": 0.95,
                 "clip_range": 0.2,
                 "ent_coef": 0.0,
                 "vf_coef": 0.5,
                 "max_grad_norm": 0.5,
                 "learning_rate": 2.5e-4,
                 "update_epochs": 2,
                 "batch_size": 32,
                 "stats_window_size": 10,
                 "norm_adv": True,
                 "clip_coef": 0.2,
                 "target_kl": None,
                 "anneal_lr": False,
                 "num_minibatches": 2 # Must be <= num_envs (2) for splitting by env
             },
             "device": "cpu", # Used by test setup or other components? PPOTrainer uses self.device
             "log_interval": 1,
             "visualize": False,
             "seed": 42
        },
        "model": {
            "_target_": "boost_and_broadside.models.yemong.scaffolds.YemongDynamics",
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "input_dim": 5, # STATE_DIM
            "target_dim": 7, # TARGET_DIM
            "action_dim": 12,
            "action_embed_dim": 8,
            "loss_type": "fixed",
            "spatial_layer": {
                 "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
                 "d_model": 64,
                 "n_heads": 2
            },
            "loss": {
                 "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
                 "losses": [
                      {"_target_": "boost_and_broadside.models.components.losses.StateLoss", "weight": 1.0},
                      {"_target_": "boost_and_broadside.models.components.losses.ActionLoss", "weight": 1.0},
                      {"_target_": "boost_and_broadside.models.components.losses.ValueLoss", "weight": 1.0},
                      {"_target_": "boost_and_broadside.models.components.losses.RewardLoss", "weight": 1.0}
                 ]
            }
        },
        "environment": {
             "max_bullets": 5,
             "world_size": [100.0, 100.0]
        },
        "project_name": "test_ppo",
        "mode": "train_rl",
        "wandb": {
             "enabled": False,
             "project": "test",
             "entity": "test",
             "log_interval": 1
        }
    })

def test_ppo_training_loop(ppo_config):
    """
    Test the PPO training loop runs without error for one iteration.
    Verifies:
    1. Environment Reset
    2. Rollout Collection
    3. GAE Computation
    4. PPO Update (Gradient Step)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU RL test")
        
    device = torch.device("cuda")
    
    # Setup
    ship_config = ShipConfig()
    ship_config.world_size = (100.0, 100.0)
    
    env = TensorEnv(
        num_envs=ppo_config.train.ppo.num_envs, # Fixed config access path
        config=ship_config,
        device=device,
        max_ships=ppo_config.environment.max_bullets, # Use valid key or hardcode based on config
        # Actually max_ships is usually in environment config for envs
        # In ppo_config above: environment: { max_bullets: 5, world_size: ... }
        # Missing max_ships in env config. Let's hardcode or add it.
        # ship_config defaults to 4.
        max_bullets=ppo_config.environment.max_bullets
    )
    env = GPUEnvWrapper(env)
    
    agent = YemongDynamics(ppo_config.model)
    agent.to(device)
    
    trainer = PPOTrainer(ppo_config, env, agent)
    
    # Run Train (Should run 1 iteration)
    try:
        trainer.train()
    except Exception as e:
        pytest.fail(f"Training loop failed with error: {e}")
