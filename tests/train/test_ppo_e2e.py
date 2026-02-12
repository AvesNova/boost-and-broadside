
import pytest
import torch
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
                "num_envs": 4,
                "num_steps": 16, # Short rollout
                "total_timesteps": 64, # 4 * 16 = 64, so 1 iteration
                "learning_rate": 2.5e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "num_minibatches": 2,
                "update_epochs": 2,
                "norm_adv": True,
                "clip_coef": 0.2,
                "clip_vloss": True,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": 0.015,
                "anneal_lr": False
            }
        },
        "model": {
             "d_model": 128,
             "n_layers": 2,
             "n_heads": 4, # Added
             "vocab_size": 100, 
             "max_ships": 4,
             "dim": 128,
             "d_state": 16,
             "d_conv": 4,
             "expand": 2,
             "dt_rank": "auto",
             "dt_min": 0.001,
             "dt_max": 0.1,
             "dt_init": "random",
             "dt_scale": 1.0,
             "dt_init_floor": 1e-4,
             "conv_bias": True,
             "bias": False,
             "use_fast_path": True, 
             # Spatial Layer config
             "spatial_layer": {
                 "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
                 "d_model": 128,
                 "n_heads": 4
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
        num_envs=ppo_config.train.ppo.num_envs,
        config=ship_config,
        device=device,
        max_ships=ppo_config.model.max_ships,
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
