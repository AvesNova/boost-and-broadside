
import pytest
from omegaconf import OmegaConf
import torch
import gymnasium as gym

from env.env import Environment
from env.sb3_wrapper import SB3Wrapper
from agents.sb3_world_model_adapter import WorldModelSB3Policy
from train.ppo_world_model import WorldModelPPO
from train.rl_trainer import create_sb3_env

class TestRLWorldModel:
    @pytest.fixture
    def config(self):
        return OmegaConf.create({
            "environment": {
                "render_mode": "none",
                "world_size": [1200, 800],
                "memory_size": 2,
                "max_ships": 4, # Reduced for test
                "agent_dt": 0.04,
                "physics_dt": 0.02,
                "random_positioning": True,
                "random_speed": True
            },
            "train": {
                "model": {
                    "transformer": {
                        "token_dim": 12 
                    }
                }
            },
            "team2": "scripted",
            "agents": {
                "scripted": {
                    "agent_type": "scripted",
                    "agent_config": {
                        "max_shooting_range": 500.0,
                        "angle_threshold": 5.0,
                        "bullet_speed": 500.0,
                        "target_radius": 10.0,
                        "radius_multiplier": 1.5,
                        "world_size": [1200, 800]
                    }
                }
            },
            "world_model": {
                "state_dim": 12,
                "action_dim": 12,
                "embed_dim": 64, # Small for test
                "n_layers": 2,
                "n_heads": 2,
                "max_ships": 4,
                "context_len": 16,
                "dropout": 0.0
            },
            "rl": {
                "policy_type": "world_model",
                "context_len": 16,
                "learning_rate": 3e-4,
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "aux_loss_coef": 1.0, 
                "freeze_world_model": False
            }
        })

    def test_policy_instantiation(self, config):
        env = SB3Wrapper(Environment(**config.environment), config)
        # Use simple FrameStackObservation
        env = gym.wrappers.FrameStackObservation(env, stack_size=config.rl.context_len)
        
        policy = WorldModelSB3Policy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda x: 3e-4,
            model_config=OmegaConf.to_container(config.world_model),
            aux_loss_config={"mask_ratio": 0.15}
        )
        assert policy is not None
        
        # Test Forward
        obs = env.reset()[0]
        # obs is Dict {"tokens": (16, 4, 12)}
        obs_tensor = {}
        for k, v in obs.items():
            obs_tensor[k] = torch.tensor(v).unsqueeze(0) # Batch dim
            
        actions, values, log_prob = policy(obs_tensor)
        assert actions.shape == (1, 4 * 3) # Flattened action space
        # 4 ships * 3 actions per ship = 12?
        # wrapper: self.action_space = spaces.MultiDiscrete([3, 7, 2] * self.max_ships)
        # But flattened? No, SB3 handles MultiDiscrete output as (Batch, N_dims)
        # So it should be (1, 4*3) = 12 dims?
        # Wait, MultiDiscrete([3, 7, 2] * 4) has 12 dimensions.
        assert actions.shape[-1] == 12 
        
        # Aux Loss
        loss = policy.get_dynamics_loss(obs_tensor)
        assert loss.item() >= 0

    def test_training_loop(self, config):
        env = SB3Wrapper(Environment(**config.environment), config)
        env = gym.wrappers.FrameStackObservation(env, stack_size=config.rl.context_len)
        
        model = WorldModelPPO(
            WorldModelSB3Policy,
            env,
            policy_kwargs={
                "model_config": OmegaConf.to_container(config.world_model),
                "aux_loss_config": {"mask_ratio": 0.15},
                "freeze_backbone": False
            },
            aux_loss_coef=0.5,
            n_steps=64,
            batch_size=32,
            verbose=1
        )
        
        # Train for a few steps
        model.learn(total_timesteps=128)
        # Should not crash
