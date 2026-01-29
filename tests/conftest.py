import os
import sys

import pytest
from omegaconf import OmegaConf
import numpy as np
import h5py

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def default_config():
    """Return a default configuration for testing."""
    return OmegaConf.create(
        {
            "mode": "play",
            "environment": {
                "world_size": [1000, 1000],
                "memory_size": 10000,
                "max_ships": 4,
                "agent_dt": 0.04,
                "physics_dt": 0.02,
                "render_mode": "none",
                "random_positioning": False,
                "random_speed": False,
            },
            "team1": "scripted",
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
                        "world_size": [1000, 1000],
                    },
                },
            },
            "collect": {
                "teams": ["scripted", "scripted"],
                "max_episode_length": 100,
                "num_workers": 1,
                "episodes_per_mode": {"1v1": 1},
                "output_dir": "data/bc_pretraining",
                "render_mode": "none",
                "save_frequency": 10,
            },
            "train": {
                "run_collect": False,
                "run_bc": False,
                "run_rl": False,
                "use_bc": True,
                "use_rl": True,
                "bc_data_path": "data/bc_pretraining/aggregated_data.pkl",
                "batch_size": 32,
                "model": {
                    "transformer": {
                        "token_dim": 15,
                        "max_ships": 4,
                        "num_actions": 6,
                        "embed_dim": 16,
                        "num_heads": 2,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "use_layer_norm": True,
                    }
                },
                "bc": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 1,
                    "validation_split": 0.2,
                    "early_stopping_patience": 10,
                    "policy_weight": 1.0,
                    "value_weight": 0.5,
                },
                "rl": {
                    "pretrained_model_path": None,
                    "n_envs": 1,
                    "total_timesteps": 1000,
                    "learning_rate": 0.0003,
                    "n_steps": 128,
                    "batch_size": 32,
                    "n_epochs": 3,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
            },
        }
    )


# Legacy fixtures removed



@pytest.fixture
def synthetic_h5_data(tmp_path):
    """Create a synthetic HDF5 dataset for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "aggregated_data.h5"

    num_episodes = 5
    episode_len = 200
    total_timesteps = num_episodes * episode_len
    # Assuming defaults from audit: max_ships=4, token_dim=15, num_actions=4 (simplified)
    # Actually checking default_config fixture: token_dim=15, max_ships=4.
    max_ships = 4
    token_dim = 15
    num_actions = 3 # 3 discrete action components (Power, Turn, Shoot)

    with h5py.File(file_path, "w") as f:
        # Attributes
        f.attrs["num_episodes"] = num_episodes
        f.attrs["total_timesteps"] = total_timesteps
        f.attrs["total_sim_time"] = 100.0
        f.attrs["max_ships"] = max_ships
        f.attrs["token_dim"] = token_dim
        f.attrs["num_actions"] = num_actions

        # Datasets
        # tokens: (N, MaxShips, TokenDim)
        f.create_dataset("tokens", data=np.random.randn(total_timesteps, max_ships, token_dim).astype(np.float32))
        
        # actions: (N, MaxShips, NumActions) - discrete indices
        # Generate integers 0..1 (Safe for all: Power=3, Turn=7, Shoot=2)
        f.create_dataset("actions", data=np.random.randint(0, 2, (total_timesteps, max_ships, num_actions)).astype(np.int64))

        # action_masks: (N, MaxShips, NumActions) - boolean
        f.create_dataset("action_masks", data=np.ones((total_timesteps, max_ships, num_actions), dtype=bool))


        # rewards: (N,)
        f.create_dataset("rewards", data=np.random.randn(total_timesteps).astype(np.float32))

        # returns: (N,)
        f.create_dataset("returns", data=np.random.randn(total_timesteps).astype(np.float32))

        # episode_ids: (N,)
        ids = np.repeat(np.arange(num_episodes), episode_len)
        f.create_dataset("episode_ids", data=ids.astype(np.int64))
        
        # episode_lengths: (NumEpisodes,)
        f.create_dataset("episode_lengths", data=np.full((num_episodes,), episode_len, dtype=np.int64))

    return str(file_path)
