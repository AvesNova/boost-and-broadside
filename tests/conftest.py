import pytest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from omegaconf import OmegaConf
from src.env.ship import Ship, default_ship_config
from src.env.state import State

@pytest.fixture
def default_config():
    """Return a default configuration for testing."""
    return OmegaConf.create({
        "mode": "play",
        "environment": {
            "world_size": [1000, 1000],
            "memory_size": 10000,
            "max_ships": 4,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
            "render_mode": "none",
        },
        "agents": {
            "team1": {
                "agent_type": "scripted",
                "agent_config": {
                    "max_shooting_range": 500.0,
                    "angle_threshold": 5.0,
                    "bullet_speed": 500.0,
                    "target_radius": 10.0,
                    "radius_multiplier": 1.5,
                    "world_size": [1000, 1000],
                }
            },
            "team2": {
                "agent_type": "scripted",
                "agent_config": {
                    "max_shooting_range": 500.0,
                    "angle_threshold": 5.0,
                    "bullet_speed": 500.0,
                    "target_radius": 10.0,
                    "radius_multiplier": 1.5,
                    "world_size": [1000, 1000],
                }
            },
        },
        "collect": {
            "teams": ["team1", "team2"],
            "max_episode_length": 100,
            "num_workers": 1,
            "episodes_per_mode": {"1v1": 1},
            "output_dir": "data/bc_pretraining",
            "render_mode": "none",
            "save_frequency": 10,
        },
        "train": {
            "batch_size": 32,
            "model": {
                "transformer": {
                    "token_dim": 10,
                    "max_ships": 4,
                    "num_actions": 6,
                    "embed_dim": 16,
                    "num_heads": 2,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "use_layer_norm": True,
                }
            }
        }
    })

@pytest.fixture
def dummy_ship():
    """Return a dummy ship instance."""
    return Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=100.0,
        initial_y=100.0,
        initial_vx=10.0,
        initial_vy=0.0,
        world_size=(1000, 1000),
    )

@pytest.fixture
def dummy_state(dummy_ship):
    """Return a dummy state with one ship."""
    return State(ships={0: dummy_ship})
