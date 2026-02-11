
import pytest
import torch
import numpy as np
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.core.config import ShipConfig

def test_random_speed_initialization():
    """Test that ships are initialized with random speed when enabled."""
    config = ShipConfig(
        random_speed=True,
        min_speed=50.0,
        max_speed=150.0,
        world_size=(1000, 1000),
        dt=0.015
    )
    
    env = TensorEnv(
        num_envs=10, 
        config=config, 
        device="cpu", # Use CPU for easier testing without CUDA req
        max_ships=8
    )
    
    # Reset environments
    obs = env.reset()
    
    # Check ship velocity
    # shape: (num_envs, max_ships)
    vel = env.state.ship_vel
    speed = vel.abs()
    
    # Filter for alive ships (although all should be alive at start)
    alive = env.state.ship_alive
    
    # Check if any speeds are non-zero (should be most if not all)
    assert (speed[alive] > 0).any(), "Speeds should be non-zero with random_speed=True"
    
    # Check range (with tolerance for float precision)
    assert (speed[alive] >= config.min_speed - 1e-4).all(), f"Speeds should be >= {config.min_speed}"
    assert (speed[alive] <= config.max_speed + 1e-4).all(), f"Speeds should be <= {config.max_speed}"
    
    # Check alignment - velocity direction should match attitude
    # attitude is complex unit vector
    # velocity should be speed * attitude
    expected_vel = speed * env.state.ship_attitude
    diff = (vel - expected_vel).abs()
    assert (diff[alive] < 1e-4).all(), "Velocity should be aligned with attitude"

def test_no_random_speed_initialization():
    """Test that ships are initialized with zero speed when disabled."""
    config = ShipConfig(
        random_speed=False,
        world_size=(1000, 1000),
        dt=0.015
    )
    
    env = TensorEnv(
        num_envs=10, 
        config=config, 
        device="cpu",
        max_ships=8
    )
    
    env.reset()
    
    vel = env.state.ship_vel
    speed = vel.abs()
    
    # Expect default speed of 100.0
    assert (speed > 99.9).all() and (speed < 100.1).all(), "Speeds should be 100.0 with random_speed=False"
