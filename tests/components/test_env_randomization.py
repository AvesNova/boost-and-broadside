
import pytest
import numpy as np
from env.env import Environment
from env.ship import default_ship_config

def test_environment_random_initialization():
    # randomized env
    env = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=10,
        max_ships=8,
        agent_dt=0.1,
        physics_dt=0.01,
        random_positioning=True,
        random_speed=True,
        random_initialization=True # Enable random health/power
    )
    
    # Reset 4v4
    state, info = env.reset(game_mode="4v4")
    
    # Check that health/power are not all equal to max
    # We expect variance.
    
    varied_health = False
    varied_power = False
    
    max_health = default_ship_config.max_health
    max_power = default_ship_config.max_power
    
    observation = env.get_observation()
    
    # ship_ids are 0..7
    for i in range(8):
        # We need to access the state directly to get exact values, 
        # as observation might be normalized (though env.get_observation currently returns raw tensors for some fields? 
        # actually env.get_observation returns a dict of tensors.
        # Let's look at env.py:
        # "health": torch.zeros((self.max_ships), dtype=torch.int64) -> Wait, dtype is int64?
        # In env.py: "health": torch.zeros((self.max_ships), dtype=torch.int64) -> This seems wrong if health is float.
        # Checking ship.py: self.health is float.
        # Checking reset_from_observation: health = float(initial_obs["health"][i, 0].item())
        # Let's check the state object directly for higher precision and certainty.
        
        ship = env.state.ships[i]
        
        if abs(ship.health - max_health) > 0.01:
            varied_health = True
        
        if abs(ship.power - max_power) > 0.01:
            varied_power = True
            
    assert varied_health, "Health should be randomized (not all max)"
    assert varied_power, "Power should be randomized (not all max)"

def test_environment_default_initialization():
    # default env
    env = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=10,
        max_ships=8,
        agent_dt=0.1,
        physics_dt=0.01,
        random_positioning=True,
        random_speed=True,
        random_initialization=False # Disable
    )
    
    env.reset(game_mode="4v4")
    
    for i in range(8):
        ship = env.state.ships[i]
        assert ship.health == default_ship_config.max_health
        assert ship.power == default_ship_config.max_power
