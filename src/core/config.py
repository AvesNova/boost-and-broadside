from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class ShipConfig:
    """
    Configuration for ship physics and capabilities.
    
    Attributes:
        collision_radius: Radius for collision detection.
        max_health: Maximum ship health.
        max_power: Maximum ship power/energy.
        base_thrust: Force applied when coasting/moving forward.
        boost_thrust: Force applied when boosting.
        reverse_thrust: Force applied when reversing.
        base_power_gain: Power regeneration rate.
        boost_power_gain: Power consumption rate when boosting.
        reverse_power_gain: Power regeneration rate when reversing.
        no_turn_drag_coeff: Drag when moving straight.
        normal_turn_drag_coeff: Drag when turning normally.
        normal_turn_lift_coeff: Lift (turning force) when turning normally.
        sharp_turn_drag_coeff: Drag when turning sharply.
        sharp_turn_lift_coeff: Lift when turning sharply.
        normal_turn_angle: Maximum wheel angle for normal turn.
        sharp_turn_angle: Maximum wheel angle for sharp turn.
        bullet_speed: Speed of fired bullets.
        bullet_energy_cost: Power cost to fire a bullet.
        bullet_damage: Damage dealt by a single bullet.
        bullet_lifetime: Duration a bullet remains active in seconds.
        bullet_spread: Angular spread of bullets.
        firing_cooldown: Time between shots.
        world_size: Dimensions of the toroidal world.
        dt: Physics simulation timestep.
    """
    # Physical Properties
    collision_radius: float = 10.0
    max_health: float = 100.0
    max_power: float = 100.0

    # Initialization
    random_speed: bool = False
    min_speed: float = 1.0
    max_speed: float = 180.0
    default_speed: float = 100.0

    # Thrust Dynamics
    base_thrust: float = 8.0
    boost_thrust: float = 80.0
    reverse_thrust: float = -10.0
    
    # Power Dynamics
    base_power_gain: float = 10.0
    boost_power_gain: float = -40.0
    reverse_power_gain: float = 20.0

    # Drag and Lift Coefficients
    no_turn_drag_coeff: float = 8e-4
    normal_turn_drag_coeff: float = 1.2e-3
    normal_turn_lift_coeff: float = 15e-3
    sharp_turn_drag_coeff: float = 5.0e-3
    sharp_turn_lift_coeff: float = 27e-3
    
    # Maneuverability Angles (radians)
    normal_turn_angle: float = np.deg2rad(5.0)
    sharp_turn_angle: float = np.deg2rad(15.0)

    # Bullet Properties
    bullet_speed: float = 500.0
    bullet_energy_cost: float = 3.0
    bullet_damage: float = 10.0
    bullet_lifetime: float = 1.0 # seconds
    bullet_spread: float = 12.0 # degrees
    firing_cooldown: float = 0.1 # seconds
    
    # World Settings
    world_size: Tuple[float, float] = (1024.0, 1024.0)
    
    # Simulation Settings
    dt: float = 1.0 / 60.0
