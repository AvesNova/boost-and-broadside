from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class StochasticAgentConfig:
    """
    Configuration for the Stochastic Scripted Agent's brain.
    
    Attributes:
        flat_action_sampling: Whether to sample from the joint 42-dim distribution 
                              or 3 independent marginal distributions (3, 7, 2).
        
        # Power Ramps (distance, probability)
        boost_speed_ramp: Range of speeds [low, high] where boost probability goes from 1.0 to 0.0.
        close_range_ramp: Range of distances [low, high] where reverse probability goes from 1.0 to 0.0.
        
        # Turn Ramps (angle, probability)
        turn_angle_ramp: Range of absolute angles [low, high] where turning probability goes from 0.0 to 1.0.
        sharp_turn_angle_ramp: Range of absolute angles [low, high] where sharp turn probability goes from 0.0 to 1.0.
        
        # Shoot Ramps
        shoot_angle_multiplier_ramp: Range of (abs_angle / shoot_threshold) [low, high] where shoot probability goes from 1.0 to 0.0.
        shoot_distance_ramp: Range of distance fractions (dist / max_range) [low, high] where shoot probability goes from 1.0 to 0.0.
        
        max_shooting_range: Maximum range to attempt shooting.
        radius_multiplier: Multiplier for target size when determining shot alignment.
        target_radius: Assumed radius of the target for aiming.
    """
    flat_action_sampling: bool = False
    
    # Power Ramps
    # If speed is below 20, 100% boost reason. If above 40, 0% boost reason.
    boost_speed_ramp: Tuple[float, float] = (20.0, 40.0)
    
    # If distance is below 30, 100% reverse. If above 50, 0% reverse.
    close_range_ramp: Tuple[float, float] = (30.0, 50.0)
    
    # Turn Ramps
    # If angle offset > 5 deg (0.087 rad), 100% turn. If < 2 deg, 0%.
    turn_angle_ramp: Tuple[float, float] = (np.deg2rad(2.0), np.deg2rad(5.0))
    
    # If angle offset > 20 deg, 100% sharp turn. If < 10 deg, 0%.
    sharp_turn_angle_ramp: Tuple[float, float] = (np.deg2rad(10.0), np.deg2rad(20.0))
    
    # Shoot Ramps
    # shoot_threshold is calculated dynamically. If abs_angle / threshold < 0.8, 100% shoot. If > 1.2, 0%.
    shoot_angle_multiplier_ramp: Tuple[float, float] = (0.8, 1.2)
    
    # Range limit: dist < 0.9 * max_range -> 100% shoot. dist > 1.1 * max_range -> 0%.
    shoot_distance_ramp: Tuple[float, float] = (0.9, 1.1)
    
    # Fixed parameters from old scripted
    max_shooting_range: float = 600.0
    radius_multiplier: float = 1.0
    target_radius: float = 20.0
