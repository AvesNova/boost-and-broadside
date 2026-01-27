
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class ShipConfig:
    """
    Configuration for ship physics and capabilities.
    Values are floats, matching `src/env/ship.py`.
    """
    # Physical
    collision_radius: float = 10.0
    max_health: float = 100.0
    max_power: float = 100.0

    # Thrust
    base_thrust: float = 8.0
    boost_thrust: float = 80.0
    reverse_thrust: float = -10.0
    
    # Power
    base_power_gain: float = 10.0
    boost_power_gain: float = -40.0
    reverse_power_gain: float = 20.0

    # Drag/Lift
    no_turn_drag_coeff: float = 8e-4
    normal_turn_drag_coeff: float = 1.2e-3
    normal_turn_lift_coeff: float = 15e-3
    sharp_turn_drag_coeff: float = 5.0e-3
    sharp_turn_lift_coeff: float = 27e-3
    
    # Angles (radians)
    normal_turn_angle: float = np.deg2rad(5.0)
    sharp_turn_angle: float = np.deg2rad(15.0)

    # Bullets
    bullet_speed: float = 500.0
    bullet_energy_cost: float = 3.0
    bullet_damage: float = 10.0
    bullet_lifetime: float = 1.0 # seconds
    bullet_spread: float = 12.0 # degrees
    firing_cooldown: float = 0.1 # seconds
    
    # World
    world_size: Tuple[float, float] = (1024.0, 1024.0)
    
    # Simulation
    dt: float = 1.0 / 60.0

@dataclass
class TensorState:
    """
    State of the environment in tensor format.
    Batch dimension B corresponds to independent environments.
    N is the max number of ships per environment.
    K is the max number of active bullets per ship.
    """
    step_count: torch.Tensor            # (B,) int32
    
    # Ships (B, N)
    ship_pos: torch.Tensor              # (B, N) complex64
    ship_vel: torch.Tensor              # (B, N) complex64
    ship_attitude: torch.Tensor         # (B, N) complex64 (unit vector)
    ship_ang_vel: torch.Tensor          # (B, N) float32 (radians/sec)
    ship_health: torch.Tensor           # (B, N) float32
    ship_power: torch.Tensor            # (B, N) float32
    ship_cooldown: torch.Tensor         # (B, N) float32 (seconds remaining)
    ship_team_id: torch.Tensor          # (B, N) int32
    ship_alive: torch.Tensor            # (B, N) bool
    ship_is_shooting: torch.Tensor      # (B, N) bool
    
    # Bullets (B, N, K) - Ring buffer per ship
    # K should be >= bullet_lifetime / firing_cooldown
    bullet_pos: torch.Tensor            # (B, N, K) complex64
    bullet_vel: torch.Tensor            # (B, N, K) complex64
    bullet_time: torch.Tensor           # (B, N, K) float32 (seconds remaining)
    bullet_active: torch.Tensor         # (B, N, K) bool
    
    # Bullet Manager State (B, N)
    bullet_cursor: torch.Tensor         # (B, N) int64 - current write index in ring buffer (0 to K-1)

    def clone(self) -> 'TensorState':
        return TensorState(
            step_count=self.step_count.clone(),
            ship_pos=self.ship_pos.clone(),
            ship_vel=self.ship_vel.clone(),
            ship_attitude=self.ship_attitude.clone(),
            ship_ang_vel=self.ship_ang_vel.clone(),
            ship_health=self.ship_health.clone(),
            ship_power=self.ship_power.clone(),
            ship_cooldown=self.ship_cooldown.clone(),
            ship_team_id=self.ship_team_id.clone(),
            ship_alive=self.ship_alive.clone(),
            ship_is_shooting=self.ship_is_shooting.clone(),
            bullet_pos=self.bullet_pos.clone(),
            bullet_vel=self.bullet_vel.clone(),
            bullet_time=self.bullet_time.clone(),
            bullet_active=self.bullet_active.clone(),
            bullet_cursor=self.bullet_cursor.clone()
        )
    
    @property
    def num_envs(self) -> int:
        return self.ship_pos.shape[0]

    @property
    def max_ships(self) -> int:
        return self.ship_pos.shape[1]
        
    @property
    def max_bullets(self) -> int:
        return self.bullet_pos.shape[2]
        
    @property
    def device(self) -> torch.device:
        return self.ship_pos.device
