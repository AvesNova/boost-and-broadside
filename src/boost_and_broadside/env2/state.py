
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from boost_and_broadside.core.config import ShipConfig

@dataclass
class TensorState:
    """
    State of the environment in tensor format.
    
    Attributes:
        step_count: (B,) Current step count for each environment.
        ship_pos: (B, N) Complex position of ships.
        ship_vel: (B, N) Complex velocity of ships.
        ship_attitude: (B, N) Complex unit vector representing ship orientation.
        ship_ang_vel: (B, N) Angular velocity (radians/sec).
        ship_health: (B, N) Current health.
        ship_power: (B, N) Current power level.
        ship_cooldown: (B, N) Time remaining until next shot.
        ship_team_id: (B, N) Team identifier.
        ship_alive: (B, N) Boolean indicating if ship is alive.
        ship_is_shooting: (B, N) Boolean indicating if ship is currently firing.
        bullet_pos: (B, N, K) Complex position of bullets.
        bullet_vel: (B, N, K) Complex velocity of bullets.
        bullet_time: (B, N, K) Time remaining for bullets.
        bullet_active: (B, N, K) Boolean indicating if bullet is active.
        bullet_cursor: (B, N) Index of the next bullet slot to use (circular buffer).
    """
    step_count: torch.Tensor            # (B,) int32
    
    # Ship State (B, N)
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
    
    # Bullet State (B, N, K) - Ring buffer per ship
    # K should be >= bullet_lifetime / firing_cooldown
    bullet_pos: torch.Tensor            # (B, N, K) complex64
    bullet_vel: torch.Tensor            # (B, N, K) complex64
    bullet_time: torch.Tensor           # (B, N, K) float32 (seconds remaining)
    bullet_active: torch.Tensor         # (B, N, K) bool
    
    # Bullet Manager State (B, N)
    bullet_cursor: torch.Tensor         # (B, N) int64 - current write index in ring buffer (0 to K-1)

    def clone(self) -> 'TensorState':
        """Creates a deep copy of the state."""
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
        """Returns the batch size (number of environments)."""
        return self.ship_pos.shape[0]

    @property
    def max_ships(self) -> int:
        """Returns the maximum number of ships per environment."""
        return self.ship_pos.shape[1]
        
    @property
    def max_bullets(self) -> int:
        """Returns the maximum buffer size for bullets per ship."""
        return self.bullet_pos.shape[2]
        
    @property
    def device(self) -> torch.device:
        """Returns the device governing the state tensors."""
        return self.ship_pos.device
