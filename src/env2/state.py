
from dataclasses import dataclass
import torch

@dataclass
class TensorState:
    """
    State of the environment in tensor format.
    Dimensions:
    - B: Batch size (number of environments)
    - N: Max ships per environment
    - K: Max bullets per ship
    """
    
    # Time
    time: torch.Tensor # (B,)
    
    # Ships (B, N)
    ships_pos: torch.Tensor       # Complex64
    ships_vel: torch.Tensor       # Complex64
    ships_power: torch.Tensor     # Float32
    ships_cooldown: torch.Tensor  # Float32
    ships_team: torch.Tensor      # Int64
    ships_alive: torch.Tensor     # Bool
    ships_health: torch.Tensor    # Float32
    
    # Kinematics (for observation)
    ships_acc: torch.Tensor       # Complex64
    ships_ang_vel: torch.Tensor   # Float32
    ships_attitude: torch.Tensor  # Complex64
    
    # Bullets (B, N, K)
    # N is the source ship index.
    bullets_pos: torch.Tensor     # Complex64
    bullets_vel: torch.Tensor     # Complex64
    bullets_time: torch.Tensor    # Float32
    bullets_team: torch.Tensor    # Int64
    
    # Bullet Ring Buffer Indices (B, N)
    # Tracks the next slot to fire into for each ship.
    bullet_cursor: torch.Tensor   # Int64
    
    def clone(self) -> 'TensorState':
        return TensorState(
            time=self.time.clone(),
            ships_pos=self.ships_pos.clone(),
            ships_vel=self.ships_vel.clone(),
            ships_power=self.ships_power.clone(),
            ships_cooldown=self.ships_cooldown.clone(),
            ships_team=self.ships_team.clone(),
            ships_alive=self.ships_alive.clone(),
            ships_health=self.ships_health.clone(),
            ships_acc=self.ships_acc.clone(),
            ships_ang_vel=self.ships_ang_vel.clone(),
            ships_attitude=self.ships_attitude.clone(),
            bullets_pos=self.bullets_pos.clone(),
            bullets_vel=self.bullets_vel.clone(),
            bullets_time=self.bullets_time.clone(),
            bullets_team=self.bullets_team.clone(),
            bullet_cursor=self.bullet_cursor.clone()
        )
