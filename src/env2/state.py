
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

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
