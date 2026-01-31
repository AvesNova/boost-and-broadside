
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym

from .state import TensorState
from core.config import ShipConfig
from .physics import update_ships, update_bullets, resolve_collisions

class TensorEnv:
    """
    Vectorized Environment for GPU execution.
    
    Manages a batch of parallel environments.
    
    Attributes:
        num_envs: Number of parallel environments.
        config: Environment configuration.
        device: Torch device.
        max_ships: Max ships per environment.
        max_bullets: Max active bullets per ship.
        state: The shared tensor state.
    """
    def __init__(
        self, 
        num_envs: int, 
        config: ShipConfig, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_ships: int = 8,
        max_bullets: int = 20
    ):
        """
        Initializes the TensorEnv.

        Args:
            num_envs: Number of environments to simulate in parallel.
            config: Physics configuration.
            device: Computation device.
            max_ships: Maximum number of ships per matchup.
            max_bullets: Maximum bullet buffer size per ship.
        """
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.max_ships = max_ships
        self.max_bullets = max_bullets
        
        self.state: Optional[TensorState] = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Resets all environments.
        
        Args:
            seed: RNG seed.
            options: Configuration options (e.g. 'team_sizes').
            
        Returns:
            The initial observation dictionary.
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        self._initialize_state()
        
        all_envs = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._reset_envs(all_envs, options)
        
        return self._get_obs()
        
    def _initialize_state(self):
        """Allocates memory for the environment state."""
        self.state = TensorState(
            step_count=torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device),
            
            ship_pos=torch.zeros((self.num_envs, self.max_ships), dtype=torch.complex64, device=self.device),
            ship_vel=torch.zeros((self.num_envs, self.max_ships), dtype=torch.complex64, device=self.device),
            ship_attitude=torch.zeros((self.num_envs, self.max_ships), dtype=torch.complex64, device=self.device),
            ship_ang_vel=torch.zeros((self.num_envs, self.max_ships), dtype=torch.float32, device=self.device),
            ship_health=torch.zeros((self.num_envs, self.max_ships), dtype=torch.float32, device=self.device),
            ship_power=torch.zeros((self.num_envs, self.max_ships), dtype=torch.float32, device=self.device),
            ship_cooldown=torch.zeros((self.num_envs, self.max_ships), dtype=torch.float32, device=self.device),
            ship_team_id=torch.zeros((self.num_envs, self.max_ships), dtype=torch.int32, device=self.device),
            ship_alive=torch.zeros((self.num_envs, self.max_ships), dtype=torch.bool, device=self.device),
            ship_is_shooting=torch.zeros((self.num_envs, self.max_ships), dtype=torch.bool, device=self.device),
            
            bullet_pos=torch.zeros((self.num_envs, self.max_ships, self.max_bullets), dtype=torch.complex64, device=self.device),
            bullet_vel=torch.zeros((self.num_envs, self.max_ships, self.max_bullets), dtype=torch.complex64, device=self.device),
            bullet_time=torch.zeros((self.num_envs, self.max_ships, self.max_bullets), dtype=torch.float32, device=self.device),
            bullet_active=torch.zeros((self.num_envs, self.max_ships, self.max_bullets), dtype=torch.bool, device=self.device),
            bullet_cursor=torch.zeros((self.num_envs, self.max_ships), dtype=torch.long, device=self.device)
        )

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Steps all environments forward by one physics tick.
        
        Args:
            actions: (Batch, NumShips, 3) Int tensor [Power, Turn, Shoot] or Legacy dict.
            
        Returns:
            A tuple containing:
            - obs: New observation dictionary.
            - rewards: (Batch, NumShips) Reward tensor.
            - dones: (Batch,) Boolean tensor indicating episode completion.
            - truncated: (Batch,) Boolean tensor (always false currently).
            - info: Empty info dictionary.
        """
        if isinstance(actions, dict) and "action" in actions:
             # Support for legacy dict-based action input
             actions = actions["action"]

        # Physics Updates
        self.state = update_ships(self.state, actions, self.config)
        self.state = update_bullets(self.state, self.config)
        
        # Collision Resolution
        self.state, rewards, dones = resolve_collisions(self.state, self.config)
        
        self.state.step_count += 1
        
        # Truncation (not implemented)
        truncated = torch.zeros_like(dones)
        
        # Auto-Reset Logic
        if dones.any():
            self._reset_envs(dones)
            
        return self._get_obs(), rewards, dones, truncated, {}

    def _reset_envs(self, env_mask: torch.Tensor, options: Optional[Dict[str, Any]] = None):
        """
        Resets specific environments identified by the mask.
        
        Args:
            env_mask: (Batch,) boolean mask where True indicates environment needs reset.
            options: Optional reset configuration options.
        """
        num_resets = int(env_mask.sum().item())
        if num_resets == 0:
            return
            
        world_width, world_height = self.config.world_size
        
        # Determine team sizes (support options override if provided, else equal split)
        n_team0 = self.max_ships // 2
        n_team1 = self.max_ships - n_team0
        
        if options and "team_sizes" in options:
             n_team0, n_team1 = options["team_sizes"]
        
        reset_indices = torch.nonzero(env_mask, as_tuple=True)[0]
        
        self.state.step_count[env_mask] = 0
        
        # --- Reset Ships ---
        
        # Random Positions
        rand_x = torch.rand((num_resets, self.max_ships), device=self.device) * world_width
        rand_y = torch.rand((num_resets, self.max_ships), device=self.device) * world_height
        pos = torch.complex(rand_x, rand_y)
        
        self.state.ship_pos[reset_indices] = pos
        self.state.ship_pos[reset_indices] = pos
        
        # Random Attitude
        rand_angle = torch.rand((num_resets, self.max_ships), device=self.device) * 2 * np.pi
        att = torch.polar(torch.ones_like(rand_angle), rand_angle)
        self.state.ship_attitude[reset_indices] = att
        
        # Calculate Random Speed
        if self.config.random_speed:
            rand_speed = torch.rand((num_resets, self.max_ships), device=self.device)
            # Scale to [min, max]
            speed = self.config.min_speed + rand_speed * (self.config.max_speed - self.config.min_speed)
        else:
            speed = torch.full((num_resets, self.max_ships), self.config.default_speed, device=self.device)
            
        self.state.ship_vel[reset_indices] = speed * att
        
        # Reset Status
        self.state.ship_health[reset_indices] = self.config.max_health
        self.state.ship_power[reset_indices] = self.config.max_power
        self.state.ship_cooldown[reset_indices] = 0.0
        self.state.ship_ang_vel[reset_indices] = 0.0
        
        # Defaults
        new_team_ids = torch.zeros((num_resets, self.max_ships), dtype=torch.int32, device=self.device)
        new_alive = torch.zeros((num_resets, self.max_ships), dtype=torch.bool, device=self.device)
        
        # Team Assignment: First n0 are Team 0, next n1 are Team 1
        new_team_ids[:, n_team0:n_team0+n_team1] = 1
        new_alive[:, :n_team0+n_team1] = True
        
        self.state.ship_team_id[reset_indices] = new_team_ids
        self.state.ship_alive[reset_indices] = new_alive
        
        # --- Reset Bullets ---
        self.state.bullet_active[reset_indices] = False
        self.state.bullet_time[reset_indices] = 0.0
        self.state.bullet_cursor[reset_indices] = 0

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """
        Generates the state dictionary.
        
        Returns:
            Dictionary matching legacy environment observation format.
        """
        batch_size, num_ships = self.state.ship_pos.shape
        ship_ids = torch.arange(num_ships, device=self.device).expand(batch_size, num_ships)
        
        return {
            "ship_id": ship_ids,
            "team_id": self.state.ship_team_id,
            "alive": self.state.ship_alive,
            "health": self.state.ship_health,
            "power": self.state.ship_power,
            "position": self.state.ship_pos,
            "velocity": self.state.ship_vel,
            "attitude": self.state.ship_attitude,
            "cooldown": self.state.ship_cooldown,
            "ang_vel": self.state.ship_ang_vel,
            "is_shooting": self.state.ship_is_shooting
        }
