
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym

from .state import TensorState, ShipConfig
from .physics import update_ships, update_bullets, resolve_collisions

class TensorEnv:
    """
    Vectorized Environment for GPU execution.
    Manages B parallel environments.
    """
    def __init__(
        self, 
        num_envs: int, 
        config: ShipConfig, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_ships: int = 8,
        max_bullets: int = 20 # Safer buffer size
    ):
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.max_ships = max_ships
        self.max_bullets = max_bullets
        
        # Placeholders
        self.state: Optional[TensorState] = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize empty state tensors
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
        
        # Reset all environments
        all_envs = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._reset_envs(all_envs, options)
        
        return self._get_obs()
        
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments.
        actions: (B, N, 3) Int tensor [Power, Turn, Shoot]
        """
        # Physics Steps
        self.state = update_ships(self.state, actions, self.config)
        self.state = update_bullets(self.state, self.config)
        
        # Collisions
        self.state, rewards, dones = resolve_collisions(self.state, self.config)
        
        self.state.step_count += 1
        
        # Truncation (Time Limit)
        # Assuming max steps? Env wrapper usually handles this.
        # But we can return truncation info.
        truncated = torch.zeros_like(dones)
        
        # Auto-Reset Logic
        if dones.any():
            self._reset_envs(dones)
            
        return self._get_obs(), rewards, dones, truncated, {}

    def _reset_envs(self, env_mask: torch.Tensor, options: Optional[Dict[str, Any]] = None):
        """
        Reset specific environments (where env_mask is True).
        """
        num_resets = int(env_mask.sum().item())
        if num_resets == 0:
            return
            
        w, h = self.config.world_size
        
        # Determine team sizes (support options override if provided, else equal split)
        # If options provided during step-reset, we use them? 
        # Usually options are passed only in reset().
        # We should store options or defaults.
        # Defaults: Equal teams.
        
        n_team0 = self.max_ships // 2
        n_team1 = self.max_ships - n_team0
        
        # If options provided in init or previous reset, use those?
        # For simplicity, use 50/50 split of max_ships unless options passed here.
        if options and "team_sizes" in options:
             n_team0, n_team1 = options["team_sizes"]
        
        # Indices of resetting envs
        reset_indices = torch.nonzero(env_mask, as_tuple=True)[0]
        
        # Reset Step Count
        self.state.step_count[env_mask] = 0
        
        # Reset Ships
        # Generate random positions
        # Uniform in world
        rand_x = torch.rand((num_resets, self.max_ships), device=self.device) * w
        rand_y = torch.rand((num_resets, self.max_ships), device=self.device) * h
        pos = torch.complex(rand_x, rand_y)
        
        self.state.ship_pos[reset_indices] = pos
        self.state.ship_vel[reset_indices] = 0.0 + 0j
        
        # Random Attitude
        rand_angle = torch.rand((num_resets, self.max_ships), device=self.device) * 2 * np.pi
        att = torch.polar(torch.ones_like(rand_angle), rand_angle)
        self.state.ship_attitude[reset_indices] = att
        
        # Reset Status
        self.state.ship_health[reset_indices] = self.config.max_health
        self.state.ship_power[reset_indices] = self.config.max_power
        self.state.ship_cooldown[reset_indices] = 0.0
        self.state.ship_ang_vel[reset_indices] = 0.0
        
        # Setup Teams
        # 0..n0-1 -> Team 0
        # n0..n0+n1-1 -> Team 1
        # Rest -> Dead (masked)
        
        # Team IDs
        new_team_ids = torch.zeros((num_resets, self.max_ships), dtype=torch.int32, device=self.device)
        new_team_ids[:, n_team0:n_team0+n_team1] = 1
        self.state.ship_team_id[reset_indices] = new_team_ids
        
        # Alive Status
        new_alive = torch.zeros((num_resets, self.max_ships), dtype=torch.bool, device=self.device)
        new_alive[:, :n_team0+n_team1] = True
        self.state.ship_alive[reset_indices] = new_alive
        
        # Reset Bullets
        self.state.bullet_active[reset_indices] = False
        self.state.bullet_time[reset_indices] = 0.0
        self.state.bullet_cursor[reset_indices] = 0

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """
        Return state dictionary matching `src/env/ship.py` keys (but batched).
        KEYS: ship_id, team_id, alive, health, power, position, velocity, attitude
        """
        # Create ship_ids tensor (B, N)
        # In vectorized env, ship_id can be (BatchIdx * MaxShips + ShipIdx) or just index 0..N-1
        # Original env uses simple integer IDs.
        # We'll use 0..N-1 for each env.
        B, N = self.state.ship_pos.shape
        ship_ids = torch.arange(N, device=self.device).expand(B, N)
        
        return {
            "ship_id": ship_ids,
            "team_id": self.state.ship_team_id,
            "alive": self.state.ship_alive,
            "health": self.state.ship_health,
            "power": self.state.ship_power,
            "position": self.state.ship_pos,
            "velocity": self.state.ship_vel,
            "attitude": self.state.ship_attitude,
            # Extra fields from v2 logic or needed by agents
            "cooldown": self.state.ship_cooldown,
            "ang_vel": self.state.ship_ang_vel,
            "is_shooting": self.state.ship_is_shooting
        }

