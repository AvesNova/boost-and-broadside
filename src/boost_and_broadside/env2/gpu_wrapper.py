
import torch
import numpy as np
from typing import Dict, Tuple, Any

from boost_and_broadside.core.constants import (
    StateFeature, 
    NORM_VELOCITY, 
    NORM_ANGULAR_VELOCITY, 
    NORM_HEALTH, 
    NORM_POWER
)

class GPUEnvWrapper:
    """
    Wraps TensorEnv to provide Yemong-compatible observations on GPU.
    
    Responsibilities:
    1. Convert TensorEnv complex state to Real (x,y) stacks.
    2. Normalize raw state features into 'state' tensor.
    3. Track 'prev_action' which is missing in TensorEnv.
    4. Track episode statistics (return, length) on GPU.
    """
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_ships = env.max_ships
        
        # Track Prev Action: (B, N, 3)
        self.prev_action = torch.zeros((self.num_envs, self.max_ships, 3), device=self.device, dtype=torch.long)
        
        # Track Episode Stats
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        
    def reset(self, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        _ = self.env.reset(**kwargs)
        
        # Reset trackers
        self.prev_action.zero_()
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        
        obs = self._make_obs()
        return obs, {}
        
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        action: (B, N, 3) LongTensor
        """
        # Store action as prev_action for NEXT step (but for current step output, we use action passed in?)
        # Wait, obs['prev_action'] should correspond to action taken at t-1.
        # So we update prev_action AFTER creating the next observation?
        # No, obs at t+1 needs action at t.
        # So we update prev_action with `action` BEFORE returning obs.
        
        # 1. Step Env
        # TensorEnv expects action tensor or dict. passing tensor is fine.
        _, reward, terminated, truncated, _ = self.env.step(action)
        
        # reward is (B, N). PPO usually wants (B,) or we train multi-agent.
        # Yemong is multi-agent (N ships). PPO matches N ships.
        # If PPOTrainer treats (B, N) as batch, then reward (B, N) is correct.
        # But done is (B,).
        # We need to broadcast done to (B, N) if we treat ships as independent?
        # NO, reset happens on Env level.
        # Our PPO buffer interaction:
        # buffer.add(..., done, ...)
        # done is (B,).
        # If random ships die, they just get masked?
        # For now, we assume standard PPO env interface.
        
        # 2. Update Stats
        # reward is (B, N). We sum for team reward or mean?
        # Usually we track "Episode Return" as sum of rewards.
        # If reward is per-ship, we sum over ships? Or mean?
        # Let's track MEAN reward per ship as the "Env Score".
        self.episode_returns += reward.mean(dim=1)
        self.episode_lengths += 1
        
        # 3. Handle Done/Reset
        done = torch.logical_or(terminated, truncated)
        info = {}
        
        if done.any():
            # Populate info with final stats for done envs
            done_indices = torch.nonzero(done, as_tuple=True)[0]
            
            final_info = []
            for idx in done_indices:
                final_info.append({
                    "episode": {
                        "r": self.episode_returns[idx].item(),
                        "l": self.episode_lengths[idx].item()
                    }
                })
            info["final_info"] = final_info
            
            # Reset trackers for done envs
            self.episode_returns[done_indices] = 0
            self.episode_lengths[done_indices] = 0
            self.prev_action[done_indices] = 0
            
        # 4. Update Prev Action
        # We update `prev_action` for the NEXT observation.
        # The `action` we just took becomes `prev_action` for the new state.
        # Note: TensorEnv auto-resets. If env reset, `prev_action` should be 0.
        # We already zeroed it above for done indices.
        # For non-done indices, we copy action.
        
        not_done = ~done
        if not_done.any():
            # We copy action to prev_action for active envs
            # Note: action might be on different device if not careful, but usually same.
            # Assuming action is (B, N, 3).
            self.prev_action[not_done] = action[not_done].long()
            
        # 5. Make Obs
        obs = self._make_obs()
        
        return obs, reward, terminated, truncated, info
        
    def _make_obs(self) -> Dict[str, torch.Tensor]:
        state = self.env.state
        
        # 1. State Feature Tensor (Normalized)
        # Dimensions: (B, N, 5)
        
        # Velocity (Complex -> Re/Im -> Norm)
        vx = state.ship_vel.real / NORM_VELOCITY
        vy = state.ship_vel.imag / NORM_VELOCITY
        
        health = state.ship_health / NORM_HEALTH
        power = state.ship_power / NORM_POWER
        ang_vel = state.ship_ang_vel / NORM_ANGULAR_VELOCITY
        
        # Stack: [Health, Power, Vx, Vy, AngVel] matches StateFeature enum order
        state_tensor = torch.stack([health, power, vx, vy, ang_vel], dim=-1)
        
        # 2. Pos/Vel/Att for Relational Encoder
        # Pos (B, N, 2)
        pos_stack = torch.stack([state.ship_pos.real, state.ship_pos.imag], dim=-1)
        
        # Vel (B, N, 2) - unnormalized? Relational Encoder usually takes raw or norm? 
        # Usually raw and it learns, or encoder handles it.
        # But previous `adapter` used raw.
        vel_stack = torch.stack([state.ship_vel.real, state.ship_vel.imag], dim=-1)
        
        # Att (B, N, 2)
        att_stack = torch.stack([state.ship_attitude.real, state.ship_attitude.imag], dim=-1)
        
        return {
            "state": state_tensor,
            "pos": pos_stack,
            "vel": vel_stack,
            "att": att_stack,
            "prev_action": self.prev_action.clone(),
            "alive": state.ship_alive,
            "team_ids": state.ship_team_id
        }
