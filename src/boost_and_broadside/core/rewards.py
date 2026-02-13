import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
import logging

from boost_and_broadside.env2.state import TensorState
from boost_and_broadside.core.constants import RewardConstants

log = logging.getLogger(__name__)

class RewardComponent(ABC):
    """
    Abstract base class for a modular reward component.
    Calculates a specific type of reward (e.g. Damage, Death, Outcome).
    """
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute(
        self, 
        prev_state: TensorState, 
        actions: torch.Tensor, 
        next_state: TensorState, 
        dones: torch.Tensor,
        is_terminal: bool = False
    ) -> torch.Tensor:
        """
        Compute the reward for this component.

        Args:
            prev_state: State at t.
            actions: Actions taken at t.
            next_state: State at t+1.
            dones: Boolean tensor indicating if episode ended at t+1.
            is_terminal: Whether next_state is terminal (game over).

        Returns:
            Reward tensor of shape (Batch, NumShips).
        """
        pass

class DamageReward(RewardComponent):
    """
    Aligned Local Reward based on Health Delta.
    
    If Ally (Team 0) takes damage -> Negative Reward (Pain).
    If Enemy (Team 1) takes damage -> Positive Reward (Pleasure).
    """
    def __init__(self, name: str = "damage", weight: float = 1.0, pain_weight: float = 1.0, blood_weight: float = 1.0):
        super().__init__(name, weight)
        self.pain_weight = pain_weight
        self.blood_weight = blood_weight

    def compute(self, prev_state: TensorState, actions: torch.Tensor, next_state: TensorState, dones: torch.Tensor, is_terminal: bool = False) -> torch.Tensor:
        # Calculate Delta Health (Positive if damage taken)
        # Handle resets: If done, next_state health might be reset to full.
        # But usually 'next_state' passed here is strictly t+1 *before* reset in some contexts,
        # or we rely on 'dones' to mask.
        # However, for simple step-based calculation:
        
        delta_health = prev_state.ship_health - next_state.ship_health
        
        # We only care about damage (positive delta). Healing is not possible currently unless reset.
        # If reset occurred, delta might be negative (health went up). Clamp to 0?
        # Actually, if done is True, the transition is valid until terminal.
        # If next_state is the FRESH state after reset, delta is meaningless. 
        # But typically PPO/Env returns (next_obs, reward, done). 'next_obs' is new state.
        # If done, next_obs is reset state.
        # So we must handle 'dones'. If done, we assume the specific transition logic handled calculating final damage elsewhere
        # OR we assume next_state is the terminal state BEFORE reset. 
        
        # IN OUR ENV: env.step() updates state, THEN checks collisions (damage), THEN checks game over.
        # So 'next_state' inside env.step (before return) has the damage applied.
        # But if it resets, 'state' is overwritten.
        # We need to compute rewards *before* the auto-reset in Env.
        
        # Assumption: This function is called with the state *after* physics/damage but *before* overwrite-reset.
        
        damage_taken = torch.maximum(delta_health, torch.tensor(0.0, device=prev_state.device))
        
        # Alignment
        # Team 0 = Ally
        # Team 1 = Enemy
        
        # We define "Aligned Reward" from perspective of Team 0.
        # This might need to be parameterized if we train self-play. 
        # But usually we train "Team 0" policy.
        
        is_ally = (next_state.ship_team_id == 0)
        is_enemy = (next_state.ship_team_id == 1)
        
        rewards = torch.zeros_like(damage_taken)
        
        # Pain (Ally Damage) -> Negative
        rewards[is_ally] = -1.0 * damage_taken[is_ally] * self.pain_weight
        
        # Pleasure (Enemy Damage) -> Positive
        rewards[is_enemy] = 1.0 * damage_taken[is_enemy] * self.blood_weight
        
        return rewards * self.weight

class DeathReward(RewardComponent):
    """
    Reward for killing/dying.
    """
    def __init__(self, name: str = "death", weight: float = 1.0, die_penalty: float = 5.0, kill_reward: float = 5.0):
        super().__init__(name, weight)
        self.die_penalty = die_penalty
        self.kill_reward = kill_reward

    def compute(self, prev_state: TensorState, actions: torch.Tensor, next_state: TensorState, dones: torch.Tensor, is_terminal: bool = False) -> torch.Tensor:
        # Alive transition 1 -> 0
        just_died = prev_state.ship_alive & (~next_state.ship_alive)
        
        is_ally = (next_state.ship_team_id == 0)
        is_enemy = (next_state.ship_team_id == 1)
        
        rewards = torch.zeros_like(next_state.ship_health)
        
        # Ally Died -> Penalty
        rewards[just_died & is_ally] = -1.0 * self.die_penalty
        
        # Enemy Died -> Reward
        rewards[just_died & is_enemy] = 1.0 * self.kill_reward
        
        return rewards * self.weight

class VictoryReward(RewardComponent):
    """
    Global reward for game outcome. Added to ALL ships.
    """
    def __init__(self, name: str = "victory", weight: float = 1.0, win_reward: float = 100.0, lose_penalty: float = 100.0, draw_reward: float = 0.0):
        super().__init__(name, weight)
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty
        self.draw_reward = draw_reward

    def compute(self, prev_state: TensorState, actions: torch.Tensor, next_state: TensorState, dones: torch.Tensor, is_terminal: bool = False) -> torch.Tensor:
        rewards = torch.zeros_like(next_state.ship_health)
        
        if not is_terminal:
             return rewards

        # Determine winner
        team0_alive = (next_state.ship_team_id == 0) & next_state.ship_alive
        team1_alive = (next_state.ship_team_id == 1) & next_state.ship_alive
        
        t0_count = team0_alive.sum(dim=1)
        t1_count = team1_alive.sum(dim=1)
        
        # Vectorized outcome
        # Win: T0 > 0 and T1 == 0
        win_mask = (t0_count > 0) & (t1_count == 0)
        # Lose: T0 == 0 and T1 > 0
        lose_mask = (t0_count == 0) & (t1_count > 0)
        # Draw: Both 0 (simultaneous kill)
        draw_mask = (t0_count == 0) & (t1_count == 0)
        
        # Broadcast to (B, N)
        B, N = rewards.shape
        win_exp = win_mask.unsqueeze(1).expand(B, N)
        lose_exp = lose_mask.unsqueeze(1).expand(B, N)
        draw_exp = draw_mask.unsqueeze(1).expand(B, N)
        
        rewards[win_exp] = self.win_reward
        rewards[lose_exp] = -self.lose_penalty
        rewards[draw_exp] = self.draw_reward
        
        return rewards * self.weight

class RewardRegistry:
    """
    Manages active reward components.
    """
    def __init__(self, config: Optional[DictConfig] = None):
        self.components: List[RewardComponent] = []
        
        if config is None:
            # Default Configuration (Legacy behavior approximation or sensible default)
            self.components.append(DamageReward())
            self.components.append(DeathReward())
            self.components.append(VictoryReward())
        else:
            # Load from config
            for name, cfg in config.items():
                if name == "damage":
                    self.components.append(DamageReward(weight=cfg.get("weight", 1.0), pain_weight=cfg.get("pain_weight", 1.0), blood_weight=cfg.get("blood_weight", 1.0)))
                elif name == "death":
                    self.components.append(DeathReward(weight=cfg.get("weight", 1.0), die_penalty=cfg.get("die_penalty", 5.0), kill_reward=cfg.get("kill_reward", 5.0)))
                elif name == "victory":
                    self.components.append(VictoryReward(weight=cfg.get("weight", 1.0), win_reward=cfg.get("win_reward", 100.0), lose_penalty=cfg.get("lose_penalty", 100.0)))
                    
    def compute_all(self, prev_state: TensorState, actions: torch.Tensor, next_state: TensorState, dones: torch.Tensor, is_terminal: bool=False) -> Dict[str, torch.Tensor]:
        """
        Compute all rewards.
        
        Returns:
            Dictionary mapping reward name to reward tensor (Batch, NumShips).
        """
        results = {}
        for comp in self.components:
            results[comp.name] = comp.compute(prev_state, actions, next_state, dones, is_terminal)
            
        return results
