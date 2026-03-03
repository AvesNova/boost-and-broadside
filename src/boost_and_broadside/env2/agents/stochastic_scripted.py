import torch
import numpy as np
from boost_and_broadside.env2.state import TensorState
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.core.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env2.agents.stochastic_config import StochasticAgentConfig

class StochasticScriptedAgent:
    """
    A stochastic scripted agent that uses configurable probability ramps
    instead of hard boundaries.
    """
    def __init__(self, ship_config: ShipConfig, agent_config: StochasticAgentConfig):
        self.ship_config = ship_config
        self.config = agent_config
        
    def _linear_ramp(self, x: torch.Tensor, low: float, high: float, invert: bool = False) -> torch.Tensor:
        """
        Maps x through a piecewise linear ramp [low, high] -> [0, 1].
        If invert is True, maps [low, high] -> [1, 0].
        Values outside [low, high] are clamped to 0 or 1.
        """
        # Avoid division by zero
        if high == low:
            # Step function
            prob = torch.where(x >= low, 1.0, 0.0)
            if invert:
                return 1.0 - prob
            return prob
            
        prob = (x - low) / (high - low)
        prob = torch.clamp(prob, 0.0, 1.0)
        
        if invert:
            return 1.0 - prob
        return prob
        
    def _prob_or(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent OR logic: P(A or B) = P(A) + P(B) - P(A)*P(B)"""
        return p_a + p_b - (p_a * p_b)
        
    def _prob_and(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent AND logic: P(A and B) = P(A) * P(B)"""
        return p_a * p_b

    def _select_targets(self, state: TensorState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same logic as VectorScriptedAgent._select_targets"""
        batch_size, num_ships = state.ship_pos.shape
        world_width, world_height = self.ship_config.world_size
        device = state.device
        
        pos_targets = state.ship_pos.unsqueeze(1)
        pos_sources = state.ship_pos.unsqueeze(2)
        
        diff = pos_targets - pos_sources
        diff.real = (diff.real + world_width / 2) % world_width - world_width / 2
        diff.imag = (diff.imag + world_height / 2) % world_height - world_height / 2
        
        dist = torch.abs(diff)
        
        team_src = state.ship_team_id.unsqueeze(2)
        team_tgt = state.ship_team_id.unsqueeze(1)
        enemy_mask = team_src != team_tgt
        alive_tgt = state.ship_alive.unsqueeze(1)
        valid_tgt = enemy_mask & alive_tgt
        
        dist_masked = torch.where(valid_tgt, dist, torch.tensor(float('inf'), device=device))
        closest_dist, target_idx = torch.min(dist_masked, dim=2)
        has_target = closest_dist < float('inf')
        
        return closest_dist, target_idx, has_target

    def _predict_interception(self, state: TensorState, target_idx: torch.Tensor, closest_dist: torch.Tensor) -> torch.Tensor:
        """Same logic as VectorScriptedAgent._predict_interception"""
        world_width, world_height = self.ship_config.world_size
        target_pos = torch.gather(state.ship_pos, 1, target_idx)
        target_vel = torch.gather(state.ship_vel, 1, target_idx)
        
        t_intercept = closest_dist / self.ship_config.bullet_speed
        pred_pos = target_pos + (target_vel * t_intercept)
        
        diff_pred = pred_pos - state.ship_pos
        diff_pred.real = (diff_pred.real + world_width / 2) % world_width - world_width / 2
        diff_pred.imag = (diff_pred.imag + world_height / 2) % world_height - world_height / 2
        
        dist_pred = torch.abs(diff_pred)
        return diff_pred / (dist_pred + 1e-8)

    def _compute_action_probs(self, state: TensorState, closest_dist: torch.Tensor, dir_pred: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes marginal probability distributions for Power(3), Turn(7), and Shoot(2).
        Returns unmasked distributions of shape (B, N, C).
        """
        batch_size, num_ships = state.ship_pos.shape
        device = state.device
        
        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att)) # range (-pi, pi)
        abs_angle = torch.abs(rel_angle)
        
        # --- 1. Turn Probabilities (7 options) ---
        # Options: Straight=0, L=1, R=2, Sharp L=3, Sharp R=4, AirBrake=5, Sharp AirBrake=6
        p_needs_turn = self._linear_ramp(abs_angle, self.config.turn_angle_ramp[0], self.config.turn_angle_ramp[1])
        p_is_sharp = self._linear_ramp(abs_angle, self.config.sharp_turn_angle_ramp[0], self.config.sharp_turn_angle_ramp[1])
        
        # Determine direction logic (deterministic boundary on 0 for direction, since it's symmetric)
        p_dir_right = torch.where(rel_angle > 0, 1.0, 0.0)
        p_dir_left = torch.where(rel_angle < 0, 1.0, 0.0)
        
        # P(Turn Left) = P(Needs Turn) * P(Left)
        p_turn_left_base = self._prob_and(p_needs_turn, p_dir_left)
        p_turn_right_base = self._prob_and(p_needs_turn, p_dir_right)
        
        # Split into sharp vs normal
        p_sharp_left = self._prob_and(p_turn_left_base, p_is_sharp)
        p_sharp_right = self._prob_and(p_turn_right_base, p_is_sharp)
        
        p_normal_left = self._prob_and(p_turn_left_base, 1.0 - p_is_sharp)
        p_normal_right = self._prob_and(p_turn_right_base, 1.0 - p_is_sharp)
        
        # P(Straight) = 1.0 - P(Needs Turn)
        p_straight = 1.0 - p_needs_turn
        
        # Air brakes unused
        p_air_brake = torch.zeros_like(p_straight)
        p_sharp_air_brake = torch.zeros_like(p_straight)
        
        # Assemble Turn Probs
        turn_probs = torch.stack([
            p_straight, p_normal_left, p_normal_right, 
            p_sharp_left, p_sharp_right, p_air_brake, p_sharp_air_brake
        ], dim=-1)
        
        # Ensure sum=1 (numerical stability)
        turn_probs = turn_probs / (torch.sum(turn_probs, dim=-1, keepdim=True) + 1e-8)
        
        # --- 2. Power Probabilities (3 options) ---
        # Options: Coast=0, Boost=1, Reverse=2
        speed = state.ship_vel.abs()
        power_ratio = state.ship_power / self.ship_config.max_power
        
        # Reverse reason: close to target (invert so low dist = 1.0 prob)
        p_is_close = self._linear_ramp(closest_dist, self.config.close_range_ramp[0], self.config.close_range_ramp[1], invert=True)
        p_reverse = p_is_close
        
        # Boost reasons: moving too slow OR far away with power
        p_slow = self._linear_ramp(speed, self.config.boost_speed_ramp[0], self.config.boost_speed_ramp[1], invert=True)
        
        # "far away and have power": old logic was want_boost = power_ratio > 0.9 * (1.0 - closest_dist/max_range)
        # We can make a ramp based on this metric: M = power_ratio - (1.0 - closest_dist/max_range). If M > 0, boost.
        boost_metric = power_ratio - (1.0 - closest_dist / self.config.max_shooting_range)
        # Ramp from M=-0.2 to M=0.2 (arbitrary spread for the old deterministic boundary at 0)
        p_far_power = self._linear_ramp(boost_metric, -0.2, 0.2)
        
        p_want_boost = self._prob_or(p_slow, p_far_power)
        
        # Can boost: P(want boost AND NOT close)
        p_can_boost = self._prob_and(p_want_boost, 1.0 - p_is_close)
        p_boost = p_can_boost
        
        # Coast: whatever is left
        # Since p_reverse and p_boost aren't explicitly mutually exclusive from the ramps, we need to normalize
        # Actually p_boost = P(want_boost) * (1-P(close)), and P(close) = p_reverse
        # So p_boost + p_reverse = P(want_boost)*(1-P(close)) + P(close) 
        # = P(want_boost) - P(want_boost)P(close) + P(close) = P(want_boost OR close). This is always <= 1.0.
        p_coast = 1.0 - self._prob_or(p_want_boost, p_reverse)
        p_coast = torch.clamp(p_coast, 0.0, 1.0)
        
        power_probs = torch.stack([p_coast, p_boost, p_reverse], dim=-1)
        power_probs = power_probs / (torch.sum(power_probs, dim=-1, keepdim=True) + 1e-8)
        
        # --- 3. Shoot Probabilities (2 options) ---
        # Options: No Shoot=0, Shoot=1
        target_angular_size = 2.0 * torch.atan(self.config.target_radius / (closest_dist + 1e-8))
        shoot_threshold = self.config.radius_multiplier * target_angular_size
        shoot_threshold = torch.clamp(shoot_threshold, np.deg2rad(1.0), np.deg2rad(45.0))
        
        # Alignment prob
        angle_ratio = abs_angle / shoot_threshold
        p_aligned = self._linear_ramp(angle_ratio, self.config.shoot_angle_multiplier_ramp[0], self.config.shoot_angle_multiplier_ramp[1], invert=True)
        
        # Distance prob
        max_dist = self.config.max_shooting_range * torch.sqrt(power_ratio + 1e-8)
        dist_ratio = closest_dist / (max_dist + 1e-8)
        p_in_range = self._linear_ramp(dist_ratio, self.config.shoot_distance_ramp[0], self.config.shoot_distance_ramp[1], invert=True)
        
        p_shoot = self._prob_and(p_aligned, p_in_range)
        p_no_shoot = 1.0 - p_shoot
        
        shoot_probs = torch.stack([p_no_shoot, p_shoot], dim=-1)
        shoot_probs = shoot_probs / (torch.sum(shoot_probs, dim=-1, keepdim=True) + 1e-8)
        
        # If not active mask, override with coast/straight/no_shoot=1.0
        active_expanded = active_mask.unsqueeze(-1)
        
        def apply_mask(probs, default_idx):
            mask_probs = torch.zeros_like(probs)
            mask_probs[..., default_idx] = 1.0
            return torch.where(active_expanded, probs, mask_probs)
            
        power_probs = apply_mask(power_probs, 0)
        turn_probs = apply_mask(turn_probs, 0)
        shoot_probs = apply_mask(shoot_probs, 0)
        
        return power_probs, turn_probs, shoot_probs

    def get_actions_and_probs(self, state: TensorState) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples actions and returns the probability distribution as soft labels.
        Returns:
            actions: (B, N, 3) IntTensor
            expert_probs: (B, N, 12) or (B, N, 42) FloatTensor based on config
        """
        closest_dist, target_idx, has_target = self._select_targets(state)
        dir_pred = self._predict_interception(state, target_idx, closest_dist)
        active_mask = state.ship_alive & has_target
        
        p_power, p_turn, p_shoot = self._compute_action_probs(state, closest_dist, dir_pred, active_mask)
        
        batch_size, num_ships = state.ship_pos.shape
        device = state.device
        
        if self.config.flat_action_sampling:
            # Joint distribution 3x7x2 = 42
            # p(a, b, c) = p(a)p(b)p(c)
            # Shapes: (B, N, 3, 1, 1), (B, N, 1, 7, 1), (B, N, 1, 1, 2)
            joint_probs = p_power.unsqueeze(-1).unsqueeze(-1) * \
                          p_turn.unsqueeze(-2).unsqueeze(-1) * \
                          p_shoot.unsqueeze(-2).unsqueeze(-2)
                          
            # Flatten to (B, N, 42)
            joint_probs = joint_probs.reshape(batch_size, num_ships, 42)
            expert_probs = joint_probs
            
            # Sample from categorical
            flat_probs = joint_probs.view(-1, 42)
            sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(batch_size, num_ships)
            
            # Decompose index back to (power, turn, shoot)
            # Index = power * (7*2) + turn * 2 + shoot
            actions_shoot = sampled_flat % 2
            sampled_flat = sampled_flat // 2
            actions_turn = sampled_flat % 7
            actions_power = sampled_flat // 7
            
            actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=-1)
        else:
            # Independent marginals
            expert_probs = torch.cat([p_power, p_turn, p_shoot], dim=-1) # Size 12
            
            p_p_flat = p_power.view(-1, 3)
            p_t_flat = p_turn.view(-1, 7)
            p_s_flat = p_shoot.view(-1, 2)
            
            a_p = torch.multinomial(p_p_flat, num_samples=1).view(batch_size, num_ships)
            a_t = torch.multinomial(p_t_flat, num_samples=1).view(batch_size, num_ships)
            a_s = torch.multinomial(p_s_flat, num_samples=1).view(batch_size, num_ships)
            
            actions = torch.stack([a_p, a_t, a_s], dim=-1)
            
        return actions, expert_probs
        
    def get_actions(self, state: TensorState) -> torch.Tensor:
        """Standard interface"""
        actions, _ = self.get_actions_and_probs(state)
        return actions
