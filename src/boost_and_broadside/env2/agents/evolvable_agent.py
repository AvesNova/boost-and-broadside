import torch
import numpy as np

from boost_and_broadside.env2.state import TensorState
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.agents.stochastic_config import StochasticAgentConfig

class BatchedEvolvableAgent:
    """
    Batched wrapper logic that evaluates the stochastic scripting behavior 
    over an entire batch with *per-ship* config vectors.
    """
    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config
        self.prob_offset = 0.1
        self.flat_action_sampling = False # Match stochastic_config default or set it manually

        # Tensor representation of PARAM_BOUNDS: shape (12, 2)
        # We need to scale the [0, 1] normalized values back to physical units.
        bounds = StochasticAgentConfig.PARAM_BOUNDS
        self.bounds_lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        self.bounds_diff = torch.tensor([b[1] - b[0] for b in bounds], dtype=torch.float32)

    def _linear_ramp(self, x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, prob_lo: torch.Tensor, prob_hi: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N)
        low, high, prob_lo, prob_hi: (B, N)
        Returns: (B, N)
        """
        # Avoid division by zero
        diff = high - low
        diff = torch.where(diff == 0, torch.ones_like(diff), diff) 
        
        t = torch.clamp((x - low) / diff, 0.0, 1.0)
        
        # Where high == low, just return prob_lo
        result = prob_lo + t * (prob_hi - prob_lo)
        result = torch.where(high == low, prob_lo, result)
        return result
        
    def _prob_or(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        return p_a + p_b - (p_a * p_b)
        
    def _prob_and(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        return p_a * p_b

    def _select_targets(self, state: TensorState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _compute_action_probs(self, state: TensorState, config_tensor: torch.Tensor, closest_dist: torch.Tensor, dir_pred: torch.Tensor, active_mask: torch.Tensor):
        batch_size, num_ships = state.ship_pos.shape
        device = state.device
        
        # Scale parameters bounds
        # config_tensor is (B, N, 24)
        bounds_lo = self.bounds_lo.to(device).view(1, 1, 12).repeat_interleave(2, dim=2)
        bounds_diff = self.bounds_diff.to(device).view(1, 1, 12).repeat_interleave(2, dim=2)
        
        configs = bounds_lo + config_tensor * bounds_diff
        
        # 12 pairs of parameters:
        c_bst_spd_ramp_lo = configs[..., 0]
        c_bst_spd_ramp_hi = configs[..., 1]
        c_bst_spd_prob_lo = configs[..., 2]
        c_bst_spd_prob_hi = configs[..., 3]
        
        c_cls_rng_ramp_lo = configs[..., 4]
        c_cls_rng_ramp_hi = configs[..., 5]
        c_cls_rng_prob_lo = configs[..., 6]
        c_cls_rng_prob_hi = configs[..., 7]
        
        c_trn_ang_ramp_lo = configs[..., 8]
        c_trn_ang_ramp_hi = configs[..., 9]
        c_trn_ang_prob_lo = configs[..., 10]
        c_trn_ang_prob_hi = configs[..., 11]
        
        c_shp_trn_ramp_lo = configs[..., 12]
        c_shp_trn_ramp_hi = configs[..., 13]
        c_shp_trn_prob_lo = configs[..., 14]
        c_shp_trn_prob_hi = configs[..., 15]
        
        c_sht_ang_ramp_lo = configs[..., 16]
        c_sht_ang_ramp_hi = configs[..., 17]
        c_sht_ang_prob_lo = configs[..., 18]
        c_sht_ang_prob_hi = configs[..., 19]

        c_sht_dst_ramp_lo = configs[..., 20]
        c_sht_dst_ramp_hi = configs[..., 21]
        c_sht_dst_prob_lo = configs[..., 22]
        c_sht_dst_prob_hi = configs[..., 23]
        
        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att)) # range (-pi, pi)
        abs_angle = torch.abs(rel_angle)

        # TURN PROBABILITIES
        p_needs_turn = self._linear_ramp(abs_angle, c_trn_ang_ramp_lo, c_trn_ang_ramp_hi, c_trn_ang_prob_lo, c_trn_ang_prob_hi)
        p_is_sharp = self._linear_ramp(abs_angle, c_shp_trn_ramp_lo, c_shp_trn_ramp_hi, c_shp_trn_prob_lo, c_shp_trn_prob_hi)
        
        p_dir_right = torch.where(rel_angle > 0, 1.0, 0.0)
        p_dir_left = torch.where(rel_angle < 0, 1.0, 0.0)
        
        p_turn_left_base = self._prob_and(p_needs_turn, p_dir_left)
        p_turn_right_base = self._prob_and(p_needs_turn, p_dir_right)
        
        p_sharp_left = self._prob_and(p_turn_left_base, p_is_sharp)
        p_sharp_right = self._prob_and(p_turn_right_base, p_is_sharp)
        p_normal_left = self._prob_and(p_turn_left_base, 1.0 - p_is_sharp)
        p_normal_right = self._prob_and(p_turn_right_base, 1.0 - p_is_sharp)
        
        p_straight = 1.0 - p_needs_turn
        p_air_brake = torch.zeros_like(p_straight)
        p_sharp_air_brake = torch.zeros_like(p_straight)
        
        turn_probs = torch.stack([
            p_straight, p_normal_left, p_normal_right, 
            p_sharp_left, p_sharp_right, p_air_brake, p_sharp_air_brake
        ], dim=-1)
        turn_probs = turn_probs / (torch.sum(turn_probs, dim=-1, keepdim=True) + 1e-8)
        
        # POWER PROBABILITIES
        speed = state.ship_vel.abs()
        power_ratio = state.ship_power / self.ship_config.max_power
        
        p_is_close = self._linear_ramp(closest_dist, c_cls_rng_ramp_lo, c_cls_rng_ramp_hi, c_cls_rng_prob_lo, c_cls_rng_prob_hi)
        p_reverse = p_is_close
        
        p_slow = self._linear_ramp(speed, c_bst_spd_ramp_lo, c_bst_spd_ramp_hi, c_bst_spd_prob_lo, c_bst_spd_prob_hi)
        
        max_shooting_range = c_sht_dst_ramp_hi
        boost_metric = power_ratio - (1.0 - closest_dist / (max_shooting_range + 1e-8))
        
        # -0.2 and 0.2 spread around 0 for p_far_power
        p_far_power = torch.clamp((boost_metric - (-0.2)) / 0.4, 0.0, 1.0)
        
        p_want_boost = self._prob_or(p_slow, p_far_power)
        p_can_boost = self._prob_and(p_want_boost, 1.0 - p_is_close)
        p_boost = p_can_boost
        
        p_coast = 1.0 - self._prob_or(p_want_boost, p_reverse)
        p_coast = torch.clamp(p_coast, 0.0, 1.0)
        
        power_probs = torch.stack([p_coast, p_boost, p_reverse], dim=-1)
        power_probs = power_probs / (torch.sum(power_probs, dim=-1, keepdim=True) + 1e-8)
        
        # SHOOT PROBABILITIES
        target_angular_size = 2.0 * torch.atan(self.ship_config.collision_radius / (closest_dist + 1e-8))
        shoot_threshold = torch.clamp(target_angular_size, np.deg2rad(1.0), np.deg2rad(45.0))
        angle_ratio = abs_angle / (shoot_threshold + 1e-8)
        
        p_aligned = self._linear_ramp(angle_ratio, c_sht_ang_ramp_lo, c_sht_ang_ramp_hi, c_sht_ang_prob_lo, c_sht_ang_prob_hi)
        p_in_range = self._linear_ramp(closest_dist, c_sht_dst_ramp_lo, c_sht_dst_ramp_hi, c_sht_dst_prob_lo, c_sht_dst_prob_hi)
        
        p_shoot = self._prob_and(p_aligned, p_in_range)
        p_no_shoot = 1.0 - p_shoot
        
        shoot_probs = torch.stack([p_no_shoot, p_shoot], dim=-1)
        shoot_probs = shoot_probs / (torch.sum(shoot_probs, dim=-1, keepdim=True) + 1e-8)
        
        # Active Mask
        active_expanded = active_mask.unsqueeze(-1)
        
        def apply_mask(probs, default_idx):
            mask_probs = torch.zeros_like(probs)
            mask_probs[..., default_idx] = 1.0
            return torch.where(active_expanded, probs, mask_probs)
            
        power_probs = apply_mask(power_probs, 0)
        turn_probs = apply_mask(turn_probs, 0)
        shoot_probs = apply_mask(shoot_probs, 0)
        
        return power_probs, turn_probs, shoot_probs

    def get_actions(self, state: TensorState, config_tensor: torch.Tensor) -> torch.Tensor:
        """
        config_tensor: (B, N, 24) bounds-normalized values.
        """
        closest_dist, target_idx, has_target = self._select_targets(state)
        dir_pred = self._predict_interception(state, target_idx, closest_dist)
        active_mask = state.ship_alive & has_target
        
        p_power, p_turn, p_shoot = self._compute_action_probs(state, config_tensor, closest_dist, dir_pred, active_mask)
        
        batch_size, num_ships = state.ship_pos.shape
        
        if self.flat_action_sampling:
            joint_probs = p_power.unsqueeze(-1).unsqueeze(-1) * p_turn.unsqueeze(-2).unsqueeze(-1) * p_shoot.unsqueeze(-2).unsqueeze(-2)
            joint_probs = joint_probs.reshape(batch_size, num_ships, 42)
            flat_probs = joint_probs.view(-1, 42)
            sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(batch_size, num_ships)
            actions_shoot = sampled_flat % 2
            sampled_flat = sampled_flat // 2
            actions_turn = sampled_flat % 7
            actions_power = sampled_flat // 7
            actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=-1)
        else:
            p_p_flat = p_power.reshape(-1, 3)
            p_t_flat = p_turn.reshape(-1, 7)
            p_s_flat = p_shoot.reshape(-1, 2)
            a_p = torch.multinomial(p_p_flat, num_samples=1).view(batch_size, num_ships)
            a_t = torch.multinomial(p_t_flat, num_samples=1).view(batch_size, num_ships)
            a_s = torch.multinomial(p_s_flat, num_samples=1).view(batch_size, num_ships)
            actions = torch.stack([a_p, a_t, a_s], dim=-1)
            
        return actions
