import torch
import numpy as np
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.scripted_utils import (
    select_targets,
    predict_interception,
    compute_team_target_bearings,
)


class StochasticScriptedAgent:
    """
    A stochastic scripted agent that uses configurable probability ramps
    instead of hard boundaries.

    Outputs probability distributions over the three action heads
    (power × turn × shoot) using physics-grounded heuristics:
    aim at nearest enemy, boost when far, reverse when close, shoot when aligned.
    """

    def __init__(self, ship_config: ShipConfig, agent_config: StochasticAgentConfig):
        self.ship_config = ship_config
        self.config = agent_config

    def _linear_ramp(
        self, x: torch.Tensor, low: float, high: float, prob_lo: float, prob_hi: float
    ) -> torch.Tensor:
        """
        Maps x linearly from [low, high] to [prob_lo, prob_hi], clamped.
        prob_lo is the output probability when x <= low.
        prob_hi is the output probability when x >= high.
        For an inverted ramp, pass prob_lo=1.0, prob_hi=0.0.
        """
        if high == low:
            return torch.full_like(x, prob_lo)
        t = torch.clamp((x - low) / (high - low), 0.0, 1.0)
        return prob_lo + t * (prob_hi - prob_lo)

    def _prob_or(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent OR logic: P(A or B) = P(A) + P(B) - P(A)*P(B)"""
        return p_a + p_b - (p_a * p_b)

    def _prob_and(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Independent AND logic: P(A and B) = P(A) * P(B)"""
        return p_a * p_b

    def _compute_action_probs(
        self,
        state: TensorState,
        closest_dist: torch.Tensor,
        dir_turn: torch.Tensor,
        dir_shoot: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute marginal probability distributions for Power(3), Turn(7), Shoot(2).

        dir_turn:  direction used for turn decisions (may be team-target-blended).
        dir_shoot: direction used for shoot alignment (always personal intercept).

        Returns unmasked distributions of shape (B, N, C).
        """
        batch_size, num_ships = state.ship_pos.shape
        device = state.device

        att = state.ship_attitude
        rel_angle = torch.angle(dir_turn * torch.conj(att))  # range (-pi, pi)
        abs_angle = torch.abs(rel_angle)

        # --- 1. Turn Probabilities (7 options) ---
        p_needs_turn = self._linear_ramp(
            abs_angle,
            self.config.turn_angle_ramp[0],
            self.config.turn_angle_ramp[1],
            *self.config.turn_angle_prob,
        )
        p_is_sharp = self._linear_ramp(
            abs_angle,
            self.config.sharp_turn_angle_ramp[0],
            self.config.sharp_turn_angle_ramp[1],
            *self.config.sharp_turn_angle_prob,
        )

        p_dir_right = torch.where(
            rel_angle > 0, torch.ones_like(rel_angle), torch.zeros_like(rel_angle)
        )
        p_dir_left = torch.where(
            rel_angle < 0, torch.ones_like(rel_angle), torch.zeros_like(rel_angle)
        )

        p_turn_left_base = self._prob_and(p_needs_turn, p_dir_left)
        p_turn_right_base = self._prob_and(p_needs_turn, p_dir_right)

        p_sharp_left = self._prob_and(p_turn_left_base, p_is_sharp)
        p_sharp_right = self._prob_and(p_turn_right_base, p_is_sharp)
        p_normal_left = self._prob_and(p_turn_left_base, 1.0 - p_is_sharp)
        p_normal_right = self._prob_and(p_turn_right_base, 1.0 - p_is_sharp)
        p_straight = 1.0 - p_needs_turn

        p_air_brake = torch.zeros_like(p_straight)
        p_sharp_air_brake = torch.zeros_like(p_straight)

        turn_probs = torch.stack(
            [
                p_straight,
                p_normal_left,
                p_normal_right,
                p_sharp_left,
                p_sharp_right,
                p_air_brake,
                p_sharp_air_brake,
            ],
            dim=-1,
        )
        turn_probs = turn_probs / (turn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 2. Power Probabilities (3 options) ---
        speed = state.ship_vel.abs()
        power_ratio = state.ship_power / self.ship_config.max_power

        p_is_close = self._linear_ramp(
            closest_dist,
            self.config.close_range_ramp[0],
            self.config.close_range_ramp[1],
            *self.config.close_range_prob,
        )
        p_reverse = p_is_close

        p_slow = self._linear_ramp(
            speed,
            self.config.boost_speed_ramp[0],
            self.config.boost_speed_ramp[1],
            *self.config.boost_speed_prob,
        )

        max_shooting_range = self.config.shoot_distance_ramp[1]
        boost_metric = power_ratio - (1.0 - closest_dist / max_shooting_range)
        p_far_power = self._linear_ramp(boost_metric, -0.2, 0.2, 0.0, 1.0)

        p_want_boost = self._prob_or(p_slow, p_far_power)
        p_boost = self._prob_and(p_want_boost, 1.0 - p_is_close)
        p_coast = (1.0 - self._prob_or(p_want_boost, p_reverse)).clamp(0.0, 1.0)

        power_probs = torch.stack([p_coast, p_boost, p_reverse], dim=-1)
        power_probs = power_probs / (power_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 3. Shoot Probabilities (2 options) ---
        target_angular_size = 2.0 * torch.atan(
            self.ship_config.collision_radius / (closest_dist + 1e-8)
        )
        shoot_threshold = target_angular_size.clamp(np.deg2rad(1.0), np.deg2rad(45.0))

        shoot_rel_angle = torch.angle(dir_shoot * torch.conj(att))
        angle_ratio = shoot_rel_angle.abs() / (shoot_threshold + 1e-8)
        p_aligned = self._linear_ramp(
            angle_ratio,
            self.config.shoot_angle_ramp[0],
            self.config.shoot_angle_ramp[1],
            *self.config.shoot_angle_prob,
        )
        p_in_range = self._linear_ramp(
            closest_dist,
            self.config.shoot_distance_ramp[0],
            self.config.shoot_distance_ramp[1],
            *self.config.shoot_distance_prob,
        )

        p_shoot = self._prob_and(p_aligned, p_in_range)
        p_no_shoot = 1.0 - p_shoot

        shoot_probs = torch.stack([p_no_shoot, p_shoot], dim=-1)
        shoot_probs = shoot_probs / (shoot_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Mask inactive ships (dead or no target) → default no-op action
        active_expanded = active_mask.unsqueeze(-1)

        def apply_mask(probs: torch.Tensor, default_idx: int) -> torch.Tensor:
            mask_probs = torch.zeros_like(probs)
            mask_probs[..., default_idx] = 1.0
            return torch.where(active_expanded, probs, mask_probs)

        power_probs = apply_mask(power_probs, 0)
        turn_probs = apply_mask(turn_probs, 0)
        shoot_probs = apply_mask(shoot_probs, 0)

        return power_probs, turn_probs, shoot_probs

    def get_actions_and_probs(
        self, state: TensorState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return the expert probability distribution as soft labels.

        Returns:
            actions:      (B, N, 3) int tensor
            expert_probs: (B, N, 12) float tensor (independent marginals) or
                          (B, N, 42) float tensor (joint, if flat_action_sampling=True)
        """
        closest_dist, target_idx, has_target, _ = select_targets(state, self.ship_config)
        dir_pred = predict_interception(state, self.ship_config, target_idx, closest_dist)
        active_mask = state.ship_alive & has_target

        # Blend turn direction: personal intercept at close range, team target at far range
        team_bearing, _, _, team_has_target = compute_team_target_bearings(state, self.ship_config)
        p_team = self._linear_ramp(
            closest_dist,
            self.config.team_target_distance_ramp[0],
            self.config.team_target_distance_ramp[1],
            *self.config.team_target_distance_prob,
        ) * team_has_target.float()
        dir_turn = (1.0 - p_team) * dir_pred + p_team * team_bearing
        dir_turn = dir_turn / (torch.abs(dir_turn) + 1e-8)

        p_power, p_turn, p_shoot = self._compute_action_probs(
            state, closest_dist, dir_turn, dir_pred, active_mask
        )

        batch_size, num_ships = state.ship_pos.shape
        device = state.device

        if self.config.flat_action_sampling:
            joint_probs = (
                p_power.unsqueeze(-1).unsqueeze(-1)
                * p_turn.unsqueeze(-2).unsqueeze(-1)
                * p_shoot.unsqueeze(-2).unsqueeze(-2)
            )
            joint_probs = joint_probs.reshape(batch_size, num_ships, 42)
            expert_probs = joint_probs

            flat_probs = joint_probs.view(-1, 42)
            sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(
                batch_size, num_ships
            )

            actions_shoot = sampled_flat % 2
            sampled_flat = sampled_flat // 2
            actions_turn = sampled_flat % 7
            actions_power = sampled_flat // 7
            actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=-1)
        else:
            expert_probs = torch.cat([p_power, p_turn, p_shoot], dim=-1)  # (B, N, 12)

            a_p = torch.multinomial(p_power.view(-1, 3), num_samples=1).view(
                batch_size, num_ships
            )
            a_t = torch.multinomial(p_turn.view(-1, 7), num_samples=1).view(
                batch_size, num_ships
            )
            a_s = torch.multinomial(p_shoot.view(-1, 2), num_samples=1).view(
                batch_size, num_ships
            )
            actions = torch.stack([a_p, a_t, a_s], dim=-1)

        return actions, expert_probs

    def get_actions(self, state: TensorState) -> torch.Tensor:
        """Standard interface — returns (B, N, 3) int tensor of sampled actions."""
        actions, _ = self.get_actions_and_probs(state)
        return actions
