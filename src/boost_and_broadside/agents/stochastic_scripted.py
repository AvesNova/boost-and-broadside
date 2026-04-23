import torch
import numpy as np
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig


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

    def _select_targets(
        self, state: TensorState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the nearest alive enemy for each ship.

        Returns:
            closest_dist: (B, N) float — distance to nearest enemy
            target_idx:   (B, N) long — index of nearest enemy
            has_target:   (B, N) bool — whether a valid enemy exists
        """
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

        dist_masked = torch.where(
            valid_tgt, dist, torch.tensor(float("inf"), device=device)
        )
        closest_dist, target_idx = torch.min(dist_masked, dim=2)
        has_target = closest_dist < float("inf")

        return closest_dist, target_idx, has_target

    def _predict_interception(
        self, state: TensorState, target_idx: torch.Tensor, closest_dist: torch.Tensor
    ) -> torch.Tensor:
        """Predict the unit direction vector to the interception point.

        Accounts for both target and shooter movement during bullet travel time,
        since bullet velocity = shooter_vel + bullet_speed * aim_dir.
        Uses relative velocity (v_target - v_shooter) for the lead correction.
        """
        world_width, world_height = self.ship_config.world_size
        target_pos = torch.gather(state.ship_pos, 1, target_idx)
        target_vel = torch.gather(state.ship_vel, 1, target_idx)

        # First-order time estimate using current distance
        t_intercept = closest_dist / self.ship_config.bullet_speed

        # Project both target and shooter forward by t_intercept.
        pred_pos = target_pos + target_vel * t_intercept
        shooter_future_pos = state.ship_pos + state.ship_vel * t_intercept

        diff_pred = pred_pos - shooter_future_pos
        diff_pred.real = (
            diff_pred.real + world_width / 2
        ) % world_width - world_width / 2
        diff_pred.imag = (
            diff_pred.imag + world_height / 2
        ) % world_height - world_height / 2

        dist_pred = torch.abs(diff_pred)
        return diff_pred / (dist_pred + 1e-8)

    def _find_nearest_obstacle(
        self, state: TensorState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest obstacle (by edge distance) for each ship.

        Returns:
            min_dist_to_edge: (B, N) float — distance from ship to nearest obstacle edge
            nearest_dir: (B, N) complex — unit vector from ship toward nearest obstacle center
        """
        B, N = state.ship_pos.shape
        M = state.obstacle_pos.shape[1]
        device = state.device
        world_w, world_h = self.ship_config.world_size

        if M == 0:
            return (
                torch.full((B, N), float("inf"), device=device),
                torch.ones((B, N), dtype=torch.complex64, device=device),
            )

        # diff (B, N, M): vector from each ship to each obstacle center
        diff_real = (
            state.obstacle_pos.real.unsqueeze(1) - state.ship_pos.real.unsqueeze(2)
        )  # (B, N, M)
        diff_imag = (
            state.obstacle_pos.imag.unsqueeze(1) - state.ship_pos.imag.unsqueeze(2)
        )
        # Toroidal wrap
        diff_real = (diff_real + world_w / 2) % world_w - world_w / 2
        diff_imag = (diff_imag + world_h / 2) % world_h - world_h / 2

        dist_to_center = torch.sqrt(diff_real**2 + diff_imag**2)  # (B, N, M)
        dist_to_edge = dist_to_center - state.obstacle_radius.unsqueeze(1)  # (B, N, M)

        min_dist, nearest_m = dist_to_edge.min(dim=2)  # (B, N)

        # Direction from ship to nearest obstacle center (using center diff, not edge)
        idx = nearest_m.unsqueeze(2)  # (B, N, 1)
        nd_real = diff_real.gather(2, idx).squeeze(2)  # (B, N)
        nd_imag = diff_imag.gather(2, idx).squeeze(2)
        nd_center = dist_to_center.gather(2, idx).squeeze(2)
        nearest_dir = torch.complex(
            nd_real / (nd_center + 1e-8), nd_imag / (nd_center + 1e-8)
        )

        return min_dist, nearest_dir

    def _compute_action_probs(
        self,
        state: TensorState,
        closest_dist: torch.Tensor,
        dir_pred: torch.Tensor,
        active_mask: torch.Tensor,
        obs_min_dist: torch.Tensor | None = None,
        obs_nearest_dir: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute marginal probability distributions for Power(3), Turn(7), Shoot(2).

        Returns unmasked distributions of shape (B, N, C).
        """
        batch_size, num_ships = state.ship_pos.shape
        device = state.device

        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att))  # range (-pi, pi)
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

        angle_ratio = abs_angle / (shoot_threshold + 1e-8)
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

        # Obstacle avoidance blend (when M > 0)
        if obs_min_dist is not None and obs_nearest_dir is not None:
            # Away direction: opposite of toward-obstacle
            away = -obs_nearest_dir  # (B, N) complex
            # Cross product determines left/right: att × away
            cross = att.real * away.imag - att.imag * away.real  # (B, N)

            p_obs_turn = self._linear_ramp(
                obs_min_dist,
                self.config.obs_avoid_dist_ramp[0],
                self.config.obs_avoid_dist_ramp[1],
                *self.config.obs_avoid_turn_prob,
            )
            p_obs_sharp = self._linear_ramp(
                obs_min_dist,
                self.config.obs_sharp_dist_ramp[0],
                self.config.obs_sharp_dist_ramp[1],
                *self.config.obs_sharp_turn_prob,
            )
            p_obs_rev = self._linear_ramp(
                obs_min_dist,
                self.config.obs_reverse_dist_ramp[0],
                self.config.obs_reverse_dist_ramp[1],
                *self.config.obs_reverse_prob,
            )

            # cross = att.real * away.imag - att.imag * away.real
            # Matches the target-tracking convention: cross > 0 → away is to the RIGHT
            p_obs_right = torch.where(cross > 0, p_obs_turn, torch.zeros_like(p_obs_turn))
            p_obs_left = torch.where(cross <= 0, p_obs_turn, torch.zeros_like(p_obs_turn))
            p_obs_sharp_right = torch.where(
                cross > 0, p_obs_sharp, torch.zeros_like(p_obs_sharp)
            )
            p_obs_sharp_left = torch.where(
                cross <= 0, p_obs_sharp, torch.zeros_like(p_obs_sharp)
            )

            # Add avoidance votes to existing turn probs and renormalize
            avoid_turn = torch.stack(
                [
                    torch.zeros_like(p_obs_turn),  # straight — avoidance adds nothing
                    p_obs_left,
                    p_obs_right,
                    p_obs_sharp_left,
                    p_obs_sharp_right,
                    torch.zeros_like(p_obs_turn),
                    torch.zeros_like(p_obs_turn),
                ],
                dim=-1,
            )
            turn_probs = turn_probs + avoid_turn
            turn_probs = turn_probs / (turn_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Blend reverse: use _prob_or so either signal can trigger it
            p_reverse_blended = self._prob_or(p_reverse, p_obs_rev)
            p_boost_blended = self._prob_and(p_want_boost, 1.0 - p_reverse_blended)
            p_coast_blended = (
                1.0 - self._prob_or(p_want_boost, p_reverse_blended)
            ).clamp(0.0, 1.0)
            power_probs = torch.stack(
                [p_coast_blended, p_boost_blended, p_reverse_blended], dim=-1
            )
            power_probs = power_probs / (power_probs.sum(dim=-1, keepdim=True) + 1e-8)

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
        closest_dist, target_idx, has_target = self._select_targets(state)
        dir_pred = self._predict_interception(state, target_idx, closest_dist)
        active_mask = state.ship_alive & has_target

        M = state.obstacle_pos.shape[1]
        if M > 0:
            obs_min_dist, obs_nearest_dir = self._find_nearest_obstacle(state)
        else:
            obs_min_dist, obs_nearest_dir = None, None

        p_power, p_turn, p_shoot = self._compute_action_probs(
            state, closest_dist, dir_pred, active_mask,
            obs_min_dist=obs_min_dist,
            obs_nearest_dir=obs_nearest_dir,
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
