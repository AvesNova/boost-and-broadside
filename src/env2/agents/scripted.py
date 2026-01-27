
import torch
import numpy as np
from env2.state import TensorState, ShipConfig
from core.constants import PowerActions, TurnActions, ShootActions

class VectorScriptedAgent:
    def __init__(self, config: ShipConfig):
        self.config = config
        
        # Parameters from original scripted agent
        self.max_shooting_range = 600.0 # Default?
        self.angle_threshold = np.deg2rad(10.0)
        self.radius_multiplier = 1.0
        self.target_radius = 20.0 # Estimate
        
        # TODO: These should be passed in init or config?
        # Original ScriptedAgent takes them in init. I'll stick to defaults matching code.
        
    def get_actions(self, state: TensorState) -> torch.Tensor:
        """
        Compute actions for all ships in all environments.
        Returns: (B, N, 3) tensor
        """
        device = state.device
        B, N = state.ship_pos.shape
        w, h = self.config.world_size
        
        # 1. Target Selection
        # Compute pairwise relative positions (wrapped)
        # (B, N, 1) - (B, 1, N) -> (B, N, N)
        # Row i, Col j is vector FROM j TO i ? Or FROM i TO j?
        # diff = target_pos - source_pos
        # source i, target j.
        diff = state.ship_pos.unsqueeze(2) - state.ship_pos.unsqueeze(1) # (B, N_source, N_target)
        # Actually diff[b, i, j] = pos[b, j] - pos[b, i] (Vector i -> j)
        # Let's verify broadcasting:
        # A (N, 1), B (1, N) -> A - B = (a0-b0, a0-b1...) NO.
        # A (N, 1) has values [[p0], [p1]]. B (1, N) has [[p0, p1]].
        # A - B = [[p0-p0, p0-p1], [p1-p0, p1-p1]].
        # p0-p1 is vector FROM 1 TO 0.
        # We want vector FROM source TO target.
        # Source i. Target j. Vector i->j = pos[j] - pos[i].
        # So pos[j] (1, N) - pos[i] (N, 1).
        # Check: [p0, p1] - [[p0], [p1]]??
        # Row 0 (i=0): [p0-p0, p1-p0]. p1-p0 is vector 0->1. Correct.
        # So diff = state.ship_pos.unsqueeze(1) - state.ship_pos.unsqueeze(2)
        
        pos_targets = state.ship_pos.unsqueeze(1) # (B, 1, N)
        pos_sources = state.ship_pos.unsqueeze(2) # (B, N, 1)
        
        diff = pos_targets - pos_sources # (B, N_src, N_tgt)
        
        # Wrap
        diff.real = (diff.real + w/2) % w - w/2
        diff.imag = (diff.imag + h/2) % h - h/2
        
        dist = torch.abs(diff)
        
        # Masking
        # Ignore same team
        team_src = state.ship_team_id.unsqueeze(2)
        team_tgt = state.ship_team_id.unsqueeze(1)
        enemy_mask = team_src != team_tgt
        
        # Ignore dead targets
        alive_tgt = state.ship_alive.unsqueeze(1) # (B, 1, N)
        valid_tgt = enemy_mask & alive_tgt
        
        # Ignore self (covered by team mask usually, but strictly distance > 0)
        # Set distance to infinity where invalid
        
        dist_masked = torch.where(valid_tgt, dist, torch.tensor(float('inf'), device=device))
        
        # Find closest
        # min_dist, closest_idx = torch.min(dist_masked, dim=2)
        # (B, N_src)
        closest_dist, target_idx = torch.min(dist_masked, dim=2) # target_idx contains index of closest enemy
        
        # Identify valid targets (if infinite distance, no target found)
        has_target = closest_dist < float('inf')
        
        # 2. Prediction
        # Need target Pos and Vel
        # Gather from state using target_idx
        # target_idx is (B, N). We need to gather from (B, N) tensors.
        # gather dim=1. index must be (B, N).
        target_pos = torch.gather(state.ship_pos, 1, target_idx)
        target_vel = torch.gather(state.ship_vel, 1, target_idx)
        
        # We also need to re-calculate wrapped vector to selected target
        # Because 'diff' was full matrix.
        # Or gather from diff? 
        # Gathering from (B, N, N) using indices (B, N, 1) -> (B, N, 1)
        # diff is (B, N, N). We want diff[b, i, target_idx[b, i]].
        # gather on dim 2.
        to_target_current = torch.gather(diff, 2, target_idx.unsqueeze(2)).squeeze(2) 
        
        # Predict Intercept
        # Time = dist / bullet_speed
        t_intercept = closest_dist / self.config.bullet_speed
        
        # Predicted Pos = Target Pos + Target Vel * t
        # (This ignores wrapping for the velocity vector part? 
        # Velocity is linear in world coordinates? 
        # Yes, standard physics assumes linear extrapolation.)
        pred_displacement = target_vel * t_intercept
        pred_pos = target_pos + pred_displacement
        
        # Re-calculate vector to predicted position from source
        # We need to wrap this vector too.
        # diff_pred = pred_pos - source_pos
        source_pos = state.ship_pos
        diff_pred = pred_pos - source_pos
        
        diff_pred.real = (diff_pred.real + w/2) % w - w/2
        diff_pred.imag = (diff_pred.imag + h/2) % h - h/2
        
        dist_pred = torch.abs(diff_pred)
        dir_pred = diff_pred / (dist_pred + 1e-8)
        
        # 3. Action Logic
        # Initialize output
        actions_power = torch.zeros((B, N), dtype=torch.long, device=device) # COAST
        actions_turn = torch.zeros((B, N), dtype=torch.long, device=device) # STRAIGHT
        actions_shoot = torch.zeros((B, N), dtype=torch.long, device=device) # NO_SHOOT
        
        # Loop only active ships with targets
        active = state.ship_alive & has_target
        
        if active.any():
            # Calculate Angle to Target
            # attitude vs dir_pred
            # angle = angle(dir_pred / attitude) ?
            # concept: attitude is vector A. dir_pred is vector T.
            # We want angle difference. 
            # att = exp(i * alpha). T = exp(i * beta).
            # T / att = exp(i * (beta - alpha)).
            # Angle diff = angle(T/att).
            
            att = state.ship_attitude
            
            # Relative angle logic
            # rel_angle = angle(dir_pred * conj(att))
            rel_angle = torch.angle(dir_pred * torch.conj(att))
            # Range passed through angle is (-pi, pi).
            
            abs_angle = torch.abs(rel_angle)
            
            # Turn Logic
            # Thresholds
            # TURN if abs_angle > angle_threshold
            # SHARP if abs_angle > sharp_threshold (15 deg?)
            
            needs_turn = abs_angle > self.angle_threshold
            is_sharp = abs_angle > self.config.sharp_turn_angle
            
            # Direction: if rel_angle > 0 (Left) or < 0 (Right)?
            # angle(T * conj(A)):
            # If T is 90 deg Left of A (CCW), result is i. angle is +pi/2.
            # So + is LEFT. - is RIGHT.
            
            # Mapping:
            # LEFT=1, RIGHT=2, S_LEFT=3, S_RIGHT=4
            
            # Default 0 (STRAIGHT)
            
            # Mapping:
            # rel_angle > 0 (CCW) -> TURN_RIGHT (Positive angle)
            # rel_angle < 0 (CW) -> TURN_LEFT (Negative angle)
            
            # Create masks
            turn_right = needs_turn & (rel_angle > 0)
            turn_left = needs_turn & (rel_angle < 0)
            
            actions_turn[turn_left] = TurnActions.TURN_LEFT
            actions_turn[turn_right] = TurnActions.TURN_RIGHT
            
            # Apply Sharp Upgrade
            sharp_left = turn_left & is_sharp
            sharp_right = turn_right & is_sharp
            
            actions_turn[sharp_left] = TurnActions.SHARP_LEFT
            actions_turn[sharp_right] = TurnActions.SHARP_RIGHT
            
            # Power Logic
            # Distance based.
            # If close (< 2 radii), REVERSE.
            # Else if power high and bad alignment? Or just Boost?
            # ScriptedAgent:
            # - Close (< 20): Reverse
            # - Else if power > 0.9 * (1 - dist/max_range) OR speed < 30: Boost
            
            close_range = 2.0 * self.target_radius
            is_close = closest_dist < close_range
            
            actions_power[is_close] = PowerActions.REVERSE
            
            # Boost logic
            speed = state.ship_vel.abs()
            power_ratio = state.ship_power / self.config.max_power
            
            want_boost = (power_ratio > 0.9 * (1.0 - closest_dist / self.max_shooting_range)) | (speed < 30.0)
            can_boost = (~is_close) & want_boost
            
            actions_power[can_boost] = PowerActions.BOOST
            
            # Shoot Logic
            # Aligned?
            # Dynamic threshold = radius_multiplier * 2 * atan(r / dist)
            # Clamp between 1 deg and 45 deg.
            
            target_angular_size = 2.0 * torch.atan(self.target_radius / closest_dist)
            shoot_threshold = self.radius_multiplier * target_angular_size
            shoot_threshold = torch.clamp(shoot_threshold, np.deg2rad(1.0), np.deg2rad(45.0))
            
            aligned = abs_angle < shoot_threshold
            in_range = closest_dist < (self.max_shooting_range * torch.sqrt(power_ratio))
            
            should_shoot = aligned & in_range
            
            actions_shoot[should_shoot] = ShootActions.SHOOT
            
        # Stack inputs
        # (B, N, 3)
        final_actions = torch.stack([actions_power, actions_turn, actions_shoot], dim=2)
        
        return final_actions

