
import torch
import numpy as np
from boost_and_broadside.env2.state import TensorState
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.core.constants import PowerActions, TurnActions, ShootActions

class VectorScriptedAgent:
    """
    A scripted agent that controls ships in a vectorized environment.
    
    It uses simple heuristics to steer towards the nearest enemy, shoot when aligned,
    and manage power distribution.
    
    Attributes:
        config: The ship configuration.
        max_shooting_range: Maximum range to attempt shooting.
        angle_threshold: Angle threshold for turning logic.
        radius_multiplier: Multiplier for target size when determining shot alignment.
        target_radius: Assumed radius of the target for aiming.
    """
    def __init__(self, config: ShipConfig):
        self.config = config
        
        # Parameters matching original scripted agent defaults
        self.max_shooting_range = 600.0 
        self.angle_threshold = np.deg2rad(10.0)
        self.radius_multiplier = 1.0
        self.target_radius = 20.0 
        
    def _select_targets(self, state: TensorState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Finds the closest valid enemy target for each ship.
        
        Args:
            state: The current environment state.
            
        Returns:
            A tuple containing:
            - closest_dist: (B, N) Distances to closest targets.
            - target_idx: (B, N) Indices of closest targets.
            - has_target: (B, N) Boolean mask indicating if a valid target was found.
        """
        batch_size, num_ships = state.ship_pos.shape
        world_width, world_height = self.config.world_size
        device = state.device
        
        # Calculate pairwise differences: pos[target] - pos[source]
        # pos_targets: (B, 1, N), pos_sources: (B, N, 1) -> (B, N_src, N_tgt)
        pos_targets = state.ship_pos.unsqueeze(1)
        pos_sources = state.ship_pos.unsqueeze(2)
        
        diff = pos_targets - pos_sources
        
        # Wrap World Boundaries
        diff.real = (diff.real + world_width / 2) % world_width - world_width / 2
        diff.imag = (diff.imag + world_height / 2) % world_height - world_height / 2
        
        dist = torch.abs(diff)
        
        # Masking: Different Team AND Alive
        team_src = state.ship_team_id.unsqueeze(2)
        team_tgt = state.ship_team_id.unsqueeze(1)
        enemy_mask = team_src != team_tgt
        
        alive_tgt = state.ship_alive.unsqueeze(1)
        valid_tgt = enemy_mask & alive_tgt
        
        # Set invalid distances to infinity so they are not selected
        dist_masked = torch.where(valid_tgt, dist, torch.tensor(float('inf'), device=device))
        
        # Find closest target index
        closest_dist, target_idx = torch.min(dist_masked, dim=2)
        
        has_target = closest_dist < float('inf')
        
        return closest_dist, target_idx, has_target

    def _predict_interception(
        self, 
        state: TensorState, 
        target_idx: torch.Tensor, 
        closest_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts the unit direction vector to the interception point.
        
        Args:
            state: The current environment state.
            target_idx: Indices of the target ships.
            closest_dist: Distances to the target ships.
            
        Returns:
            A tensor of shape (B, N) containing the unit vector pointing to the 
            predicted interception location.
        """
        world_width, world_height = self.config.world_size
        
        # Gather target state
        target_pos = torch.gather(state.ship_pos, 1, target_idx)
        target_vel = torch.gather(state.ship_vel, 1, target_idx)
        
        # Estimate time to intercept
        t_intercept = closest_dist / self.config.bullet_speed
        
        # Predict future position
        pred_displacement = target_vel * t_intercept
        pred_pos = target_pos + pred_displacement
        
        # Calculate vector to predicted position
        diff_pred = pred_pos - state.ship_pos
        
        # Wrap vector
        diff_pred.real = (diff_pred.real + world_width / 2) % world_width - world_width / 2
        diff_pred.imag = (diff_pred.imag + world_height / 2) % world_height - world_height / 2
        
        dist_pred = torch.abs(diff_pred)
        dir_pred = diff_pred / (dist_pred + 1e-8)
        
        return dir_pred

    def _compute_actions(
        self, 
        state: TensorState, 
        closest_dist: torch.Tensor, 
        dir_pred: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine Power, Turn, and Shoot actions based on target physics.
        
        Args:
            state: The current environment state.
            closest_dist: Distances to targets.
            dir_pred: Predicted direction to interception.
            mask: Boolean mask of active ships with targets.
            
        Returns:
            A tensor of shape (B, N, 3) containing actions.
        """
        batch_size, num_ships = state.ship_pos.shape
        device = state.device
        
        actions_power = torch.zeros((batch_size, num_ships), dtype=torch.long, device=device)
        actions_turn = torch.zeros((batch_size, num_ships), dtype=torch.long, device=device)
        actions_shoot = torch.zeros((batch_size, num_ships), dtype=torch.long, device=device)
        
        if not mask.any():
             return torch.stack([actions_power, actions_turn, actions_shoot], dim=2)
             
        # Calculate alignment
        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att)) # range (-pi, pi)
        abs_angle = torch.abs(rel_angle)
        
        # --- Turn Logic ---
        needs_turn = abs_angle > self.angle_threshold
        is_sharp = abs_angle > self.config.sharp_turn_angle
        
        turn_right = needs_turn & (rel_angle > 0)
        turn_left = needs_turn & (rel_angle < 0)
        
        actions_turn[turn_left] = TurnActions.TURN_LEFT
        actions_turn[turn_right] = TurnActions.TURN_RIGHT
        
        sharp_left = turn_left & is_sharp
        sharp_right = turn_right & is_sharp
        
        actions_turn[sharp_left] = TurnActions.SHARP_LEFT
        actions_turn[sharp_right] = TurnActions.SHARP_RIGHT
        
        # --- Power Logic ---
        close_range = 2.0 * self.target_radius
        is_close = closest_dist < close_range
        
        actions_power[is_close] = PowerActions.REVERSE
        
        speed = state.ship_vel.abs()
        power_ratio = state.ship_power / self.config.max_power
        
        # Boost if far away and have power, or if moving too slow
        want_boost = (power_ratio > 0.9 * (1.0 - closest_dist / self.max_shooting_range)) | (speed < 30.0)
        can_boost = (~is_close) & want_boost
        
        actions_power[can_boost] = PowerActions.BOOST
        
        # --- Shoot Logic ---
        target_angular_size = 2.0 * torch.atan(self.target_radius / closest_dist)
        shoot_threshold = self.radius_multiplier * target_angular_size
        shoot_threshold = torch.clamp(shoot_threshold, np.deg2rad(1.0), np.deg2rad(45.0))
        
        aligned = abs_angle < shoot_threshold
        in_range = closest_dist < (self.max_shooting_range * torch.sqrt(power_ratio))
        
        should_shoot = aligned & in_range
        
        actions_shoot[should_shoot] = ShootActions.SHOOT
        
        # Apply mask (only update actions for active ships with targets)
        
        # We compute predictions densely for all ships (even those without targets) 
        # to maintain consistent tensor shapes, then mask invalid actions later.
        
        valid_action_mask = mask # (B, N)
        
        actions_power = torch.where(valid_action_mask, actions_power, torch.tensor(0, device=device))
        actions_turn = torch.where(valid_action_mask, actions_turn, torch.tensor(0, device=device))
        actions_shoot = torch.where(valid_action_mask, actions_shoot, torch.tensor(0, device=device))
        
        return torch.stack([actions_power, actions_turn, actions_shoot], dim=2)

    def get_actions(self, state: TensorState) -> torch.Tensor:
        """
        Compute actions for all ships in all environments.
        
        Args:
            state: The current environment state.
            
        Returns:
            A tensor of shape (batch_size, num_ships, 3) containing actions.
        """
        closest_dist, target_idx, has_target = self._select_targets(state)

        # We compute predictions densely for all ships (even those without targets) 
        # to maintain consistent tensor shapes, then mask invalid actions later.
        
        dir_pred = self._predict_interception(state, target_idx, closest_dist)
        
        active_mask = state.ship_alive & has_target
        
        return self._compute_actions(state, closest_dist, dir_pred, active_mask)
