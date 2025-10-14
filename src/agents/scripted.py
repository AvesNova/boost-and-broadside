from typing import Any
import torch
import numpy as np

from .base import Agent


class ScriptedAgent(Agent):
    """Agent that controls n ships using scripted behavior"""

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize scripted agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to
            squad: List of ship IDs this agent controls
            config: Configuration for scripted behavior
        """
        super().__init__(agent_id, team_id, squad)

        # Default configuration
        self.config: dict[str, Any] = {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
            "world_size": [1200, 800],
            "rng": np.random.default_rng(),
        }

        # Update with provided config
        if config:
            self.config.update(config)

        # Create individual ship controllers for each ship in squad
        self.ship_controllers: dict[int, ShipController] = {}
        for ship_id in squad:
            self.ship_controllers[ship_id] = ShipController(ship_id, self.config)

    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Get actions from scripted behavior

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Dictionary mapping ship_id to action tensor
        """
        actions: dict[int, torch.Tensor] = {}
        valid_ships = self.get_ship_ids_for_obs(obs_dict)

        for ship_id in valid_ships:
            if ship_id in self.ship_controllers:
                actions[ship_id] = self.ship_controllers[ship_id].get_actions(obs_dict)
            else:
                # Fallback to no action
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        return actions

    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        return "scripted"


class ShipController:
    """Controller for a single ship using scripted behavior"""

    def __init__(self, ship_id: int, config: dict[str, Any]):
        """
        Initialize ship controller

        Args:
            ship_id: ID of the ship this controller controls
            config: Configuration for behavior
        """
        self.ship_id = ship_id
        self.config = config

    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get actions for this ship

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Action tensor for this ship
        """
        # Extract ship state
        if self.ship_id >= obs_dict["ship_id"].shape[0]:
            return torch.zeros(6, dtype=torch.float32)

        if not obs_dict["alive"][self.ship_id, 0].item():
            return torch.zeros(6, dtype=torch.float32)

        # Get ship position and velocity
        position = obs_dict["position"][self.ship_id, 0].item()
        velocity = obs_dict["velocity"][self.ship_id, 0].item()
        attitude = obs_dict["attitude"][self.ship_id, 0].item()

        # Find nearest enemy ship
        enemy_ships = self._find_enemy_ships(obs_dict)

        if enemy_ships:
            target_ship = self._find_nearest_enemy(position, enemy_ships)
            return self._compute_actions_to_target(
                position, velocity, attitude, target_ship, obs_dict
            )
        else:
            # No enemies, just move forward
            return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def _find_enemy_ships(self, obs_dict: dict[str, torch.Tensor]) -> list[dict]:
        """Find all enemy ships"""
        enemies: list[dict] = []

        for ship_id in range(obs_dict["ship_id"].shape[0]):
            if obs_dict["alive"][ship_id, 0].item() > 0:
                # Check if this is an enemy ship (different team)
                # For now, assume ships with IDs in different ranges are enemies
                # This is a simplification - in a real implementation, we'd use team_id
                if ship_id != self.ship_id:  # Simplified enemy detection
                    enemies.append(
                        {
                            "id": ship_id,
                            "position": obs_dict["position"][ship_id, 0].item(),
                            "velocity": obs_dict["velocity"][ship_id, 0].item(),
                            "alive": obs_dict["alive"][ship_id, 0].item(),
                        }
                    )

        return enemies

    def _find_nearest_enemy(self, position: complex, enemies: list[dict]) -> dict:
        """Find the nearest enemy ship"""
        if not enemies:
            return None

        nearest = None
        min_distance = float("inf")

        for enemy in enemies:
            distance = abs(position - enemy["position"])
            if distance < min_distance:
                min_distance = distance
                nearest = enemy

        return nearest

    def _compute_actions_to_target(
        self,
        position: complex,
        velocity: complex,
        attitude: complex,
        target: dict,
        obs_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute actions to move toward and shoot at target"""
        target_pos = target["position"]

        # Calculate relative position
        relative_pos = target_pos - position

        # Calculate desired angle to target
        desired_angle = np.angle(relative_pos)
        current_angle = np.angle(attitude)

        # Calculate angle difference
        angle_diff = (desired_angle - current_angle + np.pi) % (2 * np.pi) - np.pi

        # Calculate distance to target
        distance = abs(relative_pos)

        # Determine actions
        actions = torch.zeros(6, dtype=torch.float32)

        # Movement
        if distance > self.config["target_radius"] * self.config["radius_multiplier"]:
            # Move forward if far from target
            actions[0] = 1.0  # Forward
        else:
            # Move backward if too close
            actions[1] = 1.0  # Backward

        # Turning
        angle_threshold_rad = np.radians(self.config["angle_threshold"])
        if abs(angle_diff) > angle_threshold_rad:
            if angle_diff > 0:
                actions[3] = 1.0  # Turn right
            else:
                actions[2] = 1.0  # Turn left

        # Shooting
        if (
            distance < self.config["max_shooting_range"]
            and abs(angle_diff) < angle_threshold_rad
        ):
            actions[5] = 1.0  # Shoot

        return actions
