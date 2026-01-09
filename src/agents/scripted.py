from typing import Any
import torch
import numpy as np
import torch.nn as nn


class ScriptedAgent(nn.Module):
    """
    Agent that controls n ships using scripted behavior.

    This agent delegates control of each ship to a ShipController, which
    implements the actual behavioral logic (targeting, movement, shooting).
    """

    def __init__(
        self,
        max_shooting_range: float,
        angle_threshold: float,
        bullet_speed: float,
        target_radius: float,
        radius_multiplier: float,
        world_size: tuple[float, float],
        rng: np.random.Generator = np.random.default_rng(),
        **kwargs: Any,
    ):
        """
        Initialize the scripted agent.

        Args:
            max_shooting_range: Maximum range to engage targets.
            angle_threshold: Base angle threshold for shooting (degrees).
            bullet_speed: Speed of bullets.
            target_radius: Assumed radius of target ships.
            radius_multiplier: Multiplier for dynamic shooting threshold.
            world_size: Dimensions of the world (width, height).
            rng: Random number generator.
            **kwargs: Ignored additional arguments.
        """
        super().__init__()

        self.ship_controller = ShipController(
            max_shooting_range=max_shooting_range,
            angle_threshold=angle_threshold,
            bullet_speed=bullet_speed,
            target_radius=target_radius,
            radius_multiplier=radius_multiplier,
            world_size=world_size,
            rng=rng,
        )

    def forward(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Forward pass for the agent.

        Args:
            obs_dict: Observation dictionary from environment.
            ship_ids: List of ship IDs to control.

        Returns:
            Dictionary mapping ship_id to action tensor.
        """
        return self.get_actions(obs_dict, ship_ids)

    def get_actions(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Get actions for all controlled ships.

        Args:
            obs_dict: Observation dictionary from environment.
            ship_ids: List of ship IDs to control.

        Returns:
            Dictionary mapping ship_id to action tensor.
        """
        return {
            ship_id: self.ship_controller.get_actions(obs_dict, ship_id)
            for ship_id in ship_ids
        }


class ShipController(nn.Module):
    """
    Advanced ship controller with predictive targeting and team awareness.

    Features:
    - Calculates bullet travel time based on distance.
    - Predicts enemy position when bullet arrives.
    - Uses predicted position for both turning and shooting.
    - Only targets ships from opposing teams (no friendly fire).
    - Targets the closest enemy ship when multiple enemies are present.
    - Supports toroidal world wrapping.
    - Power-aware movement and shooting decisions.
    - Dynamic shooting angle thresholds.
    - Sharp turn capability for large angle differences.
    """

    def __init__(
        self,
        max_shooting_range: float,
        angle_threshold: float,
        bullet_speed: float,
        target_radius: float,
        radius_multiplier: float,
        world_size: tuple[float, float],
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Initialize ship controller.

        Args:
            max_shooting_range: Maximum range to engage targets.
            angle_threshold: Base angle threshold for shooting (degrees).
            bullet_speed: Speed of bullets.
            target_radius: Assumed radius of target ships.
            radius_multiplier: Multiplier for dynamic shooting threshold.
            world_size: Dimensions of the world (width, height).
            rng: Random number generator.
        """
        super().__init__()
        self.max_shooting_range = max_shooting_range
        self.angle_threshold = np.deg2rad(angle_threshold)
        self.bullet_speed = bullet_speed
        self.target_radius = target_radius
        self.radius_multiplier = radius_multiplier
        self.world_size = world_size

    def get_actions(
        self, obs_dict: dict[str, torch.Tensor], ship_id: int
    ) -> torch.Tensor:
        """
        Get actions for a specific ship using advanced targeting logic.

        Args:
            obs_dict: Observation dictionary from environment.
            ship_id: ID of the ship to control.

        Returns:
            Action tensor for the ship.
        """
        # Get our ship's state and find closest enemy
        our_ship_data = None
        our_team_id = None
        enemy_candidates = []

        # Look through all ships to find ourselves and potential enemies
        for ship_idx in range(len(obs_dict["ship_id"])):
            ship_alive = obs_dict["alive"][ship_idx].item()
            if not ship_alive:
                continue

            current_ship_id = obs_dict["ship_id"][ship_idx].item()
            current_team_id = obs_dict["team_id"][ship_idx].item()

            if current_ship_id == ship_id:
                our_ship_data = {
                    "health": obs_dict["health"][ship_idx].item(),
                    "power": obs_dict["power"][ship_idx].item(),
                    "position": obs_dict["position"][ship_idx],
                    "velocity": obs_dict["velocity"][ship_idx],
                    "attitude": obs_dict["attitude"][ship_idx],
                }
                our_team_id = current_team_id
            else:
                # Store all potential enemies (ships not on our team)
                enemy_candidates.append(
                    {
                        "ship_id": current_ship_id,
                        "team_id": current_team_id,
                        "position": obs_dict["position"][ship_idx],
                        "velocity": obs_dict["velocity"][ship_idx],
                    }
                )

        # Find closest enemy ship (only target ships from different team)
        enemy_ship_data = None
        if our_ship_data is not None and our_team_id is not None and enemy_candidates:
            our_pos = torch.tensor(
                [our_ship_data["position"].real, our_ship_data["position"].imag],
                dtype=torch.float32,
            )

            closest_distance = float("inf")
            for candidate in enemy_candidates:
                # Skip allied ships (same team)
                if candidate["team_id"] == our_team_id:
                    continue

                candidate_pos = torch.tensor(
                    [candidate["position"].real, candidate["position"].imag],
                    dtype=torch.float32,
                )

                # Calculate distance with world wrapping
                to_candidate = self._calculate_wrapped_vector(our_pos, candidate_pos)
                distance = torch.norm(to_candidate).item()

                # Update closest enemy
                if distance < closest_distance:
                    closest_distance = distance
                    enemy_ship_data = {
                        "position": candidate["position"],
                        "velocity": candidate["velocity"],
                    }

        # Initialize action (Power, Turn, Shoot)
        # Power: COAST(0), Turn: GO_STRAIGHT(0), Shoot: NO_SHOOT(0)
        action = torch.zeros(3, dtype=torch.float32)

        # Check if we have valid data for both our ship and an enemy
        if our_ship_data is None or enemy_ship_data is None:
            return action  # Can't act without proper data

        # Convert complex numbers to 2D vectors for easier calculation
        self_pos = torch.tensor(
            [our_ship_data["position"].real, our_ship_data["position"].imag],
            dtype=torch.float32,
        )

        enemy_pos = torch.tensor(
            [enemy_ship_data["position"].real, enemy_ship_data["position"].imag],
            dtype=torch.float32,
        )

        enemy_velocity = torch.tensor(
            [enemy_ship_data["velocity"].real, enemy_ship_data["velocity"].imag],
            dtype=torch.float32,
        )

        self_attitude = torch.tensor(
            [our_ship_data["attitude"].real, our_ship_data["attitude"].imag],
            dtype=torch.float32,
        )

        # Calculate predicted enemy position when bullet arrives
        predicted_enemy_pos = self._calculate_predicted_position(
            self_pos, enemy_pos, enemy_velocity
        )

        # Calculate vector to PREDICTED enemy position (accounting for toroidal world wrapping)
        to_target = self._calculate_wrapped_vector(self_pos, predicted_enemy_pos)
        distance_to_target = torch.norm(to_target)

        # Also calculate distance to current enemy position for range checking
        to_enemy_current = self._calculate_wrapped_vector(self_pos, enemy_pos)
        current_distance = torch.norm(to_enemy_current)

        if distance_to_target < 1e-6:  # Avoid division by zero
            return action

        # Normalize direction to predicted target position
        to_target_normalized = to_target / distance_to_target

        # Calculate angle between current attitude and direction to predicted target
        # Use dot product: cos(angle) = a·b / (|a||b|)
        cos_angle = torch.dot(self_attitude, to_target_normalized)
        angle_to_target = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

        # Determine turn direction using cross product (2D cross product gives scalar)
        # If cross product is positive, target is to the left (in some coord systems)
        cross_product = (
            self_attitude[0] * to_target_normalized[1]
            - self_attitude[1] * to_target_normalized[0]
        )

        # Decide on turning action (aim at predicted position)
        if angle_to_target > self.angle_threshold:
            # Need to turn
            is_sharp = angle_to_target > np.deg2rad(15.0)
            
            if cross_product > 0:
                # Original: action[3] = 1.0 (Right)
                # Map to TurnActions.TURN_RIGHT (2) or SHARP_RIGHT (4)
                action[1] = 4.0 if is_sharp else 2.0
            else:
                # Original: action[2] = 1.0 (Left)
                # Map to TurnActions.TURN_LEFT (1) or SHARP_LEFT (3)
                action[1] = 3.0 if is_sharp else 1.0
        else:
            action[1] = 0.0 # GO_STRAIGHT

        # Calculate dynamic shooting angle threshold based on distance to target
        dynamic_shooting_threshold = self._calculate_shooting_angle_threshold(
            current_distance
        )

        current_power_ratio = our_ship_data["power"] / 100.0  # Normalize to [0,1]

        # Shoot if aligned with predicted target and current enemy is in range
        if (
            angle_to_target <= dynamic_shooting_threshold
            and current_distance
            <= self.max_shooting_range * np.sqrt(current_power_ratio)
            and our_ship_data["health"] > 0
        ):  # Only shoot if we're alive
            action[2] = 1.0  # ShootActions.SHOOT

        # Thrust management based on distance and power
        close_range_threshold = (
            2.0 * self.target_radius
        )  # 2 radii = 20 units by default

        current_speed = torch.linalg.norm(our_ship_data["velocity"]).item()

        if current_distance <= close_range_threshold:
            # Close range: use reverse thrust
            # Original: action[1] = 1.0 -> PowerActions.REVERSE (2)
            action[0] = 2.0
        else:
            # Normal range: boost if power > 90%
            if (
                current_power_ratio
                > 0.9 * (1.0 - current_distance / self.max_shooting_range)
                or current_speed < 30
            ):
                # Original: action[0] = 1.0 -> PowerActions.BOOST (1)
                action[0] = 1.0
            else:
                # COAST
                action[0] = 0.0

        return action

    def _calculate_wrapped_vector(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the shortest vector from pos1 to pos2 in a toroidal world.

        Args:
            pos1: Position 1 [x, y].
            pos2: Position 2 [x, y].

        Returns:
            Vector from pos1 to pos2 considering wrapping.
        """
        # Calculate direct difference
        diff = pos2 - pos1

        # Handle wrapping for each dimension
        for i in range(2):
            world_size = self.world_size[i]

            # If distance is more than half the world size, wrap around
            if abs(diff[i]) > world_size / 2:
                if diff[i] > 0:
                    diff[i] -= world_size
                else:
                    diff[i] += world_size

        return diff

    def _calculate_predicted_position(
        self,
        self_pos: torch.Tensor,
        enemy_pos: torch.Tensor,
        enemy_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate where the enemy will be when a bullet fired now reaches them.

        Args:
            self_pos: Our current position [x, y].
            enemy_pos: Enemy's current position [x, y].
            enemy_velocity: Enemy's velocity [vx, vy] (already in world units).

        Returns:
            Predicted enemy position [x, y] accounting for toroidal wrapping.
        """
        # Calculate current distance to enemy (accounting for wrapping)
        to_enemy = self._calculate_wrapped_vector(self_pos, enemy_pos)
        current_distance = torch.norm(to_enemy)

        if current_distance < 1e-6:
            return enemy_pos  # Already at same position

        # Calculate time for bullet to travel current distance
        bullet_travel_time = current_distance / self.bullet_speed

        # Predict where enemy will be after this time
        predicted_displacement = enemy_velocity * bullet_travel_time
        predicted_pos = enemy_pos + predicted_displacement

        # Apply world wrapping to predicted position
        predicted_pos[0] = predicted_pos[0] % self.world_size[0]  # Wrap x
        predicted_pos[1] = predicted_pos[1] % self.world_size[1]  # Wrap y

        return predicted_pos

    def _calculate_shooting_angle_threshold(self, distance: torch.Tensor) -> float:
        """
        Calculate the dynamic shooting angle threshold based on target distance.

        The angle threshold is based on the apparent angular size of the target:
        angular_size = 2 * arctan(radius / distance)

        Args:
            distance: Distance to target.

        Returns:
            Angle threshold in radians.
        """
        if distance < 1e-6:
            return self.angle_threshold  # Fallback for very close targets

        # Calculate angular size of target (half-angle from center to edge)
        # For a circular target: half_angle = arctan(radius / distance)
        half_angle = torch.atan(self.target_radius / distance)

        # Full angular threshold = radius_multiplier * 2 * half_angle
        # This gives us the angle within which we'll shoot
        dynamic_threshold = self.radius_multiplier * 2 * half_angle

        # Clamp to reasonable bounds
        max_threshold = np.deg2rad(45.0)  # Never more than 45 degrees
        min_threshold = np.deg2rad(1.0)  # Never less than 1 degree

        return float(torch.clamp(dynamic_threshold, min_threshold, max_threshold))
