import torch
import torch.nn as nn
import numpy as np
from ..env.constants import Actions


class ScriptedAgent(nn.Module):
    """
    Advanced scripted agent with predictive targeting and team awareness.

    Features:
    - Calculates bullet travel time based on distance
    - Predicts enemy position when bullet arrives
    - Uses predicted position for both turning and shooting
    - Only targets ships from opposing teams (no friendly fire)
    - Targets the closest enemy ship when multiple enemies are present

    This agent works in both 1v1 and team-based scenarios (e.g., 2v2).
    """

    def __init__(
        self,
        controlled_ship_id: int = 1,
        max_shooting_range: float = 400.0,
        angle_threshold: float = 6.0,
        bullet_speed: float = 500.0,
        target_radius: float = 10.0,
        radius_multiplier: float = 1.5,
        world_size: tuple[float, float] = (1200, 800),
    ):
        """
        Initialize the scripted agent.

        Args:
            controlled_ship_id: Which ship this agent controls (any ship ID)
            max_shooting_range: Maximum distance to shoot at enemies (in world units)
            angle_threshold: Fallback angle tolerance in degrees for shooting (used at max range)
            bullet_speed: Speed of bullets (used for travel time calculation)
            target_radius: Collision radius of enemy ships (in world units)
            radius_multiplier: Multiplier for angular size calculation (1.5 = shoot within 1.5 radii)
            world_size: Size of the world (width, height)
        """
        super().__init__()
        self.controlled_ship_id = controlled_ship_id
        self.max_shooting_range = max_shooting_range
        self.angle_threshold = np.deg2rad(angle_threshold)  # Convert to radians
        self.bullet_speed = bullet_speed
        self.target_radius = target_radius
        self.radius_multiplier = radius_multiplier
        self.world_size = world_size

        # Register as buffer so it moves with device but isn't a parameter
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, observation: dict) -> torch.Tensor:
        """
        Generate action based on observation.

        Args:
            observation: Dictionary containing tokens for all ships

        Returns:
            action: Tensor of shape (6,) with binary actions [forward, backward, left, right, sharp_turn, shoot]
        """
        # Get the raw observation data from the debug fields
        # We'll use the non-normalized values for easier calculations

        # Extract our ship's state and find closest enemy
        our_ship_data = None
        our_team_id = None
        enemy_candidates = []

        # Look through all ships to find ourselves and potential enemies
        for ship_id in range(observation["ship_id"].shape[0]):
            ship_alive = observation["alive"][ship_id, 0].item()
            if not ship_alive:
                continue

            current_ship_id = observation["ship_id"][ship_id, 0].item()
            current_team_id = observation["team_id"][ship_id, 0].item()

            if current_ship_id == self.controlled_ship_id:
                our_ship_data = {
                    "health": observation["health"][ship_id, 0].item(),
                    "power": observation["power"][ship_id, 0].item(),
                    "position": observation["position"][ship_id, 0],
                    "velocity": observation["velocity"][ship_id, 0],
                    "attitude": observation["attitude"][ship_id, 0],
                }
                our_team_id = current_team_id
            else:
                # Store all potential enemies (ships not on our team)
                enemy_candidates.append(
                    {
                        "ship_id": current_ship_id,
                        "team_id": current_team_id,
                        "position": observation["position"][ship_id, 0],
                        "velocity": observation["velocity"][ship_id, 0],
                    }
                )

        # Find closest enemy ship (only target ships from different team)
        enemy_ship_data = None
        if our_ship_data is not None and our_team_id is not None and enemy_candidates:
            our_pos = torch.tensor(
                [our_ship_data["position"].real, our_ship_data["position"].imag],
                device=self._dummy.device,
            )

            closest_distance = float("inf")
            for candidate in enemy_candidates:
                # Skip allied ships (same team)
                if candidate["team_id"] == our_team_id:
                    continue

                candidate_pos = torch.tensor(
                    [candidate["position"].real, candidate["position"].imag],
                    device=self._dummy.device,
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

        # Initialize action (all zeros)
        action = torch.zeros(
            len(Actions), dtype=torch.float32, device=self._dummy.device
        )

        # Check if we have valid data for both our ship and an enemy
        if our_ship_data is None or enemy_ship_data is None:
            return action  # Can't act without proper data

        # Convert complex numbers to 2D vectors for easier calculation
        self_pos = torch.tensor(
            [our_ship_data["position"].real, our_ship_data["position"].imag],
            device=self._dummy.device,
        )

        enemy_pos = torch.tensor(
            [enemy_ship_data["position"].real, enemy_ship_data["position"].imag],
            device=self._dummy.device,
        )

        enemy_velocity = torch.tensor(
            [enemy_ship_data["velocity"].real, enemy_ship_data["velocity"].imag],
            device=self._dummy.device,
        )

        self_attitude = torch.tensor(
            [our_ship_data["attitude"].real, our_ship_data["attitude"].imag],
            device=self._dummy.device,
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
        # If cross product is positive, target is to the left; if negative, to the right
        cross_product = (
            self_attitude[0] * to_target_normalized[1]
            - self_attitude[1] * to_target_normalized[0]
        )

        # Decide on turning action (aim at predicted position)
        # Use small fixed threshold to prevent jitter
        if angle_to_target > self.angle_threshold:
            if cross_product > 0:
                # Target is to the left (counter-clockwise), turn right to face them
                action[Actions.right] = 1.0
            else:
                # Target is to the right (clockwise), turn left to face them
                action[Actions.left] = 1.0

            # Use sharp turn for large angle differences
            if angle_to_target > np.deg2rad(15.0):
                action[Actions.sharp_turn] = 1.0

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
            action[Actions.shoot] = 1.0

        # Thrust management based on distance and power
        close_range_threshold = (
            2.0 * self.target_radius
        )  # 2 radii = 20 units by default

        if current_distance <= close_range_threshold:
            # Close range: use reverse thrust to maintain distance, regardless of power level
            action[Actions.backward] = 1.0
        else:
            # Normal range: only boost (forward) if power > 90%, otherwise maintain base thrust
            if current_power_ratio > 0.9 * (
                1.0 - current_distance / self.max_shooting_range
            ):
                action[Actions.forward] = 1.0
            # Note: When power <= 90%, we don't set forward=1, so ship uses base thrust only

        return action

    def _calculate_wrapped_vector(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the shortest vector from pos1 to pos2 in a toroidal world.

        Args:
            pos1: Position 1 [x, y]
            pos2: Position 2 [x, y]

        Returns:
            Vector from pos1 to pos2 considering wrapping
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
            self_pos: Our current position [x, y]
            enemy_pos: Enemy's current position [x, y]
            enemy_velocity: Enemy's velocity [vx, vy] (already in world units)

        Returns:
            Predicted enemy position [x, y] accounting for toroidal wrapping
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
            distance: Distance to target

        Returns:
            Angle threshold in radians
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


def create_scripted_agent(**kwargs) -> ScriptedAgent:
    """
    Factory function to create a scripted agent.

    Args:
        **kwargs: Arguments to pass to ScriptedAgent constructor

    Returns:
        ScriptedAgent instance
    """
    return ScriptedAgent(**kwargs)
