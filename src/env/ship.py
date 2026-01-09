from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.constants import PowerActions, TurnActions, ShootActions
from env.bullets import Bullets


@dataclass
class ShipConfig:
    """Configuration parameters for ship physics and capabilities."""

    # Physical Parameters
    collision_radius: float = 10.0
    max_health: float = 100.0
    max_power: float = 100.0

    # Thrust System Parameters
    base_thrust: float = 8.0
    boost_thrust: float = 80.0
    reverse_thrust: float = -10.0
    base_power_gain: float = 10.0
    boost_power_gain: float = -40.0
    reverse_power_gain: float = 20.0

    # Aerodynamic Parameters
    no_turn_drag_coeff: float = 8e-4
    normal_turn_angle: float = np.deg2rad(5.0)
    normal_turn_drag_coeff: float = 1.2e-3
    normal_turn_lift_coeff: float = 15e-3
    sharp_turn_angle: float = np.deg2rad(15.0)
    sharp_turn_drag_coeff: float = 5.0e-3
    sharp_turn_lift_coeff: float = 27e-3

    # Bullet System Parameters
    bullet_speed: float = 500.0
    bullet_energy_cost: float = 3.0
    bullet_damage: float = 10.0
    bullet_lifetime: float = 1.0
    bullet_spread: float = 12.0
    firing_cooldown: float = 0.1


default_ship_config = ShipConfig()


@dataclass
class ActionStates:
    """Parsed action states from the action vector."""

    power: int
    turn: int
    shoot: int


class Ship(nn.Module):
    """
    Represents a single ship in the environment.

    Handles physics, movement, shooting, and state management for a ship.
    Inherits from nn.Module to be compatible with PyTorch-based logic if needed,
    though primarily acts as a physics entity.
    """

    def __init__(
        self,
        ship_id: int,
        team_id: int,
        ship_config: ShipConfig,
        initial_x: float,
        initial_y: float,
        initial_vx: float,
        initial_vy: float,
        world_size: tuple[float, float] = (-1.0, -1.0),
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Initialize the ship.

        Args:
            ship_id: Unique identifier for the ship.
            team_id: Team identifier (0 or 1).
            ship_config: Configuration object for ship parameters.
            initial_x: Initial X position.
            initial_y: Initial Y position.
            initial_vx: Initial X velocity.
            initial_vy: Initial Y velocity.
            world_size: Dimensions of the game world (width, height).
            rng: Random number generator.
        """
        super().__init__()
        self.ship_id = ship_id
        self.team_id = team_id
        self.config = ship_config
        self.window_size = world_size
        self.rng = rng
        self.collision_radius_squared = ship_config.collision_radius**2

        self.alive = True
        self.health = ship_config.max_health
        self.power = ship_config.max_power
        self.turn_offset = 0.0
        self.last_fired_time = (
            -ship_config.firing_cooldown
        )  # Allow immediate first shot
        self.is_shooting = False  # Initialize shooting state

        self.position = initial_x + 1j * initial_y
        self.velocity = initial_vx + 1j * initial_vy
        self.speed = abs(self.velocity)
        assert self.speed > 1e-6, "Initial velocity cannot be too small"
        self.attitude = self.velocity / self.speed

        self._build_lookup_tables(ship_config)

    def _build_lookup_tables(self, ship_config: ShipConfig) -> None:
        """Pre-compute lookup tables for physics calculations to avoid conditionals."""

        # Power Actions: COAST (0), BOOST (1), REVERSE (2)
        self.thrust_table = np.array(
            [
                ship_config.base_thrust,  # COAST
                ship_config.boost_thrust,  # BOOST
                ship_config.reverse_thrust,  # REVERSE
            ],
            dtype=np.float32,
        )

        self.energy_cost_table = np.array(
            [
                ship_config.base_power_gain,  # COAST
                ship_config.boost_power_gain,  # BOOST
                ship_config.reverse_power_gain,  # REVERSE
            ],
            dtype=np.float32,
        )

        # Turn Actions:
        # GO_STRAIGHT (0)
        # TURN_LEFT (1), TURN_RIGHT (2)
        # SHARP_LEFT (3), SHARP_RIGHT (4)
        # AIR_BRAKE (5), SHARP_AIR_BRAKE (6)

        # Turn Offsets
        self.turn_offset_table = np.zeros(7, dtype=np.float32)
        self.turn_offset_table[TurnActions.GO_STRAIGHT] = 0.0
        self.turn_offset_table[TurnActions.TURN_LEFT] = -ship_config.normal_turn_angle
        self.turn_offset_table[TurnActions.TURN_RIGHT] = ship_config.normal_turn_angle
        self.turn_offset_table[TurnActions.SHARP_LEFT] = -ship_config.sharp_turn_angle
        self.turn_offset_table[TurnActions.SHARP_RIGHT] = ship_config.sharp_turn_angle
        self.turn_offset_table[TurnActions.AIR_BRAKE] = 0.0
        self.turn_offset_table[TurnActions.SHARP_AIR_BRAKE] = 0.0

        # Drag Coefficients
        self.drag_coeff_table = np.zeros(7, dtype=np.float32)
        self.drag_coeff_table[TurnActions.GO_STRAIGHT] = ship_config.no_turn_drag_coeff
        self.drag_coeff_table[TurnActions.TURN_LEFT] = (
            ship_config.normal_turn_drag_coeff
        )
        self.drag_coeff_table[TurnActions.TURN_RIGHT] = (
            ship_config.normal_turn_drag_coeff
        )
        self.drag_coeff_table[TurnActions.SHARP_LEFT] = (
            ship_config.sharp_turn_drag_coeff
        )
        self.drag_coeff_table[TurnActions.SHARP_RIGHT] = (
            ship_config.sharp_turn_drag_coeff
        )
        self.drag_coeff_table[TurnActions.AIR_BRAKE] = (
            ship_config.normal_turn_drag_coeff
        )
        self.drag_coeff_table[TurnActions.SHARP_AIR_BRAKE] = (
            ship_config.sharp_turn_drag_coeff
        )

        # Lift Coefficients
        self.lift_coeff_table = np.zeros(7, dtype=np.float32)
        self.lift_coeff_table[TurnActions.GO_STRAIGHT] = 0.0
        self.lift_coeff_table[TurnActions.TURN_LEFT] = (
            -ship_config.normal_turn_lift_coeff
        )
        self.lift_coeff_table[TurnActions.TURN_RIGHT] = (
            ship_config.normal_turn_lift_coeff
        )
        self.lift_coeff_table[TurnActions.SHARP_LEFT] = (
            -ship_config.sharp_turn_lift_coeff
        )
        self.lift_coeff_table[TurnActions.SHARP_RIGHT] = (
            ship_config.sharp_turn_lift_coeff
        )
        self.lift_coeff_table[TurnActions.AIR_BRAKE] = 0.0
        self.lift_coeff_table[TurnActions.SHARP_AIR_BRAKE] = 0.0

    def _extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        """
        Convert raw action tensor to ActionStates dataclass.

        Args:
            actions: Tensor of shape (3,) containing indices for [Power, Turn, Shoot].
        """
        # Handle empty or invalid action tensor
        if actions.numel() != 3:
            # Fallback to default/noop
            return ActionStates(
                power=PowerActions.COAST,
                turn=TurnActions.GO_STRAIGHT,
                shoot=ShootActions.NO_SHOOT,
            )

        return ActionStates(
            power=int(actions[0]),
            turn=int(actions[1]),
            shoot=int(actions[2]),
        )

    def _update_power(self, actions: ActionStates, delta_t: float) -> None:
        """Update ship power based on thrust actions."""
        energy_cost = self.energy_cost_table[actions.power]
        self.power += energy_cost * delta_t
        self.power = max(0.0, min(self.power, self.config.max_power))

    def _update_attitude(self, actions: ActionStates) -> None:
        """Update ship attitude (orientation) based on turn actions."""
        self.turn_offset = self.turn_offset_table[actions.turn]
        self.attitude = self.velocity / self.speed * np.exp(1j * self.turn_offset)

    def _calculate_forces(self, actions: ActionStates) -> complex:
        """Calculate total force acting on the ship."""
        if self.power > 0:
            thrust = self.thrust_table[actions.power]
            thrust_force = thrust * self.attitude
        else:
            thrust_force = 0 + 0j

        drag_coeff = self.drag_coeff_table[actions.turn]
        drag_force = -drag_coeff * self.speed * self.velocity

        lift_coeff = self.lift_coeff_table[actions.turn]
        lift_vector = self.velocity * 1j  # 90 degrees counter-clockwise
        lift_force = lift_coeff * self.speed * lift_vector

        total_force = thrust_force + drag_force + lift_force
        return total_force

    def _update_kinematics(self, actions: ActionStates, delta_t: float) -> None:
        """Update position and velocity based on forces."""
        total_force = self._calculate_forces(actions)
        acceleration = total_force  # Assuming mass = 1

        self.velocity += acceleration * delta_t
        self.position += self.velocity * delta_t
        self.speed = abs(self.velocity)
        if self.speed < 1e-6:
            self.speed = 1e-6
            self.velocity = self.speed * self.attitude

    def _shoot_bullet(
        self, actions: ActionStates, bullets: Bullets, current_time: float
    ) -> None:
        """Handle shooting logic."""
        if (
            actions.shoot == ShootActions.SHOOT
            and current_time - self.last_fired_time >= self.config.firing_cooldown
            and self.power >= self.config.bullet_energy_cost
        ):
            self.last_fired_time = current_time
            self.power -= self.config.bullet_energy_cost
            self.is_shooting = True

            bullet_x = self.position.real
            bullet_y = self.position.imag

            bullet_vx = (
                self.velocity.real
                + self.config.bullet_speed * self.attitude.real
                + self.rng.normal(0, self.config.bullet_spread)
            )
            bullet_vy = (
                self.velocity.imag
                + self.config.bullet_speed * self.attitude.imag
                + self.rng.normal(0, self.config.bullet_spread)
            )

            bullets.add_bullet(
                ship_id=self.ship_id,
                x=bullet_x,
                y=bullet_y,
                vx=bullet_vx,
                vy=bullet_vy,
                lifetime=self.config.bullet_lifetime,
            )
        else:
            self.is_shooting = False

    def forward(
        self,
        action_vector: torch.Tensor,
        bullets: Bullets,
        current_time: float,
        delta_t: float,
    ) -> None:
        """
        Advance the ship's state by one time step.

        Args:
            action_vector: Tensor containing actions of shape (3,).
            bullets: Bullet manager instance.
            current_time: Current simulation time.
            delta_t: Time step duration.
        """
        if self.health <= 0:
            self.alive = False
        if not self.alive:
            return

        actions = self._extract_action_states(action_vector)

        self._shoot_bullet(actions, bullets, current_time)
        self._update_attitude(actions)
        self._update_kinematics(actions, delta_t)
        self._update_attitude(actions)
        self._update_power(actions, delta_t)

    def damage_ship(self, damage: float) -> None:
        """Apply damage to the ship."""
        self.health -= damage
        if self.health <= 0:
            self.alive = False

    def get_state(self) -> dict[str, int | float | complex]:
        """Get the current state of the ship as a dictionary."""
        return {
            "ship_id": self.ship_id,
            "team_id": self.team_id,
            "alive": self.alive,
            "health": self.health,
            "power": self.power,
            "position": self.position,
            "velocity": self.velocity,
            "speed": self.speed,
            "attitude": self.attitude,
            "is_shooting": self.is_shooting,
        }

    def get_token(self) -> torch.Tensor:
        """
        Generate a token representation of the ship's state.

        Returns:
            Tensor of shape (10,) containing normalized state features.
        """
        return torch.tensor(
            [
                self.team_id,
                self.health / self.config.max_health,
                self.power / self.config.max_power,
                self.position.real / self.window_size[0],
                self.position.imag / self.window_size[1],
                self.velocity.real / 180.0,
                self.velocity.imag / 180.0,
                self.attitude.real,
                self.attitude.imag,
                self.is_shooting,
            ],
            dtype=torch.float32,
        )

    @property
    def max_bullets(self) -> int:
        """Calculate maximum possible active bullets based on lifetime and cooldown."""
        return int(np.ceil(self.config.bullet_lifetime / self.config.firing_cooldown))
