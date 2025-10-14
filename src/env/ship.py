from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import Actions
from .bullets import Bullets


@dataclass
class ShipConfig:
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
    forward: int
    backward: int
    left: int
    right: int
    sharp_turn: int
    shoot: int


class Ship(nn.Module):
    def __init__(
        self,
        ship_id: int,
        team_id: int,
        ship_config: ShipConfig,
        initial_x: float,
        initial_y: float,
        initial_vx: float,
        initial_vy: float,
        world_size: tuple[float, float] = (-1, -1),
        rng: np.random.Generator = np.random.default_rng(),
    ):
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
        # Indexed by [left, right, sharp] -> turn offset
        self.turn_offset_table = np.zeros((2, 2, 2), dtype=np.float32)

        self.turn_offset_table[0, 0, 0] = 0  # None
        self.turn_offset_table[0, 1, 0] = ship_config.normal_turn_angle  # R
        self.turn_offset_table[1, 0, 0] = -ship_config.normal_turn_angle  # L
        self.turn_offset_table[1, 1, 0] = 0  # LR
        self.turn_offset_table[0, 0, 1] = 0  # S
        self.turn_offset_table[0, 1, 1] = ship_config.sharp_turn_angle  # SR
        self.turn_offset_table[1, 0, 1] = -ship_config.sharp_turn_angle  # SL
        self.turn_offset_table[1, 1, 1] = 0  # SLR

        # Indexed by [forward, backward] -> thrust
        self.thrust_table = np.zeros((2, 2), dtype=np.float32)
        self.thrust_table[0, 0] = ship_config.base_thrust  # Neither
        self.thrust_table[1, 0] = ship_config.boost_thrust  # Forward only
        self.thrust_table[0, 1] = ship_config.reverse_thrust  # Backward only
        self.thrust_table[1, 1] = ship_config.base_thrust  # Both -> cancel out to base

        # Indexed by [forward, backward] -> energy cost
        self.energy_cost_table = np.zeros((2, 2), dtype=np.float32)
        self.energy_cost_table[0, 0] = ship_config.base_power_gain  # Neither
        self.energy_cost_table[1, 0] = ship_config.boost_power_gain  # Forward only
        self.energy_cost_table[0, 1] = ship_config.reverse_power_gain  # Backward only
        self.energy_cost_table[1, 1] = ship_config.base_power_gain  # Both -> base

        # Indexed by [turning, sharp] -> drag coefficient
        self.drag_coeff_table = np.zeros((2, 2), dtype=np.float32)
        self.drag_coeff_table[0, 0] = ship_config.no_turn_drag_coeff  # No turn
        self.drag_coeff_table[0, 1] = ship_config.no_turn_drag_coeff  # No turn
        self.drag_coeff_table[1, 0] = ship_config.normal_turn_drag_coeff  # Normal turn
        self.drag_coeff_table[1, 1] = ship_config.sharp_turn_drag_coeff  # Sharp turn

        # Indexed by [left, right, sharp] -> lift coefficient
        self.lift_coeff_table = np.zeros((2, 2, 2), dtype=np.float32)
        self.lift_coeff_table[0, 0, 0] = 0.0  # None
        self.lift_coeff_table[0, 0, 1] = 0.0  # S
        self.lift_coeff_table[0, 1, 0] = self.config.normal_turn_lift_coeff  # R
        self.lift_coeff_table[0, 1, 1] = self.config.sharp_turn_lift_coeff  # SR
        self.lift_coeff_table[1, 0, 0] = -self.config.normal_turn_lift_coeff  # L
        self.lift_coeff_table[1, 0, 1] = -self.config.sharp_turn_lift_coeff  # SL
        self.lift_coeff_table[1, 1, 0] = 0.0  # LR
        self.lift_coeff_table[1, 1, 1] = 0.0  # SLR

    def _extract_action_states(self, actions: torch.Tensor) -> ActionStates:
        # Handle empty action tensor
        if actions.numel() == 0:
            return ActionStates(
                left=0,
                right=0,
                forward=0,
                backward=0,
                sharp_turn=0,
                shoot=0,
            )

        # Ensure actions has at least 6 elements
        if actions.numel() < 6:
            # Pad with zeros if needed
            actions = F.pad(actions, (0, 6 - actions.numel()))

        return ActionStates(
            left=int(actions[Actions.left]),
            right=int(actions[Actions.right]),
            forward=int(actions[Actions.forward]),
            backward=int(actions[Actions.backward]),
            sharp_turn=int(actions[Actions.sharp_turn]),
            shoot=int(actions[Actions.shoot]),
        )

    def _update_power(self, actions: ActionStates, delta_t: float) -> None:
        energy_cost = self.energy_cost_table[actions.forward, actions.backward]
        self.power += energy_cost * delta_t
        self.power = max(0.0, min(self.power, self.config.max_power))

    def _update_attitude(self, actions: ActionStates) -> None:
        if not (actions.left and actions.right):
            self.turn_offset = self.turn_offset_table[
                actions.left, actions.right, actions.sharp_turn
            ]
        self.attitude = self.velocity / self.speed * np.exp(1j * self.turn_offset)

    def _calculate_forces(self, actions: ActionStates) -> complex:
        if self.power > 0:
            thrust = self.thrust_table[actions.forward, actions.backward]
            thrust_force = thrust * self.attitude
        else:
            thrust_force = 0 + 0j

        turning = int(actions.left or actions.right)
        drag_coeff = self.drag_coeff_table[turning, actions.sharp_turn]
        drag_force = -drag_coeff * self.speed * self.velocity

        lift_coeff = self.lift_coeff_table[
            actions.left, actions.right, actions.sharp_turn
        ]
        lift_vector = self.velocity * 1j  # 90 degrees counter-clockwise
        lift_force = lift_coeff * self.speed * lift_vector

        total_force = thrust_force + drag_force + lift_force
        return total_force

    def _update_kinematics(self, actions: ActionStates, delta_t: float) -> None:
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
        if (
            actions.shoot
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
        self.health -= damage
        if self.health <= 0:
            self.alive = False

    def get_state(self) -> dict:
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
            "token": self.get_token(),
        }

    def get_token(self) -> torch.Tensor:
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
        return int(np.ceil(self.config.bullet_lifetime / self.config.firing_cooldown))
