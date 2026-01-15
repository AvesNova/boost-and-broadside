from dataclasses import dataclass
from typing import Any, NamedTuple
import torch
import numpy as np
import torch.nn as nn

from env import ship
from env.constants import PowerActions, TurnActions, ShootActions


class ShipAction(NamedTuple):
    boost_action: int
    turn_action: int
    shoot_action: int


@dataclass
class Engagement:
    distance: float
    closing_speed: float

    our_speed: float
    our_angle_from_nose: float
    our_power: float

    enemy_speed: float
    enemy_angle_from_nose: float
    enemy_power: float


@dataclass
class Blackboard:
    relative_position: torch.Tensor  # 2D complex
    relative_lead_position: torch.Tensor  # 2D complex
    distance: torch.Tensor  # 2D float
    direction: torch.Tensor  # 2D complex
    angle_from_nose: torch.Tensor  # 2D float
    enemies: torch.Tensor  # 2D bool
    both_alive: torch.Tensor  # 2D bool


@dataclass
class ShipsState:
    ship_id: torch.Tensor  # shape: [num_ships], dtype: torch.int64
    team_id: torch.Tensor  # shape: [num_ships], dtype: torch.int64
    alive: torch.Tensor  # shape: [num_ships], dtype: torch.int64
    health: torch.Tensor  # shape: [num_ships], dtype: torch.int64
    power: torch.Tensor  # shape: [num_ships], dtype: torch.float32
    position: torch.Tensor  # shape: [num_ships], dtype: torch.complex64
    velocity: torch.Tensor  # shape: [num_ships], dtype: torch.complex64
    speed: torch.Tensor  # shape: [num_ships], dtype: torch.float32
    attitude: torch.Tensor  # shape: [num_ships], dtype: torch.complex64
    is_shooting: torch.Tensor  # shape: [num_ships], dtype: torch.int64

    # def __post_init__(self):
    #     mask = self.alive.bool()
    #     for key, value in self.__dict__.items():
    #         if isinstance(value, torch.Tensor):
    #             self.__dict__[key] = value[mask]

    #     self.ship_id_to_tensor_id = {ship_id: i for i, ship_id in enumerate(self.ship_id)}


@dataclass
class ShipParameters:
    # target selection weights
    offensive_distance_weight: float = 1.0
    offensive_angle_weight: float = 500.0
    defensive_distance_weight: float = 1.0
    defensive_angle_weight: float = 500.0

    # general engagement parameters
    max_engage_distance: float = 300.0
    dead_zone_fraction: float = 0.1
    min_power_for_boost: float = 10.0

    # speed targets
    disengage_speed: float = 120.0
    head_on_speed: float = 120.0
    one_circle_speed: float = 50.0
    two_circle_speed: float = 100.0

    # distance targets
    offensive_distance: float = 50.0
    defensive_distance: float = 50.0

    # turning thresholds
    shoot_angle: float = np.deg2rad(15.0)
    normal_turn_angle: float = np.deg2rad(2.5)
    sharp_turn_angle: float = np.deg2rad(10.0)


class ScriptedAgentV2(nn.Module):
    """
    Agent that controls n ships using scripted behavior.

    This agent delegates control of each ship to a ShipController, which
    implements the actual behavioral logic (targeting, movement, shooting).
    """

    def __init__(
        self,
        world_size: tuple[float, float] = (1024.0, 1024.0),
        rng: np.random.Generator = np.random.default_rng(),
        **kwargs: Any,
    ):
        super().__init__()
        self.world_size = world_size
        self.rng = rng
        self.ship_controller = ShipController()

    def forward(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        return self.get_actions(obs_dict, ship_ids)

    def get_actions(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        ships = ShipsState(**obs_dict)
        blackboard = self.calculate_blackboard(ships)
        actions = {
            ship_id: self.ship_controller.get_actions(
                ships=ships,
                blackboard=blackboard,
                ship_id=ship_id,
                team_ids=ship_ids,
            )
            for ship_id in ship_ids
        }

        # convert from v2 actions (categorical) to v1 actions (binary)
        # [forward, backward, left, right, sharp_turn, shoot]
        binary_actions = {}
        for ship_id, action in actions.items():
            act_tensor = torch.zeros(6, dtype=torch.float32)

            # Power
            if action.boost_action == PowerActions.BOOST:
                act_tensor[0] = 1.0  # Forward
            elif action.boost_action == PowerActions.BRAKE:
                act_tensor[1] = 1.0  # Backward

            # Turn
            if action.turn_action == TurnActions.TURN_LEFT:
                act_tensor[2] = 1.0  # Left
            elif action.turn_action == TurnActions.TURN_RIGHT:
                act_tensor[3] = 1.0  # Right
            elif action.turn_action == TurnActions.SHARP_LEFT:
                act_tensor[2] = 1.0  # Left
                act_tensor[4] = 1.0  # Sharp
            elif action.turn_action == TurnActions.SHARP_RIGHT:
                act_tensor[3] = 1.0  # Right
                act_tensor[4] = 1.0  # Sharp

            # Shoot
            if action.shoot_action == ShootActions.SHOOT:
                act_tensor[5] = 1.0  # Shoot

            binary_actions[ship_id] = act_tensor

        return binary_actions

    def calculate_blackboard(self, ships: ShipsState) -> Blackboard:
        relative_position = self.get_relative_position(ships.position)

        distance = torch.abs(relative_position)

        relative_lead_position = self.get_lead_position(
            relative_position, distance, ships
        )

        direction = relative_lead_position / (distance + 1e-8)

        angle_from_nose = self.get_angle_from_nose(relative_lead_position, ships)

        enemies = ships.team_id[:, None] != ships.team_id[None, :]
        both_alive = (ships.alive[:, None] & ships.alive[None, :]).bool()

        return Blackboard(
            relative_position=relative_position,
            relative_lead_position=relative_lead_position,
            distance=distance,
            direction=direction,
            angle_from_nose=angle_from_nose,
            enemies=enemies,
            both_alive=both_alive,
        )

    def get_relative_position(self, position: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positions between all pairs of ships on a 2D toroidal world.
        """
        dx = position[None, :].real - position[:, None].real
        dy = position[None, :].imag - position[:, None].imag

        width, height = self.world_size
        dx = (dx + width / 2) % width - width / 2
        dy = (dy + height / 2) % height - height / 2

        return dx + 1j * dy

    def get_lead_position(
        self,
        relative_position: torch.Tensor,
        distance: torch.Tensor,
        ships: ShipsState,
    ) -> torch.Tensor:
        base_bullet_speed = 500.0
        bullet_speed = base_bullet_speed + ships.speed
        time_to_target = distance / bullet_speed
        lead_position = relative_position + ships.velocity[None, :] * time_to_target
        return lead_position

    def get_angle_from_nose(
        self,
        relative_position: torch.Tensor,
        ships: ShipsState,
    ) -> torch.Tensor:
        velocity_dir = ships.velocity / (ships.speed + 1e-8)
        angle_from_nose = torch.angle(relative_position * velocity_dir.conj())
        return angle_from_nose


class ShipController(nn.Module):
    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__()
        self.rng = rng
        self.ship_params = ShipParameters()

    def get_actions(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        team_ids: list[int],
    ) -> ShipAction:
        target_id = self.choose_target(blackboard, ship_id)
        engagement = Engagement(
            distance=blackboard.distance[ship_id, target_id],
            closing_speed=self.get_closing_speed(ships, blackboard, ship_id, target_id),
            our_speed=ships.speed[ship_id],
            our_angle_from_nose=blackboard.angle_from_nose[ship_id, target_id],
            our_power=ships.power[ship_id],
            enemy_speed=ships.speed[target_id],
            enemy_angle_from_nose=blackboard.angle_from_nose[target_id, ship_id],
            enemy_power=ships.power[target_id],
        )
        if (
            blackboard.distance[ship_id, target_id]
            > self.ship_params.max_engage_distance
        ):
            return self.disengaged_action(engagement)

        ship_can_shoot = (
            abs(blackboard.angle_from_nose[ship_id, target_id])
            < self.ship_params.shoot_angle
        )

        enemy_can_shoot = (
            abs(blackboard.angle_from_nose[target_id, ship_id])
            < self.ship_params.shoot_angle
        )

        if ship_can_shoot and enemy_can_shoot:
            return self.head_on_action(engagement)

        if enemy_can_shoot:
            return self.defensive_action(engagement)

        if ship_can_shoot:
            return self.offensive_action(engagement)

        if torch.sign(blackboard.angle_from_nose[ship_id, target_id]) == torch.sign(
            blackboard.angle_from_nose[target_id, ship_id]
        ):
            return self.two_circle_action(engagement)

        return self.one_circle_action(engagement)

    def get_closing_speed(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> float:
        rel_vel = ships.velocity[target_id] - ships.velocity[ship_id]
        rel_pos = blackboard.relative_position[ship_id, target_id]
        closing_speed = -torch.real(rel_vel * rel_pos.conj()) / (abs(rel_pos) + 1e-8)

        return closing_speed

    def choose_target(self, blackboard: Blackboard, ship_id: int):

        distance = blackboard.distance
        angle = torch.abs(blackboard.angle_from_nose)
        enemy = blackboard.enemies
        both_alive = blackboard.both_alive

        attack = torch.where(
            (enemy & both_alive)[ship_id, :],
            self.ship_params.offensive_distance_weight * distance[ship_id, :]
            + self.ship_params.offensive_angle_weight * angle[ship_id, :],
            torch.inf,
        )

        deffense = torch.where(
            (enemy & both_alive)[:, ship_id],
            self.ship_params.defensive_distance_weight * distance[:, ship_id]
            + self.ship_params.defensive_angle_weight * angle[:, ship_id],
            torch.inf,
        )

        return torch.argmin(torch.min(attack, deffense))

    def get_boost_action(
        self,
        engagement: Engagement,
        *,
        target_speed: float | None = None,
        target_distance: float | None = None,
    ) -> int:

        if (target_speed is None) == (target_distance is None):
            raise ValueError("Provide exactly one of target_speed or target_distance")

        if target_speed is not None:
            value = engagement.our_speed
            target = target_speed
        else:
            value = engagement.distance
            target = target_distance

        if value > target * (1 + self.ship_params.dead_zone_fraction):
            return PowerActions.BRAKE

        if (
            value < target * (1 - self.ship_params.dead_zone_fraction)
            and engagement.our_power >= self.ship_params.min_power_for_boost
        ):
            return PowerActions.BOOST

        return PowerActions.COAST

    def get_turn_action(
        self,
        engagement: Engagement,
        *,
        allow_normal: bool = False,
        allow_sharp: bool = False,
        normal_turn_threshold: float = 0.0,  # in radians
        sharp_turn_threshold: float = 0.0,  # in radians
    ) -> int:
        if allow_sharp:
            if engagement.our_angle_from_nose > sharp_turn_threshold:
                return TurnActions.SHARP_RIGHT
            if engagement.our_angle_from_nose < -sharp_turn_threshold:
                return TurnActions.SHARP_LEFT

        if allow_normal:
            if engagement.our_angle_from_nose > normal_turn_threshold:
                return TurnActions.TURN_RIGHT
            if engagement.our_angle_from_nose < -normal_turn_threshold:
                return TurnActions.TURN_LEFT

        return TurnActions.GO_STRAIGHT

    def get_shoot_action(self, engagement: Engagement) -> int:
        if (
            abs(engagement.our_angle_from_nose) < self.ship_params.shoot_angle
            and engagement.distance < self.ship_params.max_engage_distance
        ):
            return ShootActions.SHOOT
        else:
            return ShootActions.NO_SHOOT

    def disengaged_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.disengage_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            allow_normal=True,
            normal_turn_threshold=self.ship_params.normal_turn_angle,
        )

        # Shoot logic
        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )

    def head_on_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.head_on_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            allow_normal=True,
            allow_sharp=True,
            normal_turn_threshold=self.ship_params.normal_turn_angle,
            sharp_turn_threshold=self.ship_params.sharp_turn_angle,
        )

        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )

    def defensive_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_distance=self.ship_params.defensive_distance
        )

        if engagement.our_angle_from_nose > 0:
            if engagement.distance < self.ship_params.defensive_distance:
                turn_action = TurnActions.SHARP_RIGHT
            else:
                turn_action = TurnActions.TURN_RIGHT
        else:
            if engagement.distance < self.ship_params.defensive_distance:
                turn_action = TurnActions.SHARP_LEFT
            else:
                turn_action = TurnActions.TURN_LEFT

        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )

    def offensive_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_distance=self.ship_params.offensive_distance
        )

        turn_action = self.get_turn_action(
            engagement,
            allow_normal=True,
            allow_sharp=True,
            normal_turn_threshold=self.ship_params.normal_turn_angle,
            sharp_turn_threshold=self.ship_params.sharp_turn_angle,
        )

        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )

    def two_circle_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.two_circle_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            allow_normal=True,
        )

        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )

    def one_circle_action(self, engagement: Engagement) -> ShipAction:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.one_circle_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            allow_sharp=True,
        )

        shoot_action = self.get_shoot_action(engagement)

        return ShipAction(
            boost_action=boost_action,
            turn_action=turn_action,
            shoot_action=shoot_action,
        )
