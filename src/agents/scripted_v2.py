from contextlib import closing
from dataclasses import dataclass
from turtle import distance
from typing import Any, NamedTuple
import torch
import numpy as np
import torch.nn as nn

from env import ship
from env.constants import PowerActions, TurnActions, ShootActions


class ShipAction(NamedTuple):
    power: int
    turn: int
    shoot: int


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

    def __post_init__(self):
        mask = self.alive.bool()
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value[mask]


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


class ScriptedAgentV2(nn.Module):
    """
    Agent that controls n ships using scripted behavior.

    This agent delegates control of each ship to a ShipController, which
    implements the actual behavioral logic (targeting, movement, shooting).
    """

    def __init__(
        self,
        world_size: tuple[float, float] = (1200.0, 800.0),
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
        return actions

    def calculate_blackboard(self, ships: ShipsState) -> Blackboard:
        relative_position = self.get_relative_position(ships.position)

        distance = torch.abs(relative_position)

        relative_lead_position = self.get_lead_position(
            relative_position, distance, ships
        )

        direction = relative_lead_position / (distance + 1e-8)

        angle_from_nose = self.get_angle_from_nose(relative_lead_position, ships)

        enemies = ships.team_id[:, None] != ships.team_id[None, :]

        return Blackboard(
            relative_position=relative_position,
            relative_lead_position=relative_lead_position,
            distance=distance,
            direction=direction,
            angle_from_nose=angle_from_nose,
            enemies=enemies,
        )

    def get_relative_position(self, position: torch.Tensor) -> torch.Tensor:
        """s
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
        lead_position = relative_position + ships.velocity * time_to_target
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
    ) -> torch.Tensor:
        target_id = self.choose_target(blackboard, ship_id)
        engagement = Engagement(
            distance=blackboard.distance[ship_id, target_id].item(),
            closing_speed=torch.norm(
                ships.velocity[ship_id] - ships.velocity[target_id]
            ).item(),
            our_speed=ships.speed[ship_id].item(),
            our_angle_from_nose=blackboard.angle_from_nose[ship_id, target_id].item(),
            our_power=ships.power[ship_id].item(),
            enemy_speed=ships.speed[target_id].item(),
            enemy_angle_from_nose=blackboard.angle_from_nose[target_id, ship_id].item(),
            enemy_power=ships.power[target_id].item(),
        )
        if (
            blackboard.distance[ship_id, target_id]
            > self.ship_params.max_engage_distance
        ):
            return self.disengaged_action(engagement)

        ship_can_shoot = abs(
            blackboard.angle_from_nose[ship_id, target_id]
        ) < torch.deg2rad(15.0)

        enemy_can_shoot = abs(
            blackboard.angle_from_nose[target_id, ship_id]
        ) < torch.deg2rad(15.0)

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

    def choose_target(self, blackboard: Blackboard, ship_id: int):

        distance = blackboard.distance
        angle = torch.abs(blackboard.angle_from_nose)
        enemy = blackboard.enemies

        attack = (
            self.ship_params.offensive_distance_weight * distance
            + self.ship_params.offensive_angle_weight * angle
        )

        deffense = (
            self.ship_params.defensive_distance_weight * distance
            + self.ship_params.defensive_angle_weight * angle
        )

        combined = torch.where(enemy, torch.min(attack, deffense), torch.inf)
        return torch.argmin(combined[ship_id]).item()

    def get_boost_action(
        self,
        engagement: Engagement,
        target_speed: float | None,
        target_distance: float | None,
    ) -> int:
        if target_speed is not None:
            variable = engagement.our_speed
        if target_distance is not None:
            variable = engagement.distance
        else:
            raise ValueError("Either target_speed or target_distance must be provided")

        if variable > target_speed * (1 + self.ship_params.dead_zone_fraction):
            return PowerActions.BRAKE

        if (
            variable < target_speed * (1 - self.ship_params.dead_zone_fraction)
            and engagement.our_power >= self.ship_params.min_power_for_boost
        ):
            return PowerActions.BOOST

        return PowerActions.COAST

    def get_turn_action(
        self,
        engagement: Engagement,
        normal_turn_threshold: float,  # in radians
        sharp_turn_threshold: float,  # in radians
    ) -> int:

        if engagement.our_angle_from_nose > sharp_turn_threshold:
            return TurnActions.SHARP_RIGHT
        elif engagement.our_angle_from_nose > normal_turn_threshold:
            return TurnActions.TURN_RIGHT
        elif engagement.our_angle_from_nose < -sharp_turn_threshold:
            return TurnActions.SHARP_LEFT
        elif engagement.our_angle_from_nose < -normal_turn_threshold:
            return TurnActions.TURN_LEFT
        else:
            return TurnActions.GO_STRAIGHT

    def get_shoot_action(self, engagement: Engagement) -> int:
        if (
            abs(engagement.our_angle_from_nose) < torch.deg2rad(15.0)
            and engagement.distance < self.ship_params.max_engage_distance
        ):
            return ShootActions.SHOOT
        else:
            return ShootActions.NO_SHOOT

    def disengaged_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.disengage_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            normal_turn_threshold=torch.deg2rad(5.0 / 2),
            sharp_turn_threshold=torch.inf,
        )

        # Shoot logic
        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )

    def head_on_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.head_on_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            normal_turn_threshold=torch.deg2rad(5.0 / 2),
            sharp_turn_threshold=torch.deg2rad((5.0 + 15.0) / 2),
        )

        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )

    def defensive_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_distance=self.ship_params.defensive_distance
        )

        if engagement.our_angle_from_nose > 0:
            if distance < self.ship_params.defensive_distance:
                turn_action = TurnActions.SHARP_RIGHT
            else:
                turn_action = TurnActions.TURN_RIGHT
        else:
            if distance < self.ship_params.defensive_distance:
                turn_action = TurnActions.SHARP_LEFT
            else:
                turn_action = TurnActions.TURN_LEFT

        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )

    def offensive_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_distance=self.ship_params.offensive_distance
        )

        turn_action = self.get_turn_action(
            engagement,
            normal_turn_threshold=torch.deg2rad(5.0 / 2),
            sharp_turn_threshold=torch.deg2rad((5.0 + 15.0) / 2),
        )

        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )

    def two_circle_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.two_circle_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            normal_turn_threshold=0.0,
            sharp_turn_threshold=torch.inf,
        )

        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )

    def one_circle_action(self, engagement: Engagement) -> torch.Tensor:
        boost_action = self.get_boost_action(
            engagement, target_speed=self.ship_params.one_circle_speed
        )

        turn_action = self.get_turn_action(
            engagement,
            normal_turn_threshold=torch.inf,
            sharp_turn_threshold=0.0,
        )

        shoot_action = self.get_shoot_action(engagement)

        return torch.tensor(
            [boost_action, turn_action, shoot_action], dtype=torch.int64
        )
