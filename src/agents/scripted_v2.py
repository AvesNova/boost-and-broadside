from dataclasses import dataclass
from typing import Any
import torch
import numpy as np
import torch.nn as nn

from env import ship
from env.constants import PowerActions, TurnActions, ShootActions


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
        # Remove dead ships
        alive_mask = self.alive.bool()
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field)[alive_mask])


@dataclass
class ShipParameters:
    offensive_distance_weight: float = 1.0
    offensive_angle_weight: float = 500.0
    defensive_distance_weight: float = 1.0
    defensive_angle_weight: float = 500.0

    max_engage_distance: float = 300.0
    dead_zone_fraction: float = 0.1
    min_power_for_boost: float = 10.0

    disengage_speed: float = 120.0
    head_on_speed: float = 120.0
    offensive_distance: float = 50.0
    two_circle_speed: float = 100.0
    one_circle_speed: float = 50.0


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
        angle_from_nose = torch.angle(
            direction * torch.conj(torch.exp(-1j * ships.velocity[:, None]))
        )
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
        """
        Compute relative positions between all pairs of ships on a 2D toroidal world.
        """
        # Convert complex64 positions into real (x, y) coordinates
        position = torch.view_as_real(position)  # [num_ships, 2]

        world_width, world_height = self.world_size
        world_dimensions = torch.tensor(
            [world_width, world_height],
            dtype=position.dtype,
            device=position.device,
        )

        delta = position[None, :, :] - position[:, None, :]  # [num_ships, num_ships, 2]

        # Wrap using centered modulo to place deltas into:
        #   [-world_width/2, +world_width/2)
        #   [-world_height/2, +world_height/2)
        delta = (delta + world_dimensions / 2) % world_dimensions - world_dimensions / 2

        # Convert back to complex
        relative_position = torch.view_as_complex(
            delta.contiguous()
        )  # [num_ships, num_ships]

        return relative_position

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

    def get_manuever_actions(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> dict[str, torch.Tensor]:
        if (
            blackboard.distance[ship_id, target_id]
            > self.ship_params.max_engage_distance
        ):
            return self.disengaged_action(ships, blackboard, ship_id, target_id)

        ship_can_shoot = abs(
            blackboard.angle_from_nose[ship_id, target_id]
        ) < torch.deg2rad(15.0)
        enemy_can_shoot = abs(
            blackboard.angle_from_nose[target_id, ship_id]
        ) < torch.deg2rad(15.0)

        if ship_can_shoot and enemy_can_shoot:
            return self.head_on_action(ships, blackboard, ship_id, target_id)

        if enemy_can_shoot:
            return self.defensive_action(ships, blackboard, ship_id, target_id)

        if ship_can_shoot:
            return self.offensive_action(ships, blackboard, ship_id, target_id)

        if torch.sign(blackboard.angle_from_nose[ship_id, target_id]) == torch.sign(
            blackboard.angle_from_nose[target_id, ship_id]
        ):
            return self.two_circle_action(ships, blackboard, ship_id, target_id)

        return self.one_circle_action(ships, blackboard, ship_id, target_id)

    def choose_target(self, blackboard: Blackboard, ship_id: int) -> int:
        ship_to_enemy_scores = torch.where(
            blackboard.enemies[ship_id],
            self.ship_params.offensive_distance_weight * blackboard.distance[ship_id]
            + self.ship_params.offensive_angle_weight
            * torch.abs(blackboard.angle_from_nose[ship_id]),
            torch.inf,
        )

        enemy_to_ship_scores = torch.where(
            blackboard.enemies[:, ship_id],
            self.ship_params.defensive_distance_weight * blackboard.distance[:, ship_id]
            + self.ship_params.defensive_angle_weight
            * torch.abs(blackboard.angle_from_nose[:, ship_id]),
            torch.inf,
        )

        combined_scores = torch.min(ship_to_enemy_scores, enemy_to_ship_scores)

        return torch.argmin(combined_scores).item()

    def disengaged_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        power = ships.power[ship_id]
        speed = ships.speed[ship_id]

        if power < self.ship_params.min_power_for_boost:
            boost_action = 

    def head_on_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        pass

    def defensive_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        pass

    def offensive_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        pass

    def two_circle_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        pass

    def one_circle_action(
        self,
        ships: ShipsState,
        blackboard: Blackboard,
        ship_id: int,
        target_id: int,
    ) -> torch.Tensor:
        pass
