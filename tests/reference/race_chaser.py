"""The race chaser (BENCHMARK.md 3.1) on the full state, kept only to check the goldens."""

import math

import torch

from bnb.game.rules import Rules
from bnb.game.state import State, by_team


def enemy(values: torch.Tensor) -> torch.Tensor:
    return by_team(values).flip(1)


def chaser(rules: Rules, state: State) -> torch.Tensor:
    """Deterministic chaser (BENCHMARK.md 3.1): turn toward and shoot at the nearest living enemy."""
    ships = state.ships
    half = rules.ships_per_team
    to_x = enemy(ships.x)[..., None, :] - by_team(ships.x)[..., :, None]  # (B, 2, self, enemy)
    to_y = enemy(ships.y)[..., None, :] - by_team(ships.y)[..., :, None]
    distance_squared = torch.where(enemy(ships.alive)[..., None, :], to_x * to_x + to_y * to_y, math.inf)
    nearest = (distance_squared == distance_squared.amin(-1, keepdim=True)) & (distance_squared < math.inf)
    index = torch.arange(half, device=distance_squared.device)
    first = torch.where(nearest, index, half).amin(-1, keepdim=True)  # ties to the lowest index
    pick = index == first
    to_x, to_y = (torch.where(pick, to_x, 0.0).sum(-1), torch.where(pick, to_y, 0.0).sum(-1))
    cos, sin = by_team(ships.heading_cos), by_team(ships.heading_sin)
    signed = torch.atan2(cos * to_y - sin * to_x, cos * to_x + sin * to_y)
    bearing, left = signed.abs(), signed >= 0.0
    distance = torch.sqrt(to_x * to_x + to_y * to_y)
    level = ((bearing >= math.radians(3.0)).int() + (bearing >= math.radians(10.0)).int()
             + (bearing >= math.radians(30.0)).int())
    stick = torch.where(level == 0, 0, 2 * level - (~left).int())
    power = torch.where(distance > 600.0, 1, torch.where(distance < 120.0, 2, 0))
    shoot = ((bearing < math.radians(8.0)) & (distance < 250.0)).int()
    action = torch.stack([power, stick, shoot], -1).to(torch.int32)
    has_target = first[..., 0] < half
    return torch.where(has_target[..., None], action, 0).reshape(ships.x.shape + (3,))
