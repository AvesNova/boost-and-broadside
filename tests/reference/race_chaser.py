"""The race chaser (BENCHMARK.md 3.1) on the full state, kept only to check the goldens."""

import math

import torch

from bnb.game.physics.constants import Preset
from bnb.game.physics.state import by_team, enemy


def chaser(p: Preset, st: dict) -> torch.Tensor:
    """Deterministic chaser (BENCHMARK.md 3.1): turn toward and shoot at the nearest living enemy."""
    tx = enemy(st["x"])[..., None, :] - by_team(st["x"])[..., :, None]  # (B, 2, self, enemy)
    ty = enemy(st["y"])[..., None, :] - by_team(st["y"])[..., :, None]
    d2 = torch.where(enemy(st["alive"])[..., None, :], tx * tx + ty * ty, math.inf)
    nearest = (d2 == d2.amin(-1, keepdim=True)) & (d2 < math.inf)
    index = torch.arange(p.half, device=d2.device)
    first = torch.where(nearest, index, p.half).amin(-1, keepdim=True)  # ties to the lowest index
    pick = index == first
    tx, ty = (torch.where(pick, tx, 0.0).sum(-1), torch.where(pick, ty, 0.0).sum(-1))
    c, s = by_team(st["c"]), by_team(st["s"])
    bearing = torch.atan2(c * ty - s * tx, c * tx + s * ty).abs()
    left = torch.atan2(c * ty - s * tx, c * tx + s * ty) >= 0.0
    dist = torch.sqrt(tx * tx + ty * ty)
    level = ((bearing >= math.radians(3.0)).int() + (bearing >= math.radians(10.0)).int()
             + (bearing >= math.radians(30.0)).int())
    stick = torch.where(level == 0, 0, 2 * level - (~left).int())
    power = torch.where(dist > 600.0, 1, torch.where(dist < 120.0, 2, 0))
    shoot = ((bearing < math.radians(8.0)) & (dist < 250.0)).int()
    act = torch.stack([power, stick, shoot], -1).to(torch.int32)
    has_target = first[..., 0] < p.half
    return torch.where(has_target[..., None], act, 0).reshape(st["x"].shape + (3,))
