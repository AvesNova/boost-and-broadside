"""Scripted controllers over a team view: random (sticky) and the chaser.

Both read only the observation, never the game state.
"""

import math

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import SHIP

REDRAW = 0.1  # sticky random: per-tick probability of a new action


def _uniform(shape: tuple, device) -> torch.Tensor:
    u = torch.rand(shape + (3,), device=device)
    scale = torch.tensor([NUM_POWER, NUM_STICK, NUM_SHOOT], dtype=torch.float32, device=device)
    return (u * scale).to(torch.int32)


class Random:
    """Uniform random actions, each ship redrawing with probability 0.1 per tick."""

    reads = ()

    def __init__(self, ships_per_team: int) -> None:
        self.S = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> torch.Tensor:
        return _uniform((views, self.S), device)

    def step(self, carry: torch.Tensor, obs: Obs, done: torch.Tensor):
        redraw = torch.rand(carry.shape[:-1], device=carry.device) < REDRAW
        new = torch.where(redraw[..., None], _uniform(carry.shape[:-1], carry.device), carry)
        return Action.unpack(new), new, Outputs()


class Chaser:
    """Turn toward and shoot at the nearest visible living enemy (race BENCHMARK 3.1 rules)."""

    reads = ("ships",)

    def __init__(self, ships_per_team: int) -> None:
        self.S = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def step(self, carry: None, obs: Obs, done: torch.Tensor):
        S = self.S
        own, foe = obs.ships[:, :S], obs.ships[:, S:]  # rows are ordered by team_rel
        tx = foe[:, None, :, SHIP["x"]] - own[:, :, None, SHIP["x"]]  # (R, own, foe)
        ty = foe[:, None, :, SHIP["y"]] - own[:, :, None, SHIP["y"]]
        target = (foe[..., SHIP["visible"]] > 0.5) & (foe[..., SHIP["alive"]] > 0.5)
        d2 = torch.where(target[:, None, :], tx * tx + ty * ty, math.inf)
        pick = torch.nn.functional.one_hot(d2.argmin(-1), d2.shape[-1]).bool()  # ties: lowest row
        tx, ty = torch.where(pick, tx, 0.0).sum(-1), torch.where(pick, ty, 0.0).sum(-1)
        c, s = own[..., SHIP["c"]], own[..., SHIP["s"]]
        signed = torch.atan2(c * ty - s * tx, c * tx + s * ty)
        bearing, dist = signed.abs(), torch.sqrt(tx * tx + ty * ty)
        level = sum((bearing >= math.radians(deg)).int() for deg in (3.0, 10.0, 30.0))
        stick = torch.where(level == 0, 0, 2 * level - (signed < 0.0).int())
        power = torch.where(dist > 600.0, 1, torch.where(dist < 120.0, 2, 0))
        shoot = ((bearing < math.radians(8.0)) & (dist < 250.0)).int()
        has = target.any(-1)[:, None]
        action = Action(*(torch.where(has, a, 0).to(torch.int32) for a in (power, stick, shoot)))
        return action, carry, Outputs()
