"""Scripted controllers over a team view: random (sticky), the chaser, and the scripted
agent (scripted spec A), which is also the stationary ladder's rungs and the BC teacher.

All read only the observation, never the game state.
"""

import math

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import NUM_JOINT, NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.handling.characteristics import Characteristics

from .scripted_combat import Marginals, combat, context
from .scripted_frontline import frontline
from .scripted_parameters import ScriptedParameters

REDRAW = 0.1  # sticky random: per-tick probability of a new action


def uniform_actions(shape: tuple, device) -> torch.Tensor:
    draws = torch.rand(shape + (3,), device=device)
    scale = torch.tensor([NUM_POWER, NUM_STICK, NUM_SHOOT], dtype=torch.float32, device=device)
    return (draws * scale).to(torch.int32)


class Random:
    """Uniform random actions, each ship redrawing with probability 0.1 per tick."""

    reads = ()

    def __init__(self, ships_per_team: int) -> None:
        self.S = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> torch.Tensor:
        return uniform_actions((views, self.S), device)

    def step(self, carry: torch.Tensor, obs: Obs, reset: torch.Tensor):
        redraw = torch.rand(carry.shape[:-1], device=carry.device) < REDRAW
        new = torch.where(redraw[..., None], uniform_actions(carry.shape[:-1], carry.device), carry)
        return Action.unpack(new), new, Outputs()


class Chaser:
    """Turn toward and shoot at the nearest visible living enemy (race BENCHMARK 3.1 rules)."""

    reads = ("ships",)

    def __init__(self, ships_per_team: int) -> None:
        self.S = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor):
        S = self.S
        own, foe = obs.ships[:, :S], obs.ships[:, S:]  # rows are ordered by team_rel
        tx = foe[:, None, :, SHIP["x"]] - own[:, :, None, SHIP["x"]]  # (R, own, foe)
        ty = foe[:, None, :, SHIP["y"]] - own[:, :, None, SHIP["y"]]
        target = (foe[..., SHIP["visible"]] > 0.5) & (foe[..., SHIP["alive"]] > 0.5)
        d2 = torch.where(target[:, None, :], tx * tx + ty * ty, math.inf)
        pick = torch.nn.functional.one_hot(d2.argmin(-1), d2.shape[-1]).bool()  # ties: lowest row
        tx, ty = torch.where(pick, tx, 0.0).sum(-1), torch.where(pick, ty, 0.0).sum(-1)
        c, s = own[..., SHIP["heading_cos"]], own[..., SHIP["heading_sin"]]
        signed = torch.atan2(c * ty - s * tx, c * tx + s * ty)
        bearing, dist = signed.abs(), torch.sqrt(tx * tx + ty * ty)
        level = sum((bearing >= math.radians(deg)).int() for deg in (3.0, 10.0, 30.0))
        stick = torch.where(level == 0, 0, 2 * level - (signed < 0.0).int())
        power = torch.where(dist > 600.0, 1, torch.where(dist < 120.0, 2, 0))
        shoot = ((bearing < math.radians(8.0)) & (dist < 250.0)).int()
        has = target.any(-1)[:, None]
        action = Action(*(torch.where(has, a, 0).to(torch.int32) for a in (power, stick, shoot)))
        return action, carry, Outputs()


def marginals(
    obs: Obs, characteristics: Characteristics, parameters: ScriptedParameters
) -> Marginals:
    """The scripted agent's power, stick and shoot marginals per own ship, from the
    observation alone: the combat layer, with the Frontline layer on top in Frontline. A
    dead ship (or one waiting to respawn) takes the no-op action."""
    fleet = context(obs, characteristics, parameters)
    fighting, has_target, distance = combat(fleet, characteristics, parameters)
    strategy = frontline(
        fleet, obs.zones, characteristics, parameters, fighting, has_target, distance
    )
    in_frontline = (obs.globals[:, None, GLOBAL["mode_frontline"]] > 0.5)[..., None]
    alive = (fleet.own[..., SHIP["alive"]] > 0.5)[..., None]

    def chosen(combat_part: torch.Tensor, frontline_part: torch.Tensor) -> torch.Tensor:
        part = torch.where(in_frontline, frontline_part, combat_part)
        no_op = torch.nn.functional.one_hot(
            torch.zeros_like(part[..., 0], dtype=torch.long), part.shape[-1]
        ).float()
        return torch.where(alive, part, no_op)

    return Marginals(*(chosen(a, b) for a, b in zip(fighting, strategy, strict=True)))


def joint(parts: Marginals) -> torch.Tensor:
    """(R, S, 42) the outer product of the marginals, in `to_joint` order."""
    product = (
        parts.power[..., :, None, None]
        * parts.stick[..., None, :, None]
        * parts.shoot[..., None, None, :]
    )
    return product.flatten(-3)


def distribution(
    obs: Obs, characteristics: Characteristics, parameters: ScriptedParameters
) -> torch.Tensor:
    """(R, S, 42) the scripted agent's joint action probabilities, `to_joint` order."""
    return joint(marginals(obs, characteristics, parameters))


def sample(probabilities: torch.Tensor) -> torch.Tensor:
    """One index per row of (..., K) probabilities, by the inverse CDF (a category of
    probability 0 is never drawn)."""
    cumulative = probabilities.cumsum(-1)
    draw = torch.rand(probabilities.shape[:-1] + (1,), device=probabilities.device)
    return (draw * cumulative[..., -1:] >= cumulative[..., :-1]).sum(-1)


class Scripted:
    """The scripted agent, and with it every rung of the stationary ladder (spec A6).

    scripted_probability: a float, or a tensor (R,) per view: each ship plays the scripted
    action with that probability and a uniform one otherwise (0 is random, 1 the scripted
    agent), so its distribution is the scripted probability times the scripted
    distribution plus the rest spread uniformly over the 42 joint actions. The league may
    change a tensor in place. parameters: one set, or per view (`scripted_parameters.unpack`
    of a table). Memoryless: the carry is None.
    """

    reads = ("ships", "zones", "globals")

    def __init__(
        self,
        characteristics: Characteristics,
        parameters: ScriptedParameters = ScriptedParameters(),
        scripted_probability: float | torch.Tensor = 1.0,
    ) -> None:
        self.characteristics, self.parameters = characteristics, parameters
        self.scripted_probability = scripted_probability

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def _probability(self) -> torch.Tensor | float:
        """The scripted probability per view, (R, 1) for a tensor."""
        probability = self.scripted_probability
        return probability[:, None] if isinstance(probability, torch.Tensor) else probability

    def action_distribution(self, obs: Obs) -> torch.Tensor:
        """(R, S, 42) the distribution `step` samples from."""
        probability = self._probability()
        if isinstance(probability, torch.Tensor):
            probability = probability[..., None]
        return (
            probability * distribution(obs, self.characteristics, self.parameters)
            + (1.0 - probability) / NUM_JOINT
        )

    def step(self, carry: None, obs: Obs, reset: torch.Tensor):
        parts = marginals(obs, self.characteristics, self.parameters)
        action = torch.stack([sample(part) for part in parts], -1).to(torch.int32)
        probability = self._probability()
        if isinstance(probability, torch.Tensor) or probability < 1.0:
            # a rung: one uniform draw for every view
            scripted = torch.rand(action.shape[:-1], device=action.device) < probability
            action = torch.where(
                scripted[..., None], action, uniform_actions(action.shape[:-1], action.device)
            )
        return Action.unpack(action), carry, Outputs()
