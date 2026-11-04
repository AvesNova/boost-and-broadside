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
    """Actions (*shape, 3) int32 drawn uniformly from the global generator, in the order power,
    stick, shoot."""
    draws = torch.rand(shape + (3,), device=device)
    # scalar factors, not a tensor built from host values: a host copy cannot run inside a
    # CUDA graph capture (an eager rollout's, such as the head-to-head's)
    scaled = [
        draws[..., index] * count for index, count in enumerate((NUM_POWER, NUM_STICK, NUM_SHOOT))
    ]
    return torch.stack(scaled, -1).to(torch.int32)


class Random:
    """Uniform random actions, each ship redrawing with probability `REDRAW` per tick."""

    reads = ()

    def __init__(self, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> torch.Tensor:
        """The held actions (R, S, 3) int32, drawn uniformly."""
        return uniform_actions((views, self.ships_per_team), device)

    def step(self, carry: torch.Tensor, obs: Obs, reset: torch.Tensor):
        """One decision for views (R, ...): each ship keeps its held action or, with probability
        REDRAW, draws a new one. The carry is the held actions."""
        redraw = torch.rand(carry.shape[:-1], device=carry.device) < REDRAW
        drawn = uniform_actions(carry.shape[:-1], carry.device)
        held = torch.where(redraw[..., None], drawn, carry)
        return Action.unpack(held), held, Outputs()


class Chaser:
    """Turn toward and shoot at the nearest visible living enemy (race BENCHMARK 3.1 rules)."""

    reads = ("ships",)

    def __init__(self, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the chaser is stateless."""
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor):
        """Each own ship turns toward and shoots at its nearest visible living enemy."""
        S = self.ships_per_team
        own, enemy = obs.ships[:, :S], obs.ships[:, S:]  # rows are ordered by team_relative
        to_x = enemy[:, None, :, SHIP["x"]] - own[:, :, None, SHIP["x"]]  # (R, own, enemy)
        to_y = enemy[:, None, :, SHIP["y"]] - own[:, :, None, SHIP["y"]]
        target = (enemy[..., SHIP["visible"]] > 0.5) & (enemy[..., SHIP["alive"]] > 0.5)
        squared = torch.where(target[:, None, :], to_x * to_x + to_y * to_y, math.inf)
        pick = torch.nn.functional.one_hot(squared.argmin(-1), squared.shape[-1]).bool()
        # ties go to the lowest row
        to_x, to_y = torch.where(pick, to_x, 0.0).sum(-1), torch.where(pick, to_y, 0.0).sum(-1)
        cos, sin = own[..., SHIP["heading_cos"]], own[..., SHIP["heading_sin"]]
        signed = torch.atan2(cos * to_y - sin * to_x, cos * to_x + sin * to_y)
        bearing, distance = signed.abs(), torch.sqrt(to_x * to_x + to_y * to_y)
        level = sum((bearing >= math.radians(degrees)).int() for degrees in (3.0, 10.0, 30.0))
        stick = torch.where(level == 0, 0, 2 * level - (signed < 0.0).int())
        power = torch.where(distance > 600.0, 1, torch.where(distance < 120.0, 2, 0))
        shoot = ((bearing < math.radians(8.0)) & (distance < 250.0)).int()
        any_target = target.any(-1)[:, None]
        action = Action(
            *(torch.where(any_target, part, 0).to(torch.int32) for part in (power, stick, shoot))
        )
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
    """(R, S, NUM_JOINT) the outer product of the marginals, in `to_joint` order."""
    product = (
        parts.power[..., :, None, None]
        * parts.stick[..., None, :, None]
        * parts.shoot[..., None, None, :]
    )
    return product.flatten(-3)


def distribution(
    obs: Obs, characteristics: Characteristics, parameters: ScriptedParameters
) -> torch.Tensor:
    """(R, S, NUM_JOINT) the scripted agent's joint action probabilities, `to_joint` order."""
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
        """No carry: the scripted agent is memoryless."""
        return None

    def _probability(self) -> torch.Tensor | float:
        """The scripted probability per view, (R, 1) for a tensor."""
        probability = self.scripted_probability
        return probability[:, None] if isinstance(probability, torch.Tensor) else probability

    def action_distribution(self, obs: Obs) -> torch.Tensor:
        """(R, S, NUM_JOINT) the distribution `step` samples from."""
        probability = self._probability()
        if isinstance(probability, torch.Tensor):
            probability = probability[..., None]
        return (
            probability * distribution(obs, self.characteristics, self.parameters)
            + (1.0 - probability) / NUM_JOINT
        )

    def step(self, carry: None, obs: Obs, reset: torch.Tensor):
        """One decision for views (R, ...): an action sampled from each marginal, each ship's
        replaced by a uniform one with probability 1 - scripted_probability (per view when it is a
        tensor)."""
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
