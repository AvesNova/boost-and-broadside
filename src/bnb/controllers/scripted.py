"""Scripted controllers over a team view that are not Imugi: random (sticky) and the chaser,
with the helpers they share with Imugi (`bnb.controllers.imugi`).

All read only the observation, never the game state.
"""

import math

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import SHIP
from bnb.game.handling.characteristics import Characteristics
from bnb.streams import action_uniforms, purpose_key, row_uniforms

REDRAW = 0.1  # sticky random: per-tick probability of a new action
# The standard card's lengths from its characteristics at 30 Hz, to the last bit
# (`gun_range` and `corner_radius`): its longest gun range (the fighter's) and its largest
# corner radius (the heavy's: the corner speed squared over the g limit's acceleration, the
# radius of a full-stick turn at the corner speed). Distances below are stated in these, so
# the standard card plays exactly the distances in metres they were set in, and any other
# card the same distances in its own lengths.
STANDARD_GUN_RANGE = 220.78790283203125  # m
STANDARD_CORNER_RADIUS = 45.45454775076505  # m


RANDOM_REDRAW = purpose_key("random redraw")
FLOOR_DRAWS = purpose_key("uniform floor")


def uniform_actions(uniforms: torch.Tensor) -> torch.Tensor:
    """Actions (..., 3) int32 drawn uniformly by uniforms (..., 3) in [0, 1), in the order
    power, stick, shoot."""
    # scalar factors, not a tensor built from host values: a host copy cannot run inside a
    # CUDA graph capture (an eager rollout's, such as the head-to-head's)
    scaled = [
        uniforms[..., index] * count
        for index, count in enumerate((NUM_POWER, NUM_STICK, NUM_SHOOT))
    ]
    return torch.stack(scaled, -1).to(torch.int32)


def floored_actions(action: torch.Tensor, floor: float | torch.Tensor, decision_key: torch.Tensor):
    """The packed actions (R, S, 3) with each ship's replaced, with probability `floor` (a
    float, or (R,) per row), by a uniform joint action. The choice is keyed by the decision's
    key (R,) apart from the actions' uniforms, so the mixture is the floor's whatever the
    agent's action; the uniform action is drawn by the actions' uniforms
    (`bnb.streams.action_uniforms`)."""
    floor = floor[:, None] if isinstance(floor, torch.Tensor) else floor
    ships = action.shape[1]
    chosen = row_uniforms(FLOOR_DRAWS, decision_key, (ships,)) < floor
    uniform = uniform_actions(action_uniforms(decision_key, ships))
    return torch.where(chosen[..., None], uniform, action)


class Random:
    """Uniform random actions, each ship redrawing with probability `REDRAW` per tick."""

    reads = ()

    def __init__(self, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> torch.Tensor:
        """The held actions (R, S, 3) int32: none yet (-1), so the first decision draws."""
        return torch.full((views, self.ships_per_team, 3), -1, dtype=torch.int32, device=device)

    def step(self, carry: torch.Tensor, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for views (R, ...): each ship keeps its held action or, with probability
        REDRAW (always without one), draws a new one. The carry is the held actions."""
        redraw = row_uniforms(RANDOM_REDRAW, decision_key, carry.shape[1:-1]) < REDRAW
        redraw = redraw | (carry[..., 0] < 0)
        drawn = uniform_actions(action_uniforms(decision_key, carry.shape[1]))
        held = torch.where(redraw[..., None], drawn, carry)
        return Action.unpack(held), held, Outputs()


def gun_range(characteristics: Characteristics) -> float:
    """The card's longest gun range (m): the length the agents' gun distances are stated in."""
    return float(characteristics.gun_range.amax())


def corner_radius(characteristics: Characteristics) -> float:
    """The card's largest corner radius (m), corner speed^2 / a_max in float64: the length
    the chaser's flight distances are stated in."""
    speed = characteristics.corner_speed.double()
    return float((speed * speed / characteristics.max_acceleration.double()).amax())


# The chaser's distances (race BENCHMARK 3.1: boost beyond 600 m, reverse inside 120 m and
# shoot inside 250 m, on the standard card): the flight's in corner radii, the gun's in gun
# ranges.
BOOST_IN_CORNER_RADII = 600.0 / STANDARD_CORNER_RADIUS
REVERSE_IN_CORNER_RADII = 120.0 / STANDARD_CORNER_RADIUS
SHOOT_IN_GUN_RANGES = 250.0 / STANDARD_GUN_RANGE
FIRE_BEARING = math.radians(8.0)  # the chaser shoots inside this bearing to its target
STICK_BEARINGS = tuple(math.radians(degrees) for degrees in (3.0, 10.0, 30.0))  # per level


def stick_toward(signed: torch.Tensor) -> torch.Tensor:
    """The stick action (int64, same shape) that turns the nose toward a bearing (rad, positive
    to the left): straight inside the first level's bearing, else the level by the bearing's
    size, odd indices to the right and even to the left (`Action` order)."""
    level = sum((signed.abs() >= bearing).int() for bearing in STICK_BEARINGS)
    return torch.where(level == 0, 0, 2 * level - (signed < 0.0).int())


class Chaser:
    """Turn toward and shoot at the nearest visible living enemy (race BENCHMARK 3.1 rules),
    at distances in the card's lengths (`BOOST_IN_CORNER_RADII`, `REVERSE_IN_CORNER_RADII`,
    `SHOOT_IN_GUN_RANGES`): the race's 600, 120 and 250 m on the standard card."""

    reads = ("ships",)

    def __init__(self, characteristics: Characteristics, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team
        self.retune(characteristics)

    def retune(self, characteristics: Characteristics) -> None:
        """Play at the distances of this card's characteristics."""
        radius, reach = corner_radius(characteristics), gun_range(characteristics)
        self.boost_distance = BOOST_IN_CORNER_RADII * radius  # m
        self.reverse_distance = REVERSE_IN_CORNER_RADII * radius  # m
        self.shoot_distance = SHOOT_IN_GUN_RANGES * reach  # m

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the chaser is stateless."""
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """Each own ship turns toward and shoots at its nearest visible living enemy (no
        draw)."""
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
        stick = stick_toward(signed)
        power = torch.where(
            distance > self.boost_distance, 1, torch.where(distance < self.reverse_distance, 2, 0)
        )
        shoot = ((bearing < FIRE_BEARING) & (distance < self.shoot_distance)).int()
        any_target = target.any(-1)[:, None]
        action = Action(
            *(torch.where(any_target, part, 0).to(torch.int32) for part in (power, stick, shoot))
        )
        return action, carry, Outputs()
