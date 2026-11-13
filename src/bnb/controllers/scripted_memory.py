"""The scripted agent's memory (its carry): per own ship the plan choices it has committed
to, and per enemy the last seen position and velocity, from which a hidden enemy is dead
reckoned for a short time.

A plan choice (`commit`) keeps its committed option and switches to the best alternative
with the probability of its `Switch` over one decision; it re-chooses outright (the
memoryless choice) when the ship is fresh (after an episode reset, its death or its spawn)
or the committed option is gone. With memory off, every ship is fresh at every decision and
only visible enemies are known, which is the memoryless agent: the same code path.
"""

import dataclasses
import math
from typing import NamedTuple

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.contracts.tree import contract
from bnb.game.handling.characteristics import Characteristics
from bnb.game.ship_types import per_ship

from .scripted_parameters import ScriptedParameters, Switch, logistic, switch_probability

# A remembered enemy is forgotten once its weight as a target falls below this: its
# prediction is then worth nothing to aim or fire at.
FORGET_WEIGHT = 0.01


@contract
class ScriptedCarry:
    """Per view (R leading); own ships (R, S), enemies (R, E).

    target (R, S) int64: the committed target's enemy row, -1 for none. combat_left and
    frontline_left (R, S) bool: each layer's committed turn side. defending, recovering (R,
    S) bool: the Frontline goal and recovery. fresh (R, S) bool: the ship's plan choices are
    unset (it re-chooses each outright). agent (R, S) bool: the ship plays the agent (else
    uniformly: a mixture rung's share). sense_draw (R, S): the uniform drawn for the
    sense's choice when its gate opened, negative while it is shut.
    enemy_x, enemy_y, enemy_velocity_x,
    enemy_velocity_y (R, E): each enemy's last seen state; known (R, E) bool: remembered."""

    target: torch.Tensor
    combat_left: torch.Tensor
    frontline_left: torch.Tensor
    defending: torch.Tensor
    recovering: torch.Tensor
    fresh: torch.Tensor
    agent: torch.Tensor
    sense_draw: torch.Tensor
    enemy_x: torch.Tensor
    enemy_y: torch.Tensor
    enemy_velocity_x: torch.Tensor
    enemy_velocity_y: torch.Tensor
    known: torch.Tensor


def fresh_carry(views: int, ships: int, enemies: int, device) -> ScriptedCarry:
    """A carry with nothing committed or remembered, every ship playing the agent."""

    def flag(count: int, value: bool) -> torch.Tensor:
        return torch.full((views, count), value, dtype=torch.bool, device=device)

    def zero() -> torch.Tensor:
        return torch.zeros((views, enemies), device=device)

    return ScriptedCarry(
        target=torch.full((views, ships), -1, dtype=torch.int64, device=device),
        combat_left=flag(ships, False),
        frontline_left=flag(ships, False),
        defending=flag(ships, False),
        recovering=flag(ships, False),
        fresh=flag(ships, True),
        agent=flag(ships, True),
        sense_draw=torch.full((views, ships), -1.0, device=device),
        enemy_x=zero(),
        enemy_y=zero(),
        enemy_velocity_x=zero(),
        enemy_velocity_y=zero(),
        known=flag(enemies, False),
    )


def forgotten(carry: ScriptedCarry) -> ScriptedCarry:
    """The carry with every plan choice unset and every enemy forgotten (memory off); the
    ships' agent flags kept."""
    return dataclasses.replace(
        carry, fresh=torch.ones_like(carry.fresh), known=torch.zeros_like(carry.known)
    )


def time_unit(
    characteristics: Characteristics, own_type: torch.Tensor, parameters: ScriptedParameters
) -> torch.Tensor:
    """Each own ship's time unit (R, S), in seconds: its corner turn time (the corner speed
    over the g limit's acceleration: the time a full-stick turn at corner speed takes to
    turn the path one radian) over the card's first type's (the type each set was tuned
    on), raised to time_unit_exponent. At the exponent 0 it is exactly 1 (seconds)."""
    turn_time = characteristics.corner_speed / characteristics.max_acceleration
    relative = per_ship(turn_time / turn_time[..., :1], own_type)
    return torch.pow(relative, parameters.time_unit_exponent)


class Recall(NamedTuple):
    """Each enemy (R, E) as the agent believes it: its position and velocity (seen, or dead
    reckoned), its weight as a target (1 seen, fading with the time since seen, 0
    unknown) and whether it is known."""

    x: torch.Tensor
    y: torch.Tensor
    velocity_x: torch.Tensor
    velocity_y: torch.Tensor
    weight: torch.Tensor
    known: torch.Tensor


def remember(
    carry: ScriptedCarry,
    obs: Obs,
    reset: torch.Tensor,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    temperature,
) -> tuple[ScriptedCarry, Recall]:
    """The carry after this observation (the plan choices still to make): fresh ships
    (reset, dead or just spawned) and the enemy memory, and what it recalls.

    An enemy seen now is stored; a hidden one is dead reckoned, its last seen position plus
    its last seen velocity times the time since seen, held inside the arena, and weighted by
    the fade logistic in that time. A hidden enemy is forgotten when it dies, when its weight
    falls below FORGET_WEIGHT, or when its prediction falls inside an own living ship's
    vision (it was wrong: it would be seen there)."""
    ships = obs.ships
    S = ships.shape[-2] // 2
    own, enemy = ships[:, :S], ships[:, S:]
    reset_views = reset[:, None]
    own_alive = own[..., SHIP["alive"]] > 0.5
    fresh = carry.fresh | reset_views | ~own_alive | (own[..., SHIP["spawned"]] > 0.5)
    enemy_alive = enemy[..., SHIP["alive"]] > 0.5
    seen = (enemy[..., SHIP["visible"]] > 0.5) & enemy_alive
    known = carry.known & ~reset_views

    def stored(name: str, last: torch.Tensor) -> torch.Tensor:
        return torch.where(seen, enemy[..., SHIP[name]], last)

    last_x, last_y = stored("x", carry.enemy_x), stored("y", carry.enemy_y)
    velocity_x = stored("velocity_x", carry.enemy_velocity_x)
    velocity_y = stored("velocity_y", carry.enemy_velocity_y)
    since = torch.where(seen, 0.0, enemy[..., SHIP["time_since_seen"]])
    # the team's memory fades in its ships' own time: their mean unit (every ship's own,
    # where the team flies one type)
    own_type = own[..., SHIP["ship_type"]].to(torch.int32)
    unit = time_unit(characteristics, own_type, parameters).mean(-1, keepdim=True)
    x, y = last_x + velocity_x * since, last_y + velocity_y * since
    arena_radius = obs.globals[:, None, GLOBAL["arena_radius"]]
    radius = torch.sqrt(x * x + y * y).clamp_min(1e-6)
    inside = (arena_radius / radius).clamp_max(1.0)
    x = torch.where(seen, last_x, x * inside)
    y = torch.where(seen, last_y, y * inside)
    weight = torch.where(
        seen, 1.0, logistic(since / unit, parameters.memory_fade_in_seconds, temperature)
    )
    # a prediction inside an own living ship's vision, where it would be seen, was wrong
    vision = per_ship(characteristics.vision_range, own_type)
    offset_x = x[:, None, :] - own[..., SHIP["x"]][..., None]  # (R, S, E)
    offset_y = y[:, None, :] - own[..., SHIP["y"]][..., None]
    in_view = (
        (offset_x * offset_x + offset_y * offset_y < (vision * vision)[..., None])
        & own_alive[..., None]
    ).any(1)
    known = seen | (known & enemy_alive & ~in_view & (weight >= FORGET_WEIGHT))
    weight = torch.where(known, weight, 0.0)
    carry = dataclasses.replace(
        carry,
        fresh=fresh,
        enemy_x=last_x,
        enemy_y=last_y,
        enemy_velocity_x=velocity_x,
        enemy_velocity_y=velocity_y,
        known=known,
    )
    return carry, Recall(x, y, velocity_x, velocity_y, weight, known)


def commit(
    committed: torch.Tensor,
    best: torch.Tensor,
    advantage: torch.Tensor,
    forced: torch.Tensor,
    shape: Switch,
    temperature,
    seconds: float,
    switching: bool,
) -> torch.Tensor:
    """A plan choice after one decision: `best` where `forced`, else `best` with the switch's
    probability at `advantage` (best's over the committed option's), else `committed`.
    Without `switching` (re-reading a carry that has made its choices) only `forced`
    changes it."""
    if not switching:
        return torch.where(forced, best, committed)
    probability = switch_probability(advantage, shape, temperature, seconds)
    switch = torch.rand_like(probability) < probability
    return torch.where(forced | switch, best, committed)


def ways_round(error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The left and right turns (radians, in [0, 2 pi]) to an angle error (positive
    left)."""
    left = torch.where(error >= 0.0, error, error + 2.0 * math.pi)
    return left, 2.0 * math.pi - left
