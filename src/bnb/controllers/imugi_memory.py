"""Imugi's memory (its carry): per own ship the plan choices it has committed
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
from bnb.models.yemong.density import hex_cells

from .imugi_parameters import ImugiParameters, Switch, logistic, switch_probability

# A remembered enemy is forgotten once its weight as a target falls below this: its
# prediction is then worth nothing to aim or fire at.
FORGET_WEIGHT = 0.01
# s: a hidden committed target is pursued (kept known, dead reckoned) this long after it was
# last seen (`remember`, `imugi_combat.choose_target`).
TARGET_LOST_SECONDS = 3.0
# The explore grid (`track_seen`, `imugi_combat.explore_target`): the hex grid of
# EXPLORE_RINGS rings over the arena (cell centres in arena radii, about 1 km apart on the
# 7.8 km map), and the time a seen cell's value takes to fall by e: about 20 s, after which
# an enemy at cruise could have crossed a cell's surroundings several times over (vision is
# 1 km at real scale), so a cell seen empty that long ago says little.
EXPLORE_RINGS = 8
EXPLORE_CELLS = hex_cells(EXPLORE_RINGS)[1]
EXPLORE_SECONDS = 20.0
_EXPLORE_CELLS_ON: dict[torch.device, torch.Tensor] = {}


def explore_cells(device: torch.device) -> torch.Tensor:
    """(C, 2) EXPLORE_CELLS on `device`, copied once per device: a copy inside a step would
    be a host-to-device copy in every tick, which a CUDA graph's capture refuses."""
    if device not in _EXPLORE_CELLS_ON:
        _EXPLORE_CELLS_ON[device] = EXPLORE_CELLS.to(device)
    return _EXPLORE_CELLS_ON[device]


@contract
class ImugiCarry:
    """Per view (R leading); own ships (R, S), enemies (R, E).

    target (R, S) int64: the committed target's enemy row, -1 for none. combat_left and
    frontline_left (R, S) bool: each layer's committed turn side. defending (R, S)
    bool: the Frontline goal. fresh (R, S) bool: the ship's plan choices are
    unset (it re-chooses each outright). agent (R, S) bool: the ship plays the agent (else
    uniformly: a mixture rung's share).
    enemy_x, enemy_y, enemy_velocity_x,
    enemy_velocity_y (R, E): each enemy's last seen state; known (R, E) bool: remembered.
    boundary_radius, boundary_progress and boundary_speed (R,): the boundary's radius (m)
    and the episode progress when last observed (-1 none), and its closing speed (m/s)
    from the last two observations (`track_boundary`). seen (R, C): how recently the team
    has seen each cell of the explore grid (`EXPLORE_CELLS`; 1 in view now, decaying with
    EXPLORE_SECONDS: `track_seen`). target_doubt (R, S): the seconds a reason to leave
    the committed target has held without a break (`imugi_combat.choose_target`).
    extend_phase (R, S): the extend-and-return rule's clock (`imugi_combat.extend_and_return`):
    positive, the seconds of extension left; negative, the seconds before another may
    start; 0, free. fight_sense (R, S): the turning fight's committed rotation sense (+1
    left, -1 right, 0 none; `imugi_combat.turn_commitment`), and fight_clock (R, S) the
    seconds since it was committed."""

    target: torch.Tensor
    combat_left: torch.Tensor
    frontline_left: torch.Tensor
    defending: torch.Tensor
    fresh: torch.Tensor
    agent: torch.Tensor
    enemy_x: torch.Tensor
    enemy_y: torch.Tensor
    enemy_velocity_x: torch.Tensor
    enemy_velocity_y: torch.Tensor
    known: torch.Tensor
    boundary_radius: torch.Tensor
    boundary_progress: torch.Tensor
    boundary_speed: torch.Tensor
    seen: torch.Tensor
    target_doubt: torch.Tensor
    extend_phase: torch.Tensor
    fight_sense: torch.Tensor
    fight_clock: torch.Tensor


def fresh_carry(views: int, ships: int, enemies: int, device) -> ImugiCarry:
    """A carry with nothing committed or remembered, every ship playing the agent."""

    def flag(count: int, value: bool) -> torch.Tensor:
        return torch.full((views, count), value, dtype=torch.bool, device=device)

    def zero() -> torch.Tensor:
        return torch.zeros((views, enemies), device=device)

    return ImugiCarry(
        target=torch.full((views, ships), -1, dtype=torch.int64, device=device),
        combat_left=flag(ships, False),
        frontline_left=flag(ships, False),
        defending=flag(ships, False),
        fresh=flag(ships, True),
        agent=flag(ships, True),
        enemy_x=zero(),
        enemy_y=zero(),
        enemy_velocity_x=zero(),
        enemy_velocity_y=zero(),
        known=flag(enemies, False),
        boundary_radius=torch.zeros(views, device=device),
        boundary_progress=torch.full((views,), -1.0, device=device),
        boundary_speed=torch.zeros(views, device=device),
        seen=torch.zeros((views, EXPLORE_CELLS.shape[0]), device=device),
        target_doubt=torch.zeros((views, ships), device=device),
        extend_phase=torch.zeros((views, ships), device=device),
        fight_sense=torch.zeros((views, ships), device=device),
        fight_clock=torch.zeros((views, ships), device=device),
    )


def track_boundary(carry: ImugiCarry, obs: Obs, reset: torch.Tensor, seconds: float) -> ImugiCarry:
    """The carry with the boundary's closing speed (m/s, never negative) from its radius now
    and at the last observation a decision's length earlier, as if it keeps closing at that
    speed (the edge steer's assumption: `imugi_combat.context`). A new episode (a reset, a
    first observation, or the episode progress going back) starts at 0. Re-reading the
    observation the carry was made from (the same episode progress) keeps the speed, so a
    carry `decide` returned gives the same marginals again."""
    radius = obs.globals[:, GLOBAL["arena_radius"]] * obs.globals[:, GLOBAL["boundary_fraction"]]
    # contiguous: a column of the globals is a strided view, which stored in the carry
    # made the compiled tick's second call recompile for the carry's new strides
    progress = obs.globals[:, GLOBAL["episode_progress"]].contiguous()
    last = carry.boundary_progress
    fresh = reset | (last < 0.0) | (progress < last)
    same = ~fresh & (progress == last)
    closing = ((carry.boundary_radius - radius) / seconds).clamp_min(0.0)
    speed = torch.where(fresh, 0.0, torch.where(same, carry.boundary_speed, closing))
    return dataclasses.replace(
        carry,
        boundary_radius=torch.where(same, carry.boundary_radius, radius),
        boundary_progress=progress,
        boundary_speed=speed,
    )


def track_seen(
    carry: ImugiCarry,
    obs: Obs,
    reset: torch.Tensor,
    characteristics: Characteristics,
    seconds: float,
) -> ImugiCarry:
    """The carry with the explore grid's seen values: a cell within vision range of a living
    own ship is 1, every other cell decays by exp(-seconds / EXPLORE_SECONDS). A new episode
    starts from nothing seen. Re-reading the observation the carry was made from (the same
    episode progress as `track_boundary` recorded: call this before it) keeps the values."""
    progress = obs.globals[:, GLOBAL["episode_progress"]]
    last = carry.boundary_progress
    fresh = reset | (last < 0.0) | (progress < last)
    same = ~fresh & (progress == last)
    own = obs.ships[:, : obs.ships.shape[-2] // 2]
    alive = own[..., SHIP["alive"]] > 0.5
    vision = per_ship(characteristics.vision_range, own[..., SHIP["ship_type"]].to(torch.int32))
    cells = explore_cells(own.device)
    arena_radius = obs.globals[:, None, GLOBAL["arena_radius"]]
    # (R, C, S): each cell's offset from each own ship
    off_x = (cells[:, 0] * arena_radius)[..., None] - own[..., SHIP["x"]][:, None, :]
    off_y = (cells[:, 1] * arena_radius)[..., None] - own[..., SHIP["y"]][:, None, :]
    in_view = (off_x * off_x + off_y * off_y < (vision * vision)[:, None, :]) & alive[:, None, :]
    visible = in_view.any(-1).float()
    before = torch.where(fresh[:, None], 0.0, carry.seen)
    decayed = before * math.exp(-seconds / EXPLORE_SECONDS)
    seen = torch.where(same[:, None], carry.seen, torch.maximum(decayed, visible))
    return dataclasses.replace(carry, seen=seen)


def forgotten(carry: ImugiCarry) -> ImugiCarry:
    """The carry with every plan choice unset and every enemy forgotten (memory off); the
    ships' agent flags kept."""
    return dataclasses.replace(
        carry, fresh=torch.ones_like(carry.fresh), known=torch.zeros_like(carry.known)
    )


def time_unit(
    characteristics: Characteristics, own_type: torch.Tensor, parameters: ImugiParameters
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
    carry: ImugiCarry,
    obs: Obs,
    reset: torch.Tensor,
    characteristics: Characteristics,
    parameters: ImugiParameters,
    temperature,
) -> tuple[ImugiCarry, Recall]:
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
    # an own ship's committed target stays known while hidden up to the target's lost time
    # (`imugi_combat.TARGET_LOST_SECONDS`), its weight fading as any other's: the ship
    # presses the fight past a short loss of sight rather than turning to another
    committed = carry.target[..., None] == torch.arange(enemy.shape[1], device=enemy.device)
    pursued = (committed & ~fresh[..., None]).any(1) & (since < TARGET_LOST_SECONDS)
    known = seen | (known & enemy_alive & ~in_view & ((weight >= FORGET_WEIGHT) | pursued))
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
    uniform: torch.Tensor | None,
) -> torch.Tensor:
    """A plan choice after one decision: `best` where `forced`, else `best` with the switch's
    probability at `advantage` (best's over the committed option's), drawn by `uniform`
    (shaped like the advantage; `imugi_combat.choice_uniforms`), else `committed`.
    Without a uniform (re-reading a carry that has made its choices) only `forced`
    changes it."""
    if uniform is None:
        return torch.where(forced, best, committed)
    probability = switch_probability(advantage, shape, temperature, seconds)
    switch = uniform < probability
    return torch.where(forced | switch, best, committed)


def ways_round(error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The left and right turns (radians, in [0, 2 pi]) to an angle error (positive
    left)."""
    left = torch.where(error >= 0.0, error, error + 2.0 * math.pi)
    return left, 2.0 * math.pi - left
