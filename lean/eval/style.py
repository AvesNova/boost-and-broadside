"""Style counters: how a side flew a game, beside who won it.

Per environment and per seat, sums over the ship-ticks of the seat's ships alive at tick
start (`tick_sums`, inside the runner's compiled tick: no host read). The runner keeps them per
slot, zeroes them at each deal and writes a finished game's sums to its record. `measures`
turns a game's sums into the stored style (per game and seat, small floats):

- minutes: ship-minutes flown (the weight of the means below).
- turning: share of the time in a turning fight: the ship and its nearest living enemy both
  turning (`turn_sense`).
- two_circle: share of that turning-fight time in two-circle flow (both turn the same way
  round); the rest is one-circle (opposite senses). None without a turning fight.
- store: mean store, fraction of full.
- speed: mean speed, in corner speeds.
- shots: real shots fired per ship-minute.
- hits: damaging hits on enemies, a real bullet's or a collision's (one per shooter, target
  and tick; a hit that does no damage, as inside the nose armour (`nose_cone`, on by default), is not
  counted).
- bullet_hits, collision_hits: the same count by cause, a real bullet's hit or a collision's
  (a shooter and target with both in one tick count once in each, once in `hits`, so `hits`
  is at most their sum). The game's `hit_damage` lumps both; `bullet_damage` and
  `collision_damage` are its parts. A collision hits both ships, each credited to the other.
- tail_hits: bullet hits whose shooter is in the target's rear cone: the target's train angle
  to the shooter (`lean.game.train_angle`, the one `pair_geometry` gives; from its nose to the
  line to the shooter, at the start of the hit's tick) is beyond `TAIL_CHASE_ANGLE` either way.
  The tail-chase share of hits is tail_hits / bullet_hits (the `styles` view's `tail_share`).
  Collisions are not part of it (they have no shooter).
- hit_range: mean range at hits (shooter to target at the start of the hit's tick), in gun
  ranges. None without a hit.
- climb, dive: share of the time with climb or dive applied.

The flow is own sense x target sense (the sense from the nose rate against a floor of
sustained rates), with a hard floor
(`TURNING_FLOOR`) and the nearest living enemy as the target, so that it is one definition for
every agent (a network has no target).
"""

import math

import torch
from torch import Tensor

from lean.game import train_angle

# The shooter is "behind" its target when the target's train angle to the shooter is more than
# this (radians, 120 degrees): the shooter is in the 120-degree cone behind the target's tail.
TAIL_CHASE_ANGLE = math.radians(120.0)
# The nose rate, in sustained rates, below which a ship counts as flying straight.
TURNING_FLOOR = 0.25
# The sums per seat, in this order.
SUMS = (
    "ticks",
    "turning",
    "two_circle",
    "store",
    "speed",
    "shots",
    "hits",
    "bullet_hits",
    "collision_hits",
    "tail_hits",
    "hit_range",
    "climb",
    "dive",
)
MEASURES = (
    "minutes",
    "turning",
    "two_circle",
    "store",
    "speed",
    "shots",
    "hits",
    "bullet_hits",
    "collision_hits",
    "tail_hits",
    "hit_range",
    "climb",
    "dive",
)
COAST, DIVE, CLIMB = 0, 1, 2  # power indices (lean.game.card)


def turn_sense(
    nose_rate: Tensor, sustained_rate: Tensor | float, floor: float = TURNING_FLOOR
) -> Tensor:
    """-1 turning right, +1 left, 0 straight (|nose rate| under `floor` sustained rates)."""
    turning = nose_rate.abs() >= floor * sustained_rate
    return torch.where(turning, torch.sign(nose_rate), 0.0)


def nearest_enemy(state) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Each ship's nearest living enemy: its index (E, N) long, its distance (E, N), whether
    there is one (E, N) bool; and the distance between every pair of ships (E, N, N)."""
    dx = state.x[:, None, :] - state.x[:, :, None]
    dy = state.y[:, None, :] - state.y[:, :, None]
    distance = torch.sqrt(dx * dx + dy * dy)
    enemy = (state.team[:, :, None] != state.team[:, None, :]) & state.alive[:, None, :]
    masked = torch.where(enemy, distance, torch.inf)
    nearest = masked.min(-1)
    return nearest.indices, nearest.values, enemy.any(-1), distance


def per_ship(
    state,
    applied_power: Tensor,
    fired: Tensor,
    hit_damage: Tensor,
    bullet_damage: Tensor,
    collision_damage: Tensor,
) -> Tensor:
    """(E, N, len(SUMS)) float32 for every ship over one tick: `state` at tick start (the one
    the actions were chosen on), `applied_power` (E, N) the power index the tick applied,
    `fired` (E, N) a real shot this tick, `hit_damage` (E, N, N) StepOut's damage (real bullets
    and collisions) by shooter and target, `bullet_damage` and `collision_damage` its parts.
    Rows of ships not alive at tick start are zero."""
    c = state.constants
    sustained = c["sustained_rate"]
    target, _, has_enemy, distance = nearest_enemy(state)
    own = turn_sense(state.nose_rate, sustained)
    target_rate = state.nose_rate.gather(1, target)
    target_sustained = sustained.gather(1, target) if isinstance(sustained, Tensor) else sustained
    theirs = turn_sense(target_rate, target_sustained)
    flow = own * theirs * has_enemy
    hit = (hit_damage > 0.0) & (state.team[:, :, None] != state.team[:, None, :])
    enemies = state.team[:, :, None] != state.team[:, None, :]
    bullet_hit = (bullet_damage > 0.0) & enemies
    collision_hit = (collision_damage > 0.0) & enemies
    # The target's train angle to the shooter, [shooter, target]: the target's nose against the
    # line from the target to the shooter.
    angle = train_angle(
        state.heading_cos[:, None, :],
        state.heading_sin[:, None, :],
        state.x[:, :, None] - state.x[:, None, :],
        state.y[:, :, None] - state.y[:, None, :],
    )
    tail_hit = bullet_hit & (angle.abs() > TAIL_CHASE_ANGLE)
    hit_range = (hit * distance).sum(-1) / c["gun_range"]
    alive = state.alive
    one = torch.ones_like(state.speed)
    rows = torch.stack(
        [
            one,
            (flow != 0.0).float(),
            (flow > 0.0).float(),
            state.energy * c["inverse_max_energy"],
            state.speed / c["corner_speed"],
            fired.float(),
            hit.sum(-1).float(),
            bullet_hit.sum(-1).float(),
            collision_hit.sum(-1).float(),
            tail_hit.sum(-1).float(),
            hit_range,
            (applied_power == CLIMB).float(),
            (applied_power == DIVE).float(),
        ],
        -1,
    )
    return torch.where(alive[..., None], rows, 0.0)


def tick_sums(
    state,
    seat_a: Tensor,
    playing: Tensor,
    applied_power: Tensor,
    fired: Tensor,
    hit_damage: Tensor,
    bullet_damage: Tensor,
    collision_damage: Tensor,
) -> Tensor:
    """(E, 2, len(SUMS)): one tick's sums for seat A (row 0) and seat B (row 1) of each
    environment, zero where `playing` (E,) is false. `seat_a` (E, N) the ship plays for seat A."""
    rows = (
        per_ship(state, applied_power, fired, hit_damage, bullet_damage, collision_damage)
        * playing[:, None, None]
    )
    seats = torch.stack([seat_a, ~seat_a], 1).float()  # (E, 2, N)
    return torch.bmm(seats, rows)


def measures(sums: Tensor, tick_length: float) -> list[tuple]:
    """Per game, the stored style of both seats from the record's sums (G, 2, len(SUMS)):
    a tuple of 2 x len(MEASURES) values (seat A's, then seat B's), None where undefined."""
    rows = []
    for game in sums.double().tolist():
        values = []
        for (
            ticks,
            turning,
            two_circle,
            store,
            speed,
            shots,
            hits,
            bullet_hits,
            collision_hits,
            tail_hits,
            hit_range,
            climb,
            dive,
        ) in game:
            if ticks <= 0:
                values += [0.0] + [None] * (len(MEASURES) - 1)
                continue
            minutes = ticks * tick_length / 60.0
            values += [
                minutes,
                turning / ticks,
                two_circle / turning if turning > 0 else None,
                store / ticks,
                speed / ticks,
                shots / minutes,
                int(round(hits)),
                int(round(bullet_hits)),
                int(round(collision_hits)),
                int(round(tail_hits)),
                hit_range / hits if hits > 0 else None,
                climb / ticks,
                dive / ticks,
            ]
        rows.append(tuple(values))
    return rows
