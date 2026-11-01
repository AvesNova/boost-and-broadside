"""Shot labels (learner spec 6): re-simulate the bullet every living ship would fire on every
tick, with the game's own launch, flight and swept-hit code, against the stored ships of the
following ticks.

The scan keeps one bullet per (row, age), newest first, as the game keeps one per slot: each
tick every row fires a fresh bullet from its captured launch state, every bullet moves, and
the swept test runs against the tick's captured ship motion. Every bullet flies all its
moves; a shot's first contact is its hit (the game spends the bullet there), so its later
ages are invalid. Rows are in the view's order, so a shooter's side is its block.
"""

import dataclasses

import torch

from bnb.contracts.tree import contract
from bnb.game.physics.bullets import Launch, launch, move_bullets
from bnb.game.physics.combat import contacts, first_contact
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes
from bnb.game.state import Bullets, Fields, Sweep, apply, replace
from bnb.models.yemong.shots import HIT_ALLY, HIT_ENEMY, HIT_NONE, SIDE_ALLY, SIDE_ENEMY


@contract
class ShotCapture:
    """What the rollout stores each tick for the re-simulation, per row (..., N): launch,
    the game's launch state (`bnb.game.physics.bullets.launch_state`) at tick start;
    displacement_x, displacement_y (m), the ship's move this tick (the hit test's sweep);
    alive, the ship was alive at tick start (it fires, and it can be hit)."""

    launch: Launch
    displacement_x: torch.Tensor
    displacement_y: torch.Tensor
    alive: torch.Tensor


@contract
class Shots:
    """One shot per (tick, row), (L, R, N) leading: valid, the shooter was alive and the
    flight (to its hit, or all its moves) ends inside the segment and the episode;
    snapshots (L, R, N, A, 5), after each move: the displacement and velocity (m, m/s) in
    the launch heading's frame (forward, left) and the refractive index; age_valid
    (L, R, N, A), the shot is valid and the age is not after its hit; hit (HIT_*), the first
    hit's side relative to the shooter; distance (m), the closest pass of a shot that hit
    nothing, and side (SIDE_*) that ship's."""

    valid: torch.Tensor
    snapshots: torch.Tensor
    age_valid: torch.Tensor
    hit: torch.Tensor
    distance: torch.Tensor
    side: torch.Tensor


@contract
class _Tick:
    """One tick of the scan, per bullet (R, N, A) by age: where it is after its move,
    whether it touched a ship and which first, and its closest pass."""

    x: torch.Tensor
    y: torch.Tensor
    velocity_x: torch.Tensor
    velocity_y: torch.Tensor
    index: torch.Tensor
    hit: torch.Tensor
    target: torch.Tensor
    miss_squared: torch.Tensor
    nearest: torch.Tensor


BULLET_FIELDS = tuple(field.name for field in dataclasses.fields(Bullets))


def _at(tree, time: int):
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x[time], tree)


def _fields(rows: torch.Tensor) -> Fields:
    """Observation field rows (..., M, 5) as the game's Fields (the same values)."""
    return Fields(x=rows[..., 0], y=rows[..., 1], radius=rows[..., 2], band_width=rows[..., 3],
                  level=rows[..., 4].to(torch.int32))


def _tick(rules: Rules, types: ShipTypes, bullets: Bullets, capture: ShotCapture,
          fields: Fields) -> tuple[Bullets, _Tick]:
    """Move every bullet (R, N, A) once and test it against this tick's ships."""
    moved = move_bullets(rules, bullets, fields)
    start = capture.launch
    sweep = Sweep(ship_x=start.x, ship_y=start.y, ship_displacement_x=capture.displacement_x,
                  ship_displacement_y=capture.displacement_y, bullet_x=bullets.x,
                  bullet_y=bullets.y, bullet_displacement_x=moved.x - bullets.x,
                  bullet_displacement_y=moved.y - bullets.y)
    pairs = contacts(types, sweep, start.ship_type, bullets.moves_left > 0, capture.alive)
    _, target, hit = first_contact(pairs)
    rows, ages = bullets.moves_left.shape[-2:]
    shooter = torch.arange(rows, device=hit.device).repeat_interleave(ages)[:, None]
    passable = capture.alive[..., None, :] & (shooter != torch.arange(rows, device=hit.device))
    miss_squared, nearest = torch.where(passable, pairs.miss_squared, torch.inf).min(-1)
    index = torch.exp(moved.log_refractive_index)
    speed = moved.proper_speed / index
    shape = bullets.moves_left.shape
    record = _Tick(x=moved.x, y=moved.y, velocity_x=moved.direction_x * speed,
                   velocity_y=moved.direction_y * speed, index=index, hit=hit.reshape(shape),
                   target=target.reshape(shape), miss_squared=miss_squared.reshape(shape),
                   nearest=nearest.reshape(shape))
    return moved, record


def resimulate(rules: Rules, types: ShipTypes, capture: ShotCapture, fields: torch.Tensor,
               done: torch.Tensor, spread: torch.Tensor, own: int) -> Shots:
    """The shots of a stored segment. capture (L, R, N); fields (L, R, M, 5), the
    observation's field rows (the map of each tick); done (L, R), the episode ended after
    the tick; spread (L, R, N, 2) standard normals (the learner's draw); own: S, the
    shooter's side is rows [0, S) or [S, N)."""
    length, views, rows = capture.alive.shape
    ages = rules.bullet_moves
    device = capture.alive.device
    empty = apply(lambda x: x.new_zeros((views, rows, ages)),
                  launch(rules, types, _at(capture.launch, 0), spread[0]))
    bullets = replace(empty, moves_left=torch.ones_like(empty.moves_left))
    records = []
    for time in range(length):
        fresh = launch(rules, types, _at(capture.launch, time), spread[time])
        fresh = replace(fresh, moves_left=torch.ones_like(fresh.moves_left))
        bullets = Bullets(**{name: torch.cat([getattr(fresh, name)[..., None],
                                              getattr(bullets, name)[..., :-1]], -1)
                             for name in BULLET_FIELDS})
        bullets, record = _tick(rules, types, bullets, _at(capture, time),
                                _fields(fields[time]))
        records.append(record)
    by_tick = torch.utils._pytree.tree_map(lambda *x: torch.stack(x), *records)  # (L, R, N, A)
    # By launch: the shot fired at t is at age a + 1 on tick t + a.
    launch_tick = torch.arange(length, device=device)[:, None]
    tick = launch_tick + torch.arange(ages, device=device)  # (L, A)
    inside = tick < length
    tick = tick.clamp_max(length - 1)

    def by_launch(values: torch.Tensor) -> torch.Tensor:
        index = tick.reshape(length, 1, 1, ages).expand(values.shape)
        age = torch.arange(ages, device=device).expand(values.shape)
        return values.gather(0, index).gather(-1, age)  # [t, ..., a] = values[t + a, ..., a]

    shot = _Tick(**{field.name: by_launch(getattr(by_tick, field.name))
                    for field in dataclasses.fields(_Tick)})
    inside = inside[:, None, None, :]
    contact = shot.hit & inside
    hit_age = torch.where(contact, torch.arange(ages, device=device), ages).min(-1).values
    hit_any = hit_age < ages
    flight_end = torch.where(hit_any, hit_age, ages - 1)  # the age index of its last move
    # Episode ends strictly inside the flight: done on ticks t .. end - 1, from the count of
    # ends before each tick.
    ended = torch.cat([done.new_zeros((1, views)), done], 0).to(torch.int32).cumsum(0)
    ended = ended[:, :, None].expand(-1, -1, rows)  # (L + 1, R, N)
    end_tick = launch_tick[:, :, None] + flight_end  # (L, R, N)
    crossed = ended.gather(0, end_tick.clamp_max(length)) - ended[:-1]
    valid = capture.alive & (end_tick < length) & (crossed == 0)
    age_valid = valid[..., None] & (torch.arange(ages, device=device) <= flight_end[..., None])

    start = capture.launch
    cos, sin = start.heading_cos[..., None], start.heading_sin[..., None]
    offset_x, offset_y = shot.x - start.x[..., None], shot.y - start.y[..., None]
    snapshots = torch.stack([offset_x * cos + offset_y * sin, offset_y * cos - offset_x * sin,
                             shot.velocity_x * cos + shot.velocity_y * sin,
                             shot.velocity_y * cos - shot.velocity_x * sin, shot.index], -1)

    side_of = torch.arange(rows, device=device) < own  # (N,) the row's block
    shooter_side = side_of  # the shooter is the last axis of a shot

    def same_side(row: torch.Tensor) -> torch.Tensor:
        return side_of[row] == shooter_side

    first_target = shot.target.gather(-1, hit_age.clamp_max(ages - 1)[..., None])[..., 0]
    hit = torch.where(hit_any, torch.where(same_side(first_target), HIT_ALLY, HIT_ENEMY),
                      HIT_NONE)
    miss = torch.where(inside, shot.miss_squared, torch.inf)
    closest, closest_age = miss.min(-1)
    nearest = shot.nearest.gather(-1, closest_age[..., None])[..., 0]
    side = torch.where(same_side(nearest), SIDE_ALLY, SIDE_ENEMY)
    return Shots(valid=valid, snapshots=snapshots, age_valid=age_valid, hit=hit,
                 distance=torch.sqrt(closest), side=side)
