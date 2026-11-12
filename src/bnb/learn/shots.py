"""Shot labels (learner spec 6): re-simulate the bullet every living ship would fire on every
tick, with the game's own launch, flight and swept-hit code, against the stored ships of the
following ticks.

The scan keeps one bullet per (row, age), newest first, as the game keeps one per slot: each
tick every row fires a fresh bullet from its captured launch state, every bullet moves, and
the swept test runs against the tick's captured ship motion. Every bullet flies the card's
longest life (`Rules.longest_bullet_moves`); a shot's first contact within its own type's
life is its hit (the game spends the bullet there) and its outcome,
while its path is labelled at every age: past a hit, the straight-on flight it would have
had. The trajectory head learns a shot at QUERIES of its ages (`query_ages`). Rows are in
the view's order, so a shooter's side is its block.
"""

import dataclasses
from collections.abc import Callable

import torch

from bnb.contracts.tree import contract
from bnb.game.physics.bullets import Launch, launch, move_bullets
from bnb.game.physics.combat import contacts, first_contact
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, bullet_moves, per_ship
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


# The ages the trajectory head is queried at per shot (`query_ages`).
QUERIES = 4


@contract
class Shots:
    """One shot per (tick, row), (L, R, N) leading: valid, the shooter was alive and its
    whole flight (all its moves) ends inside the window and inside the episode, so every
    age of a valid shot is labelled; snapshots (L, R, N, A, 5), after each move: the
    displacement and velocity (m, m/s) in the launch heading's frame (forward, left) and
    the refractive index; hit (HIT_*), the first hit's side relative to the shooter, and
    hit_age its age index (0: after the first move; A for none); distance (m), the closest
    pass to another living ship (any side; inf with none), side (SIDE_*) that ship's and
    closest_age its age index; bend_age, the age index of the first move whose refractive
    index differs from the one before it (the launch's, for the first move): the first
    move into or through a field's band, -1 for none."""

    valid: torch.Tensor
    snapshots: torch.Tensor
    hit: torch.Tensor
    hit_age: torch.Tensor
    distance: torch.Tensor
    side: torch.Tensor
    closest_age: torch.Tensor
    bend_age: torch.Tensor


@contract
class ShotLabels:
    """What the loss reads of the shots, (L, R, N) leading: ages (..., QUERIES) uint8, the
    queried age indices (`query_ages`); snapshots (..., QUERIES, 5) at them; hit, distance
    and side as `Shots`'."""

    ages: torch.Tensor
    snapshots: torch.Tensor
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
    return Fields(
        x=rows[..., 0],
        y=rows[..., 1],
        radius=rows[..., 2],
        band_width=rows[..., 3],
        level=rows[..., 4].to(torch.int32),
    )


def tick(
    rules: Rules,
    types: ShipTypes,
    bullets: Bullets,
    capture: ShotCapture,
    spread: torch.Tensor,
    field_rows: torch.Tensor,
) -> tuple[Bullets, _Tick]:
    """One tick of the scan, every view (R, ...): each row fires a fresh bullet from its
    captured launch state (the newest age), every bullet (R, N, A) moves once, and each is
    tested against this tick's ships. The runtime may compile it (the scan calls it once
    per tick)."""
    fresh = launch(rules, types, capture.launch, spread)
    fresh = replace(fresh, moves_left=torch.ones_like(fresh.moves_left))
    bullets = Bullets(
        **{
            name: torch.cat([getattr(fresh, name)[..., None], getattr(bullets, name)[..., :-1]], -1)
            for name in BULLET_FIELDS
        }
    )
    moved = move_bullets(rules, types, bullets, _fields(field_rows), capture.launch.ship_type)
    start = capture.launch
    sweep = Sweep(
        ship_x=start.x,
        ship_y=start.y,
        ship_displacement_x=capture.displacement_x,
        ship_displacement_y=capture.displacement_y,
        bullet_x=bullets.x,
        bullet_y=bullets.y,
        bullet_displacement_x=moved.x - bullets.x,
        bullet_displacement_y=moved.y - bullets.y,
    )
    pairs = contacts(types, sweep, start.ship_type, bullets.moves_left > 0, capture.alive)
    _, target, hit = first_contact(pairs)
    rows, ages = bullets.moves_left.shape[-2:]
    shooter = torch.arange(rows, device=hit.device).repeat_interleave(ages)[:, None]
    passable = capture.alive[..., None, :] & (shooter != torch.arange(rows, device=hit.device))
    miss_squared, nearest = torch.where(passable, pairs.miss_squared, torch.inf).min(-1)
    index = torch.exp(moved.log_refractive_index)
    speed = moved.proper_speed / index
    shape = bullets.moves_left.shape
    record = _Tick(
        x=moved.x,
        y=moved.y,
        velocity_x=moved.direction_x * speed,
        velocity_y=moved.direction_y * speed,
        index=index,
        hit=hit.reshape(shape),
        target=target.reshape(shape),
        miss_squared=miss_squared.reshape(shape),
        nearest=nearest.reshape(shape),
    )
    return moved, record


def resimulate(
    rules: Rules,
    types: ShipTypes,
    capture: ShotCapture,
    fields: torch.Tensor,
    done: torch.Tensor,
    spread: torch.Tensor,
    own: int,
    step: Callable = tick,
    available: torch.Tensor | None = None,
) -> Shots:
    """The shots of a stored segment. capture (L, R, N); fields (L, R, M, 5), the
    observation's field rows (the map of each tick); done (L, R), the episode ended after
    the tick; spread (L, R, N, 2) standard normals (the learner's draw); own: S, the
    shooter's side is rows [0, S) or [S, N); step: `tick`, or the runtime's compiled
    copy of it; available (R,): the ticks of each view that hold played ticks (a view whose
    window was extended past its segment by the ticks after it), L when None."""
    length, views, rows = capture.alive.shape
    ages = rules.longest_bullet_moves
    device = capture.alive.device
    empty = apply(
        lambda x: x.new_zeros((views, rows, ages)),
        launch(rules, types, _at(capture.launch, 0), spread[0]),
    )
    bullets = replace(empty, moves_left=torch.ones_like(empty.moves_left))
    records = []
    for time in range(length):
        bullets, record = step(
            rules, types, bullets, _at(capture, time), spread[time], fields[time]
        )
        records.append(record)
    by_tick = torch.utils._pytree.tree_map(lambda *x: torch.stack(x), *records)  # (L, R, N, A)
    # By launch: the shot fired at t is at age a + 1 on tick t + a.
    launch_tick = torch.arange(length, device=device)[:, None]
    flown = launch_tick + torch.arange(ages, device=device)  # (L, A): the tick of each age
    inside = flown < length
    flown = flown.clamp_max(length - 1)

    def by_launch(values: torch.Tensor) -> torch.Tensor:
        index = flown.reshape(length, 1, 1, ages).expand(values.shape)
        age = torch.arange(ages, device=device).expand(values.shape)
        return values.gather(0, index).gather(-1, age)  # [t, ..., a] = values[t + a, ..., a]

    shot = _Tick(
        **{
            field.name: by_launch(getattr(by_tick, field.name))
            for field in dataclasses.fields(_Tick)
        }
    )
    # A shot hits, and passes ships, only within its shooter's bullet life (the scan flies
    # every bullet the card's longest).
    life = bullet_moves(per_ship(types.bullet_life, capture.launch.ship_type), rules.tick_length)
    inside = inside[:, None, None, :] & (torch.arange(ages, device=device) < life[..., None])
    contact = shot.hit & inside
    hit_age = torch.where(contact, torch.arange(ages, device=device), ages).min(-1).values
    hit_any = hit_age < ages
    # Episode ends strictly inside the whole flight (every age is labelled, past a hit too):
    # done on ticks t .. t + A - 2, from the count of ends before each tick.
    ended = torch.cat([done.new_zeros((1, views)), done], 0).to(torch.int32).cumsum(0)
    end_tick = (launch_tick + ages - 1).clamp_max(length)  # (L, 1)
    crossed = (ended.gather(0, end_tick.expand(-1, views)) - ended[:-1])[:, :, None]
    # A shot launched within the longest flight of the window's end is invalid whatever it
    # hit: keeping the ones that hit in time would bias the tail's labels toward hits.
    end = length if available is None else available[:, None]  # () or (R, 1)
    whole_flight = launch_tick[:, :, None] + ages - 1 < end  # (L, 1 or R, 1)
    valid = capture.alive & whole_flight & (crossed == 0)

    start = capture.launch
    cos, sin = start.heading_cos[..., None], start.heading_sin[..., None]
    offset_x, offset_y = shot.x - start.x[..., None], shot.y - start.y[..., None]
    snapshots = torch.stack(
        [
            offset_x * cos + offset_y * sin,
            offset_y * cos - offset_x * sin,
            shot.velocity_x * cos + shot.velocity_y * sin,
            shot.velocity_y * cos - shot.velocity_x * sin,
            shot.index,
        ],
        -1,
    )

    side_of = torch.arange(rows, device=device) < own  # (N,) the row's block
    shooter_side = side_of  # the shooter is the last axis of a shot

    def same_side(row: torch.Tensor) -> torch.Tensor:
        return side_of[row] == shooter_side

    first_target = shot.target.gather(-1, hit_age.clamp_max(ages - 1)[..., None])[..., 0]
    hit = torch.where(hit_any, torch.where(same_side(first_target), HIT_ALLY, HIT_ENEMY), HIT_NONE)
    miss = torch.where(inside, shot.miss_squared, torch.inf)
    closest, closest_age = miss.min(-1)
    nearest = shot.nearest.gather(-1, closest_age[..., None])[..., 0]
    side = torch.where(same_side(nearest), SIDE_ALLY, SIDE_ENEMY)
    before = torch.cat([torch.exp(start.log_refractive_index)[..., None], shot.index[..., :-1]], -1)
    bent = shot.index != before  # the index is constant outside the fields' bands
    bend_age = torch.where(bent.any(-1), torch.argmax(bent.to(torch.int32), -1), -1)
    return Shots(
        valid=valid,
        snapshots=snapshots,
        hit=hit,
        hit_age=hit_age,
        distance=torch.sqrt(closest),
        side=side,
        closest_age=closest_age,
        bend_age=bend_age,
    )


def query_ages(shots: Shots, drawn: torch.Tensor) -> torch.Tensor:
    """(L, R, N, QUERIES) the age indices each shot's trajectory is learnt at: its last
    move; one uniform over its whole flight (it may repeat another); its closest pass to
    another ship; its first move into a field's band (`Shots.bend_age`). The last two fall
    back to a uniform age of their own when the shot passed no ship or met no band.
    drawn (L, R, N, 3): uniform age indices in [0, A) (the learner's draw), one per use."""
    ages = shots.snapshots.shape[-2]
    drawn = drawn.long()
    return torch.stack(
        [
            torch.full_like(shots.closest_age, ages - 1),
            drawn[..., 0],
            torch.where(torch.isfinite(shots.distance), shots.closest_age, drawn[..., 1]),
            torch.where(shots.bend_age >= 0, shots.bend_age, drawn[..., 2]),
        ],
        -1,
    )


def labels(shots: Shots, drawn: torch.Tensor) -> ShotLabels:
    """The loss's labels of `shots`: the snapshots at `query_ages(shots, drawn)`."""
    ages = query_ages(shots, drawn)
    snapshots = shots.snapshots.gather(
        -2, ages[..., None].expand(*ages.shape, shots.snapshots.shape[-1])
    )
    return ShotLabels(
        ages=ages.to(torch.uint8),
        snapshots=snapshots,
        hit=shots.hit,
        distance=shots.distance,
        side=shots.side,
    )
