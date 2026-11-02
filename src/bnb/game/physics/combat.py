"""Hits (game spec 7.1) and damage, death and shields (game spec 7.2).

Friendly fire is on: every bullet is paired with every ship but its own shooter, as
(B, N K, N) pair tensors that the compiled stage reduces without storing.
"""

import math
from typing import NamedTuple

import torch

from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, Ships, Sweep, replace

NO_HIT = torch.iinfo(torch.int64).max


class Hits(NamedTuple):
    """One tick's swept hits, before shields and spawn protection."""

    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship, -1 if none


class Contacts(NamedTuple):
    """Every (bullet, ship) pair of one tick's swept test, (B, N K, N): tau, the fraction of
    the move at the closest approach; miss_squared, that distance squared (m^2); contact,
    the pair is a hit candidate (an active bullet, a ship alive at tick start that is not
    its shooter, within the ship's hit radius)."""

    tau: torch.Tensor
    miss_squared: torch.Tensor
    contact: torch.Tensor


def contacts(
    types: ShipTypes,
    sweep: Sweep,
    ship_type: torch.Tensor,
    active: torch.Tensor,
    alive_at_start: torch.Tensor,
) -> Contacts:
    """The relative swept test between this tick's bullet and ship motions.

    sweep: the motions from `move`; ship_type, alive_at_start: (B, N); active: (B, N, K),
    the bullets with moves left. Bullet o K + k belongs to ship o.
    """
    B, N, K = active.shape
    per_bullet = (B, N * K, 1)
    start_x = sweep.bullet_x.reshape(per_bullet) - sweep.ship_x[:, None, :]  # (B, N K, N): D0
    start_y = sweep.bullet_y.reshape(per_bullet) - sweep.ship_y[:, None, :]
    relative_x = (
        sweep.bullet_displacement_x.reshape(per_bullet) - sweep.ship_displacement_x[:, None, :]
    )  # Delta
    relative_y = (
        sweep.bullet_displacement_y.reshape(per_bullet) - sweep.ship_displacement_y[:, None, :]
    )
    # abs() turns a -0.0 from the clamp into +0.0, so ties at tau = 0 order by bullet index
    # in the integer key below, eagerly and compiled alike.
    tau = (
        (
            -(start_x * relative_x + start_y * relative_y)
            / (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
        )
        .clamp(0.0, 1.0)
        .abs()
    )
    miss_x, miss_y = start_x + tau * relative_x, start_y + tau * relative_y
    miss_squared = miss_x * miss_x + miss_y * miss_y
    # The target's hit radius squared; -1 (no contact possible) for a ship dead at tick start.
    reach = torch.where(alive_at_start, per_ship(types.hit_radius, ship_type) ** 2, -1.0)
    ship = torch.arange(N, device=alive_at_start.device)
    not_shooter = ship.repeat_interleave(K)[:, None] != ship  # (N K, N)
    contact = (miss_squared < reach[:, None, :]) & not_shooter & active.reshape(per_bullet)
    return Contacts(tau=tau, miss_squared=miss_squared, contact=contact)


def contact_order(tau: torch.Tensor, miss_squared: torch.Tensor) -> torch.Tensor:
    """The int64 key that orders contacts in time: smallest tau first, ties (the clamp puts
    many contacts at exactly tau = 0 or 1) to the closer pass, the smaller miss distance.
    Both are +0.0 or more, so their float bits order like the floats: tau's 30 bits above
    the miss's 31. An order by sim index here would favour a team: team-major storage puts
    team 0's ships and bullets first."""
    return (tau.view(torch.int32).long() << 31) | miss_squared.view(torch.int32).long()


def first_contact(pairs: Contacts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each bullet hits only its first contact by `contact_order` (an exact tie in both to
    the lowest ship index: min returns the first minimal index). Returns (order, target,
    hit), each (B, N K): the first contact's order key, NO_HIT for a bullet that hit
    nothing."""
    order, target = torch.where(
        pairs.contact, contact_order(pairs.tau, pairs.miss_squared), NO_HIT
    ).min(-1)
    return order, target, order < NO_HIT


def hits(
    rules: Rules,
    types: ShipTypes,
    sweep: Sweep,
    ships: Ships,
    bullets: Bullets,
    alive_at_start: torch.Tensor,
) -> Hits:
    """Hits of this tick's swept test (`contacts`), each bullet on its first contact.

    sweep: the motions from `move`; ships, bullets: after the moves; alive_at_start:
    (B, N). Active bullets are those with moves_left > 0.
    """
    B, N, K = bullets.moves_left.shape
    order, target, hit = first_contact(
        contacts(types, sweep, ships.ship_type, bullets.moves_left > 0, alive_at_start)
    )
    ship = torch.arange(N, device=alive_at_start.device)

    # Damage by aspect: theta between the bullet's reversed direction and its target's nose.
    direction_x = bullets.direction_x.reshape(B, N * K)
    direction_y = bullets.direction_y.reshape(B, N * K)
    nose_cos, nose_sin = ships.heading_cos.gather(1, target), ships.heading_sin.gather(1, target)
    theta = torch.atan2(
        (direction_x * nose_sin - direction_y * nose_cos).abs(),
        -(direction_x * nose_cos + direction_y * nose_sin),
    )
    full = per_ship(types.hit_damage, ships.ship_type).repeat_interleave(K, 1)  # the shooter's
    head_on = rules.head_on_fraction * per_ship(types.head_on_fraction_scale, ships.ship_type)
    head_on = head_on.gather(1, target)  # the target's armour
    aspect = 1.0 - (1.0 - head_on) * torch.exp(-4.0 * theta * theta / math.pi)
    bullet_damage = torch.where(hit, full * aspect, 0.0)

    # Scatter each bullet's hit to its (shooter, target): O(bullets), not O(bullets x ships).
    # A bullet that hit nothing adds 0. The sum's order can matter only when three or more
    # of one shooter's bullets hit one target in the same tick.
    owner = ship.repeat_interleave(K)
    damage = (
        torch.zeros((B, N * N), dtype=bullet_damage.dtype, device=bullet_damage.device)
        .scatter_add(1, owner * N + target, bullet_damage)
        .reshape(B, N, N)
    )
    # The earliest hit on each target: the smallest contact order (tau, then the miss
    # distance), an exact tie in both to the lowest bullet index o K + k. Two exact int64
    # mins: the order, then the bullet among those that reach it.
    bullet = torch.arange(N * K, device=bullet_damage.device)
    blank = torch.full((B, N), NO_HIT, dtype=torch.int64, device=bullet_damage.device)
    earliest_order = blank.scatter_reduce(1, target, order, "amin")  # a miss is NO_HIT
    first = hit & (order == earliest_order.gather(1, target))
    earliest = blank.scatter_reduce(1, target, torch.where(first, bullet, NO_HIT), "amin")
    shooter = torch.where(earliest < NO_HIT, earliest // K, -1)
    return Hits(damage=damage, bullet_hit=hit.reshape(B, N, K), first_shooter=shooter)


def boundary_damage(
    rules: Rules, arena_radius: torch.Tensor, ships: Ships, alive_at_start: torch.Tensor
) -> torch.Tensor:
    """Game spec 5: (boundary_damage_base + boundary_damage_per_metre depth) tick_length for
    a living ship outside the arena (radius (B,)) after its move."""
    depth = torch.sqrt(ships.x * ships.x + ships.y * ships.y) - arena_radius[:, None]
    rate = rules.boundary_damage_base + rules.boundary_damage_per_metre * depth
    return torch.where(alive_at_start & (depth > 0.0), rate * rules.tick_length, 0.0)


class Damage(NamedTuple):
    """Phase 5's results: the ships after shields, recharge and deaths."""

    ships: Ships
    died: torch.Tensor  # (B, N)
    shield_lost: torch.Tensor  # (B, N) what the shield actually lost, overkill ignored
    shield_restored: torch.Tensor  # (B, N) the recharge applied


def apply_damage(
    rules: Rules, types: ShipTypes, ships: Ships, alive_at_start: torch.Tensor, total: torch.Tensor
) -> Damage:
    """Phase 5 for ships alive at tick start: shields, recharge, and who dies.

    Damage beyond the shield breaks it (the overkill is ignored); any damage to a shield
    already broken at tick start kills. Ships dead at tick start lose and restore nothing.
    """
    tick_length = rules.tick_length
    shield0, wait0 = ships.shield, ships.shield_recharge_wait
    hurt = total > 0.0
    died = alive_at_start & hurt & (shield0 <= 0.0)
    shield = (shield0 - total).clamp_min(0.0)
    rate = rules.shield_recharge_rate * per_ship(types.shield_recharge_rate_scale, ships.ship_type)
    delay = rules.shield_recharge_delay * per_ship(
        types.shield_recharge_delay_scale, ships.ship_type
    )
    recharge = torch.minimum(
        per_ship(types.max_shield, ships.ship_type) - shield,
        rate * (tick_length - wait0).clamp(0.0, tick_length),
    )
    restored = torch.where(alive_at_start & ~hurt, recharge, 0.0)  # an unhurt ship cannot have died
    wait = torch.where(hurt, delay, (wait0 - tick_length).clamp_min(0.0))
    ships = replace(
        ships,
        shield=torch.where(alive_at_start, shield + restored, shield0),
        shield_recharge_wait=torch.where(alive_at_start, wait, wait0),
    )
    return Damage(
        ships=ships,
        died=died,
        shield_lost=torch.where(alive_at_start, shield0 - shield, 0.0),
        shield_restored=restored,
    )
