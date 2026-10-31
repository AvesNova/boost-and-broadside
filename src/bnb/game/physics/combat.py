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

INFINITY = float("inf")
NO_HIT = torch.iinfo(torch.int64).max


class Hits(NamedTuple):
    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    taken: torch.Tensor  # (B, N) hit damage taken by each ship
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship, -1 if none


def hits(rules: Rules, types: ShipTypes, sweep: Sweep, ships: Ships, bullets: Bullets,
         alive_at_start: torch.Tensor) -> Hits:
    """Relative swept test between this tick's bullet and ship motions.

    sweep: the motions from `move`; ships, bullets: after the moves; alive_at_start:
    (B, N). Active bullets are those with moves_left > 0.
    """
    B, N, K = bullets.moves_left.shape
    per_bullet = (B, N * K, 1)
    start_x = sweep.bullet_x.reshape(per_bullet) - sweep.ship_x[:, None, :]  # (B, N K, N): D0
    start_y = sweep.bullet_y.reshape(per_bullet) - sweep.ship_y[:, None, :]
    relative_x = (sweep.bullet_displacement_x.reshape(per_bullet)
                  - sweep.ship_displacement_x[:, None, :])  # Delta
    relative_y = (sweep.bullet_displacement_y.reshape(per_bullet)
                  - sweep.ship_displacement_y[:, None, :])
    # abs() turns a -0.0 from the clamp into +0.0, so ties at tau = 0 order by bullet index
    # in the integer key below, eagerly and compiled alike.
    tau = (-(start_x * relative_x + start_y * relative_y)
           / (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
           ).clamp(0.0, 1.0).abs()
    miss_x, miss_y = start_x + tau * relative_x, start_y + tau * relative_y
    # The target's hit radius squared; -1 (no contact possible) for a ship dead at tick start.
    reach = torch.where(alive_at_start, per_ship(types.hit_radius, ships.ship_type) ** 2, -1.0)
    ship = torch.arange(N, device=alive_at_start.device)
    not_shooter = ship.repeat_interleave(K)[:, None] != ship  # (N K, N)
    contact = (((miss_x * miss_x + miss_y * miss_y) < reach[:, None, :]) & not_shooter
               & (bullets.moves_left > 0).reshape(per_bullet))

    # Each bullet hits only its first contact: smallest tau, ties to the lowest ship index
    # (min returns the first minimal index).
    tau_first, target = torch.where(contact, tau, INFINITY).min(-1)  # (B, N K)
    hit = tau_first < INFINITY

    # Damage by aspect: theta between the bullet's reversed direction and its target's nose.
    direction_x = bullets.direction_x.reshape(B, N * K)
    direction_y = bullets.direction_y.reshape(B, N * K)
    nose_cos, nose_sin = ships.heading_cos.gather(1, target), ships.heading_sin.gather(1, target)
    theta = torch.atan2((direction_x * nose_sin - direction_y * nose_cos).abs(),
                        -(direction_x * nose_cos + direction_y * nose_sin))
    full = per_ship(types.hit_damage, ships.ship_type).repeat_interleave(K, 1)  # the shooter's
    aspect = 1.0 - (1.0 - rules.head_on_fraction) * torch.exp(-4.0 * theta * theta / math.pi)
    bullet_damage = torch.where(hit, full * aspect, 0.0)

    # Scatter each bullet's hit to its (shooter, target): O(bullets), not O(bullets x ships).
    # A bullet that hit nothing adds 0. The sum's order can matter only when three or more
    # of one shooter's bullets hit one target in the same tick.
    owner = ship.repeat_interleave(K)
    damage = (torch.zeros((B, N * N), dtype=bullet_damage.dtype, device=bullet_damage.device)
              .scatter_add(1, owner * N + target, bullet_damage).reshape(B, N, N))
    # The earliest hit on each target: smallest tau, ties to the lowest bullet index o K + k,
    # as one int64 key (tau >= +0.0, so its float bits order like the floats) and an exact
    # min.
    bullet = torch.arange(N * K, device=bullet_damage.device)
    key = torch.where(hit, tau_first.view(torch.int32).long() * (N * K) + bullet, NO_HIT)
    earliest = (torch.full((B, N), NO_HIT, dtype=torch.int64, device=bullet_damage.device)
                .scatter_reduce(1, target, key, "amin"))
    shooter = torch.where(earliest < NO_HIT, (earliest % (N * K)) // K, -1)
    return Hits(damage=damage, taken=damage.sum(1), bullet_hit=hit.reshape(B, N, K),
                first_shooter=shooter)


def boundary_damage(rules: Rules, ships: Ships, alive_at_start: torch.Tensor) -> torch.Tensor:
    """Game spec 5: (5 + 0.05 depth) tick_length for a living ship outside the arena after its move."""
    depth = torch.sqrt(ships.x * ships.x + ships.y * ships.y) - rules.arena_radius
    rate = rules.boundary_damage_base + rules.boundary_damage_per_metre * depth
    return torch.where(alive_at_start & (depth > 0.0), rate * rules.tick_length, 0.0)


def apply_damage(rules: Rules, types: ShipTypes, ships: Ships, alive_at_start: torch.Tensor,
                 total: torch.Tensor) -> tuple[Ships, torch.Tensor]:
    """Phase 5 for ships alive at tick start: shields, recharge, and who dies.

    Damage beyond the shield breaks it (the overkill is ignored); any damage to a shield
    already broken at tick start kills. Returns (ships, died).
    """
    tick_length = rules.tick_length
    shield0, wait0 = ships.shield, ships.shield_recharge_wait
    hurt = total > 0.0
    died = alive_at_start & hurt & (shield0 <= 0.0)
    shield = (shield0 - total).clamp_min(0.0)
    recharge = torch.minimum(per_ship(types.max_shield, ships.ship_type) - shield,
                             rules.shield_recharge_rate * (tick_length - wait0).clamp(0.0, tick_length))
    shield = torch.where(hurt, shield, shield + recharge)  # an unhurt ship cannot have died
    wait = torch.where(hurt, rules.shield_recharge_delay, (wait0 - tick_length).clamp_min(0.0))
    return replace(ships, shield=torch.where(alive_at_start, shield, shield0),
                   shield_recharge_wait=torch.where(alive_at_start, wait, wait0)), died
