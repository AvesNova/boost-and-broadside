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

INF = float("inf")
NONE = torch.iinfo(torch.int64).max


class Hits(NamedTuple):
    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    taken: torch.Tensor  # (B, N) hit damage taken by each ship
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship, -1 if none


def hits(rules: Rules, types: ShipTypes, sweep: Sweep, ships: Ships, bullets: Bullets,
         alive0: torch.Tensor) -> Hits:
    """Relative swept test between this tick's bullet and ship motions.

    sweep: the motions from `move`; ships, bullets: after the moves; alive0: (B, N) ships
    alive at tick start. Active bullets are those with ttl > 0.
    """
    B, N, K = bullets.ttl.shape
    flat = (B, N * K, 1)
    d0x = sweep.bx.reshape(flat) - sweep.x[:, None, :]  # (B, N K, N) bullet minus ship
    d0y = sweep.by.reshape(flat) - sweep.y[:, None, :]
    deltax = sweep.bdx.reshape(flat) - sweep.dx[:, None, :]
    deltay = sweep.bdy.reshape(flat) - sweep.dy[:, None, :]
    tau = (-(d0x * deltax + d0y * deltay)
           / (deltax * deltax + deltay * deltay).clamp_min(1e-12)).clamp(0.0, 1.0)
    missx, missy = d0x + tau * deltax, d0y + tau * deltay
    # The target's hit radius squared; -1 (no contact possible) for a ship dead at tick start.
    reach = torch.where(alive0, per_ship(types.hit_radius, ships.ship_type) ** 2, -1.0)
    ship = torch.arange(N, device=alive0.device)
    not_shooter = ship.repeat_interleave(K)[:, None] != ship  # (N K, N)
    contact = ((missx * missx + missy * missy) < reach[:, None, :]) & not_shooter \
        & (bullets.ttl > 0).reshape(flat)

    # Each bullet hits only its first contact: smallest tau, ties to the lowest ship index
    # (min returns the first minimal index).
    tau_first, first = torch.where(contact, tau, INF).min(-1)  # (B, N K)
    hit = tau_first < INF

    # Damage by aspect: theta between the bullet's reversed direction and its target's nose.
    dx, dy = bullets.dx.reshape(B, N * K), bullets.dy.reshape(B, N * K)
    nc, ns = ships.c.gather(1, first), ships.s.gather(1, first)
    theta = torch.atan2((dx * ns - dy * nc).abs(), -(dx * nc + dy * ns))
    full = per_ship(types.damage, ships.ship_type).repeat_interleave(K, 1)  # the shooter's D
    aspect = 1.0 - (1.0 - rules.head_on_fraction) * torch.exp(-4.0 * theta * theta / math.pi)
    dmg = torch.where(hit, full * aspect, 0.0)

    # Scatter each bullet's hit to its (shooter, target): O(bullets), not O(bullets x ships).
    # A bullet that hit nothing adds 0. The sum's order can matter only when three or more
    # of one shooter's bullets hit one target in the same tick.
    owner = ship.repeat_interleave(K)
    damage = torch.zeros((B, N * N), dtype=dmg.dtype, device=dmg.device) \
        .scatter_add(1, owner * N + first, dmg).reshape(B, N, N)
    # The earliest hit on each target: smallest tau, ties to the lowest bullet index o K + k,
    # as one int64 key (tau >= 0, so its float bits order like the floats) and an exact min.
    bullet = torch.arange(N * K, device=dmg.device)
    key = torch.where(hit, tau_first.view(torch.int32).long() * (N * K) + bullet, NONE)
    earliest = torch.full((B, N), NONE, dtype=torch.int64, device=dmg.device) \
        .scatter_reduce(1, first, key, "amin")
    shooter = torch.where(earliest < NONE, (earliest % (N * K)) // K, -1)
    return Hits(damage=damage, taken=damage.sum(1), bullet_hit=hit.reshape(B, N, K),
                first_shooter=shooter)


def boundary_damage(rules: Rules, ships: Ships, alive0: torch.Tensor) -> torch.Tensor:
    """Game spec 5: (5 + 0.05 depth) h for a living ship outside the arena after its move."""
    depth = torch.sqrt(ships.x * ships.x + ships.y * ships.y) - rules.arena_radius
    return torch.where(alive0 & (depth > 0.0),
                       (rules.boundary_base + rules.boundary_per_m * depth) * rules.tick, 0.0)


def apply_damage(rules: Rules, types: ShipTypes, ships: Ships, alive0: torch.Tensor,
                 total: torch.Tensor) -> tuple[Ships, torch.Tensor]:
    """Phase 5 for ships alive at tick start: shields, recharge, and who dies.

    Damage beyond the shield breaks it (the overkill is ignored); any damage to a shield
    already broken at tick start kills. Returns (ships, died).
    """
    h = rules.tick
    shield0, wait0 = ships.shield, ships.shield_wait
    hurt = total > 0.0
    died = alive0 & hurt & (shield0 <= 0.0)
    shield = (shield0 - total).clamp_min(0.0)
    recharge = torch.minimum(per_ship(types.shield_max, ships.ship_type) - shield,
                             rules.shield_regen * (h - wait0).clamp(0.0, h))
    shield = torch.where(hurt, shield, shield + recharge)  # an unhurt ship cannot have died
    wait = torch.where(hurt, rules.shield_delay, (wait0 - h).clamp_min(0.0))
    return replace(ships, shield=torch.where(alive0, shield, shield0),
                   shield_wait=torch.where(alive0, wait, wait0)), died
