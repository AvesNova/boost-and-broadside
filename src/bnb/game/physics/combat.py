"""Hits (game spec 7.1) and damage, death and shields (game spec 7.2).

Friendly fire is on: every bullet is paired with every ship but its own shooter, as
(B, N K, N) pair tensors that the compiled stage reduces without storing.
"""

import math
from typing import NamedTuple

import torch

from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, Origin, Ships, replace

INF = float("inf")


class Hits(NamedTuple):
    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    taken: torch.Tensor  # (B, N) hit damage taken by each ship
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship, -1 if none


def hits(rules: Rules, types: ShipTypes, origin: Origin, ships: Ships, bullets: Bullets,
         alive0: torch.Tensor) -> Hits:
    """Relative swept test between this tick's bullet and ship motions.

    origin: tick-start positions (bullets after firing); ships, bullets: after the moves;
    alive0: (B, N) ships alive at tick start. Active bullets are those with ttl > 0.
    """
    B, N, K = bullets.ttl.shape
    flat = (B, N * K, 1)
    d0x = origin.bx.reshape(flat) - origin.x[:, None, :]  # (B, N K, N) bullet minus ship
    d0y = origin.by.reshape(flat) - origin.y[:, None, :]
    deltax = bullets.x.reshape(flat) - ships.x[:, None, :] - d0x
    deltay = bullets.y.reshape(flat) - ships.y[:, None, :] - d0y
    tau = (-(d0x * deltax + d0y * deltay)
           / (deltax * deltax + deltay * deltay).clamp_min(1e-12)).clamp(0.0, 1.0)
    missx, missy = d0x + tau * deltax, d0y + tau * deltay
    radius = per_ship(types.hit_radius, ships.ship_type)[:, None, :]  # the target's
    ship = torch.arange(N, device=alive0.device)
    not_shooter = ship.repeat_interleave(K)[:, None] != ship  # (N K, N)
    contact = ((missx * missx + missy * missy) < radius * radius) & not_shooter \
        & (bullets.ttl > 0).reshape(flat) & alive0[:, None, :]

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

    on_target = hit[..., None] & (first[..., None] == ship)  # (B, N K, N), never stored
    damage = torch.where(on_target, dmg[..., None], 0.0).reshape(B, N, K, N).sum(2)

    # The earliest hit on each target: smallest tau, ties to the lowest bullet index o K + k.
    tau_earliest, earliest = torch.where(on_target, tau_first[..., None], INF).min(1)  # (B, N)
    shooter = torch.where(tau_earliest < INF, earliest // K, -1)
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
