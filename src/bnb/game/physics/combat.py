"""Hits (SPEC 7.1) and damage, death and shields (SPEC 7.2).

Bullets are paired only with the ships of the other team: with the team axis first, team
T's bullets (B, 2, N/2 K) meet `enemy` ships (B, 2, N/2), so every pair below is a
legal one and allied fire never needs masking.
"""

import math
from typing import NamedTuple

import torch

from .constants import Preset
from .ship_types import type_value
from .state import by_team, enemy, per_ship

INF = float("inf")


class Hits(NamedTuple):
    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    taken: torch.Tensor  # (B, N) hit damage taken by each ship
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship (-1 if none)


def hits(p: Preset, start: dict, end: dict, active: torch.Tensor, alive0: torch.Tensor) -> Hits:
    """Relative swept test between the bullet and ship motions of this tick.

    start: tick-start positions x, y, bx, by (bullets after firing); end: the moved state;
    active: (B, N, K) bullets that moved; alive0: (B, N) ships alive at tick start.
    """
    B, N, K = active.shape
    half = p.half

    def pair(key_b, key_s, st):  # bullet minus enemy-ship coordinate, (B, 2, N/2 K, N/2)
        return by_team(st[key_b])[..., :, None] - enemy(st[key_s])[..., None, :]

    d0x, d0y = pair("bx", "x", start), pair("by", "y", start)
    deltax, deltay = pair("bx", "x", end) - d0x, pair("by", "y", end) - d0y
    tau = (-(d0x * deltax + d0y * deltay) / (deltax * deltax + deltay * deltay).clamp_min(1e-12)).clamp(0.0, 1.0)
    missx, missy = d0x + tau * deltax, d0y + tau * deltay
    radius = enemy(type_value(p, "hit_radius", end["ship_type"]))[..., None, :]  # the target's
    contact = ((missx * missx + missy * missy) < radius * radius) & by_team(active)[..., None] \
        & enemy(alive0)[..., None, :]

    # Each bullet hits only its first contact: smallest tau, ties to the lowest ship index
    # (torch.min returns the first minimal index).
    tau_first, first = torch.where(contact, tau, INF).min(-1)  # (B, 2, N/2 K)
    hit = tau_first < INF

    # Damage by aspect, per bullet: theta between its reversed direction and its target's nose.
    dx, dy = by_team(end["bdx"]), by_team(end["bdy"])
    nc, ns = enemy(end["c"]).gather(2, first), enemy(end["s"]).gather(2, first)
    theta = torch.atan2((dx * ns - dy * nc).abs(), -(dx * nc + dy * ns))
    full = by_team(type_value(p, "damage", end["ship_type"])[..., None].expand(B, N, K))  # the shooter's D
    dmg = torch.where(hit, full * (1.0 - (1.0 - p.head_on_fraction) * torch.exp(-4.0 * theta * theta / math.pi)),
                      0.0)

    # Spread each bullet's hit over its target column: (B, 2, N/2 K, N/2), never stored.
    on_target = hit[..., None] & (first[..., None] == torch.arange(half, device=first.device))
    by_shooter = torch.where(on_target, dmg[..., None], 0.0).reshape(B, 2, half, K, half).sum(3)
    zeros = torch.zeros_like(by_shooter[:, 0])
    damage = torch.cat([torch.cat([zeros, by_shooter[:, 0]], -1), torch.cat([by_shooter[:, 1], zeros], -1)], 1)

    # The earliest hit on each target: smallest tau, ties to the lowest bullet index o K + slot.
    tau_earliest, earliest = torch.where(on_target, tau_first[..., None], INF).min(2)  # (B, 2, N/2) targets
    team_base = torch.tensor([0, half], device=first.device)[None, :, None]
    shooter = torch.where(tau_earliest < INF, team_base + earliest // K, -1)
    return Hits(damage=damage, taken=per_ship(by_shooter.sum(2).flip(1)), bullet_hit=hit.reshape(B, N, K),
                first_shooter=per_ship(shooter.flip(1)))


def boundary_damage(p: Preset, st: dict, alive0: torch.Tensor) -> torch.Tensor:
    """SPEC 5: (5 + 0.05 depth) h for a living ship outside the arena after its move."""
    depth = torch.sqrt(st["x"] * st["x"] + st["y"] * st["y"]) - p.arena_radius
    return torch.where(alive0 & (depth > 0.0), (p.boundary_base + p.boundary_per_m * depth) * p.h, 0.0)


def apply_damage(p: Preset, st: dict, alive0: torch.Tensor, total: torch.Tensor):
    """Phase 5 for ships alive at tick start: shields, recharge, and who dies. Returns (state, died)."""
    shield0, wait0 = st["shield"], st["shield_wait"]
    hurt = total > 0.0
    died = alive0 & hurt & (shield0 <= 0.0)  # an exposed core dies to anything
    shield = (shield0 - total).clamp_min(0.0)
    recharge = torch.minimum(type_value(p, "shield_max", st["ship_type"]) - shield,
                             p.shield_regen * (p.h - wait0).clamp(0.0, p.h))
    shield = torch.where(hurt, shield, shield + recharge)  # an unhurt ship cannot have died
    wait = torch.where(hurt, p.shield_delay, (wait0 - p.h).clamp_min(0.0))
    out = dict(st, shield=torch.where(alive0, shield, shield0), shield_wait=torch.where(alive0, wait, wait0))
    return out, died
