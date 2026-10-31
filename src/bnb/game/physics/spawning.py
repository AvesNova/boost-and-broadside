"""New maps for reset environments (game spec 9) and spawning (game spec 8)."""

import math

import torch

from ..rules import FIELD_LEVELS, Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Fields, Ships, replace, where
from .medium import medium
from .numerics import lookup


class TickUniforms:
    """Slices of the (B, U) tick uniforms (game spec 2): the map, a formation rotation per
    team, a formation key per ship."""

    def __init__(self, rules: Rules, u: torch.Tensor) -> None:
        m, T = rules.map_uniforms, rules.num_teams
        self.map = u[:, :m]
        self.rotation = u[:, m:m + T]
        self.keys = u[:, m + T:m + T + rules.num_ships]


def new_map(rules: Rules, u: torch.Tensor) -> tuple[Fields, torch.Tensor, torch.Tensor]:
    """Fields and the spawn axis (axis_x, axis_y) from (B, 4M + 1) map uniforms.

    Point-symmetric: field P + i is field i rotated by 180 degrees about the centre
    (P = M // 2), and an odd last field sits at the centre.
    """
    B, M = u.shape[0], rules.num_fields
    P = M // 2
    f = u[:, :4 * M].reshape(B, M, 4)
    dist = 0.85 * rules.arena_radius * torch.sqrt(f[:, :P, 0])
    angle = 2.0 * math.pi * f[:, :P, 1]
    x, y = dist * torch.cos(angle), dist * torch.sin(angle)
    drawn = torch.cat([f[:, :P], f[:, :P], f[:, 2 * P:]], 1)  # mirrored pairs, then the centre
    radius = rules.map_scale * (30.0 + 460.0 * drawn[..., 2])
    level = lookup(FIELD_LEVELS, (4.0 * drawn[..., 3]).to(torch.int32).clamp_max(3))
    centre = torch.zeros((B, M - 2 * P), device=u.device)
    fields = Fields(x=torch.cat([x, -x, centre], 1), y=torch.cat([y, -y, centre], 1),
                    radius=radius, width=torch.full_like(radius, rules.field_width), level=level)
    axis = 2.0 * math.pi * u[:, 4 * M]
    return fields, torch.cos(axis), torch.sin(axis)


def formation_rank(rules: Rules, keys: torch.Tensor) -> torch.Tensor:
    """(B, N) int: each ship's rank among its team's keys, ties to the lower slot."""
    B, T, S = keys.shape[0], rules.num_teams, rules.ships_per_team
    k = keys.reshape(B, T, S)
    slot = torch.arange(S, device=keys.device)
    before = (k[..., None, :] < k[..., :, None]) \
        | ((k[..., None, :] == k[..., :, None]) & (slot < slot[:, None]))  # [.., i, j]: j first
    return before.sum(-1).reshape(B, T * S)


def spawn(rules: Rules, types: ShipTypes, ships: Ships, fields: Fields, axis_x, axis_y,
          mask: torch.Tensor, rotation: torch.Tensor, keys: torch.Tensor) -> Ships:
    """Spawn the ships in mask (B, N) on their team's formation (game spec 8).

    axis_x, axis_y (B,): the spawn axis; rotation (B, T) and keys (B, N): tick uniforms.
    """
    N, S = mask.shape[1], rules.ships_per_team
    team = torch.arange(N, device=mask.device) // S
    side = (2 * team - 1).to(torch.float32)  # team 0 spawns at -0.8 R_a axis, team 1 at +
    phase = formation_rank(rules, keys) / S + rotation.repeat_interleave(S, 1)
    angle = 2.0 * math.pi * torch.frac(phase)
    ax, ay = axis_x[:, None], axis_y[:, None]
    rho = rules.spawn_radius * rules.map_scale
    offset = side * (rules.spawn_offset * rules.arena_radius)
    x = offset * ax + rho * torch.cos(angle)
    y = offset * ay + rho * torch.sin(angle)
    logn, gx, gy = medium(fields, x, y)
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    izeros = torch.zeros_like(ships.gun_wait)
    fresh = Ships(x=x, y=y, c=-side * ax, s=-side * ay, w=rules.spawn_speed * torch.exp(logn),
                  alpha=zeros, ca=ones, sa=zeros, r=zeros,
                  E=per_ship(types.spawn_energy, ships.ship_type), logn=logn, gx=gx, gy=gy,
                  shield=per_ship(types.shield_max, ships.ship_type), shield_wait=zeros,
                  alive=torch.ones_like(mask), fired=ships.fired, spawned=mask,
                  respawn=izeros, gun_wait=izeros, cursor=ships.cursor,
                  ship_type=ships.ship_type)
    return replace(where(mask, fresh, ships), spawned=mask)
