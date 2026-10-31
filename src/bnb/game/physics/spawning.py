"""New maps for reset environments (game spec 9) and spawning (game spec 8)."""

import math

import torch

from ..rules import FIELD_LEVELS, TEAMS, Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Fields, Ships, replace, where
from .medium import medium
from .numerics import lookup


class TickUniforms:
    """Slices of the (B, U) tick uniforms (game spec 2): the map, a formation rotation per
    team, a formation key per ship."""

    def __init__(self, rules: Rules, uniforms: torch.Tensor) -> None:
        start = rules.map_uniform_count
        self.map = uniforms[:, :start]
        self.rotation = uniforms[:, start:start + TEAMS]
        self.keys = uniforms[:, start + TEAMS:start + TEAMS + rules.num_ships]


def new_map(rules: Rules, uniforms: torch.Tensor) -> tuple[Fields, torch.Tensor, torch.Tensor]:
    """Fields and the spawn axis (x, y) from (B, 4M + 1) map uniforms.

    Point-symmetric: field P + i is field i rotated by 180 degrees about the centre
    (P = M // 2), and an odd last field sits at the centre.
    """
    B, M = uniforms.shape[0], rules.num_fields
    P = M // 2
    per_field = uniforms[:, :4 * M].reshape(B, M, 4)
    distance = 0.85 * rules.arena_radius * torch.sqrt(per_field[:, :P, 0])
    angle = 2.0 * math.pi * per_field[:, :P, 1]
    x, y = distance * torch.cos(angle), distance * torch.sin(angle)
    drawn = torch.cat([per_field[:, :P], per_field[:, :P], per_field[:, 2 * P:]], 1)  # pairs, centre
    radius = rules.map_scale * (30.0 + 460.0 * drawn[..., 2])
    level = lookup(FIELD_LEVELS, (4.0 * drawn[..., 3]).to(torch.int32).clamp_max(3))
    centre = torch.zeros((B, M - 2 * P), device=uniforms.device)
    fields = Fields(x=torch.cat([x, -x, centre], 1), y=torch.cat([y, -y, centre], 1),
                    radius=radius, band_width=torch.full_like(radius, rules.field_band_width),
                    level=level)
    axis = 2.0 * math.pi * uniforms[:, 4 * M]
    return fields, torch.cos(axis), torch.sin(axis)


def formation_rank(rules: Rules, keys: torch.Tensor) -> torch.Tensor:
    """(B, N) int: each ship's rank among its team's keys, ties to the lower slot."""
    B, S = keys.shape[0], rules.ships_per_team
    key = keys.reshape(B, TEAMS, S)
    slot = torch.arange(S, device=keys.device)
    before = ((key[..., None, :] < key[..., :, None])
              | ((key[..., None, :] == key[..., :, None]) & (slot < slot[:, None])))  # [i, j]
    return before.sum(-1).reshape(B, TEAMS * S)


def spawn(rules: Rules, types: ShipTypes, ships: Ships, fields: Fields, axis_x, axis_y,
          mask: torch.Tensor, rotation: torch.Tensor, keys: torch.Tensor) -> Ships:
    """Spawn the ships in mask (B, N) on their team's formation (game spec 8).

    axis_x, axis_y (B,): the spawn axis; rotation (B, 2) and keys (B, N): tick uniforms.
    Computed for every ship and selected by mask, so the graph has no data-dependent shape.
    """
    N, S = mask.shape[1], rules.ships_per_team
    team = torch.arange(N, device=mask.device) // S
    side = (2 * team - 1).to(torch.float32)  # team 0 spawns at -0.8 R_a axis, team 1 at +
    phase = formation_rank(rules, keys) / S + rotation.repeat_interleave(S, 1)
    angle = 2.0 * math.pi * torch.frac(phase)
    axis_x, axis_y = axis_x[:, None], axis_y[:, None]
    radius = rules.spawn_radius * rules.map_scale
    offset = side * (rules.spawn_offset * rules.arena_radius)
    x = offset * axis_x + radius * torch.cos(angle)
    y = offset * axis_y + radius * torch.sin(angle)
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    integer_zeros = torch.zeros_like(ships.gun_cooldown)
    fresh = Ships(
        x=x, y=y, heading_cos=-side * axis_x, heading_sin=-side * axis_y,
        proper_speed=rules.spawn_speed * torch.exp(log_index), angle_of_attack=zeros,
        angle_of_attack_cos=ones, angle_of_attack_sin=zeros, nose_rate=zeros,
        energy=per_ship(types.spawn_energy, ships.ship_type), log_refractive_index=log_index,
        log_index_gradient_x=gradient_x, log_index_gradient_y=gradient_y,
        shield=per_ship(types.max_shield, ships.ship_type), shield_recharge_wait=zeros,
        alive=torch.ones_like(mask), fired=ships.fired, spawned=mask, respawn_wait=integer_zeros,
        gun_cooldown=integer_zeros, next_slot=ships.next_slot, ship_type=ships.ship_type)
    return replace(where(mask, fresh, ships), spawned=mask)
