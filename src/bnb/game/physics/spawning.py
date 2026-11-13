"""New maps for reset environments (game spec 9) and spawning (game spec 8).

Spawning keeps game spec 8's spawn state; where a ship appears and which way it points
come from the mode (`modes.spawns`). The formation of game spec 8 is sandbox's placement.
"""

import math
from typing import NamedTuple

import torch

from ..rules import FIELD_LEVELS, TEAMS, Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Fields, Ships, replace, ship_team, where
from .medium import medium
from .numerics import lookup


class TickUniforms:
    """Slices of the (B, U) tick uniforms (game spec 2): the map, a formation rotation per
    team, a formation key per ship; then the mode's spawn draws (modes spec 2), as wide as
    the widest mode of the batch needs (each mode reads its own prefix)."""

    def __init__(self, rules: Rules, uniforms: torch.Tensor) -> None:
        start = rules.map_uniform_count
        self.map = uniforms[:, :start]
        self.rotation = uniforms[:, start : start + TEAMS]
        self.keys = uniforms[:, start + TEAMS : start + TEAMS + rules.num_ships]
        self.mode = uniforms[:, rules.game_uniform_count :]


class Placement(NamedTuple):
    """Where spawning ships appear and which way they point, (B, N) each."""

    x: torch.Tensor
    y: torch.Tensor
    heading_cos: torch.Tensor
    heading_sin: torch.Tensor


def uniform_disc(radius, radius_uniform: torch.Tensor, angle_uniform: torch.Tensor):
    """(x, y) uniform points in a disc of this radius (a number, or broadcasting against the
    uniforms) around the origin, from two uniforms in [0, 1) each."""
    distance = radius * torch.sqrt(radius_uniform)
    angle = 2.0 * math.pi * angle_uniform
    return distance * torch.cos(angle), distance * torch.sin(angle)


def field_level(rules: Rules, uniform: torch.Tensor) -> torch.Tensor:
    """Field levels (FIELD_LEVELS) from uniforms in [0, 1): low with probability
    1 - field_high_share (then strong, -2, with field_strong_share), high otherwise. The
    default shares of 1/2 give the four levels a quarter each."""
    high, strong = rules.field_high_share, rules.field_strong_share
    edges = ((1.0 - high) * strong, 1.0 - high, 1.0 - high + high * (1.0 - strong))
    index = sum((uniform >= edge).to(torch.int32) for edge in edges)
    return lookup(FIELD_LEVELS, index)


def new_map(
    rules: Rules, arena_radius: torch.Tensor, map_scale: torch.Tensor, uniforms: torch.Tensor
) -> tuple[Fields, torch.Tensor, torch.Tensor]:
    """Fields and the spawn axis (x, y) from (B, 4M + 1) map uniforms, in arenas of this
    radius and map scale (B,).

    Point-symmetric: field P + i is field i rotated by 180 degrees about the centre
    (P = M // 2), and an odd last field sits at the centre.
    """
    B, M = uniforms.shape[0], rules.num_fields
    P = M // 2
    per_field = uniforms[:, : 4 * M].reshape(B, M, 4)
    x, y = uniform_disc(
        (rules.field_disc_fraction * arena_radius)[:, None],
        per_field[:, :P, 0],
        per_field[:, :P, 1],
    )
    drawn = torch.cat(
        [per_field[:, :P], per_field[:, :P], per_field[:, 2 * P :]], 1
    )  # pairs, centre
    radius = map_scale[:, None] * field_radius(rules, drawn[..., 2])
    level = field_level(rules, drawn[..., 3])
    centre = torch.zeros((B, M - 2 * P), device=uniforms.device)
    fields = Fields(
        x=torch.cat([x, -x, centre], 1),
        y=torch.cat([y, -y, centre], 1),
        radius=radius,
        band_width=torch.zeros_like(radius) + rules.field_band_width,
        level=level,
    )
    axis = 2.0 * math.pi * uniforms[:, 4 * M]
    return fields, torch.cos(axis), torch.sin(axis)


def _live(value, like: torch.Tensor, dtype=None) -> torch.Tensor:
    """A live rule value (a number or a 0-d tensor) as a tensor on `like`'s device: a number
    is filled there, not copied from the host, which a CUDA graph capture forbids."""
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=dtype or value.dtype)
    return torch.full((), value, device=like.device, dtype=dtype)


def field_radius(rules: Rules, u: torch.Tensor) -> torch.Tensor:
    """A field's radius in metres at map scale 1 from its uniform draw `u` in [0, 1), by
    the rules' `field_radius_distribution` (every branch is computed and the rule's value
    selects one, so the choice is live data in the compiled tick)."""
    low, high = rules.field_radius_min, rules.field_radius_max
    span = high - low
    code = _live(rules.field_radius_distribution, u)
    uniform = low + span * u
    triangular = low + span * (1.0 - torch.sqrt(1.0 - u))
    eps = 1e-6
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u.clamp(eps, 1.0 - eps) - 1.0)
    lognormal = rules.field_radius_median * torch.exp(rules.field_radius_log_sigma * z)
    lognormal = torch.maximum(
        torch.minimum(lognormal, _live(high, u, u.dtype)), _live(low, u, u.dtype)
    )
    return torch.where(code == 0, uniform, torch.where(code == 1, triangular, lognormal))


def field_radius_summary(rules: Rules) -> tuple[float, float, float]:
    """(mean, 10th percentile, 90th percentile) of a field's radius in metres at map scale
    1 under these rules, on the CPU from the same formula (a midpoint grid of the uniform)."""
    grid = (torch.arange(4000, dtype=torch.float64) + 0.5) / 4000.0
    radius = field_radius(rules, grid)
    return radius.mean().item(), radius[400].item(), radius[3599].item()


def formation_rank(rules: Rules, keys: torch.Tensor) -> torch.Tensor:
    """(B, N) int: each ship's rank among its team's keys, ties to the lower slot."""
    B, S = keys.shape[0], rules.ships_per_team
    key = keys.reshape(B, TEAMS, S)
    slot = torch.arange(S, device=keys.device)
    before = (key[..., None, :] < key[..., :, None]) | (
        (key[..., None, :] == key[..., :, None]) & (slot < slot[:, None])
    )  # [i, j]
    return before.sum(-1).reshape(B, TEAMS * S)


def formation(
    rules: Rules,
    arena_radius: torch.Tensor,
    map_scale: torch.Tensor,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    rotation: torch.Tensor,
    keys: torch.Tensor,
) -> Placement:
    """Game spec 8's formation for every ship: around its team's centre on the spawn axis
    (B,), at its rank's phase plus its team's rotation (B, 2), nose toward the other
    centre. keys (B, N): the formation keys; arena_radius and map_scale (B,)."""
    S = rules.ships_per_team
    team = ship_team(S, keys.device)
    side = (2 * team - 1).to(torch.float32)  # team 0 on the -axis side, team 1 on the +
    phase = formation_rank(rules, keys) / S + rotation.repeat_interleave(S, 1)
    angle = 2.0 * math.pi * torch.frac(phase)
    axis_x, axis_y = axis_x[:, None], axis_y[:, None]
    radius = (rules.spawn_radius * map_scale)[:, None]
    offset = side * (rules.spawn_offset * arena_radius)[:, None]
    return Placement(
        x=offset * axis_x + radius * torch.cos(angle),
        y=offset * axis_y + radius * torch.sin(angle),
        heading_cos=-side * axis_x.expand_as(angle),
        heading_sin=-side * axis_y.expand_as(angle),
    )


def spawn(
    rules: Rules,
    types: ShipTypes,
    ships: Ships,
    fields: Fields,
    mask: torch.Tensor,
    placement: Placement,
) -> Ships:
    """Spawn the ships in mask (B, N) at their placement (game spec 8's spawn state).

    Computed for every ship and selected by mask, so the graph has no data-dependent shape.
    """
    x, y = placement.x, placement.y
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    integer_zeros = torch.zeros_like(ships.gun_cooldown)
    fresh = Ships(
        x=x,
        y=y,
        heading_cos=placement.heading_cos,
        heading_sin=placement.heading_sin,
        proper_speed=rules.spawn_speed * torch.exp(log_index),
        angle_of_attack=zeros,
        angle_of_attack_cos=ones,
        angle_of_attack_sin=zeros,
        nose_rate=zeros,
        energy=per_ship(types.spawn_energy, ships.ship_type),
        log_refractive_index=log_index,
        log_index_gradient_x=gradient_x,
        log_index_gradient_y=gradient_y,
        shield=per_ship(types.max_shield, ships.ship_type),
        shield_recharge_wait=zeros,
        alive=torch.ones_like(mask),
        fired=ships.fired,
        spawned=mask,
        respawn_wait=integer_zeros,
        gun_cooldown=integer_zeros,
        next_slot=ships.next_slot,
        ship_type=ships.ship_type,
    )
    return replace(where(mask, fresh, ships), spawned=mask)
