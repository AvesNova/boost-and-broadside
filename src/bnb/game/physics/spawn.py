"""Spawning (SPEC 8) and new maps for reset environments (SPEC 9)."""

import math

import torch

from .constants import FIELD_LEVELS, Preset
from .medium import medium
from .ship_types import type_value
from .state import BULLET_KEYS, FIELD_KEYS, fields_of, select

GOLDEN = 0.61803398875  # spawn phase step per ship index
GOLDEN_T = 0.38196601125  # spawn phase step per tick


def new_map(p: Preset, uniforms: torch.Tensor) -> dict:
    """Fields and spawn axis from (B, 4M + 1) uniforms."""
    B, M = uniforms.shape[0], p.num_fields
    u = uniforms[:, : 4 * M].reshape(B, M, 4)
    dist = 0.85 * p.arena_radius * torch.sqrt(u[..., 0])
    angle = 2.0 * math.pi * u[..., 1]
    levels = torch.tensor(FIELD_LEVELS, dtype=torch.int32, device=uniforms.device)
    axis = 2.0 * math.pi * uniforms[:, 4 * M]
    return dict(fx=dist * torch.cos(angle), fy=dist * torch.sin(angle),
                fr=p.map_scale * (30.0 + 460.0 * u[..., 2]), fw=torch.full_like(dist, p.field_width),
                flevel=levels[(4.0 * u[..., 3]).long().clamp_max(3)],
                axis_x=torch.cos(axis), axis_y=torch.sin(axis))


def clear_envs(p: Preset, st: dict, done: torch.Tensor, fresh: dict) -> dict:
    """Section 9 except the spawns: the new map, t = 0, no bullets, cursors and fired cleared."""
    out = dict(st)
    out.update(select(done, fresh, st, FIELD_KEYS + ("axis_x", "axis_y")))
    out.update(select(done, {k: torch.zeros_like(st[k]) for k in BULLET_KEYS + ("cursor", "t")}, st,
                      BULLET_KEYS + ("cursor", "t")))
    out["fired"] = st["fired"] & ~done[:, None]
    return out


def spawn(p: Preset, st: dict, mask: torch.Tensor, t: torch.Tensor) -> dict:
    """Spawn the ships in mask (B, N) at tick counter t (B,), at their team's spawn disc."""
    B, N = mask.shape
    index = torch.arange(N, device=mask.device)
    side = torch.where(index < p.half, -1.0, 1.0)  # team 0 spawns at -0.8 R_a axis, team 1 at +
    # The phase in float64: frac() of a float32 0.382 t near T_ep is off by ~1e-4 turns,
    # which would move a battle spawn by ~4 cm (a game error, not a tolerance one).
    phase = torch.frac(GOLDEN * index.double() + GOLDEN_T * t.double()[:, None])
    angle = (2.0 * math.pi * phase).float()
    ax, ay = st["axis_x"][:, None], st["axis_y"][:, None]
    rho = p.spawn_radius * p.map_scale
    x = side * p.spawn_offset * p.arena_radius * ax + rho * torch.cos(angle)
    y = side * p.spawn_offset * p.arena_radius * ay + rho * torch.sin(angle)
    logn, gx, gy = medium(fields_of(st), x, y)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    fresh = dict(x=x, y=y, c=-side * ax, s=-side * ay, alpha=zeros, ca=ones, sa=zeros, r=zeros,
                 logn=logn, gx=gx, gy=gy, w=p.spawn_speed * torch.exp(logn),
                 E=type_value(p, "spawn_energy", st["ship_type"]), shield=type_value(p, "shield_max", st["ship_type"]),
                 shield_wait=zeros,
                 gun_wait=torch.zeros_like(st["gun_wait"]), respawn=torch.zeros_like(st["respawn"]),
                 alive=torch.ones_like(mask))
    out = dict(st)
    out.update(select(mask, fresh, st, tuple(fresh)))
    out["spawned"] = mask
    return out
