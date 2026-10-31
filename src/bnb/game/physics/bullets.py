"""Bullets (SPEC 6): firing and motion."""

import torch

from .medium import medium
from .numerics import cs, rotate
from .ship_types import type_value

BULLET_MOTION_KEYS = ("bx", "by", "bdx", "bdy", "bw", "blogn", "bgx", "bgy")


def _half_bend(h: float, b: dict, n):
    """Turn the direction toward higher n by half a tick's worth: rate (w/n) (d x grad ln n)."""
    speed = b["bw"] / n
    turn = 0.5 * h * speed * (b["bdx"] * b["bgy"] - b["bdy"] * b["bgx"])
    return rotate(b["bdx"], b["bdy"], *cs(turn))


def bullet_motion(h: float, b: dict, fields: dict) -> dict:
    """One move of every bullet given (SPEC 6.2, normative). The caller freezes inactive ones."""
    n = torch.exp(b["blogn"])
    dx, dy = _half_bend(h, b, n)
    speed = b["bw"] / n
    xi = 0.5 * h * speed * (dx * b["bgx"] + dy * b["bgy"])
    step = speed * h / (1.0 + xi * (1.0 + xi * (0.5 + xi / 6.0)))
    x, y = b["bx"] + dx * step, b["by"] + dy * step
    logn, gx, gy = medium(fields, x, y)
    moved = dict(b, bx=x, by=y, bdx=dx, bdy=dy, blogn=logn, bgx=gx, bgy=gy)
    dx, dy = _half_bend(h, moved, torch.exp(logn))
    return dict(moved, bdx=dx, bdy=dy)


def fire(p, st: dict, shoot: torch.Tensor, spread: torch.Tensor) -> dict:
    """Phase 1 (SPEC 6.1), from the tick-start state. shoot (B, N) bool; spread (B, N, 2) normals.
    The gun's constants are the shooter's (SPEC 1.1)."""
    cost, muzzle = type_value(p, "bullet_cost", st["ship_type"]), type_value(p, "muzzle_speed", st["ship_type"])
    fired = st["alive"] & shoot & (st["gun_wait"] == 0) & (st["E"] >= cost)
    c, s, w, ca, sa = st["c"], st["s"], st["w"], st["ca"], st["sa"]
    # Proper velocity: the ship's own (along its flight path) plus the muzzle velocity plus spread.
    vx = w * (c * ca + s * sa) + muzzle * c + p.spread * spread[..., 0]
    vy = w * (s * ca - c * sa) + muzzle * s + p.spread * spread[..., 1]
    speed = torch.sqrt(vx * vx + vy * vy)
    new = dict(bx=st["x"], by=st["y"], bdx=vx / speed, bdy=vy / speed, bw=speed, blogn=st["logn"], bgx=st["gx"],
               bgy=st["gy"], ttl=torch.full_like(st["gun_wait"], p.bullet_moves))
    slot = fired[..., None] & (torch.arange(p.slots, device=fired.device) == st["cursor"][..., None])
    out = dict(st)
    for k, v in new.items():
        out[k] = torch.where(slot, v[..., None], st[k])
    out["E"] = torch.where(fired, st["E"] - cost, st["E"])
    out["gun_wait"] = torch.where(fired, type_value(p, "fire_interval", st["ship_type"]), st["gun_wait"])
    out["cursor"] = torch.where(fired, (st["cursor"] + 1) % p.slots, st["cursor"])
    out["fired"] = fired
    return out


def move_bullets(p, st: dict) -> dict:
    """Phase 3: every active bullet moves; inactive ones are frozen. Returns the new state."""
    B, N, K = st["ttl"].shape
    flat = {k: st[k].reshape(B, N * K) for k in BULLET_MOTION_KEYS}
    moved = bullet_motion(p.h, flat, {k: st[k] for k in ("fx", "fy", "fr", "fw", "flevel")})
    active = st["ttl"] > 0
    out = dict(st)
    for k in BULLET_MOTION_KEYS:
        out[k] = torch.where(active, moved[k].reshape(B, N, K), st[k])
    return out
