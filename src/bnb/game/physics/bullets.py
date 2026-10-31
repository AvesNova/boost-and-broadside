"""Bullets (game spec 6): firing and motion."""

import torch

from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, Fields, Ships, apply, replace, where
from .medium import medium
from .numerics import cs, rotate


def _half_bend(h: float, b: Bullets, n: torch.Tensor):
    """Turn the direction toward higher n by half a tick: rate (w/n) (d x grad ln n)."""
    turn = 0.5 * h * (b.w / n) * (b.dx * b.gy - b.dy * b.gx)
    return rotate(b.dx, b.dy, *cs(turn))


def _half_drag(rules: Rules, w: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Half a tick of quadratic drag on proper speed, dw/dt = -k w^2 / n, n held: exact."""
    return w / (1.0 + rules.bullet_drag * 0.5 * rules.tick * w / n)


def bullet_motion(rules: Rules, b: Bullets, fields: Fields) -> Bullets:
    """One move of every bullet given (game spec 6.2, normative); ttl is untouched and the
    caller freezes inactive bullets. Shapes (B, P)."""
    h = rules.tick
    n = torch.exp(b.logn)
    b = replace(b, w=_half_drag(rules, b.w, n))
    dx, dy = _half_bend(h, b, n)
    speed = b.w / n
    xi = 0.5 * h * speed * (dx * b.gx + dy * b.gy)
    step = speed * h / (1.0 + xi * (1.0 + xi * (0.5 + xi / 6.0)))
    x, y = b.x + dx * step, b.y + dy * step
    logn, gx, gy = medium(fields, x, y)
    n = torch.exp(logn)
    b = replace(b, x=x, y=y, dx=dx, dy=dy, logn=logn, gx=gx, gy=gy)
    dx, dy = _half_bend(h, b, n)
    return replace(b, dx=dx, dy=dy, w=_half_drag(rules, b.w, n))


def fire(rules: Rules, types: ShipTypes, ships: Ships, bullets: Bullets, shoot: torch.Tensor,
         spread: torch.Tensor) -> tuple[Ships, Bullets]:
    """Phase 1 (game spec 6.1), from the tick-start state. shoot (B, N) bool; spread
    (B, N, 2) standard normals. The gun's values are the shooter's."""
    cost = per_ship(types.bullet_cost, ships.ship_type)
    fired = ships.alive & shoot & (ships.gun_wait == 0) & (ships.E >= cost)
    c, s, w, ca, sa = ships.c, ships.s, ships.w, ships.ca, ships.sa
    muzzle = per_ship(types.muzzle_speed, ships.ship_type)
    # Proper velocity: the ship's own along its flight path, plus the muzzle velocity along
    # the nose, plus the spread.
    vx = w * (c * ca + s * sa) + muzzle * c + rules.spread * spread[..., 0]
    vy = w * (s * ca - c * sa) + muzzle * s + rules.spread * spread[..., 1]
    speed = torch.sqrt(vx * vx + vy * vy)
    new = Bullets(x=ships.x[..., None], y=ships.y[..., None], dx=(vx / speed)[..., None],
                  dy=(vy / speed)[..., None], w=speed[..., None], logn=ships.logn[..., None],
                  gx=ships.gx[..., None], gy=ships.gy[..., None],
                  ttl=torch.full_like(ships.gun_wait, rules.bullet_moves)[..., None])
    slot = fired[..., None] & (torch.arange(rules.slots, device=fired.device)
                               == ships.cursor[..., None])
    bullets = where(slot, apply(lambda v: v.expand(bullets.ttl.shape), new), bullets)
    ships = replace(
        ships, E=torch.where(fired, ships.E - cost, ships.E),
        gun_wait=torch.where(fired, per_ship(types.fire_interval, ships.ship_type),
                             ships.gun_wait),
        cursor=torch.where(fired, (ships.cursor + 1) % rules.slots, ships.cursor), fired=fired)
    return ships, bullets


def move_bullets(rules: Rules, bullets: Bullets, fields: Fields) -> Bullets:
    """Phase 3: every active bullet moves; inactive ones are frozen."""
    B, N, K = bullets.ttl.shape
    moved = bullet_motion(rules, apply(lambda v: v.reshape(B, N * K), bullets), fields)
    moved = apply(lambda v: v.reshape(B, N, K), moved)
    return where(bullets.ttl > 0, moved, bullets)
