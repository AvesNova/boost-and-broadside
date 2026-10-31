"""Bullets (game spec 6): firing and motion."""

import torch

from ..rules import Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, Fields, Ships, apply, replace, where
from .medium import medium
from .numerics import rotate, small_angle_cos_sin


def _half_bend(tick_length: float, bullet: Bullets, index: torch.Tensor):
    """Turn the direction toward higher n by half a tick: rate (w/n) (d x grad ln n)."""
    turn = 0.5 * tick_length * (bullet.proper_speed / index) * (
        bullet.direction_x * bullet.log_index_gradient_y
        - bullet.direction_y * bullet.log_index_gradient_x)
    return rotate(bullet.direction_x, bullet.direction_y, *small_angle_cos_sin(turn))


def _half_drag(rules: Rules, speed: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Half a tick of quadratic drag on proper speed, dw/dt = -k w^2 / n, n held: exact."""
    return speed / (1.0 + rules.bullet_drag * 0.5 * rules.tick_length * speed / index)


def bullet_motion(rules: Rules, bullet: Bullets, fields: Fields) -> Bullets:
    """One move of every bullet given (game spec 6.2, normative); moves_left is untouched
    and the caller freezes inactive bullets. Shapes (B, P)."""
    tick_length = rules.tick_length
    index = torch.exp(bullet.log_refractive_index)
    bullet = replace(bullet, proper_speed=_half_drag(rules, bullet.proper_speed, index))
    direction_x, direction_y = _half_bend(tick_length, bullet, index)
    speed = bullet.proper_speed / index
    correction = 0.5 * tick_length * speed * (direction_x * bullet.log_index_gradient_x
                                    + direction_y * bullet.log_index_gradient_y)  # xi
    step = speed * tick_length / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    x, y = bullet.x + direction_x * step, bullet.y + direction_y * step
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    index = torch.exp(log_index)
    bullet = replace(bullet, x=x, y=y, direction_x=direction_x, direction_y=direction_y,
                     log_refractive_index=log_index, log_index_gradient_x=gradient_x,
                     log_index_gradient_y=gradient_y)
    direction_x, direction_y = _half_bend(tick_length, bullet, index)
    return replace(bullet, direction_x=direction_x, direction_y=direction_y,
                   proper_speed=_half_drag(rules, bullet.proper_speed, index))


def fire(rules: Rules, types: ShipTypes, ships: Ships, bullets: Bullets, shoot: torch.Tensor,
         spread: torch.Tensor) -> tuple[Ships, Bullets]:
    """Phase 1 (game spec 6.1), from the tick-start state. shoot (B, N) bool; spread
    (B, N, 2) standard normals. The gun's values are the shooter's."""
    cost = per_ship(types.bullet_energy_cost, ships.ship_type)
    fired = ships.alive & shoot & (ships.gun_cooldown == 0) & (ships.energy >= cost)
    cos, sin, speed = ships.heading_cos, ships.heading_sin, ships.proper_speed
    angle_cos, angle_sin = ships.angle_of_attack_cos, ships.angle_of_attack_sin
    muzzle = per_ship(types.muzzle_speed, ships.ship_type)
    # Proper velocity: the ship's own along its flight path, plus the muzzle velocity along
    # the nose, plus the spread.
    velocity_x = (speed * (cos * angle_cos + sin * angle_sin) + muzzle * cos
                  + rules.bullet_spread * spread[..., 0])
    velocity_y = (speed * (sin * angle_cos - cos * angle_sin) + muzzle * sin
                  + rules.bullet_spread * spread[..., 1])
    bullet_speed = torch.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)

    def slot_value(value):
        return value[..., None].expand(bullets.moves_left.shape)

    new = apply(slot_value, Bullets(
        x=ships.x, y=ships.y, direction_x=velocity_x / bullet_speed,
        direction_y=velocity_y / bullet_speed, proper_speed=bullet_speed,
        log_refractive_index=ships.log_refractive_index,
        log_index_gradient_x=ships.log_index_gradient_x,
        log_index_gradient_y=ships.log_index_gradient_y,
        moves_left=torch.full_like(ships.gun_cooldown, rules.bullet_moves)))
    slot = torch.arange(rules.bullet_slots, device=fired.device)
    bullets = where(fired[..., None] & (slot == ships.next_slot[..., None]), new, bullets)
    ships = replace(
        ships, energy=torch.where(fired, ships.energy - cost, ships.energy),
        gun_cooldown=torch.where(fired, per_ship(types.ticks_between_shots, ships.ship_type),
                                 ships.gun_cooldown),
        next_slot=torch.where(fired, (ships.next_slot + 1) % rules.bullet_slots, ships.next_slot),
        fired=fired)
    return ships, bullets


def move_bullets(rules: Rules, bullets: Bullets, fields: Fields) -> Bullets:
    """Phase 3: every active bullet moves; inactive ones are frozen."""
    B, N, K = bullets.moves_left.shape
    moved = bullet_motion(rules, apply(lambda v: v.reshape(B, N * K), bullets), fields)
    moved = apply(lambda v: v.reshape(B, N, K), moved)
    return where(bullets.moves_left > 0, moved, bullets)
