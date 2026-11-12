"""Bullets (game spec 6): firing and motion."""

import dataclasses

import torch

from bnb.contracts.tree import contract

from ..rules import Rules
from ..ship_types import ShipTypes, bullet_moves, per_ship
from ..state import Bullets, Fields, Ships, apply, replace, where
from .medium import medium
from .numerics import flight_path_velocity, rotate, small_angle_cos_sin


def _half_bend(tick_length: float, bullet: Bullets, index: torch.Tensor):
    """Turn the direction toward higher n by half a tick: rate (w/n) (d x grad ln n)."""
    turn = (
        0.5
        * tick_length
        * (bullet.proper_speed / index)
        * (
            bullet.direction_x * bullet.log_index_gradient_y
            - bullet.direction_y * bullet.log_index_gradient_x
        )
    )
    return rotate(bullet.direction_x, bullet.direction_y, *small_angle_cos_sin(turn))


def _half_drag(rules: Rules, drag_scale, speed: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Half a tick of quadratic drag on proper speed, dw/dt = -k w^2 / n, n held: exact."""
    return speed / (1.0 + rules.bullet_drag * 0.5 * rules.tick_length * drag_scale * speed / index)


def bullet_motion(rules: Rules, bullet: Bullets, fields: Fields, drag_scale=1.0) -> Bullets:
    """One move of every bullet given (game spec 6.2, normative); moves_left is untouched
    and the caller freezes inactive bullets. Shapes (B, P). drag_scale: each bullet's
    shooter's scale of the shared drag (B, P), or one number (1: the shared drag)."""
    tick_length = rules.tick_length
    index = torch.exp(bullet.log_refractive_index)
    bullet = replace(bullet, proper_speed=_half_drag(rules, drag_scale, bullet.proper_speed, index))
    direction_x, direction_y = _half_bend(tick_length, bullet, index)
    speed = bullet.proper_speed / index
    correction = (
        0.5
        * tick_length
        * speed
        * (direction_x * bullet.log_index_gradient_x + direction_y * bullet.log_index_gradient_y)
    )  # xi
    step = speed * tick_length / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    x, y = bullet.x + direction_x * step, bullet.y + direction_y * step
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    index = torch.exp(log_index)
    bullet = replace(
        bullet,
        x=x,
        y=y,
        direction_x=direction_x,
        direction_y=direction_y,
        log_refractive_index=log_index,
        log_index_gradient_x=gradient_x,
        log_index_gradient_y=gradient_y,
    )
    direction_x, direction_y = _half_bend(tick_length, bullet, index)
    return replace(
        bullet,
        direction_x=direction_x,
        direction_y=direction_y,
        proper_speed=_half_drag(rules, drag_scale, bullet.proper_speed, index),
    )


@contract
class Launch:
    """What `fire` reads of each ship to launch its bullet, (B, N): the position (m), the
    nose (heading_cos, heading_sin), proper_speed (m/s), the angle of attack's cos and sin,
    the medium cached at the ship (log_refractive_index and its gradient, 1/m) and
    ship_type (int32, for the muzzle speed). The game reads it from the tick-start state;
    the learner's shot re-simulation stores it and fires the same bullet."""

    x: torch.Tensor
    y: torch.Tensor
    heading_cos: torch.Tensor
    heading_sin: torch.Tensor
    proper_speed: torch.Tensor
    angle_of_attack_cos: torch.Tensor
    angle_of_attack_sin: torch.Tensor
    log_refractive_index: torch.Tensor
    log_index_gradient_x: torch.Tensor
    log_index_gradient_y: torch.Tensor
    ship_type: torch.Tensor


LAUNCH_FIELDS = tuple(field.name for field in dataclasses.fields(Launch))  # (traced as constants)


def launch_state(ships: Ships) -> Launch:
    """Each ship's launch state, as `fire` reads it."""
    return Launch(**{name: getattr(ships, name) for name in LAUNCH_FIELDS})


def launch(rules: Rules, types: ShipTypes, state: Launch, spread: torch.Tensor) -> Bullets:
    """The bullet each ship fires from its launch state, (B, N), with all its moves left (its
    type's bullet life). spread (B, N, 2) standard normals. The gun's values are the
    shooter's."""
    cos, sin, speed = state.heading_cos, state.heading_sin, state.proper_speed
    angle_cos, angle_sin = state.angle_of_attack_cos, state.angle_of_attack_sin
    muzzle = per_ship(types.muzzle_speed, state.ship_type)
    spread_speed = rules.bullet_spread * per_ship(types.bullet_spread_scale, state.ship_type)
    # Proper velocity: the ship's own along its flight path, plus the muzzle velocity along
    # the nose, plus the spread.
    path_x, path_y = flight_path_velocity(speed, cos, sin, angle_cos, angle_sin)
    velocity_x = path_x + muzzle * cos + spread_speed * spread[..., 0]
    velocity_y = path_y + muzzle * sin + spread_speed * spread[..., 1]
    bullet_speed = torch.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
    return Bullets(
        x=state.x,
        y=state.y,
        direction_x=velocity_x / bullet_speed,
        direction_y=velocity_y / bullet_speed,
        proper_speed=bullet_speed,
        log_refractive_index=state.log_refractive_index,
        log_index_gradient_x=state.log_index_gradient_x,
        log_index_gradient_y=state.log_index_gradient_y,
        moves_left=bullet_moves(per_ship(types.bullet_life, state.ship_type), rules.tick_length),
    )


def fire(
    rules: Rules,
    types: ShipTypes,
    ships: Ships,
    bullets: Bullets,
    shoot: torch.Tensor,
    spread: torch.Tensor,
) -> tuple[Ships, Bullets]:
    """Phase 1 (game spec 6.1), from the tick-start state. shoot (B, N) bool; spread
    (B, N, 2) standard normals."""
    cost = per_ship(types.bullet_energy_cost, ships.ship_type)
    fired = ships.alive & shoot & (ships.gun_cooldown == 0) & (ships.energy >= cost)

    def slot_value(value):
        return value[..., None].expand(bullets.moves_left.shape)

    new = apply(slot_value, launch(rules, types, launch_state(ships), spread))
    # The tick rate over the fire rate, whole ticks for any card reset accepted
    # (ship_types.ticks_between_shots, in exact fractions): rounding removes only the float error.
    rate = per_ship(types.fire_rate, ships.ship_type)
    interval = torch.round(1.0 / (rules.tick_length * rate)).to(torch.int32)
    slot = torch.arange(rules.bullet_slots, device=fired.device)
    bullets = where(fired[..., None] & (slot == ships.next_slot[..., None]), new, bullets)
    ships = replace(
        ships,
        energy=torch.where(fired, ships.energy - cost, ships.energy),
        gun_cooldown=torch.where(fired, interval, ships.gun_cooldown),
        next_slot=torch.where(fired, (ships.next_slot + 1) % rules.bullet_slots, ships.next_slot),
        fired=fired,
    )
    return ships, bullets


def move_bullets(
    rules: Rules, types: ShipTypes, bullets: Bullets, fields: Fields, ship_type: torch.Tensor
) -> Bullets:
    """Phase 3: every active bullet moves, with its shooter's drag; inactive ones are
    frozen. ship_type (B, N), the owners' types."""
    B, N, K = bullets.moves_left.shape
    drag_scale = per_ship(types.bullet_drag_scale, ship_type)[..., None].expand(B, N, K)
    moved = bullet_motion(
        rules, apply(lambda v: v.reshape(B, N * K), bullets), fields, drag_scale.reshape(B, N * K)
    )
    moved = apply(lambda v: v.reshape(B, N, K), moved)
    return where(bullets.moves_left > 0, moved, bullets)
