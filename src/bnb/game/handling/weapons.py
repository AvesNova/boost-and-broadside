"""The combat handles (ship-card spec 3) and the card's weapons, shields and armour, in closed
form both ways.

Times are stated in seconds, and the fire rate in shots per second; each must be a whole
number of ticks at the rules' tick, or the conversion fails with the reason (spec 5). The
gun's range is the distance a bullet fired at cruise along the flight path travels in its
life: s = ln(1 + k_b w0 t) / k_b at n = 1, with w0 = cruise + muzzle speed and k_b the
type's own bullet drag (the `bullet_drag` handle; the card holds it as a scale of the
rules' drag, the floor). The life is the whole ticks at or above the time the type's drag
takes to cover the range, t = expm1(k_b R) / (k_b w0) rounded up, so the flown range is the
handle's to within one tick of travel; the card's handles read back the flown one.

The hit damage is a centre hit's by a bullet at its reference speed: the speed a shot at
cruise has at the rules' damage reference distance d, w0 exp(-k_b d) (the card's
`reference_bullet_speed`). The game scales it by the bullet's kinetic energy at the hit and
the closeness of the pass (`physics.combat.kinetic`, `closeness`).
"""

import math

import torch

from bnb.contracts import ticks_of

from ..rules import LONGEST_BULLET_LIFE, Rules
from ..ship_types import ShipTypes, bullet_moves, ticks_between_shots
from .errors import Infeasible

# Ticks: a range's flight time this close above a whole tick is that tick (the float error of
# a range measured from a card, whose life covers it exactly).
LIFE_TOLERANCE = 1e-4


def whole_tick_life(
    rules: Rules, distance: torch.Tensor, launch_speed: torch.Tensor, drag=None
) -> torch.Tensor:
    """The bullet life (s, whole ticks) that covers `distance` at this launch speed under
    this drag (1/m; the rules' by default): t = expm1(k_b R) / (k_b w0) rounded up to a
    tick."""
    drag = rules.bullet_drag if drag is None else drag
    flight = torch.expm1(drag * distance) / (drag * launch_speed)
    return torch.ceil(flight / rules.tick_length - LIFE_TOLERANCE) * rules.tick_length


def bullet_life(
    rules: Rules, distance: torch.Tensor, launch_speed: torch.Tensor, drag=None
) -> torch.Tensor:
    """`whole_tick_life`, or Infeasible with the reason past LONGEST_BULLET_LIFE."""
    life = whole_tick_life(rules, distance, launch_speed, drag)
    if (life > LONGEST_BULLET_LIFE + 0.5 * rules.tick_length).any():
        raise Infeasible(
            f"a gun range needs a bullet life of {float(life.max()):.3g} s at this cruise and "
            f"muzzle speed and bullet drag, beyond the longest, {LONGEST_BULLET_LIFE:g} s: "
            "shorten it, or raise the muzzle speed or lower the bullet drag"
        )
    return life


def gun_range(drag: torch.Tensor, launch_speed: torch.Tensor, life) -> torch.Tensor:
    """The distance a bullet launched at this proper speed travels in its life (s), n = 1."""
    return torch.log1p(drag * launch_speed * life) / drag


def reference_bullet_speed(
    rules: Rules, drag: torch.Tensor, launch_speed: torch.Tensor
) -> torch.Tensor:
    """The proper speed (m/s) of a bullet launched at this speed once it has flown the rules'
    damage reference distance, n = 1: w(s) = w0 exp(-k_b s). Past its range it is the speed
    the bullet would have there."""
    return launch_speed * torch.exp(-drag * rules.damage_reference_distance)


def whole_ticks(seconds: torch.Tensor, rules: Rules, what: str) -> torch.Tensor:
    """seconds (any shape) as int32 whole ticks of the rules' tick, or Infeasible with the
    reason."""
    try:
        ticks = [ticks_of(float(value), rules.tick_length) for value in seconds.flatten()]
    except ValueError as error:
        raise Infeasible(f"{what}: {error} (at {1.0 / rules.tick_length:g} Hz)") from error
    return torch.tensor(ticks, dtype=torch.int32, device=seconds.device).reshape(seconds.shape)


def combat_card(rules: Rules, handles, cruise_speed: torch.Tensor) -> dict[str, torch.Tensor]:
    """The card's combat fields from the combat handles (and the store and cruise, which the
    gun's cost and range depend on)."""
    if ((handles.head_on_fraction < 0.0) | (handles.head_on_fraction > 1.0)).any():
        raise Infeasible("a head-on fraction must lie in [0, 1]")
    if ((handles.head_on_angle <= 0.0) | (handles.head_on_angle > math.pi)).any():
        raise Infeasible("a head-on angle must lie in (0, 180] degrees")
    if ((handles.explosive_share < 0.0) | (handles.explosive_share > 1.0)).any():
        raise Infeasible("an explosive share must lie in [0, 1]")
    if (handles.bullet_drag <= 0.0).any():
        raise Infeasible("a bullet drag must be positive")
    if (handles.shots_per_store < 1.0).any():
        raise Infeasible("a full store must pay for at least one shot")
    whole_ticks(handles.shield_recharge_delay, rules, "shield recharge delay")
    try:
        ticks_between_shots(handles.fire_rate, rules.tick_length)
    except ValueError as error:
        raise Infeasible(str(error)) from error
    launch_speed = cruise_speed + handles.muzzle_speed
    drag = handles.bullet_drag
    life = bullet_life(rules, handles.gun_range, launch_speed, drag)
    return dict(
        hit_damage=handles.hit_damage,
        explosive_share=handles.explosive_share,
        reference_bullet_speed=reference_bullet_speed(rules, drag, launch_speed),
        fire_rate=handles.fire_rate,
        bullet_life=life,
        muzzle_speed=handles.muzzle_speed,
        bullet_spread_scale=handles.bullet_spread / rules.bullet_spread,
        bullet_drag_scale=drag / rules.bullet_drag,
        bullet_energy_cost=handles.ceiling / handles.shots_per_store,
        max_shield=handles.max_shield,
        shield_recharge_rate_scale=handles.shield_recharge_rate / rules.shield_recharge_rate,
        shield_recharge_delay_scale=handles.shield_recharge_delay / rules.shield_recharge_delay,
        head_on_fraction_scale=handles.head_on_fraction / rules.head_on_fraction,
        head_on_angle_scale=handles.head_on_angle / rules.head_on_angle,
        hit_radius=handles.hit_radius,
        spawn_energy=handles.spawn_energy,
    )


def combat_handles(rules: Rules, types: ShipTypes, cruise_speed: torch.Tensor) -> dict:
    """The combat handles of a card (fields (B, num_types)) flying at this cruise: the gun
    range is the one flown in the card's life (the handle's to within a tick of travel)."""
    drag = rules.bullet_drag * types.bullet_drag_scale
    life = bullet_moves(types.bullet_life, rules.tick_length) * rules.tick_length
    return dict(
        hit_damage=types.hit_damage,
        explosive_share=types.explosive_share,
        fire_rate=types.fire_rate,
        muzzle_speed=types.muzzle_speed,
        bullet_spread=rules.bullet_spread * types.bullet_spread_scale,
        gun_range=gun_range(drag, cruise_speed + types.muzzle_speed, life),  # the flown range
        bullet_drag=drag,
        shots_per_store=types.max_energy / types.bullet_energy_cost,
        max_shield=types.max_shield,
        shield_recharge_rate=rules.shield_recharge_rate * types.shield_recharge_rate_scale,
        # Held to whole ticks: the float32 card leaves a delay a hair off its tick.
        shield_recharge_delay=torch.round(
            rules.shield_recharge_delay * types.shield_recharge_delay_scale / rules.tick_length
        )
        * rules.tick_length,
        head_on_fraction=rules.head_on_fraction * types.head_on_fraction_scale,
        head_on_angle=rules.head_on_angle * types.head_on_angle_scale,
        hit_radius=types.hit_radius,
        spawn_energy=types.spawn_energy,
    )
