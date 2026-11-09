"""The combat handles (ship-card spec 3) and the card's weapons, shields and armour, in closed
form both ways.

Times are stated in seconds, and the fire rate in shots per second; each must be a whole
number of ticks at the rules' tick, or the conversion fails with the reason (spec 5). The
gun's range is the distance a bullet fired at cruise along the flight path travels in its
life: s = ln(1 + k_b w0 t) / k_b at n = 1, with w0 = cruise + muzzle speed, solved for the
drag k_b by bisection.
"""

import torch

from bnb.contracts import ticks_of

from ..rules import Rules
from ..ship_types import ShipTypes, ticks_between_shots
from .errors import Infeasible

DRAG_BISECTIONS = 200
DRAG_RANGE = (1e-12, 10.0)  # 1/m


def bullet_life(rules: Rules) -> float:
    """Every bullet's life in seconds: its moves at the rules' tick."""
    return rules.bullet_moves * rules.tick_length


def gun_range(rules: Rules, drag: torch.Tensor, launch_speed: torch.Tensor) -> torch.Tensor:
    """The distance a bullet launched at this proper speed travels in its life, n = 1."""
    reach = drag * launch_speed * bullet_life(rules)
    return torch.log1p(reach) / drag


def range_drag(rules: Rules, distance: torch.Tensor, launch_speed: torch.Tensor) -> torch.Tensor:
    """The drag k_b at which a bullet launched at this speed travels `distance` in its life
    (bisection in log k_b: the range falls as the drag grows)."""
    reach = launch_speed * bullet_life(rules)
    if (distance >= reach).any():
        raise Infeasible(
            "a gun range at or beyond the drag-free reach (cruise + muzzle speed) "
            "x bullet life cannot be met"
        )
    low = torch.full_like(distance, DRAG_RANGE[0]).log()
    high = torch.full_like(distance, DRAG_RANGE[1]).log()
    for _ in range(DRAG_BISECTIONS):
        middle = 0.5 * (low + high)
        too_far = gun_range(rules, middle.exp(), launch_speed) > distance
        low, high = torch.where(too_far, middle, low), torch.where(too_far, high, middle)
    return (0.5 * (low + high)).exp()


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
    if (handles.shots_per_store < 1.0).any():
        raise Infeasible("a full store must pay for at least one shot")
    whole_ticks(handles.shield_recharge_delay, rules, "shield recharge delay")
    try:
        ticks_between_shots(handles.fire_rate, rules.tick_length)
    except ValueError as error:
        raise Infeasible(str(error)) from error
    drag = range_drag(rules, handles.gun_range, cruise_speed + handles.muzzle_speed)
    return dict(
        hit_damage=handles.hit_damage,
        fire_rate=handles.fire_rate,
        muzzle_speed=handles.muzzle_speed,
        bullet_spread_scale=handles.bullet_spread / rules.bullet_spread,
        bullet_drag_scale=drag / rules.bullet_drag,
        bullet_energy_cost=handles.ceiling / handles.shots_per_store,
        max_shield=handles.max_shield,
        shield_recharge_rate_scale=handles.shield_recharge_rate / rules.shield_recharge_rate,
        shield_recharge_delay_scale=handles.shield_recharge_delay / rules.shield_recharge_delay,
        head_on_fraction_scale=handles.head_on_fraction / rules.head_on_fraction,
        hit_radius=handles.hit_radius,
        spawn_energy=handles.spawn_energy,
    )


def combat_handles(rules: Rules, types: ShipTypes, cruise_speed: torch.Tensor) -> dict:
    """The combat handles of a card (fields (B, num_types)) flying at this cruise."""
    drag = rules.bullet_drag * types.bullet_drag_scale
    return dict(
        hit_damage=types.hit_damage,
        fire_rate=types.fire_rate,
        muzzle_speed=types.muzzle_speed,
        bullet_spread=rules.bullet_spread * types.bullet_spread_scale,
        gun_range=gun_range(rules, drag, cruise_speed + types.muzzle_speed),
        shots_per_store=types.max_energy / types.bullet_energy_cost,
        max_shield=types.max_shield,
        shield_recharge_rate=rules.shield_recharge_rate * types.shield_recharge_rate_scale,
        # Held to whole ticks: the float32 card leaves a delay a hair off its tick.
        shield_recharge_delay=torch.round(
            rules.shield_recharge_delay * types.shield_recharge_delay_scale / rules.tick_length
        )
        * rules.tick_length,
        head_on_fraction=rules.head_on_fraction * types.head_on_fraction_scale,
        hit_radius=types.hit_radius,
        spawn_energy=types.spawn_energy,
    )
