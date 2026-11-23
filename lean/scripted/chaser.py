"""The chaser: a weak, deterministic baseline. It points at its nearest living enemy and
fires when the nose is on it and it is within gun range. No lead, no soft gates, no
boundary steer, no power management, no memory."""

from dataclasses import dataclass

import torch
from torch import Tensor

from lean.game.core import team_view

from .opponent import ScriptedConfig, engage, signed_angle


@dataclass
class ChaserConfig:
    """The chaser's thresholds."""

    fine_error: float = 0.02  # radians: below this fly straight
    sustained_error: float = 0.15  # radians: above this use the sustained stick level
    full_error: float = 0.5  # radians: above this use the full stick level
    fire_error: float = 0.05  # radians: fire when the nose is within this of the target
    fire_range_in_gun_range: float = 1.0


def chaser_act(
    config: ScriptedConfig, state, team: int, chaser: ChaserConfig | None = None
) -> Tensor:
    """The chaser's actions for the rows of `team`: (E, N, 3) long [stick, power, fire];
    other rows, dead rows and rows that do not exist are 0. Power is always coast.
    `config` supplies the stick layout; the gun range is the game's constant."""
    chaser = chaser or ChaserConfig()
    state = team_view(state, team)  # under fog, its team's view only
    fight = engage(state, team)
    error = signed_angle(state.heading_cos, state.heading_sin, fight.offset_x, fight.offset_y)
    size = error.abs()
    level = (
        (size > chaser.fine_error).long()
        + (size > chaser.sustained_error).long()
        + (size > chaser.full_error).long()
    )
    turns_to_even = (error > 0.0) == config.left_is_even
    stick = torch.where(level > 0, 2 * level - 1 + turns_to_even.long(), 0)
    ready = state.cooldown <= 0  # a shot costs no store
    in_range = fight.distance <= chaser.fire_range_in_gun_range * state.constants["gun_range"]
    fire = (size < chaser.fire_error) & in_range & ready
    action = torch.stack([stick, torch.zeros_like(stick), fire.long()], -1)
    return action * fight.has_target[..., None]
