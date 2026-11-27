"""The chaser: a weak, deterministic baseline. It points at its nearest known living enemy and
fires when the nose is on it and it is within gun range. No lead, no soft gates, no boundary
steer, no power management, no memory."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lean.game import card
from lean.game.core import enemy_view, team_view


@dataclass
class ChaserConfig:
    """The chaser's thresholds."""

    gradual_error: float = 0.02  # radians: below this fly straight
    middle_error: float = 0.15  # radians: above this use the middle stick level
    full_error: float = 0.5  # radians: above this use the full stick level
    fire_error: float = 0.05  # radians: fire when the nose is within this of the target
    fire_range_in_gun_range: float = 1.0


def chaser_act(state, team: int, chaser: ChaserConfig | None = None) -> Tensor:
    """The chaser's actions for the rows of `team`: (E, N, 3) long [stick, power, fire];
    other rows, dead rows and rows that do not exist are 0. Power is always coast. Under fog it
    reads its team's view only; the stick's side is the game's (card.STICK_LEVELS)."""
    chaser = chaser or ChaserConfig()
    state = team_view(state, team)
    enemies = enemy_view(state) if state.seen is not None else state
    present = state.exists & state.alive
    controlled = present & (state.team == team)
    enemy = (enemies.exists & enemies.alive & (state.team != team))[:, None, :]
    offset_x = enemies.x[:, None, :] - state.x[:, :, None]
    offset_y = enemies.y[:, None, :] - state.y[:, :, None]
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y)
    nearest = torch.where(enemy, distance, math.inf).min(-1)
    target = nearest.indices[..., None]
    target_x, target_y = offset_x.gather(-1, target)[..., 0], offset_y.gather(-1, target)[..., 0]
    error = torch.atan2(
        state.heading_cos * target_y - state.heading_sin * target_x,
        state.heading_cos * target_x + state.heading_sin * target_y,
    )
    size = error.abs()
    level = (
        (size > chaser.gradual_error).long()
        + (size > chaser.middle_error).long()
        + (size > chaser.full_error).long()
    )
    # The stick index of |level| on the error's side (positive levels turn left).
    signed = torch.where(error > 0.0, level, -level)
    stick = torch.zeros_like(level)
    for index, stick_level in enumerate(card.STICK_LEVELS):
        stick = torch.where(signed == int(stick_level), index, stick)
    ready = state.cooldown <= 0
    in_range = nearest.values <= chaser.fire_range_in_gun_range * state.constants["gun_range"]
    fire = (size < chaser.fire_error) & in_range & ready
    action = torch.stack([stick, torch.zeros_like(stick), fire.long()], -1)
    has_target = controlled & torch.isfinite(nearest.values)
    return action * has_target[..., None]
