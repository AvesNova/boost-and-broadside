"""The scripted agent's assessment of its fight: dimensionless quantities of each ship and its
target (docs/rewrite/scripted-bfm-design.md B2). Under fog the target's heading, nose rate,
speed and store are its last sighting (no dead reckoning, decided)."""

from dataclasses import dataclass

import torch
from torch import Tensor

TINY = 1e-6


def signed_angle(cos: Tensor, sin: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """The angle from the nose (cos, sin) to the direction (x, y), positive to the left."""
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


@dataclass
class Situation:
    """Each ship against its target, (E, N) each; arbitrary where there is no target."""

    bearing: Tensor  # lambda: the angle off my nose to him, radians, positive to the left
    aspect: Tensor  # alpha: the angle off his tail to me, 0 (I am dead astern) to pi
    angles_off: Tensor  # alpha + |lambda|, 0 (I am on his tail, nose on) to 2 pi (he is on
    # mine): pi minus Shaw's angular advantage, so his advantage is angles_off - pi
    sight_rate: Tensor  # the line of sight's turn rate, in sustained rates, positive to the left
    own_sense: Tensor  # my turn sense: -1 right .. +1 left (tanh of the nose rate over the floor)
    target_sense: Tensor  # his
    flow: Tensor  # own x target sense: +1 the same rotation (two-circle), -1 opposite (one-circle)
    energy_advantage: Tensor  # (my energy - his) / full store: kinetic (v^2 / 2) plus store
    store: Tensor  # my store, fraction of full


def assess(state, fight, turning_floor: Tensor | float) -> Situation:
    """The situation of every ship of `state` against its target in `fight` (an Engagement).
    turning_floor: the nose rate, in sustained rates, below which a ship counts as straight
    (the sense is tanh(nose rate / (floor x sustained rate)))."""
    c = state.constants
    sustained_rate = c["sustained_rate"]
    inverse_store = c["inverse_max_energy"]
    distance = fight.distance.clamp_min(TINY)
    bearing = signed_angle(state.heading_cos, state.heading_sin, fight.offset_x, fight.offset_y)
    aspect = signed_angle(
        fight.target_heading_cos, fight.target_heading_sin, fight.offset_x, fight.offset_y
    ).abs()
    relative_x = fight.target_velocity_x - fight.velocity_x
    relative_y = fight.target_velocity_y - fight.velocity_y
    inverse_distance = 1.0 / distance
    sight_rate = (fight.offset_x * relative_y - fight.offset_y * relative_x) * (
        inverse_distance * inverse_distance
    )
    sense_scale = 1.0 / (turning_floor * sustained_rate)
    own_sense = torch.tanh(state.nose_rate * sense_scale)
    target_sense = torch.tanh(fight.target_nose_rate * sense_scale)
    kinetic = 0.5 * (state.speed * state.speed - fight.target_speed * fight.target_speed)
    store = state.energy * inverse_store
    return Situation(
        bearing=bearing,
        aspect=aspect,
        angles_off=aspect + bearing.abs(),
        sight_rate=sight_rate / sustained_rate,
        own_sense=own_sense,
        target_sense=target_sense,
        flow=own_sense * target_sense,
        energy_advantage=kinetic * inverse_store + store - fight.target_energy * inverse_store,
        store=store,
    )
