"""Team deathmatch rewards (skeleton stub; the modes section replaces it).

Per ship: kills - deaths + DAMAGE_WEIGHT * (damage dealt - damage taken) / SHIELD_UNIT.
Hits are zero-sum across the two teams; a boundary death has no killer and is not.
"""

import torch

from bnb.contracts import Events

DAMAGE_WEIGHT = 0.25
SHIELD_UNIT = 100.0  # damage counted in fighter shields


def reward(events: Events) -> torch.Tensor:
    """(B, N) reward of each ship (sim order) for one tick."""
    N = events.died.shape[1]
    ships = torch.arange(N, device=events.died.device)
    kills = (events.killer[:, :, None] == ships).sum(1).float()  # (B, N) by shooter
    dealt, taken = events.damage.sum(2), events.damage.sum(1)
    return kills - events.died.float() + DAMAGE_WEIGHT * (dealt - taken) / SHIELD_UNIT
