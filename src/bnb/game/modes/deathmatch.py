"""Team deathmatch rewards (skeleton stub; the modes section replaces it).

Per ship: kills - deaths + DAMAGE_WEIGHT * (damage dealt - damage taken) / SHIELD_UNIT.
With friendly fire, damage dealt to an ally and the kill of an ally count against the
shooter, so shooting a teammate is never rewarded. A boundary death has no killer.
"""

import torch

from bnb.contracts import Events

DAMAGE_WEIGHT = 0.25
SHIELD_UNIT = 100.0  # damage counted in fighter shields


def reward(events: Events) -> torch.Tensor:
    """(B, N) reward of each ship (sim order) for one tick; two equal teams."""
    N = events.died.shape[1]
    ships = torch.arange(N, device=events.died.device)
    team = ships // (N // 2)
    sign = torch.where(team[:, None] == team, -1.0, 1.0)  # (shooter, target): -1 for allies
    killed = events.killer[:, :, None] == ships  # (B, target, shooter)
    kills = (killed * sign.T).sum(1)  # by shooter
    dealt, taken = (events.damage * sign).sum(2), events.damage.sum(1)
    return kills - events.died.float() + DAMAGE_WEIGHT * (dealt - taken) / SHIELD_UNIT
