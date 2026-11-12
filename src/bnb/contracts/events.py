"""What happened in one tick. Read by rewards and diagnostics, never by controllers.

Indices are sim indices (team-major), not observation rows.
"""

import math
from typing import ClassVar

import torch

from .tree import contract

# A hit's aspect class, for the statistics (the damage itself follows the smooth aspect of
# `bnb.game.physics.combat.hits`): theta, the angle between the bullet's reversed direction
# and the target's nose (0 head-on), under 45 degrees is head-on, over 135 tail, between
# them the flank. 45 degrees is near the armour's half-strength angle (42 degrees at the
# standard head-on angle).
ASPECTS = ("head_on", "flank", "tail")
ASPECT_BOUNDS = (math.pi / 4, 3 * math.pi / 4)


@contract
class Events:
    """Per tick; (B, N) unless noted.

    damage: (B, N, N) float hit damage from shooter to target. A per-tick transient: reward
        functions consume it and trajectories never store it.
    boundary_damage: float. died, fired, spawned: bool.
    killer: int32 sim index of the shooter whose hit killed the ship, -1 if none.
    done: (B,) bool, the environment reset after this tick.
    shield_lost: the shield a ship alive at tick start actually lost to this tick's damage
        (overkill ignored). shield_restored: the recharge applied this tick.
    objective_progress, objective: the mode's two objective amounts, signed per ship, each
        summing to zero over the environment (Frontline: meter fractions, captures).
    result: (B,) int32, the match result for team 0 (+1 win, -1 loss, 0 draw), nonzero
        only where done.
    enemy_damage_by_aspect: (B, N, len(ASPECTS)) float, the hit damage each ship took from
        the enemy team this tick by the hit's aspect class (friendly fire left out).
    kill_aspect: int32, the aspect class (an index of ASPECTS) of the hit that killed the
        ship (the killer's earliest hit), -1 if none.
    """

    damage: torch.Tensor
    boundary_damage: torch.Tensor
    died: torch.Tensor
    fired: torch.Tensor
    spawned: torch.Tensor
    killer: torch.Tensor
    done: torch.Tensor
    shield_lost: torch.Tensor
    shield_restored: torch.Tensor
    objective_progress: torch.Tensor
    objective: torch.Tensor
    result: torch.Tensor
    enemy_damage_by_aspect: torch.Tensor
    kill_aspect: torch.Tensor
    SCHEMA_VERSION: ClassVar[int] = 3  # 3: the aspect statistics
