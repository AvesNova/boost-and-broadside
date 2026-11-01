"""What happened in one tick. Read by rewards and diagnostics, never by controllers.

Indices are sim indices (team-major), not observation rows.
"""

from typing import ClassVar

import torch

from .tree import contract


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
    SCHEMA_VERSION: ClassVar[int] = 2
