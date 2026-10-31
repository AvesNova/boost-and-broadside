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
    """

    damage: torch.Tensor
    boundary_damage: torch.Tensor
    died: torch.Tensor
    fired: torch.Tensor
    spawned: torch.Tensor
    killer: torch.Tensor
    done: torch.Tensor
    SCHEMA_VERSION: ClassVar[int] = 1
